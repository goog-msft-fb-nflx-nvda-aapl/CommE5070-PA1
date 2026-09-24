"""Audio Language Model closed-set classification (Qwen2-Audio-7B-Instruct primary;
Audio Flamingo 3 secondary -- its HF loading API is newer/less stable, verify class
names at run time). Runs every sample in a chosen split, >=2 prompt designs, reports
invalid-output rate alongside top-1/top-3/confusion per Description.md.
"""
import argparse
import json
import os
import re
import sys

import librosa
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.data import load_manifest, load_audio_normalized
from src.metrics import summarize

DECADE_ALIASES = {
    "1960s": ["1960s", "60s", "sixties"],
    "1970s": ["1970s", "70s", "seventies"],
    "1980s": ["1980s", "80s", "eighties"],
    "1990s": ["1990s", "90s", "nineties"],
    "2000s": ["2000s", "00s", "two thousands", "noughties"],
    "2010s": ["2010s", "10s", "twenty tens"],
}
MARKET_ALIASES = {
    "US": ["us", "usa", "united states", "america"],
    "UK": ["uk", "united kingdom", "britain", "england"],
    "Brazil": ["brazil", "brazilian"],
    "Spain": ["spain", "spanish"],
    "Germany": ["germany", "german"],
    "Italy": ["italy", "italian"],
}

PROMPTS = {
    "direct": {
        "template": (
            "Listen to this 30-second music excerpt. Which {axis} is it most likely from: "
            "{choices}? Answer with exactly one of those labels and nothing else -- no punctuation, "
            "no explanation."
        ),
        "answer_prefix": "",  # scored candidates: bare label immediately after the prompt
    },
    "cot_then_answer": {
        "template": (
            "Listen to this 30-second music excerpt. Briefly consider its production style, "
            "instrumentation, and mastering, then decide which {axis} it is most likely from: "
            "{choices}. End your response with a final line in exactly this format: "
            "'Answer: <label>' using exactly one of the given labels."
        ),
        # scored candidates use the same "Answer: <label>" format this prompt asks
        # for, rather than a bare label with no reasoning trace -- teacher-forced
        # scoring only conditions on the prompt (no free-form reasoning is actually
        # generated before scoring), so matching the declared answer format keeps
        # the comparison against `direct` fair rather than penalizing a prompt for
        # a format mismatch that has nothing to do with the model's real knowledge.
        "answer_prefix": "Answer: ",
    },
}


def build_alias_pattern(aliases):
    return {canon: re.compile(r"\b(" + "|".join(re.escape(a) for a in al) + r")\b", re.IGNORECASE)
            for canon, al in aliases.items()}


def parse_label(text, canonical_labels, aliases):
    patterns = build_alias_pattern(aliases)
    # prefer text after "Answer:" if present (cot_then_answer format)
    m = re.search(r"answer\s*:\s*(.+)", text, re.IGNORECASE)
    search_text = m.group(1) if m else text
    for canon in canonical_labels:
        if patterns[canon].search(search_text):
            return canon
    for canon in canonical_labels:
        if patterns[canon].search(text):
            return canon
    return None


def _build_generate_and_score(model, processor, device, target_sr=16000):
    """Shared generate()/score_candidates() pair for any HF audio-chat model whose
    processor takes (text=..., audio=[...], sampling_rate=...) and supports
    apply_chat_template with a {"type": "audio"} content block. Verified against
    each model's real processor signature before use -- see WORKLOG.md for the
    Qwen2-Audio kwarg-naming bug this generalization was built to avoid repeating."""

    def _prep(audio, sr, prompt_text):
        if sr != target_sr:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        conversation = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": prompt_text}]}]
        prefix_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        prefix_inputs = processor(text=prefix_text, audio=[audio], sampling_rate=target_sr,
                                   return_tensors="pt", padding=True).to(device)
        return prefix_inputs

    def generate(audio, sr, prompt_text, max_new_tokens=64):
        inputs = _prep(audio, sr, prompt_text)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen = out_ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def score_candidates(audio, sr, prompt_text, candidates, answer_prefix=""):
        """Teacher-forced average log-prob per candidate label -- gives a genuine
        ranked top-k over the closed label set, unlike parsing a single free-form
        generation (which only yields one answer, not a ranking). answer_prefix
        matches the prompt's own declared output format (e.g. "Answer: ")."""
        prefix_inputs = _prep(audio, sr, prompt_text)
        prefix_ids = prefix_inputs["input_ids"]
        prefix_len = prefix_ids.shape[1]
        scores = []
        for cand in candidates:
            cand_text = " " + answer_prefix + cand if answer_prefix else " " + cand
            cand_ids = processor.tokenizer(cand_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            full_ids = torch.cat([prefix_ids, cand_ids], dim=1)
            model_inputs = dict(prefix_inputs)
            model_inputs["input_ids"] = full_ids
            model_inputs["attention_mask"] = torch.ones_like(full_ids)
            with torch.no_grad():
                out = model(**model_inputs)
            logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)
            total = 0.0
            for j in range(cand_ids.shape[1]):
                pos = prefix_len - 1 + j
                tok = full_ids[0, prefix_len + j]
                total += logprobs[pos, tok].item()
            scores.append(total / max(1, cand_ids.shape[1]))
        return np.array(scores)

    return generate, score_candidates


def load_qwen2_audio(device="cuda"):
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    return _build_generate_and_score(model, processor, device, target_sr=16000)


def load_audio_flamingo3(device="cuda"):
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
    model_id = "nvidia/audio-flamingo-3-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    # bfloat16 hits a dtype mismatch between the processor's float32 input_features
    # and the audio tower's bf16 conv weights (RuntimeError: Input type (float) and
    # bias type (c10::BFloat16) should be the same) -- use float32 instead, plenty
    # of VRAM headroom on the H200s regardless.
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    model.eval()
    return _build_generate_and_score(model, processor, device, target_sr=16000)


def load_music_flamingo(device="cuda"):
    """Music Flamingo (round-3 queue item 4, IMPROVEMENT_FINDINGS_ROUND3.md section 4) --
    NVIDIA's music-specialized successor built on the Audio Flamingo 3 backbone,
    arXiv:2511.10289. Structurally analogous to AudioFlamingo3ForConditionalGeneration
    (same author/library pattern) -- verified the class + model id actually exist in
    the installed transformers (5.16.1) before writing this, not assumed from the
    research summary alone."""
    from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor
    model_id = "nvidia/music-flamingo-2601-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    # same float32 rationale as Audio Flamingo 3 above -- try float32 first given the
    # identical dtype-mismatch bug hit there; verified via smoke test before full run.
    model = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    model.eval()
    return _build_generate_and_score(model, processor, device, target_sr=16000)


def run(dataset_key, model_name="qwen2audio", split="validation", device="cuda", out_dir=None):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    aliases = DECADE_ALIASES if dataset_key == "A" else MARKET_ALIASES
    axis = "decade this track was released in" if dataset_key == "A" else "release market of this track"
    choices_str = ", ".join(labels)

    if model_name == "qwen2audio":
        generate_fn, score_fn = load_qwen2_audio(device)
    elif model_name == "audioflamingo3":
        generate_fn, score_fn = load_audio_flamingo3(device)
    elif model_name == "musicflamingo":
        generate_fn, score_fn = load_music_flamingo(device)
    else:
        raise NotImplementedError(f"wire up {model_name} loader (verify current HF class name at run time)")

    rows, _ = load_manifest(dataset_key, split)
    # split-suffix only for non-validation splits, so existing validation-split
    # result paths (results/alm/{model}_{dataset}/) stay exactly as they were --
    # this was previously unsuffixed regardless of split, which meant a --split train
    # run would silently overwrite the validation results at the same path.
    dir_name = f"{model_name}_{dataset_key}" if split == "validation" else f"{model_name}_{dataset_key}_{split}"
    out_dir = out_dir or os.path.join(RESULTS_DIR, "alm", dir_name)
    os.makedirs(out_dir, exist_ok=True)

    for prompt_name, prompt_spec in PROMPTS.items():
        prompt_text = prompt_spec["template"].format(axis=axis, choices=choices_str)
        answer_prefix = prompt_spec["answer_prefix"]
        n_class = len(labels)
        label_to_int = spec["label_to_int"]
        probs, y_true, ids, invalid = [], [], [], 0
        raw_log = []
        for row in tqdm(rows, desc=f"{model_name}/{dataset_key}/{prompt_name}"):
            audio_path = os.path.join(spec["dir"], row["audio_path"])
            y = load_audio_normalized(audio_path)
            # free-form generation: used only to measure how often the model's
            # natural-language answer is unparseable (invalid-output rate).
            raw_text = generate_fn(y, 24000, prompt_text)
            pred_label = parse_label(raw_text, labels, aliases)
            if pred_label is None:
                invalid += 1
            # teacher-forced label scoring: used for the actual top-1/top-3/confusion
            # metrics -- a genuine ranking over the 6 candidates, not a one-hot
            # stand-in from a single free-form guess (which can't yield a real top-3).
            scores = score_fn(y, 24000, prompt_text, labels, answer_prefix=answer_prefix)
            raw_log.append({"sample_id": row["sample_id"], "raw_output": raw_text, "parsed": pred_label,
                             "label_scores": {lab: float(s) for lab, s in zip(labels, scores)}})
            probs.append(np.exp(scores - scores.max()) / np.exp(scores - scores.max()).sum())
            y_true.append(label_to_int[row["label"]])
            ids.append(row["sample_id"])

        probs = np.stack(probs)
        y_true = np.array(y_true)
        metrics = summarize(probs, y_true, n_class, ordinal=spec["ordinal"], label_names=labels)
        metrics["invalid_output_rate"] = invalid / len(rows)
        metrics["prompt"] = prompt_text
        metrics["n_samples"] = len(rows)
        metrics["note"] = ("top1/top3/confusion computed from teacher-forced label log-prob "
                            "scoring (genuine ranking); invalid_output_rate computed separately "
                            "from free-form generation parsing.")

        with open(os.path.join(out_dir, f"{prompt_name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(out_dir, f"{prompt_name}_raw.json"), "w") as f:
            json.dump(raw_log, f, indent=2)
        print(f"[{dataset_key}/{model_name}/{prompt_name}] top1={metrics['top1']:.4f} "
              f"invalid_rate={metrics['invalid_output_rate']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--model", default="qwen2audio")
    p.add_argument("--split", default="validation")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(args.dataset, model_name=args.model, split=args.split, device=args.device)
