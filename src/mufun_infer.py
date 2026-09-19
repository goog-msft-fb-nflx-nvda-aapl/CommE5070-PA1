"""MuFun (round-3 lower-priority optional item, revisited on explicit user request) --
`Yi3852/MuFun-Instruct` (arXiv:2508.01178), a ~9B audio-text LLM built on a TinyLLaVA-
style architecture (not the standard HF AutoProcessor pattern the other ALMs in this
project use). ChatGPT's round-3 pick for "most architecturally independent" new ALM.

MuFun's own documented API is `model.chat(tokenizer, prompt, audio_files)`, a free-form
generation-only interface with no exposed teacher-forced scoring path. Inspected
`modeling_mufun.py` directly (downloaded via hf_hub_download, not guessed) to confirm
`chat()` internally just builds `input_ids` + an audio tensor via `TextPreprocess`/
`AudioPreprocess`/`Message` helper classes bundled in the model's own trust_remote_code
files, then calls the model's own `forward()` (standard `CausalLMOutputWithPast`, has
`.logits`) via `generate()`. This module replicates that same input-building step but
calls `forward()` directly with a candidate label's tokens appended, exactly the
teacher-forced label-scoring pattern already used for Qwen2-Audio/AF3/Music Flamingo in
`alm_infer.py` -- avoids the invalid-rate-only limitation a naive free-form-generation-
only evaluation would have had.
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.data import load_manifest
from src.metrics import summarize

MUFUN_MODEL_ID = "Yi3852/MuFun-Instruct"

DECADE_ALIASES = {
    "1960s": ["1960s", "60s", "sixties"], "1970s": ["1970s", "70s", "seventies"],
    "1980s": ["1980s", "80s", "eighties"], "1990s": ["1990s", "90s", "nineties"],
    "2000s": ["2000s", "00s", "two thousands"], "2010s": ["2010s", "10s", "twenty tens"],
}
MARKET_ALIASES = {
    "US": ["us", "usa", "united states"], "UK": ["uk", "united kingdom", "britain"],
    "Brazil": ["brazil"], "Spain": ["spain"], "Germany": ["germany"], "Italy": ["italy"],
}


def load_mufun(device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    import transformers.integrations.accelerate as acc_mod

    # Two real transformers-version-drift bugs in MuFun's own trust_remote_code,
    # found by direct traceback inspection (not assumed) and fixed with the smallest
    # possible monkey-patches, same "verify before trusting" discipline as every
    # other new pipeline this project -- same class of bug as MERT/MuQ/MusicFM, but
    # in model *construction* rather than a specific forward-pass API:
    # 1. MuFun's __init__ calls WhisperForConditionalGeneration.from_pretrained(...)
    #    for its internal audio tower WHILE already inside the outer from_pretrained's
    #    meta-device init context -- current transformers refuses this nested call
    #    ("You are using from_pretrained with a meta device context manager").
    #    check_and_set_device_map is the specific function that raises; no-op it.
    acc_mod.check_and_set_device_map = lambda device_map: device_map
    # 2. MuFun's own TinyLlavaForConditionalGeneration.tie_weights() override has an
    #    outdated signature (no **kwargs) -- current transformers calls it from
    #    multiple internal sites with keyword args it doesn't expect
    #    (recompute_mapping=, missing_keys=), raising TypeError. Patch the class
    #    itself (fetched via the same dynamic-module mechanism transformers uses
    #    internally) so it silently accepts and ignores those kwargs.
    AutoConfig.from_pretrained(MUFUN_MODEL_ID, trust_remote_code=True)
    tinyllava_cls = get_class_from_dynamic_module("modeling_mufun.TinyLlavaForConditionalGeneration", MUFUN_MODEL_ID)
    _orig_tie_weights = tinyllava_cls.tie_weights
    tinyllava_cls.tie_weights = lambda self, *args, **kwargs: _orig_tie_weights(self)

    tokenizer = AutoTokenizer.from_pretrained(MUFUN_MODEL_ID, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MUFUN_MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16)
    # third real bug, same class as the AF3 dtype fix elsewhere in this project: the
    # internally-constructed Whisper audio tower loads its own weights independently
    # (its own from_pretrained call inside MuFun's __init__, with no dtype propagated
    # from the outer bf16 load) and ends up in fp16, mismatched against the bf16 audio
    # features -- force everything to a single consistent dtype after loading.
    model = model.to(device=device, dtype=torch.float32).eval()
    # the model's own remote-code module already imported these classes at its top
    # level (`from .text_preprocess import TextPreprocess`, etc.) -- pulling them off
    # the already-loaded module avoids hardcoding the dynamic transformers_modules
    # cache path, and guarantees the exact same classes `chat()` itself would use.
    mod = importlib.import_module(model.__class__.__module__)
    return model, tokenizer, mod


def _build_inputs(model, tokenizer, mod, audio_path, prompt_text):
    text_processor = mod.TextPreprocess(tokenizer, "qwen2_instruct")
    audio_processor = mod.AudioPreprocess(model.vision_tower._image_processor, model.config)
    msg = mod.Message()
    audio_tensor, audio_size = mod.load_audios(audio_processor, audio_path, None)
    # load_audios returns a list -- the model's own generate() (line 338-339 of
    # modeling_mufun.py) does `images = torch.cat(images, dim=0)` before calling
    # prepare_inputs_labels_for_multimodal, but plain forward() does not do this
    # conversion itself and assumes an already-concatenated tensor -- do it once
    # here so both the generation and teacher-forced-scoring paths get a tensor.
    if isinstance(audio_tensor, list) and audio_tensor:
        audio_tensor = torch.cat(audio_tensor, dim=0)
    prompt = prompt_text
    if audio_tensor is not None and "<audio>" not in prompt:
        prompt = "<audio>\n" + prompt
    msg.add_message(prompt)
    result = text_processor(msg.messages, mode="eval")
    input_ids = result["input_ids"].unsqueeze(0).to(model.device)
    return input_ids, audio_tensor, audio_size


def generate(model, tokenizer, mod, audio_path, prompt_text, max_new_tokens=32):
    input_ids, audio_tensor, audio_size = _build_inputs(model, tokenizer, mod, audio_path, prompt_text)
    with torch.inference_mode():
        out_ids = model.generate(
            input_ids, images=audio_tensor, do_sample=False, max_new_tokens=max_new_tokens,
            use_cache=True, pad_token_id=tokenizer.eos_token_id,
            image_sizes=[audio_size] if audio_tensor is not None else None,
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


def score_candidates(model, tokenizer, mod, audio_path, prompt_text, candidates):
    """Teacher-forced average log-prob per candidate label, same methodology as
    alm_infer.py's score_candidates -- a genuine ranking over all 6 classes, not a
    one-hot stand-in from a single free-form generation."""
    input_ids, audio_tensor, audio_size = _build_inputs(model, tokenizer, mod, audio_path, prompt_text)
    prefix_len = input_ids.shape[1]
    scores = []
    for cand in candidates:
        cand_ids = tokenizer(" " + cand, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        full_ids = torch.cat([input_ids, cand_ids], dim=1)
        with torch.no_grad():
            out = model(input_ids=full_ids, images=audio_tensor,
                        image_sizes=[audio_size] if audio_tensor is not None else None)
        logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)
        total = 0.0
        for j in range(cand_ids.shape[1]):
            pos = prefix_len - 1 + j
            tok = full_ids[0, prefix_len + j]
            total += logprobs[pos, tok].item()
        scores.append(total / max(1, cand_ids.shape[1]))
    return np.array(scores)


def parse_label(text, canonical_labels, aliases):
    import re
    text_lower = text.lower()
    for canon in canonical_labels:
        for alias in aliases[canon]:
            if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
                return canon
    return None


def run(dataset_key, split="validation", device="cuda", out_dir=None):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    label_to_int = spec["label_to_int"]
    aliases = DECADE_ALIASES if dataset_key == "A" else MARKET_ALIASES
    axis = "decade this track was released in" if dataset_key == "A" else "release market of this track"
    choices_str = ", ".join(labels)
    prompt_text = (f"Listen to this music excerpt. Which {axis} is it most likely from: {choices_str}? "
                    "Answer with exactly one of those labels and nothing else.")

    model, tokenizer, mod = load_mufun(device)

    rows, _ = load_manifest(dataset_key, split)
    dirname = f"mufun_{dataset_key}" if split == "validation" else f"mufun_{dataset_key}_{split}"
    out_dir = out_dir or os.path.join(RESULTS_DIR, "alm", dirname)
    os.makedirs(out_dir, exist_ok=True)

    n_class = len(labels)
    probs, y_true, ids, invalid = [], [], [], 0
    raw_log = []
    for row in tqdm(rows, desc=f"mufun/{dataset_key}/{split}"):
        audio_path = os.path.join(spec["dir"], row["audio_path"])
        raw_text = generate(model, tokenizer, mod, audio_path, prompt_text)
        pred_label = parse_label(raw_text, labels, aliases)
        if pred_label is None:
            invalid += 1
        scores = score_candidates(model, tokenizer, mod, audio_path, prompt_text, labels)
        raw_log.append({"sample_id": row["sample_id"], "raw_output": raw_text, "parsed": pred_label,
                         "label_scores": {lab: float(s) for lab, s in zip(labels, scores)}})
        p = np.exp(scores - scores.max())
        probs.append(p / p.sum())
        y_true.append(label_to_int[row["label"]])
        ids.append(row["sample_id"])

    probs = np.stack(probs)
    y_true = np.array(y_true)
    metrics = summarize(probs, y_true, n_class, ordinal=spec["ordinal"], label_names=labels)
    metrics["invalid_output_rate"] = invalid / len(rows)
    metrics["n_samples"] = len(rows)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "raw.json"), "w") as f:
        json.dump(raw_log, f, indent=2)
    print(f"[{dataset_key}/mufun/{split}] top1={metrics['top1']:.4f} top3={metrics['top3']:.4f} "
          f"invalid_rate={metrics['invalid_output_rate']:.4f}")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(args.dataset, split=args.split, device=args.device)
