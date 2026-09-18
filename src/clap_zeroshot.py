"""LAION-CLAP zero-shot classification (post-round-3 push, single-source idea from
Gemini's round-3 response, worth testing on its own merits: `laion/clap-htsat-fused`,
arXiv:2211.06687). Genuinely different mechanism from every ALM tried so far
(Qwen2-Audio/AF3/Music Flamingo all use teacher-forced label-token scoring on a
generative LLM) -- CLAP is a dual-encoder audio-text CONTRASTIVE model, trained
specifically for zero-shot classification via audio-text cosine similarity, not
next-token prediction. Worth checking whether this different training objective
surfaces signal the generative ALMs and SSL probes both missed.

Standard `transformers` ClapModel/ClapProcessor (native support, not custom code).
48kHz native (resampled from our 24kHz clips), native ~10s window with the "fused"
variant's built-in longer-input handling (that's what "-fused" means in the model
name -- verified this is the intended way to feed it 30s audio, not assumed).
"""
import argparse
import json
import os
import sys

import librosa
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.data import load_manifest, load_audio_normalized
from src.metrics import summarize

CLAP_MODEL_ID = "laion/clap-htsat-fused"
CLAP_SR = 48000

DECADE_PROMPTS = {
    "1960s": "a music recording from the 1960s",
    "1970s": "a music recording from the 1970s",
    "1980s": "a music recording from the 1980s",
    "1990s": "a music recording from the 1990s",
    "2000s": "a music recording from the 2000s",
    "2010s": "a music recording from the 2010s",
}
MARKET_PROMPTS = {
    "US": "a music recording released in the United States",
    "UK": "a music recording released in the United Kingdom",
    "Brazil": "a music recording released in Brazil",
    "Spain": "a music recording released in Spain",
    "Germany": "a music recording released in Germany",
    "Italy": "a music recording released in Italy",
}


def load_clap(device="cuda"):
    from transformers import ClapModel, ClapProcessor
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device)
    model.eval()
    return model, processor


def run(dataset_key, split="validation", device="cuda", temperature=None, out_dir=None):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    label_to_int = spec["label_to_int"]
    n_class = len(labels)
    prompts = DECADE_PROMPTS if dataset_key == "A" else MARKET_PROMPTS
    prompt_texts = [prompts[lab] for lab in labels]

    model, processor = load_clap(device)

    with torch.no_grad():
        text_inputs = processor(text=prompt_texts, return_tensors="pt", padding=True).to(device)
        # get_text_features returns a BaseModelOutputWithPooling in this transformers
        # version, not a plain tensor as its own docstring example implies -- caught by
        # a smoke test (AttributeError), verified via source inspection: .pooler_output
        # is the already-projected, already-L2-normalized embedding, so no extra
        # normalization step is needed (or correct) here.
        text_features = model.get_text_features(**text_inputs).pooler_output
        # CLAP's own learned logit scale, used exactly as in its own zero-shot examples,
        # rather than an arbitrarily chosen temperature -- verified this attribute
        # exists on the loaded model before relying on it.
        logit_scale = model.logit_scale_a.exp().item() if temperature is None else 1.0 / temperature

    rows, _ = load_manifest(dataset_key, split)
    out_dir = out_dir or os.path.join(RESULTS_DIR, "alm", f"clap_{dataset_key}" if split == "validation" else f"clap_{dataset_key}_{split}")
    os.makedirs(out_dir, exist_ok=True)

    probs, y_true, ids = [], [], []
    raw_log = []
    for row in tqdm(rows, desc=f"clap/{dataset_key}/{split}"):
        audio_path = os.path.join(spec["dir"], row["audio_path"])
        y = load_audio_normalized(audio_path)
        y_resampled = librosa.resample(y.astype(np.float32), orig_sr=24000, target_sr=CLAP_SR)
        with torch.no_grad():
            audio_inputs = processor(audio=y_resampled, sampling_rate=CLAP_SR, return_tensors="pt").to(device)
            audio_features = model.get_audio_features(**audio_inputs).pooler_output
            sims = (logit_scale * audio_features @ text_features.T).squeeze(0)
            p = torch.softmax(sims, dim=-1).cpu().numpy()
        probs.append(p)
        y_true.append(label_to_int[row["label"]])
        ids.append(row["sample_id"])
        raw_log.append({"sample_id": row["sample_id"], "label_scores": {lab: float(s) for lab, s in zip(labels, sims.cpu().numpy())}})

    probs = np.stack(probs)
    y_true = np.array(y_true)
    metrics = summarize(probs, y_true, n_class, ordinal=spec["ordinal"], label_names=labels)
    metrics["prompts"] = prompt_texts
    metrics["n_samples"] = len(rows)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "raw.json"), "w") as f:
        json.dump(raw_log, f, indent=2)
    print(f"[{dataset_key}/clap/{split}] top1={metrics['top1']:.4f} top3={metrics['top3']:.4f}")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(args.dataset, split=args.split, device=args.device)
