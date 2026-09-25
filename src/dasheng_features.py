"""Dasheng-1.2B (mispeech/dasheng-1.2B) frozen probe, both tasks -- old backlog item
("MiDashengLM/Dasheng as a diverse frozen extractor"), explicitly deprioritized twice in
this project's own notes based on the lecture's GTZAN comparison table (81.4 vs MuQ's
83.8) -- per this project's standing "no assumed won't-work" practice (don't prune a
cheaply-testable candidate on predicted reasoning), actually run rather than left as a
permanent skip: a static external benchmark number is not the same as our own measurement
on our own artist-disjoint, editorial-label tasks.

General-purpose audio SSL encoder (1.2B params, 272k hours of speech+music+environmental
audio -- NOT music-specialized, unlike every encoder that's beaten MuQ/MERT-v2 so far).
Standard transformers AutoModel + AutoFeatureExtractor (trust_remote_code=True), 16kHz,
handles our full 30s clip natively (no windowing). Only the final-layer mean-pooled
embedding (`logits`, 1536-dim) is exposed without forward hooks -- `output_hidden_states`
isn't supported by the custom forward() (a real, minor kwarg-compatibility gap, not
pursued further given the item's already-low priority) -- so this is a single global
embedding, no layer sweep, matching the CLaMP3/PupuM2D convention.
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR
from src.data import load_manifest, load_audio_normalized

DASHENG_SR = 16000
DASHENG_MODEL_ID = "mispeech/dasheng-1.2B"


def load_dasheng(device="cuda"):
    from transformers import AutoModel, AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained(DASHENG_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(DASHENG_MODEL_ID, outputdim=None, trust_remote_code=True).eval().to(device)
    return model, fe


def extract(dataset_key, device="cuda", out_dir=None):
    import librosa

    spec = DATASETS[dataset_key]
    out_dir = out_dir or os.path.join(CACHE_DIR, "dasheng", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    model, fe = load_dasheng(device)

    rows, _ = load_manifest(dataset_key)
    for row in tqdm(rows, desc=f"{dataset_key}/dasheng"):
        sample_id = row["sample_id"]
        cache_path = os.path.join(out_dir, f"{sample_id}.npz")
        if os.path.exists(cache_path):
            continue
        audio_path = os.path.join(spec["dir"], row["audio_path"])
        y = load_audio_normalized(audio_path)
        y16 = librosa.resample(y.astype(np.float32), orig_sr=24000, target_sr=DASHENG_SR)
        inputs = fe(y16, sampling_rate=DASHENG_SR, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        emb = out.logits.squeeze(0).cpu().numpy().astype(np.float32)  # (1536,)
        np.savez(cache_path, embedding=emb[None, :], label=row["label"], sample_id=sample_id, split=row["split"])
    print(f"[{dataset_key}] Dasheng features cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    extract(args.dataset, device=args.device)
