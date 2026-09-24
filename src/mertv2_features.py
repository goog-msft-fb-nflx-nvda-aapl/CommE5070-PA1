"""Extract MERT-v2-30s (round-5 top recommendation, `round5_lecture_grounded_research.md`
section 2) frame-level hidden states, time-mean-pool per layer (using the model's own
`feature_attention_mask`, per its documented usage snippet), cache to disk. 24kHz native,
trained on exactly our 30s clip length (no resampling/windowing needed, unlike MERT-v1's
5s training context). 24 layers, 1024-dim.

Unlike every other custom-code model integrated this project (MERT-v1, MuQ, MusicFM,
CultureMERT, MuFun -- all needed at least one transformers-version-drift workaround),
MERT-v2-30s's `output_hidden_states=True` worked correctly out of the box on the first
smoke test (verified via a real forward pass before writing this extraction loop, not
assumed) -- no forward-hook workaround needed here.
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

MERTV2_MODEL_ID = "m-a-p/MERT-v2-30s"


def load_mertv2(device="cuda"):
    from transformers import AutoFeatureExtractor, AutoModel
    processor = AutoFeatureExtractor.from_pretrained(MERTV2_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MERTV2_MODEL_ID, trust_remote_code=True).eval().to(device)
    return model, processor


def extract(dataset_key, device="cuda", out_dir=None, splits=("train", "validation", "test")):
    out_dir = out_dir or os.path.join(CACHE_DIR, "mertv2_30s", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    model, processor = load_mertv2(device)

    spec = DATASETS[dataset_key]
    for split in splits:
        rows, _ = load_manifest(dataset_key, split)
        for row in tqdm(rows, desc=f"{dataset_key}/{split}"):
            sample_id = row["sample_id"]
            cache_path = os.path.join(out_dir, f"{sample_id}.npz")
            if os.path.exists(cache_path):
                continue
            audio_path = os.path.join(spec["dir"], row["audio_path"])
            y = load_audio_normalized(audio_path)
            inputs = processor(y, sampling_rate=processor.sampling_rate, return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model(**inputs, output_hidden_states=True)
            mask = output.feature_attention_mask[..., None].float()
            layer_means = torch.stack([
                (h * mask).sum(1) / mask.sum(1).clamp_min(1) for h in output.hidden_states
            ]).squeeze(1)  # (24, 1024)
            assert layer_means.shape == (24, 1024), f"unexpected shape {layer_means.shape}"
            np.savez(cache_path, embedding=layer_means.cpu().numpy().astype(np.float32),
                     label=row["label"], split=split, sample_id=sample_id)
    print(f"[{dataset_key}] MERT-v2-30s features cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    extract(args.dataset, device=args.device)
