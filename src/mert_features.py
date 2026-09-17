"""Extract MERT-v1-330M frame-level hidden states, time-mean-pool per layer, cache to disk.
Native 24kHz — this dataset needs no resampling for MERT.
Caches per-clip a (n_layers, hidden_dim) array of layer-pooled means so the
classifier/probe training script can sweep layers or a learned weighted sum
without re-running the encoder.
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR, SAMPLE_RATE
from src.data import load_manifest, load_audio_normalized, multi_crop

MERT_MODEL_ID = "m-a-p/MERT-v1-330M"


def cache_is_complete(dataset_key, cache_dir):
    """Guards against a partial/interrupted extraction being mistaken for a
    finished one (non-empty != complete) -- compares cached .npz count against
    the full manifest row count (train+validation+test)."""
    if not os.path.isdir(cache_dir):
        return False
    from src.config import DATASETS as _DATASETS
    spec = _DATASETS[dataset_key]
    total_expected = sum(1 for _ in open(os.path.join(spec["dir"], "manifest.csv"))) - 1  # minus header
    n_cached = len([f for f in os.listdir(cache_dir) if f.endswith(".npz")])
    return n_cached >= total_expected


def _install_layer_hooks(model):
    """MERT's trust_remote_code forward doesn't reliably populate
    out.hidden_states under transformers 5.x (verified: returns None even with
    output_hidden_states=True/return_dict=True -- likely an internal Hubert API
    drift the pinned custom modeling_MERT.py wasn't updated for). Capture each
    of the 24 encoder.layers' outputs plus the feature_projection (embedding
    layer) directly via forward hooks instead, which is version-independent."""
    captured = []

    def make_hook():
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured.append(hs.detach())
        return hook

    handles = [model.feature_projection.register_forward_hook(make_hook())]
    for layer in model.encoder.layers:
        handles.append(layer.register_forward_hook(make_hook()))
    return captured, handles


def _run_one(model, captured, wav_np, device):
    wav = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
    captured.clear()
    with torch.no_grad():
        model(wav)
    assert len(captured) == 25, f"expected 25 layer outputs (1 embed + 24 transformer), got {len(captured)}"
    return torch.stack([h.mean(dim=1).squeeze(0) for h in captured])  # (25, 1024)


def extract(dataset_key, device="cuda", model_id=MERT_MODEL_ID, out_dir=None, splits=("train", "validation", "test"),
            crop_seconds=None, n_crops=1):
    """crop_seconds=None: full 30s clip (default, original behaviour, cached under
    mert_v1_330m/{dataset}). Otherwise: n_crops overlapping crop_seconds-long segments
    per clip (via multi_crop), layer-means averaged across crops -- used for the
    segment-length / multi-excerpt required experiments, cached under a
    length/crop-count-specific subdir so it never collides with the full-clip cache."""
    if out_dir is None:
        if crop_seconds is None:
            out_dir = os.path.join(CACHE_DIR, "mert_v1_330m", dataset_key)
        else:
            out_dir = os.path.join(CACHE_DIR, f"mert_v1_330m_seg{crop_seconds}s_{n_crops}crop", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    captured, handles = _install_layer_hooks(model)

    spec = DATASETS[dataset_key]
    for split in splits:
        rows, _ = load_manifest(dataset_key, split)
        for row in tqdm(rows, desc=f"{dataset_key}/{split}/{crop_seconds or 'full'}s"):
            sample_id = row["sample_id"]
            cache_path = os.path.join(out_dir, f"{sample_id}.npz")
            if os.path.exists(cache_path):
                continue
            audio_path = os.path.join(spec["dir"], row["audio_path"])
            y = load_audio_normalized(audio_path)
            if crop_seconds is None:
                layer_means = _run_one(model, captured, y, device)
            else:
                crops = multi_crop(y, crop_seconds, n_crops)
                per_crop = torch.stack([_run_one(model, captured, c, device) for c in crops])
                layer_means = per_crop.mean(dim=0)
            np.savez(cache_path, embedding=layer_means.cpu().numpy().astype(np.float32),
                     label=row["label"], split=split, sample_id=sample_id)
    for h in handles:
        h.remove()
    print(f"[{dataset_key}] MERT features ({crop_seconds or 'full'}s x{n_crops}) cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--crop-seconds", type=float, default=None)
    p.add_argument("--n-crops", type=int, default=1)
    args = p.parse_args()
    extract(args.dataset, device=args.device, crop_seconds=args.crop_seconds, n_crops=args.n_crops)
