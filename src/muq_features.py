"""Extract MuQ (OpenMuQ/MuQ-large-msd-iter) frame-level hidden states, time-mean-pool
per layer, cache to disk. Native 24kHz -- this dataset needs no resampling for MuQ.

Two real library bugs found and fixed here (both caught by direct verification, not
assumed to work just because the docs say so -- see reference_alm_pipeline_verification
memory):
1. MuQ's internal Wav2Vec2ConformerEncoder config is a stale EasyDict (transformers_version
   4.19.0.dev0 baked into the checkpoint) missing `_attn_implementation`, which our current
   transformers (5.16.1) requires -- AttributeError on the very first forward call. Fixed by
   patching `_attn_implementation = "eager"` onto that config object after loading.
2. Even after that fix, `output_hidden_states=True` on the conformer silently doesn't
   populate the output (KeyError: 'hidden_states') -- the exact same class of transformers-
   version-drift bug hit with MERT. Fixed the same way: forward hooks on each of the 12
   conformer layers (+ the pre-layer embedding) instead of trusting output_hidden_states.
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

MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"


def load_muq(device="cuda"):
    from muq import MuQ
    muq = MuQ.from_pretrained(MUQ_MODEL_ID)
    # bug 1: stale EasyDict config from the checkpoint predates transformers' current
    # attention-masking API, which requires this attribute to exist.
    muq.model.conformer.config._attn_implementation = "eager"
    muq = muq.to(device).eval()
    return muq


def _install_layer_hooks(muq):
    """bug 2 fix: capture the embedding-stage output (pre-conformer, via the
    feature-projection module `muq.model.conv`... actually the conformer's own
    pos-embed-processed input isn't cleanly hookable pre-layers, so instead hook
    the 12 conformer transformer layers only -- gives 12 layer representations,
    which is what MuQ's own paper analyzes layer-wise anyway (12-layer Conformer)."""
    captured = []

    def make_hook():
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured.append(hs.detach())
        return hook

    handles = [layer.register_forward_hook(make_hook()) for layer in muq.model.conformer.layers]
    return captured, handles


def cache_is_complete(dataset_key, cache_dir):
    if not os.path.isdir(cache_dir):
        return False
    spec = DATASETS[dataset_key]
    total_expected = sum(1 for _ in open(os.path.join(spec["dir"], "manifest.csv"))) - 1
    n_cached = len([f for f in os.listdir(cache_dir) if f.endswith(".npz")])
    return n_cached >= total_expected


def _run_one(muq, captured, wav_np, device):
    wav = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
    captured.clear()
    with torch.no_grad():
        try:
            muq(wav, output_hidden_states=True)
        except KeyError as e:
            # MuQ's own wrapper crashes reading out["hidden_states"] from the
            # conformer's return value (transformers-version drift, see module
            # docstring) -- but that happens *after* the conformer's forward
            # pass (and our hooks) already completed, so `captured` is valid;
            # only re-raise if hooks genuinely didn't fire (a different bug).
            if str(e) != "'hidden_states'" or len(captured) == 0:
                raise
    assert len(captured) == 12, f"expected 12 conformer layer outputs, got {len(captured)}"
    return torch.stack([h.mean(dim=1).squeeze(0) for h in captured])  # (12, 1024)


def extract(dataset_key, device="cuda", out_dir=None, splits=("train", "validation", "test"),
            crop_seconds=None, n_crops=1):
    """crop_seconds=None: full 30s clip (default, cached under muq_large_msd/{dataset}).
    Otherwise: n_crops overlapping crop_seconds-long segments per clip (via multi_crop),
    layer-means averaged across crops -- multi-excerpt TTA, mirroring
    mert_features.py's identical crop_seconds/n_crops mechanism (never applied to MuQ
    before now), cached under a length/crop-count-specific subdir."""
    if out_dir is None:
        if crop_seconds is None:
            out_dir = os.path.join(CACHE_DIR, "muq_large_msd", dataset_key)
        else:
            out_dir = os.path.join(CACHE_DIR, f"muq_large_msd_seg{crop_seconds}s_{n_crops}crop", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    muq = load_muq(device)
    captured, handles = _install_layer_hooks(muq)

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
                layer_means = _run_one(muq, captured, y, device)
            else:
                from src.data import multi_crop
                crops = multi_crop(y, crop_seconds, n_crops)
                per_crop = torch.stack([_run_one(muq, captured, c, device) for c in crops])
                layer_means = per_crop.mean(dim=0)
            np.savez(cache_path, embedding=layer_means.cpu().numpy().astype(np.float32),
                     label=row["label"], split=split, sample_id=sample_id)
    for h in handles:
        h.remove()
    print(f"[{dataset_key}] MuQ features ({crop_seconds or 'full'}s x{n_crops}) cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--crop-seconds", type=float, default=None)
    p.add_argument("--n-crops", type=int, default=1)
    args = p.parse_args()
    extract(args.dataset, device=args.device, crop_seconds=args.crop_seconds, n_crops=args.n_crops)
