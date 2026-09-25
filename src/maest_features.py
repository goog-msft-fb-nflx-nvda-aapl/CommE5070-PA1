"""MAEST (mtg-upf/discogs-maest-30s-pw-129e, arXiv:2309.16418) frozen probe, both tasks --
a Discogs-*supervised* embedding (pretrained on style classification over 400 Discogs styles),
the same metadata universe our own dataset's labels (Discogs-VI) are drawn from. Distinguished
from every other encoder tried this project (all self-supervised) -- this is the one encoder
whose pretraining signal is directly, explicitly about Discogs taxonomy.

Official API (`from maest import get_maest`, no HF AutoModel path): accepts a raw 16kHz
waveform directly (1D), model does its own resampling-free mel-spectrogram internally.
`model(data, transformer_block=i)` returns (logits, embeddings) where embeddings is a
2304-dim vector (CLS+DIST+avg-of-rest tokens stacked, ViT-base hidden size 768x3) for
block i (0-11, 12 blocks total) -- pooled per-block, unlike our other encoders where we do
our own mean-pooling over a raw per-token hidden-state sequence.

Real integration issues hit and fixed: (1) MAEST's declared `timm~=0.9` pin conflicts with
PupuM2D's need for timm>=1.0 (EVA rope support) -- runs in a dedicated `maest_env` conda
env. (2) `pip install -e .` failed with `ModuleNotFoundError: No module named 'pkg_resources'`
-- recent setuptools (>=81) dropped it; pinned `setuptools<81` to restore it (a real,
now-common transformers-adjacent-ecosystem compatibility issue, not MAEST-specific).
(3) `model.to(device)` doesn't move the internal melspectrogram's STFT window buffer to
the same device, causing `RuntimeError: stft input and window must be on the same device`
on any CUDA device -- given the model is small (86M params) and fast even on CPU (~0.4s/clip,
verified via smoke test), ran CPU-only rather than patching the library's device handling.
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

MAEST_SR = 16000
N_BLOCKS = 12
# MAEST's transformer_block=i argument early-exits after running blocks 0..i (verified by
# reading forward_features()'s loop -- NOT a single forward pass with all hidden states
# returned, unlike our other encoders' AutoModel(output_hidden_states=True) path). Calling
# all 12 blocks per clip would cost ~78 "block-units" (1+2+...+12) vs 2 units for just the
# two we actually want -- extract only the paper's reported-best layer (block 6, "layer 7")
# and the last block (11), not a full sweep, to keep extraction time reasonable.
BLOCKS_TO_EXTRACT = [6, 11]


def load_maest():
    from maest import get_maest
    model = get_maest(arch="discogs-maest-30s-pw-129e").eval()
    return model


def extract(dataset_key, out_dir=None):
    import librosa

    spec = DATASETS[dataset_key]
    out_dir = out_dir or os.path.join(CACHE_DIR, "maest", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    model = load_maest()

    rows, _ = load_manifest(dataset_key)
    for row in tqdm(rows, desc=f"{dataset_key}/maest"):
        sample_id = row["sample_id"]
        cache_path = os.path.join(out_dir, f"{sample_id}.npz")
        if os.path.exists(cache_path):
            continue
        audio_path = os.path.join(spec["dir"], row["audio_path"])
        y = load_audio_normalized(audio_path)
        y16 = librosa.resample(y.astype(np.float32), orig_sr=24000, target_sr=MAEST_SR)
        y_t = torch.from_numpy(y16).unsqueeze(0)
        block_embs = []
        with torch.no_grad():
            for block in BLOCKS_TO_EXTRACT:
                _, emb = model(y_t, transformer_block=block)
                block_embs.append(emb.squeeze(0).numpy())
        arr = np.stack(block_embs).astype(np.float32)  # (len(BLOCKS_TO_EXTRACT), 2304)
        np.savez(cache_path, embedding=arr, label=row["label"], sample_id=sample_id, split=row["split"])
    print(f"[{dataset_key}] MAEST features cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    args = p.parse_args()
    extract(args.dataset)
