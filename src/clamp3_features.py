"""CLaMP 3 (sander-wood/clamp3, arXiv:2502.10362) audio embedding as a frozen probe
(round-5 queue item 6). Trained contrastively across text/audio/MIDI/sheet-music/image
on music *metadata* (M4-RAG, 194 countries/27 languages) -- distinguished from our CLAP
zero-shot negative (sound-event captions, not music metadata); here we probe the audio
tower's global embedding, we don't repeat contrastive zero-shot.

CLaMP3's own audio branch is itself a two-stage pipeline (MERT-v1-95M mean-pooled features
-> CLaMP3's own 12-layer contrastive audio encoder), CLI-oriented, with its own pinned
requirements (transformers==4.40.0, conflicts with this project's main env) -- runs in a
dedicated `clamp3_env` conda env, repo cloned to external/clamp3/. This wrapper shells out
to their clamp3_embd.py (mirroring utils.py's own extract_audio_features orchestration)
pointed directly at our raw dataset audio dir, then re-keys the resulting global embeddings
(768-dim, one per clip, no layer structure) into this project's cache/{sample_id}.npz
convention so train_probe.py's generic run() can consume them unmodified.
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR

CLAMP3_REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "clamp3")
CLAMP3_PYTHON = "/home/jtan/miniconda3/envs/clamp3_env/bin/python"


def run_clamp3_embd(audio_dir, raw_out_dir, cuda_visible_devices="1"):
    if os.path.exists(raw_out_dir):
        shutil.rmtree(raw_out_dir)  # clamp3_embd.py skips extraction entirely if output_dir already exists
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    clamp3_bin = os.path.dirname(CLAMP3_PYTHON)
    env["PATH"] = clamp3_bin + os.pathsep + env.get("PATH", "")  # utils.py's run_command shells out to bare "python"
    cmd = [CLAMP3_PYTHON, "clamp3_embd.py", audio_dir, raw_out_dir, "--get_global"]
    subprocess.run(cmd, cwd=CLAMP3_REPO, env=env, check=True)


def extract(dataset_key, cuda_visible_devices="1", out_dir=None):
    spec = DATASETS[dataset_key]
    audio_dir = os.path.join(spec["dir"], "audio")
    raw_out_dir = os.path.join(CACHE_DIR, "clamp3_raw", dataset_key)
    out_dir = out_dir or os.path.join(CACHE_DIR, "clamp3", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    run_clamp3_embd(audio_dir, raw_out_dir, cuda_visible_devices)

    # clamp3_embd.py writes <basename>.npy per input file, shape (1, 1, 768) with --get_global
    manifest_path = os.path.join(spec["dir"], "manifest.csv")
    import csv
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n_written, n_missing = 0, 0
    for row in rows:
        sample_id = row["sample_id"]
        audio_basename = os.path.splitext(os.path.basename(row["audio_path"]))[0]
        npy_path = os.path.join(raw_out_dir, audio_basename + ".npy")
        cache_path = os.path.join(out_dir, f"{sample_id}.npz")
        if os.path.exists(cache_path):
            continue
        if not os.path.exists(npy_path):
            print(f"[{dataset_key}] MISSING clamp3 output for {sample_id} ({npy_path})")
            n_missing += 1
            continue
        emb = np.load(npy_path).reshape(-1).astype(np.float32)  # (768,)
        assert emb.shape == (768,), f"unexpected embedding shape {emb.shape} for {sample_id}"
        np.savez(cache_path, embedding=emb[None, :], label=row["label"], sample_id=sample_id, split=row["split"])
        n_written += 1

    print(f"[{dataset_key}] clamp3 features: {n_written} written, {n_missing} missing, cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--cuda-visible-devices", default="1")
    args = p.parse_args()
    extract(args.dataset, cuda_visible_devices=args.cuda_visible_devices)
