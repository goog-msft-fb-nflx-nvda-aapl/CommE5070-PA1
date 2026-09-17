"""Demucs-based mixture/vocal/accompaniment stem separation for Task 2's required
comparison. Uses htdemucs's --two-stems=vocals mode: outputs vocals.wav + no_vocals.wav
(= accompaniment) per clip. Run only on Dataset B (the assignment's Task 2 requirement).
Batches many files per demucs invocation (avoids reloading the model per clip)."""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS
from src.data import load_manifest


def separate_all(dataset_key="B", out_dir=None, model="htdemucs", device="cuda", batch_size=32):
    spec = DATASETS[dataset_key]
    out_dir = out_dir or os.path.join(os.path.dirname(spec["dir"]), f"{dataset_key}_stems")
    os.makedirs(out_dir, exist_ok=True)

    rows, _ = load_manifest(dataset_key)
    pending = []
    for row in rows:
        sample_id = row["sample_id"]
        vocals_path = os.path.join(out_dir, model, sample_id, "vocals.wav")
        if os.path.exists(vocals_path):
            continue
        pending.append(os.path.join(spec["dir"], row["audio_path"]))

    print(f"[{dataset_key}] {len(pending)}/{len(rows)} clips need separation")
    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        cmd = ["python3", "-m", "demucs", "--two-stems", "vocals", "-n", model,
               "-d", device, "-o", out_dir, "--filename", "{track}/{stem}.{ext}"] + chunk
        subprocess.run(cmd, check=True)
        print(f"[{dataset_key}] batch {i // batch_size + 1}/{(len(pending) - 1) // batch_size + 1} done "
              f"({min(i + batch_size, len(pending))}/{len(pending)} clips)")

    print(f"[{dataset_key}] stems written to {out_dir}/{model}/<sample_id>/{{vocals,no_vocals}}.wav")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="B", choices=["A", "B"])
    p.add_argument("--model", default="htdemucs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    separate_all(args.dataset, model=args.model, device=args.device, batch_size=args.batch_size)
