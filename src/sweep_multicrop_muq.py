"""Multi-excerpt/TTA sweep, MuQ encoder (post-round-3 push). The original
sweep_multicrop.py only ever tested this with MERT (before MuQ existed in this
project). MERT showed a real, measured gain on Task A from multi-crop TTA
(n_crops=3/5 beat the 30s baseline) -- worth checking whether the same technique
helps MuQ, our current best frozen-probe encoder for both tasks, since this exact
combination has never been tried.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR
from src.muq_features import extract, cache_is_complete
from src.train_probe import run

# each dataset's established best MuQ frozen-probe config (layer/classifier/pca),
# matching af3_stack.py's FROZEN_CONFIGS -- the full-clip (30s) baseline this sweep
# is measured against.
WINNING_CONFIG = {
    "A": {"layer": 1, "classifier": "svm", "pca_dim": None},
    "B": {"layer": 2, "classifier": "logreg", "pca_dim": None},
}


def sweep(dataset_key, crop_seconds=10, n_crops_list=(1, 3, 5, 8), device="cuda"):
    cfg = WINNING_CONFIG[dataset_key]
    rows = []
    for n_crops in n_crops_list:
        encoder_name = f"muq_large_msd_seg{crop_seconds}s_{n_crops}crop"
        cache_dir = os.path.join("cache", encoder_name, dataset_key)
        if not cache_is_complete(dataset_key, cache_dir):
            extract(dataset_key, device=device, crop_seconds=crop_seconds, n_crops=n_crops)

        out_dir = os.path.join(RESULTS_DIR, f"multicrop_sweep_muq_{dataset_key}", f"{crop_seconds}s_{n_crops}crop")
        metrics, _fitted = run(dataset_key, layer=cfg["layer"], classifier=cfg["classifier"],
                                pca_dim=cfg["pca_dim"], encoder_name=encoder_name, out_dir=out_dir)
        row = {"crop_seconds": crop_seconds, "n_crops": n_crops, "top1": metrics["top1"], "top3": metrics["top3"]}
        if "mean_abs_decade_error" in metrics:
            row["mean_abs_decade_error"] = metrics["mean_abs_decade_error"]
        rows.append(row)
        print(f"[{dataset_key}] crop={crop_seconds}s n_crops={n_crops} top1={metrics['top1']:.4f} top3={metrics['top3']:.4f}")

    summary_dir = os.path.join(RESULTS_DIR, f"multicrop_sweep_muq_{dataset_key}")
    os.makedirs(summary_dir, exist_ok=True)
    csv_path = os.path.join(summary_dir, f"summary_{crop_seconds}s.csv")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{dataset_key}] MuQ multi-crop sweep summary -> {csv_path}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--crop-seconds", type=int, default=10)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    sweep(args.dataset, crop_seconds=args.crop_seconds, device=args.device)
