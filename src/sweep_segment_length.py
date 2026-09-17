"""Required experiment: compare 5/10/15/30s input length. For each dataset, uses
its own winning MERT layer+classifier config (from the earlier layer-sweep deep-dive)
and re-extracts embeddings at each length with a single center-ish crop (n_crops=1,
pure length effect -- multi-excerpt/TTA is a separate axis, see --n-crops)."""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR
from src.mert_features import extract, cache_is_complete
from src.train_probe import run

WINNING_CONFIG = {
    "A": {"layer": 4, "classifier": "logreg", "pca_dim": 128},
    "B": {"layer": 7, "classifier": "logreg", "pca_dim": None},
}


def sweep(dataset_key, lengths=(5, 10, 15, 30), n_crops=1, device="cuda"):
    cfg = WINNING_CONFIG[dataset_key]
    rows = []
    for length in lengths:
        crop_seconds = None if length == 30 else length  # 30s == full clip, reuse existing cache
        encoder_name = "mert_v1_330m" if crop_seconds is None else f"mert_v1_330m_seg{crop_seconds}s_{n_crops}crop"
        cache_dir = os.path.join("cache", encoder_name, dataset_key)
        if not cache_is_complete(dataset_key, cache_dir):
            extract(dataset_key, device=device, crop_seconds=crop_seconds, n_crops=n_crops)

        out_dir = os.path.join(RESULTS_DIR, f"seglen_sweep_{dataset_key}", f"{length}s_{n_crops}crop")
        metrics, _fitted = run(dataset_key, layer=cfg["layer"], classifier=cfg["classifier"],
                                pca_dim=cfg["pca_dim"], encoder_name=encoder_name, out_dir=out_dir)
        row = {"length_s": length, "n_crops": n_crops, "top1": metrics["top1"], "top3": metrics["top3"]}
        if "mean_abs_decade_error" in metrics:
            row["mean_abs_decade_error"] = metrics["mean_abs_decade_error"]
            row["quadratic_weighted_kappa"] = metrics["quadratic_weighted_kappa"]
        rows.append(row)
        print(f"[{dataset_key}] length={length}s n_crops={n_crops} top1={metrics['top1']:.4f} top3={metrics['top3']:.4f}")

    summary_dir = os.path.join(RESULTS_DIR, f"seglen_sweep_{dataset_key}")
    os.makedirs(summary_dir, exist_ok=True)
    csv_path = os.path.join(summary_dir, f"summary_{n_crops}crop.csv")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{dataset_key}] segment-length sweep summary -> {csv_path}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--n-crops", type=int, default=1)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    sweep(args.dataset, n_crops=args.n_crops, device=args.device)
