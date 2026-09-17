"""Deep-dive ablation: sweep MERT-v1-330M layers (0=feature_projection embedding,
1-24=transformer blocks) + mean_all/concat_last4, for one dataset, one classifier,
to find which layer(s) carry the most decade/market signal before touching
classifier-type or PCA ablations. Saves a CSV summary."""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR
from src.train_probe import run


def sweep(dataset_key, classifier="logreg", out_dir=None):
    out_dir = out_dir or os.path.join(RESULTS_DIR, f"mert_layer_sweep_{dataset_key}")
    os.makedirs(out_dir, exist_ok=True)
    layers = list(range(25)) + ["mean_all", "concat_last4"]
    rows = []
    for layer in layers:
        metrics, _fitted = run(dataset_key, layer=layer, classifier=classifier, out_dir=out_dir)
        row = {"layer": layer, "top1": metrics["top1"], "top3": metrics["top3"],
               "cv_best_score": metrics["cv_best_score"], "best_params": metrics["best_params"]}
        if "mean_abs_decade_error" in metrics:
            row["mean_abs_decade_error"] = metrics["mean_abs_decade_error"]
            row["quadratic_weighted_kappa"] = metrics["quadratic_weighted_kappa"]
        rows.append(row)

    csv_path = os.path.join(out_dir, "layer_sweep_summary.csv")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{dataset_key}] layer sweep summary -> {csv_path}")

    best = max(rows, key=lambda r: r["top1"])
    print(f"[{dataset_key}] best layer={best['layer']} top1={best['top1']:.4f} top3={best['top3']:.4f}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--classifier", default="logreg")
    args = p.parse_args()
    sweep(args.dataset, classifier=args.classifier)
