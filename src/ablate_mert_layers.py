"""Deep-dive ablation: sweep an encoder's cached layers (0=embedding, 1..N=transformer
blocks) + mean_all/concat_last4, for one dataset, one classifier, to find which layer(s)
carry the most decade/market signal before touching classifier-type or PCA ablations.
Saves a CSV summary. Generic over encoder_name/n_layers -- reused for MERT-v1 (25),
MERT-v2-30s (25), Whisper-large-v3 encoder (33), etc."""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR
from src.train_probe import run


def sweep(dataset_key, classifier="logreg", out_dir=None, encoder_name="mert_v1_330m", n_layers=25):
    out_dir = out_dir or os.path.join(RESULTS_DIR, f"{encoder_name}_layer_sweep_{dataset_key}")
    os.makedirs(out_dir, exist_ok=True)
    layers = list(range(n_layers)) + ["mean_all", "concat_last4"]
    rows = []
    for layer in layers:
        metrics, _fitted = run(dataset_key, layer=layer, classifier=classifier, out_dir=out_dir, encoder_name=encoder_name)
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
    p.add_argument("--encoder-name", default="mert_v1_330m")
    p.add_argument("--n-layers", type=int, default=25)
    args = p.parse_args()
    sweep(args.dataset, classifier=args.classifier, encoder_name=args.encoder_name, n_layers=args.n_layers)
