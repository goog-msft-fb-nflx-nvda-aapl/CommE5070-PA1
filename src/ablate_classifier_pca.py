"""Classifier-type x PCA-dim ablation on a single (already layer-sweep-chosen) layer,
for one encoder/dataset. Generic over encoder_name -- reused across every encoder this
project (MuQ, MERT-v1, MERT-v2-30s, MusicFM, CultureMERT, Whisper-encoder, ...)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR
from src.train_probe import run


def ablate(dataset_key, layer, encoder_name, out_dir=None,
           classifiers=("logreg", "mlp", "svm"), pca_dims=(None, 64, 128, 256)):
    out_dir = out_dir or os.path.join(RESULTS_DIR, f"{encoder_name}_ablation_{dataset_key}")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for classifier in classifiers:
        for pca_dim in pca_dims:
            metrics, _fitted = run(dataset_key, layer=layer, classifier=classifier,
                                    pca_dim=pca_dim, out_dir=out_dir, encoder_name=encoder_name)
            rows.append({"classifier": classifier, "pca_dim": pca_dim,
                         "top1": metrics["top1"], "top3": metrics["top3"],
                         "cv_best_score": metrics["cv_best_score"]})

    best = max(rows, key=lambda r: r["top1"])
    summary = {"dataset": dataset_key, "encoder_name": encoder_name, "layer": str(layer),
               "rows": rows, "best": best}
    summary_path = os.path.join(RESULTS_DIR, f"{encoder_name}_ablation_summary_{dataset_key}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{dataset_key}/{encoder_name}] layer={layer} ablation best: "
          f"classifier={best['classifier']} pca_dim={best['pca_dim']} "
          f"top1={best['top1']:.4f} top3={best['top3']:.4f}")
    print(f"[{dataset_key}/{encoder_name}] summary -> {summary_path}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--layer", required=True, help="int layer index, 'mean_all', or 'concat_last4'")
    p.add_argument("--encoder-name", required=True)
    args = p.parse_args()
    layer = args.layer if args.layer in ("mean_all", "concat_last4") else int(args.layer)
    ablate(args.dataset, layer, args.encoder_name)
