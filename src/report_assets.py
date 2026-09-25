"""Generates report-ready assets for the submitted configs: confusion matrix heatmaps
(required by the assignment, "at least one confusion matrix for each task") and a
summary comparison chart across every major method tried this project. Computed on the
*validation* split (ground truth available) using the exact same pipeline as
final_predictions.py, since that's what our reported top1/top3 numbers are based on.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.final_predictions import predict_task_a, predict_task_b
from src.metrics import summarize
from src.data import load_manifest

OUT_DIR = os.path.join(RESULTS_DIR, "report_assets")


def plot_confusion(cm_norm, cm_raw, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            frac = cm_norm[i][j]
            count = cm_raw[i][j]
            color = "white" if frac > 0.5 else "black"
            ax.text(j, i, f"{frac:.2f}\n(n={count})", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="row-normalized fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def generate_confusion_matrices():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for dataset_key, predict_fn, title in (
        ("A", predict_task_a, "Task 1 (Decade) — MERT-v2×PupuM2D fusion, validation"),
        ("B", predict_task_b, "Task 2 (Market) — MERT-v2+calibrated-AF3 fusion, validation"),
    ):
        spec = DATASETS[dataset_key]
        labels = spec["labels"]
        probs, ids, _labels = predict_fn(target_split="validation")
        rows, _ = load_manifest(dataset_key, "validation")
        row_by_id = {r["sample_id"]: r for r in rows}
        y_val = np.array([spec["label_to_int"][row_by_id[sid]["label"]] for sid in ids])
        m = summarize(probs, y_val, len(labels), ordinal=spec["ordinal"], label_names=labels)

        out_path = os.path.join(OUT_DIR, f"confusion_{dataset_key}_submitted.png")
        plot_confusion(m["confusion_matrix_row_normalized"], m["confusion_matrix"], labels, title, out_path)

        results[dataset_key] = {"top1": m["top1"], "top3": m["top3"], "labels": labels,
                                 "confusion_matrix": m["confusion_matrix"],
                                 "confusion_matrix_row_normalized": m["confusion_matrix_row_normalized"]}
        if "mean_abs_decade_error" in m:
            results[dataset_key]["mean_abs_decade_error"] = m["mean_abs_decade_error"]
            results[dataset_key]["acc_within_1_decade"] = m["acc_within_1_decade"]
            results[dataset_key]["quadratic_weighted_kappa"] = m["quadratic_weighted_kappa"]
        print(f"[{dataset_key}] submitted-config validation: top1={m['top1']:.4f} top3={m['top3']:.4f}")

    with open(os.path.join(OUT_DIR, "submitted_config_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {os.path.join(OUT_DIR, 'submitted_config_metrics.json')}")


def plot_method_comparison():
    """Bar chart of top1 across every major method tried, both tasks -- gives the
    report a single at-a-glance figure for the "what did we try" narrative."""
    task_a = [
        ("Random baseline", 0.167), ("Wide-deep hand-crafted", 0.4091),
        ("Short-Chunk CNN", 0.500), ("MuQ frozen probe", 0.508),
        ("Whisper-encoder", 0.4318), ("CLaMP3", 0.4924), ("Dasheng", 0.4394),
        ("MAEST", 0.5152), ("PupuM2D", 0.5152), ("MERT-v2-30s", 0.5455),
        ("MERT-v2×PupuM2D fusion (submitted)", 0.5606),
    ]
    task_b = [
        ("Random baseline", 0.167), ("kNN retrieval", 0.4706),
        ("Caption-as-features", 0.5392), ("Dasheng", 0.5392),
        ("MuQ frozen probe", 0.500), ("CLaMP3", 0.5098), ("MAEST", 0.5098),
        ("PupuM2D", 0.4804), ("AF3 zero-shot (raw)", 0.5882),
        ("AF3 zero-shot (calibrated)", 0.6275), ("AF3 LoRA fine-tune", 0.6078),
        ("MERT-v2-30s", 0.6471), ("MERT-v2+calibrated-AF3 fusion (submitted)", 0.6569),
    ]
    for key, data, title in (("A", task_a, "Task 1 (Decade): top-1 accuracy by method"),
                              ("B", task_b, "Task 2 (Market): top-1 accuracy by method")):
        data = sorted(data, key=lambda x: x[1])
        names = [d[0] for d in data]
        vals = [d[1] for d in data]
        colors = ["#d62728" if "submitted" in n else ("#7f7f7f" if "Random" in n else "#1f77b4") for n in names]
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(names) + 1))
        ax.barh(names, vals, color=colors)
        ax.axvline(0.167, color="gray", linestyle="--", linewidth=1, label="random chance (1/6)")
        ax.set_xlabel("Validation top-1 accuracy")
        ax.set_title(title)
        ax.set_xlim(0, max(vals) * 1.15)
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f"method_comparison_{key}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved {out_path}")


if __name__ == "__main__":
    generate_confusion_matrices()
    plot_method_comparison()
