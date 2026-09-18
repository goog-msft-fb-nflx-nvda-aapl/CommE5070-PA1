"""Confusion-matrix / pairwise-AUC diagnostic, Task 2 (IMPROVEMENT_FINDINGS.md queue item 4)
-- analysis only, no training. Runs on the current best Task 2 config (MuQ+AF3 fusion,
src/af3_stack.py) to check whether errors concentrate in specific market pairs (the
"US/UK/Germany/Italy share 1980s Anglo/Euro-pop production conventions" hypothesis raised
earlier in WORKLOG.md's Task 2 stem-comparison section) or spread evenly across all 6
classes. Also run on Task 1 for completeness (not presumed to be uninteresting there).
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.af3_stack import frozen_probe_probs, af3_probs
from src.ensemble import finetuned_mert_probs, cnn_probs, align


def pairwise_auc(probs, y_true, labels):
    """One-vs-one AUC per class pair, using only samples belonging to that pair and
    each class's own probability mass renormalized over just the two candidates --
    the standard pairwise-AUC diagnostic for finding which specific classes get
    confused, which a single multi-class confusion matrix or macro-AUC number hides."""
    n_class = len(labels)
    out = {}
    for i, j in itertools.combinations(range(n_class), 2):
        mask = (y_true == i) | (y_true == j)
        if mask.sum() < 4 or len(set(y_true[mask])) < 2:
            out[f"{labels[i]}_vs_{labels[j]}"] = None
            continue
        sub_y = (y_true[mask] == j).astype(int)  # 1 if class j
        sub_p = probs[mask][:, j] / (probs[mask][:, i] + probs[mask][:, j] + 1e-12)
        try:
            auc = roc_auc_score(sub_y, sub_p)
        except ValueError:
            auc = None
        out[f"{labels[i]}_vs_{labels[j]}"] = None if auc is None else float(auc)
    return out


def diagnose(dataset_key, probs, y_true, ids, tag, out_dir):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    n_class = len(labels)
    preds = probs.argmax(axis=1)

    cm = confusion_matrix(y_true, preds, labels=list(range(n_class)))
    prec, rec, f1, support = precision_recall_fscore_support(y_true, preds, labels=list(range(n_class)), zero_division=0)
    pw_auc = pairwise_auc(probs, y_true, labels)

    print(f"\n[{dataset_key}/{tag}] confusion matrix (rows=true, cols=pred), labels={labels}")
    print("      " + " ".join(f"{l[:6]:>6}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l[:6]:>6} " + " ".join(f"{cm[i, j]:>6}" for j in range(n_class)))

    print(f"[{dataset_key}/{tag}] per-class precision/recall/f1/support:")
    for i, l in enumerate(labels):
        print(f"  {l}: P={prec[i]:.3f} R={rec[i]:.3f} F1={f1[i]:.3f} n={support[i]}")

    print(f"[{dataset_key}/{tag}] pairwise AUC (lower = more confused pair), sorted ascending:")
    ranked = sorted(((v, k) for k, v in pw_auc.items() if v is not None))
    for auc, pair in ranked:
        print(f"  {pair}: AUC={auc:.3f}")

    out = {
        "confusion_matrix": cm.tolist(), "labels": labels,
        "per_class": {labels[i]: {"precision": float(prec[i]), "recall": float(rec[i]),
                                   "f1": float(f1[i]), "support": int(support[i])} for i in range(n_class)},
        "pairwise_auc": pw_auc, "top1": float((preds == y_true).mean()),
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{dataset_key}/{tag}] saved -> {out_path}")
    return out


def run_B(device="cuda:1"):
    out_dir = os.path.join(RESULTS_DIR, "confusion_diagnostic")
    muq_p, y, ids = frozen_probe_probs("B", "muq_large_msd")
    af3cot_p, af3_ids = af3_probs("B", "cot_then_answer")
    af3cot_p = align(muq_p, ids, af3cot_p, af3_ids)
    fused_best = 0.4 * af3cot_p + 0.6 * muq_p
    diagnose("B", fused_best, y, ids, "fused_best_muq_af3cot", out_dir)
    diagnose("B", muq_p, y, ids, "muq_alone", out_dir)


def run_A(device="cuda:1"):
    out_dir = os.path.join(RESULTS_DIR, "confusion_diagnostic")
    muq_p, y, ids = frozen_probe_probs("A", "muq_large_msd")
    af3dir_p, af3_ids = af3_probs("A", "direct")
    af3dir_p = align(muq_p, ids, af3dir_p, af3_ids)
    fused_best = 0.5 * af3dir_p + 0.5 * muq_p
    diagnose("A", fused_best, y, ids, "fused_best_muq_af3direct", out_dir)
    diagnose("A", muq_p, y, ids, "muq_alone", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--device", default="cuda:1")
    args = p.parse_args()
    if args.dataset == "A":
        run_A(device=args.device)
    else:
        run_B(device=args.device)
