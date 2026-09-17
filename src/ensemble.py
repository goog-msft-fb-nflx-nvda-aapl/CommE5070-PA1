"""Probability-average ensemble of already-trained models. Motivated by the prior
semester's HW1 report (singer classification): their graded best result was a
weighted ensemble of 7 from-scratch models, not any single model. We test whether
combining our already-trained, architecturally-diverse models (from-scratch CNN +
pretrained-SSL-encoder probe [+ fine-tuned encoder once available]) helps here too
-- not presumed, measured.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.scnn import ShortChunkCNN
from src.train_scnn import eval_multicrop
from src.train_probe import load_cached, run as run_probe
from src.finetune_mert import MERTClassifier, CropDataset, get_probs as get_finetune_probs
from src.metrics import summarize

WINNING_FROZEN = {
    "A": {"layer": 4, "classifier": "svm", "pca_dim": 64},
    "B": {"layer": 7, "classifier": "logreg", "pca_dim": None},
}


def cnn_probs(dataset_key, device="cuda"):
    spec = DATASETS[dataset_key]
    ckpt = os.path.join(RESULTS_DIR, f"scnn_{dataset_key}", "best.pt")
    model = ShortChunkCNN(n_class=len(spec["labels"])).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    probs, labels, ids, _ = eval_multicrop(model, dataset_key, "validation", device)
    return probs, labels, ids


def frozen_mert_probs(dataset_key):
    """Reuse train_probe.run()'s actual GridSearchCV-selected best classifier
    (not a hand-guessed hyperparameter) to get predict_proba on validation,
    aligned to the same sample order returned by load_cached."""
    cfg = WINNING_FROZEN[dataset_key]
    _metrics, fitted = run_probe(dataset_key, layer=cfg["layer"], classifier=cfg["classifier"],
                                  pca_dim=cfg["pca_dim"], out_dir=os.path.join(RESULTS_DIR, "ensemble_refit"))
    cache_dir = os.path.join("cache", "mert_v1_330m", dataset_key)
    data = load_cached(dataset_key, cache_dir, cfg["layer"])
    X_val, y_val, val_ids = data["validation"]
    X_val_s = fitted["scaler"].transform(X_val)
    if fitted["pca"] is not None:
        X_val_s = fitted["pca"].transform(X_val_s)
    probs = fitted["clf"].predict_proba(X_val_s)
    return probs, y_val, list(val_ids)


def finetuned_mert_probs(dataset_key, device="cuda"):
    spec = DATASETS[dataset_key]
    ckpt = os.path.join(RESULTS_DIR, f"mert_finetune_{dataset_key}", "best.pt")
    model = MERTClassifier(n_class=len(spec["labels"])).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    val_ds = CropDataset(dataset_key, "validation")
    probs, labels, ids = get_finetune_probs(model, val_ds, device, verbose=False)
    return probs, labels, ids


def align(probs_a, ids_a, probs_b, ids_b):
    """Re-order probs_b's rows to match ids_a's order (models may iterate the
    manifest in different orders)."""
    idx_b = {sid: i for i, sid in enumerate(ids_b)}
    order = [idx_b[sid] for sid in ids_a]
    return probs_b[order]


def run_ensemble(dataset_key, device="cuda"):
    spec = DATASETS[dataset_key]
    n_class = len(spec["labels"])

    cnn_p, cnn_labels, cnn_ids = cnn_probs(dataset_key, device)
    mert_p, mert_labels, mert_ids = frozen_mert_probs(dataset_key)
    mert_p_aligned = align(cnn_p, cnn_ids, mert_p, mert_ids)
    mert_labels_aligned = align(cnn_labels.reshape(-1, 1), cnn_ids, mert_labels.reshape(-1, 1), mert_ids).ravel()
    assert np.array_equal(cnn_labels, mert_labels_aligned), "label mismatch after alignment -- sample order bug"

    results = {}
    for w_cnn in (0.0, 0.3, 0.5, 0.7, 1.0):
        w_mert = 1.0 - w_cnn
        ens_probs = w_cnn * cnn_p + w_mert * mert_p_aligned
        metrics = summarize(ens_probs, cnn_labels, n_class, ordinal=spec["ordinal"], label_names=spec["labels"])
        results[f"cnn{w_cnn}_mert{w_mert:.1f}"] = {"top1": metrics["top1"], "top3": metrics["top3"]}
        print(f"[{dataset_key}] w_cnn={w_cnn} w_mert={w_mert:.1f} top1={metrics['top1']:.4f} top3={metrics['top3']:.4f}")

    out_dir = os.path.join(RESULTS_DIR, f"ensemble_{dataset_key}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cnn_mert_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_ensemble_3way(dataset_key, device="cuda"):
    """Adds the fine-tuned MERT checkpoint as a third ensemble member. Not
    presumed to beat the single fine-tuned model (which is already the best
    single result for A) -- swept and measured."""
    spec = DATASETS[dataset_key]
    n_class = len(spec["labels"])

    cnn_p, cnn_labels, cnn_ids = cnn_probs(dataset_key, device)
    mert_p, _mert_labels, mert_ids = frozen_mert_probs(dataset_key)
    ft_p, _ft_labels, ft_ids = finetuned_mert_probs(dataset_key, device)

    mert_p = align(cnn_p, cnn_ids, mert_p, mert_ids)
    ft_p = align(cnn_p, cnn_ids, ft_p, ft_ids)

    weight_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}
    best = (None, -1.0)
    for w_ft in weight_grid:
        remaining = 1.0 - w_ft
        for split in (0.0, 0.25, 0.5, 0.75, 1.0):  # fraction of `remaining` given to cnn; rest to frozen mert
            w_cnn = remaining * split
            w_mert = remaining * (1 - split)
            ens_probs = w_ft * ft_p + w_cnn * cnn_p + w_mert * mert_p
            metrics = summarize(ens_probs, cnn_labels, n_class, ordinal=spec["ordinal"], label_names=spec["labels"])
            key = f"ft{w_ft:.2f}_cnn{w_cnn:.2f}_mert{w_mert:.2f}"
            results[key] = {"top1": metrics["top1"], "top3": metrics["top3"]}
            if metrics["top1"] > best[1]:
                best = (key, metrics["top1"])
            print(f"[{dataset_key}] {key} top1={metrics['top1']:.4f} top3={metrics['top3']:.4f}")

    out_dir = os.path.join(RESULTS_DIR, f"ensemble_{dataset_key}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "3way_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{dataset_key}] best 3-way: {best[0]} top1={best[1]:.4f}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--mode", choices=["2way", "3way"], default="2way")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.mode == "2way":
        run_ensemble(args.dataset, device=args.device)
    else:
        run_ensemble_3way(args.dataset, device=args.device)
