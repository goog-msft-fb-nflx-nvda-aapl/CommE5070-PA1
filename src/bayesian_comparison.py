"""Repeated stratified k-fold CV + Bayesian correlated t-test (Corani/Benavoli via
`baycomp`, Nadeau-Bengio variance correction) -- round-4 queue item 5. Applied to the
single highest-stakes open question from this round: is the full MuQ+LangID+calibrated-
AF3 fusion (0.647) actually better than calibrated-AF3 alone (0.6275) for Task 2, or are
they practically equivalent given our small sample size?

Honest limitation, stated plainly (not silently worked around): we have no artist IDs
in the manifest, so this is repeated *stratified* (not artist-*grouped*) k-fold CV on
the pooled train+validation set -- within-fold artist leakage is possible, unlike our
project's actual artist-disjoint train/val/test splits. AF3's calibrated label scores
are a fixed, already-computed zero-shot output (no retraining needed per fold, so no
leakage risk from AF3 itself); only the MuQ+LangID probe component is refit per fold
(genuinely held-out within each fold), and the fixed fusion weight (fit once, OOF, on
the original train split) is reused across all folds/repeats -- not refit per fold,
since the question here is about a single frozen pipeline's evaluation variance, not
about re-deriving the fusion weight each time.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.train_probe import load_cached
from src.langid_probe import load_langid_features, fit_probe
from src.af3_stack_oof import calibrated_softmax

FUSION_ALPHA = 0.400  # accuracy-optimal weight already fit (OOF, on the original train split) -- reused as-is


def run(dataset_key="B", n_repeats=5, n_splits=5, rope=0.01):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    n_class = len(labels)

    lid = load_langid_features(dataset_key, stem="vocals")
    Xtr_lid, ytr, train_ids = lid["train"]
    Xval_lid, yval, val_ids = lid["validation"]
    enc = load_cached(dataset_key, os.path.join("cache", "muq_large_msd", dataset_key), 2)
    Xtr_enc, _, _ = enc["train"]
    Xval_enc, _, _ = enc["validation"]

    # pool train+validation (900 samples) for repeated CV -- both feature sets and
    # the fixed calibrated-AF3 scores, aligned by sample_id
    X_cat_train = np.concatenate([Xtr_enc, Xtr_lid], axis=1)
    X_cat_val = np.concatenate([Xval_enc, Xval_lid], axis=1)
    X_pool = np.concatenate([X_cat_train, X_cat_val], axis=0)
    y_pool = np.concatenate([ytr, yval])
    ids_pool = list(train_ids) + list(val_ids)

    null = json.load(open(os.path.join(RESULTS_DIR, "contextual_calibration", f"{dataset_key}_direct.json")))["null_scores"]
    null_vec = np.array([null[l] for l in labels])
    af3_by_id = {}
    for split, dirname in [("train", f"audioflamingo3_{dataset_key}_train"), ("validation", f"audioflamingo3_{dataset_key}")]:
        raw = json.load(open(os.path.join(RESULTS_DIR, "alm", dirname, "direct_raw.json")))
        for row in raw:
            scores = np.array([row["label_scores"][lab] for lab in labels]) - null_vec
            p = np.exp(scores - scores.max())
            af3_by_id[row["sample_id"]] = p / p.sum()
    af3_pool = np.stack([af3_by_id[sid] for sid in ids_pool])

    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    fused_scores, af3_scores = [], []
    for train_idx, test_idx in rkf.split(X_pool, y_pool):
        scaler, clf, _ = fit_probe(X_pool[train_idx], y_pool[train_idx], classifier="logreg")
        probe_probs_test = clf.predict_proba(scaler.transform(X_pool[test_idx]))
        af3_probs_test = af3_pool[test_idx]
        fused_test = FUSION_ALPHA * probe_probs_test + (1 - FUSION_ALPHA) * af3_probs_test

        y_test = y_pool[test_idx]
        fused_scores.append((fused_test.argmax(1) == y_test).mean())
        af3_scores.append((af3_probs_test.argmax(1) == y_test).mean())

    fused_scores = np.array(fused_scores)
    af3_scores = np.array(af3_scores)
    print(f"[{dataset_key}] {n_repeats}x{n_splits}-fold CV: fused mean={fused_scores.mean():.4f} "
          f"af3_alone mean={af3_scores.mean():.4f} (n_folds={len(fused_scores)})")

    from baycomp import CorrelatedTTest
    p_left, p_rope, p_right = CorrelatedTTest.probs(fused_scores, af3_scores, rope=rope, runs=n_repeats)
    print(f"[{dataset_key}] Bayesian correlated t-test (Nadeau-Bengio corrected, ROPE=+/-{rope}): "
          f"P(af3_alone better)={p_left:.4f} P(practically equivalent)={p_rope:.4f} P(fused better)={p_right:.4f}")

    out = {"fused_fold_scores": fused_scores.tolist(), "af3_fold_scores": af3_scores.tolist(),
           "fused_mean": float(fused_scores.mean()), "af3_mean": float(af3_scores.mean()),
           "n_repeats": n_repeats, "n_splits": n_splits, "rope": rope,
           "p_af3_better": float(p_left), "p_practically_equivalent": float(p_rope), "p_fused_better": float(p_right),
           "limitation": "repeated STRATIFIED (not artist-grouped) k-fold on pooled train+val -- no artist IDs available"}
    out_dir = os.path.join(RESULTS_DIR, "bayesian_comparison")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}_fused_vs_af3.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{dataset_key}] saved -> {out_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="B", choices=["B"])
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--rope", type=float, default=0.01)
    args = p.parse_args()
    run(args.dataset, n_repeats=args.n_repeats, n_splits=args.n_splits, rope=args.rope)
