"""Generates the actual submission JSON (<studentID>.json) against the real held-out
test split, using this project's two best point-estimate configs (see WORKLOG.md's
"CURRENT overall-best configs" table):

  Task A (decade): MERT-v2-30s (layer10, logreg) OOF-fitted fusion with PupuM2D-Large
                    (mean-pooled global embedding, logreg, PCA128), alpha=0.6 MERT-v2.
                    top1=0.5606, top3=0.8485 on the *validation* split (n=132) -- this
                    is our best measured estimate of test-set performance, not a new
                    test-set number (test labels are never available to us).

  Task B (market): MERT-v2-30s (mean_all, logreg, PCA128) frozen probe, OOF-fitted
                    fusion with contextually-calibrated Audio Flamingo 3 (direct
                    prompt), alpha=0.8 MERT-v2. top1=0.6569, top3=0.8922 on validation
                    (n=102).

Every hyperparameter below (classifier C via GridSearchCV, PCA dims, fusion alpha,
calibration null-scores) is frozen exactly as already established and reported in
WORKLOG.md -- nothing is re-tuned here. Classifiers are refit on the training set only
(not train+validation), exactly reproducing the pipeline whose validation numbers are
quoted above, so this script's own validation-set predictions can be used as a sanity
check against the numbers already reported (see verify_on_validation()).
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR, CACHE_DIR
from src.train_probe import load_cached
from src.metrics import summarize
from src.contextual_calibration import apply_calibration

ALPHA_A = 0.6   # weight on MERT-v2 in the A fusion (accuracy-optimal OOF fit)
ALPHA_B = 0.8   # weight on MERT-v2 in the B fusion (accuracy-optimal OOF fit)
AF3_PROMPT = "direct"
# cached null (content-free) label scores from the original calibration run
# (results/contextual_calibration/B_direct.json) -- reused exactly, not recomputed,
# so this matches the already-reported 0.6275/0.6569 numbers precisely.
AF3_NULL_SCORES_PATH = os.path.join(RESULTS_DIR, "contextual_calibration", "B_direct.json")


def fit_probe(X_train, y_train, classifier="logreg", pca_dim=None):
    scaler = StandardScaler().fit(X_train)
    X_s = scaler.transform(X_train)
    pca = None
    if pca_dim:
        pca = PCA(n_components=pca_dim, random_state=0).fit(X_s)
        X_s = pca.transform(X_s)
    base = LogisticRegression(max_iter=5000)
    grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    search = GridSearchCV(base, grid, cv=cv, scoring="accuracy")
    search.fit(X_s, y_train)
    return scaler, pca, search.best_estimator_


def transform(scaler, pca, X):
    X_s = scaler.transform(X)
    return pca.transform(X_s) if pca is not None else X_s


def reorder(X_b, ids_b, ids_a):
    by_id = {sid: i for i, sid in enumerate(ids_b)}
    idx = [by_id[sid] for sid in ids_a]
    return X_b[idx]


def probs_to_top3(probs, ids, label_names):
    out = {}
    for i, sample_id in enumerate(ids):
        top3_idx = np.argsort(-probs[i])[:3]
        out[sample_id] = [label_names[j] for j in top3_idx]
    return out


def predict_task_a(target_split="test"):
    spec = DATASETS["A"]
    labels = spec["labels"]

    mert_dir = os.path.join(CACHE_DIR, "mertv2_30s", "A")
    mert = load_cached("A", mert_dir, 10)
    X_train_m, y_train, _train_ids = mert["train"]
    X_tgt_m, _y_tgt, tgt_ids = mert[target_split]

    pupu_dir = os.path.join(CACHE_DIR, "pupum2d_large", "A")
    pupu = load_cached("A", pupu_dir, 0)
    X_train_p, y_train_p, train_ids_p = pupu["train"]
    X_tgt_p, _y_tgt_p, tgt_ids_p = pupu[target_split]

    X_train_p = reorder(X_train_p, train_ids_p, _train_ids)
    X_tgt_p = reorder(X_tgt_p, tgt_ids_p, tgt_ids)

    scaler_m, pca_m, clf_m = fit_probe(X_train_m, y_train, "logreg", None)
    scaler_p, pca_p, clf_p = fit_probe(X_train_p, y_train, "logreg", 128)

    probs_m = clf_m.predict_proba(transform(scaler_m, pca_m, X_tgt_m))
    probs_p = clf_p.predict_proba(transform(scaler_p, pca_p, X_tgt_p))
    fused = ALPHA_A * probs_m + (1 - ALPHA_A) * probs_p
    return fused, tgt_ids, labels


def predict_task_b(target_split="test"):
    spec = DATASETS["B"]
    labels = spec["labels"]

    mert_dir = os.path.join(CACHE_DIR, "mertv2_30s", "B")
    mert = load_cached("B", mert_dir, "mean_all")
    X_train_m, y_train, train_ids = mert["train"]
    X_tgt_m, _y_tgt, tgt_ids = mert[target_split]
    scaler_m, pca_m, clf_m = fit_probe(X_train_m, y_train, "logreg", 128)
    probs_m = clf_m.predict_proba(transform(scaler_m, pca_m, X_tgt_m))

    with open(AF3_NULL_SCORES_PATH) as f:
        null_scores = json.load(f)["null_scores"]
    _probs_uncal, probs_cal, af3_ids = apply_calibration("B", AF3_PROMPT, target_split, null_scores)
    probs_af3 = reorder(probs_cal, af3_ids, tgt_ids)

    fused = ALPHA_B * probs_m + (1 - ALPHA_B) * probs_af3
    return fused, tgt_ids, labels


def verify_on_validation():
    """Sanity check: rerun the identical pipeline on the validation split and confirm
    it reproduces the already-reported top1 numbers (0.5606 for A, 0.6569 for B) before
    trusting the test-split predictions built the same way."""
    from src.data import load_manifest

    for dataset_key, predict_fn, expected_top1 in (("A", predict_task_a, 0.5606), ("B", predict_task_b, 0.6569)):
        probs, ids, labels = predict_fn(target_split="validation")
        spec = DATASETS[dataset_key]
        rows, _ = load_manifest(dataset_key, "validation")
        row_by_id = {r["sample_id"]: r for r in rows}
        y_val = np.array([spec["label_to_int"][row_by_id[sid]["label"]] for sid in ids])
        m = summarize(probs, y_val, len(labels), ordinal=spec["ordinal"], label_names=labels)
        # expected_top1 is a value rounded to 4 decimals as quoted in WORKLOG.md, not
        # the exact fraction (e.g. 0.5606 vs the true 74/132 = 0.560606...) -- compare
        # at that same rounding precision, not to float tolerance.
        match = "MATCH" if abs(round(m["top1"], 4) - expected_top1) < 1e-4 else "MISMATCH"
        print(f"[{dataset_key}] validation reproduction: top1={m['top1']:.4f} (expected {expected_top1:.4f}) -- {match}")
        if match == "MISMATCH":
            raise RuntimeError(f"[{dataset_key}] pipeline does not reproduce the reported validation number -- "
                                f"do not trust the test predictions until this is fixed")


def save_checkpoints(checkpoint_dir):
    """Export the small fitted classifier objects (scaler/PCA/logreg -- not the base
    encoder weights, which are large, public, and downloaded fresh from HuggingFace by
    the extraction scripts) for the TA's convenience/as a determinism safety net. Fit on
    the training set only, identical to what predict_task_a/b already do internally."""
    import joblib

    os.makedirs(checkpoint_dir, exist_ok=True)

    mert_a = load_cached("A", os.path.join(CACHE_DIR, "mertv2_30s", "A"), 10)
    X_train_m, y_train, _ = mert_a["train"]
    scaler_m, pca_m, clf_m = fit_probe(X_train_m, y_train, "logreg", None)
    joblib.dump({"scaler": scaler_m, "pca": pca_m, "clf": clf_m},
                os.path.join(checkpoint_dir, "task_a_mertv2_layer10.joblib"))

    pupu_a = load_cached("A", os.path.join(CACHE_DIR, "pupum2d_large", "A"), 0)
    X_train_p, y_train_p, _ = pupu_a["train"]
    scaler_p, pca_p, clf_p = fit_probe(X_train_p, y_train_p, "logreg", 128)
    joblib.dump({"scaler": scaler_p, "pca": pca_p, "clf": clf_p},
                os.path.join(checkpoint_dir, "task_a_pupum2d.joblib"))

    mert_b = load_cached("B", os.path.join(CACHE_DIR, "mertv2_30s", "B"), "mean_all")
    X_train_mb, y_train_b, _ = mert_b["train"]
    scaler_mb, pca_mb, clf_mb = fit_probe(X_train_mb, y_train_b, "logreg", 128)
    joblib.dump({"scaler": scaler_mb, "pca": pca_mb, "clf": clf_mb},
                os.path.join(checkpoint_dir, "task_b_mertv2_meanall.joblib"))

    print(f"checkpoints saved to {checkpoint_dir}: task_a_mertv2_layer10.joblib, "
          f"task_a_pupum2d.joblib, task_b_mertv2_meanall.joblib "
          f"(alpha_A={ALPHA_A} on mertv2, alpha_B={ALPHA_B} on mertv2 -- fusion weights "
          f"are fixed constants, not fitted objects, see this file's module docstring)")


def run(student_id, out_dir=None, checkpoint_dir=None):
    verify_on_validation()

    probs_a, ids_a, labels_a = predict_task_a(target_split="test")
    probs_b, ids_b, labels_b = predict_task_b(target_split="test")

    if checkpoint_dir:
        save_checkpoints(checkpoint_dir)

    submission = {
        "dataset_A": probs_to_top3(probs_a, ids_a, labels_a),
        "dataset_B": probs_to_top3(probs_b, ids_b, labels_b),
    }
    assert len(submission["dataset_A"]) == 132, f"expected 132 dataset_A test samples, got {len(submission['dataset_A'])}"
    assert len(submission["dataset_B"]) == 102, f"expected 102 dataset_B test samples, got {len(submission['dataset_B'])}"

    out_dir = out_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{student_id}.json")
    with open(out_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"wrote {out_path} ({len(submission['dataset_A'])} dataset_A + {len(submission['dataset_B'])} dataset_B predictions)")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--student-id", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--checkpoint-dir", default=None, help="if set, also export fitted scaler/PCA/logreg objects here")
    args = p.parse_args()
    run(args.student_id, out_dir=args.out_dir, checkpoint_dir=args.checkpoint_dir)
