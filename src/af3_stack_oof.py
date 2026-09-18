"""Round-3 queue item 1: OOF-fitted, temperature-calibrated AF3+probe fusion for Task 2.

Our existing fusion (src/af3_stack.py) picks its weight by sweeping the 102-sample
validation set directly -- fine for a first pass, but the significance testing in
src/significance.py showed the resulting config isn't confirmed to beat AF3 alone
(p=0.21). IMPROVEMENT_FINDINGS_ROUND3.md's top convergent recommendation (3-4 of 4
independent deep-research sources) is to fit the fusion on out-of-fold predictions from
the 798-sample TRAINING set instead, and to calibrate AF3's zero-shot probabilities
(known to be overconfident) with a single temperature scalar before combining -- both
fit on ~8x more data than the validation-only sweep, without touching the validation
set until the final, single evaluation.

Three quantities are learned, all on the training set only:
1. AF3 temperature T (NLL-fit on AF3's own train-set predictions vs true labels)
2. Probe out-of-fold predictions (5-fold CV, so every training sample's "probe
   prediction" comes from a fold that never saw it during fitting)
3. Fusion weight alpha (NLL-fit on OOF-probe + T-calibrated-AF3 train predictions)

Then (T, alpha) are frozen and applied once to validation: probe refit on the FULL
training set, AF3 validation probs calibrated with the same T, combined with the same
alpha -- evaluated once, no further tuning on validation.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.train_probe import load_cached
from src.metrics import summarize
from src.significance import bootstrap_ci_top1, compare

# must match af3_stack.py's FROZEN_CONFIGS exactly -- these are the actual established
# winning frozen-probe configs per dataset (from the muq layer-sweep + classifier
# ablation), NOT a single hardcoded layer/classifier reused across both datasets.
PROBE_CONFIGS = {
    "A": {"layer": 1, "classifier": "svm"},
    "B": {"layer": 2, "classifier": "logreg"},
}
CLASSIFIER_CV = {
    "logreg": (LogisticRegression(max_iter=5000), {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}),
    "svm": (SVC(probability=True, random_state=0, max_iter=20000),
            {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "kernel": ["rbf", "linear"]}),
}


def fit_probe(X_train, y_train, classifier="logreg"):
    scaler = StandardScaler().fit(X_train)
    X_s = scaler.transform(X_train)
    base, grid = CLASSIFIER_CV[classifier]
    search = GridSearchCV(base, grid, cv=5, scoring="accuracy")
    search.fit(X_s, y_train)
    return scaler, search.best_estimator_


def oof_probe_probs(X_train, y_train, classifier="logreg", n_splits=5, seed=0):
    """5-fold OOF predict_proba -- every row's prediction comes from a model that
    never saw that row during fitting, so these are honest held-out predictions
    usable as "features" for fitting the fusion weight without leakage."""
    n_class = len(set(y_train))
    oof = np.zeros((len(y_train), n_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, held_idx in cv.split(X_train, y_train):
        scaler, clf = fit_probe(X_train[train_idx], y_train[train_idx], classifier=classifier)
        oof[held_idx] = clf.predict_proba(scaler.transform(X_train[held_idx]))
    return oof


def load_af3_scores(dataset_key, prompt_name, split, sample_id_order=None):
    """Load AF3's raw teacher-forced label log-prob scores (not yet softmaxed --
    calibration needs the raw scores, not an already-fixed-temperature softmax)."""
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    dirname = f"audioflamingo3_{dataset_key}" if split == "validation" else f"audioflamingo3_{dataset_key}_{split}"
    path = os.path.join(RESULTS_DIR, "alm", dirname, f"{prompt_name}_raw.json")
    with open(path) as f:
        raw = json.load(f)
    by_id = {row["sample_id"]: np.array([row["label_scores"][lab] for lab in labels]) for row in raw}
    if sample_id_order is None:
        sample_id_order = list(by_id.keys())
    scores = np.stack([by_id[sid] for sid in sample_id_order])
    return scores, sample_id_order


def calibrated_softmax(scores, T):
    z = scores / T
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(scores, y_true, T_bounds=(0.05, 10.0)):
    def nll(T):
        p = calibrated_softmax(scores, T)
        return -np.mean(np.log(np.clip(p[np.arange(len(y_true)), y_true], 1e-12, 1.0)))
    res = minimize_scalar(nll, bounds=T_bounds, method="bounded")
    return float(res.x), float(res.fun)


def fit_fusion_weight(probe_probs, af3_probs, y_true):
    def nll(alpha):
        p = alpha * probe_probs + (1 - alpha) * af3_probs
        return -np.mean(np.log(np.clip(p[np.arange(len(y_true)), y_true], 1e-12, 1.0)))
    res = minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded")
    return float(res.x), float(res.fun)


def fit_fusion_weight_accuracy(probe_probs, af3_probs, y_true):
    """Secondary check: NLL-optimal and accuracy-optimal alpha can disagree (they
    optimize different objectives) -- grid search accuracy directly as a cross-check,
    since accuracy is what's ultimately reported."""
    best_acc, best_alpha = -1.0, None
    for a in np.arange(0.0, 1.001, 0.05):
        combo = a * probe_probs + (1 - a) * af3_probs
        acc = (combo.argmax(1) == y_true).mean()
        if acc > best_acc:
            best_acc, best_alpha = acc, a
    return float(best_alpha), float(best_acc)


def run(dataset_key="B", prompt_name="cot_then_answer", encoder_name="muq_large_msd", device="cuda"):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    label_to_int = spec["label_to_int"]
    n_class = len(labels)
    probe_cfg = PROBE_CONFIGS[dataset_key]
    layer, classifier = probe_cfg["layer"], probe_cfg["classifier"]
    print(f"[{dataset_key}] probe config: layer={layer} classifier={classifier} (must match af3_stack.py's FROZEN_CONFIGS)")

    cache_dir = os.path.join("cache", encoder_name, dataset_key)
    data = load_cached(dataset_key, cache_dir, layer)
    X_train, y_train, train_ids = data["train"]
    X_val, y_val, val_ids = data["validation"]

    # 1. AF3 train-set scores + temperature fit (train set only)
    af3_train_scores, af3_train_ids = load_af3_scores(dataset_key, prompt_name, "train")
    # align AF3's row order to the probe's train_ids order
    af3_by_id = {sid: af3_train_scores[i] for i, sid in enumerate(af3_train_ids)}
    af3_train_scores_aligned = np.stack([af3_by_id[sid] for sid in train_ids])
    T, train_nll_at_T = fit_temperature(af3_train_scores_aligned, y_train)
    print(f"[{dataset_key}] AF3 temperature fit on train: T={T:.3f} (train NLL={train_nll_at_T:.4f})")

    # 2. Probe OOF predictions on train set
    oof_probe = oof_probe_probs(X_train, y_train, classifier=classifier)

    # 3. Fusion weight fit on train set (OOF probe + T-calibrated AF3)
    af3_train_calibrated = calibrated_softmax(af3_train_scores_aligned, T)
    alpha, train_nll_at_alpha = fit_fusion_weight(oof_probe, af3_train_calibrated, y_train)
    print(f"[{dataset_key}] fusion weight fit on train (OOF, NLL-optimal): alpha(probe)={alpha:.3f} "
          f"1-alpha(AF3)={1 - alpha:.3f} (train NLL={train_nll_at_alpha:.4f})")
    alpha_acc, train_acc_at_alpha = fit_fusion_weight_accuracy(oof_probe, af3_train_calibrated, y_train)
    print(f"[{dataset_key}] fusion weight fit on train (OOF, accuracy-optimal): alpha(probe)={alpha_acc:.3f} "
          f"1-alpha(AF3)={1 - alpha_acc:.3f} (train OOF acc={train_acc_at_alpha:.4f})")

    # 4. Apply frozen (T, alpha) once to validation -- probe refit on FULL train set
    scaler, clf = fit_probe(X_train, y_train, classifier=classifier)
    probe_val_probs = clf.predict_proba(scaler.transform(X_val))
    af3_val_scores, af3_val_ids = load_af3_scores(dataset_key, prompt_name, "validation")
    af3_by_id_val = {sid: af3_val_scores[i] for i, sid in enumerate(af3_val_ids)}
    af3_val_scores_aligned = np.stack([af3_by_id_val[sid] for sid in val_ids])
    af3_val_calibrated = calibrated_softmax(af3_val_scores_aligned, T)

    fused_val = alpha * probe_val_probs + (1 - alpha) * af3_val_calibrated
    fused_val_acc = alpha_acc * probe_val_probs + (1 - alpha_acc) * af3_val_calibrated
    metrics = summarize(fused_val, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
    metrics_acc = summarize(fused_val_acc, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
    print(f"[{dataset_key}] OOF-fused NLL-optimal (T={T:.3f}, alpha={alpha:.3f}) validation: "
          f"top1={metrics['top1']:.4f} top3={metrics['top3']:.4f}")
    print(f"[{dataset_key}] OOF-fused accuracy-optimal (T={T:.3f}, alpha={alpha_acc:.3f}) validation: "
          f"top1={metrics_acc['top1']:.4f} top3={metrics_acc['top3']:.4f}")

    # comparisons: OOF-fused (both variants) vs probe-alone, AF3-alone (uncalibrated,
    # matching af3_stack.py), AF3-alone (calibrated)
    af3_val_uncalibrated = calibrated_softmax(af3_val_scores_aligned, 1.0)
    configs = {
        "probe_alone": probe_val_probs,
        "af3_alone_uncalibrated": af3_val_uncalibrated,
        "af3_alone_calibrated": af3_val_calibrated,
        "oof_fused_nll_optimal": fused_val,
        "oof_fused_accuracy_optimal": fused_val_acc,
    }
    cis = {name: bootstrap_ci_top1(p, y_val, n_boot=10000) for name, p in configs.items()}
    for name, ci in cis.items():
        print(f"  {name}: top1={ci['top1']:.4f} 95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

    comps = {}
    for fused_name in ("oof_fused_nll_optimal", "oof_fused_accuracy_optimal"):
        for name in ("probe_alone", "af3_alone_uncalibrated", "af3_alone_calibrated"):
            comps[f"{fused_name}_vs_{name}"] = compare(fused_name, configs[fused_name], name, configs[name], y_val, n_boot=10000)

    out = {"T": T, "alpha_nll_optimal": alpha, "alpha_accuracy_optimal": alpha_acc,
           "prompt": prompt_name, "encoder": encoder_name, "layer": layer, "classifier": classifier,
           "val_top1_nll_optimal": metrics["top1"], "val_top3_nll_optimal": metrics["top3"],
           "val_top1_accuracy_optimal": metrics_acc["top1"], "val_top3_accuracy_optimal": metrics_acc["top3"],
           "individual_ci": cis, "paired_comparisons": comps}
    out_dir = os.path.join(RESULTS_DIR, "af3_stack_oof")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}_{encoder_name}_af3{prompt_name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{dataset_key}] saved -> {out_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], default="B")
    p.add_argument("--prompt", default="cot_then_answer")
    p.add_argument("--encoder", default="muq_large_msd")
    args = p.parse_args()
    run(args.dataset, prompt_name=args.prompt, encoder_name=args.encoder)
