"""Round-5 item 8 analysis: (1) standalone probe on the hand-crafted wide/deep features,
(2) the actual "wide & deep" fusion -- concatenate with MERT-v2-30s's best-layer embedding
(current Task 1 best standalone encoder) and refit, (3) leave-one-group-out feature-
importance ablation (timbre/pitch/dynamics/rhythm) on the standalone features, per the
source doc's own suggested analysis (L02 slide 101's confusion-matrix-diagnostics spirit).
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR, RESULTS_DIR
from src.train_probe import load_cached
from src.metrics import summarize
from src.wide_deep_features import FEATURE_NAMES, FEATURE_GROUPS

MERTV2_CFG = {"encoder_name": "mertv2_30s", "layer": 10}


def fit_and_eval(X_train, y_train, X_val, y_val, n_class, ordinal, label_names):
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)
    base = LogisticRegression(max_iter=5000)
    grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    search = GridSearchCV(base, grid, cv=cv, scoring="accuracy")
    search.fit(X_train_s, y_train)
    probs = search.best_estimator_.predict_proba(X_val_s)
    return summarize(probs, y_val, n_class, ordinal=ordinal, label_names=label_names), search.best_params_


def run(dataset_key="A"):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    n_class = len(labels)

    wd_dir = os.path.join(CACHE_DIR, "wide_deep_features", dataset_key)
    wd_data = load_cached(dataset_key, wd_dir, layer=0)
    X_train_wd, y_train, train_ids_wd = wd_data["train"]
    X_val_wd, y_val, val_ids_wd = wd_data["validation"]

    mert_dir = os.path.join(CACHE_DIR, MERTV2_CFG["encoder_name"], dataset_key)
    mert_data = load_cached(dataset_key, mert_dir, layer=MERTV2_CFG["layer"])
    X_train_m, y_train_m, train_ids_m = mert_data["train"]
    X_val_m, y_val_m, val_ids_m = mert_data["validation"]

    def reorder(X_b, ids_b, y_b, ids_a, y_a):
        by_id = {sid: i for i, sid in enumerate(ids_b)}
        idx = [by_id[sid] for sid in ids_a]
        assert (y_b[idx] == y_a).all()
        return X_b[idx]

    X_train_m = reorder(X_train_m, train_ids_m, y_train_m, train_ids_wd, y_train)
    X_val_m = reorder(X_val_m, val_ids_m, y_val_m, val_ids_wd, y_val)

    results = {}

    # 1. wide-deep hand-crafted features, standalone
    metrics, params = fit_and_eval(X_train_wd, y_train, X_val_wd, y_val, n_class, spec["ordinal"], labels)
    print(f"[{dataset_key}] wide_deep_alone: top1={metrics['top1']:.4f} top3={metrics['top3']:.4f} params={params}")
    results["wide_deep_alone"] = {**metrics, "params": params}

    # 2. wide & deep fusion: concat with MERT-v2 layer10
    X_train_cat = np.concatenate([X_train_wd, X_train_m], axis=1)
    X_val_cat = np.concatenate([X_val_wd, X_val_m], axis=1)
    metrics, params = fit_and_eval(X_train_cat, y_train, X_val_cat, y_val, n_class, spec["ordinal"], labels)
    print(f"[{dataset_key}] wide_and_deep (concat with MERT-v2 layer10): top1={metrics['top1']:.4f} top3={metrics['top3']:.4f} params={params}")
    results["wide_and_deep_concat_mertv2"] = {**metrics, "params": params}

    # 3. leave-one-group-out ablation on the standalone wide-deep features
    group_results = {}
    for group in FEATURE_GROUPS:
        keep_idx = [i for i in range(len(FEATURE_NAMES)) if i not in FEATURE_GROUPS[group]]
        metrics, params = fit_and_eval(X_train_wd[:, keep_idx], y_train, X_val_wd[:, keep_idx], y_val,
                                        n_class, spec["ordinal"], labels)
        print(f"[{dataset_key}] wide_deep minus {group} ({len(FEATURE_GROUPS[group])} feats removed): top1={metrics['top1']:.4f}")
        group_results[f"minus_{group}"] = {**metrics, "params": params}
    results["leave_one_group_out"] = group_results

    out_dir = os.path.join(RESULTS_DIR, "wide_deep_analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{dataset_key}] saved -> {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="A", choices=["A", "B"])
    args = p.parse_args()
    run(args.dataset)
