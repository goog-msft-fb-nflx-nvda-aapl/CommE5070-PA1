"""Required experiment: treat Task 1 (decade) as year-regression or hierarchical
prediction instead of flat 6-way softmax. Reuses the already-cached MERT layer-4
embeddings (winning config from the layer-sweep deep-dive) -- no new extraction.

(a) Regression: Ridge regression on decade-index (0-5), rounded to nearest class
    for top1/MAE comparison against the flat-classification baseline (top1=0.4621,
    MAE=0.947 decades).
(b) Hierarchical: coarse 3-way grouping of adjacent decade pairs
    ({1960s,1970s}, {1980s,1990s}, {2000s,2010s}) then a fine 2-way classifier
    within the predicted coarse group.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CACHE_DIR, RESULTS_DIR, DECADE_LABELS
from src.train_probe import load_cached

LAYER = 4
PCA_DIM = 128
COARSE_GROUPS = [[0, 1], [2, 3], [4, 5]]  # decade indices grouped in adjacent pairs
IDX_TO_COARSE = {i: g for g, idxs in enumerate(COARSE_GROUPS) for i in idxs}
IDX_TO_FINE = {i: idxs.index(i) for idxs in COARSE_GROUPS for i in idxs}  # 0/1 within group


def prep():
    cache_dir = os.path.join(CACHE_DIR, "mert_v1_330m", "A")
    data = load_cached("A", cache_dir, LAYER)
    X_train, y_train, _ = data["train"]
    X_val, y_val, _ = data["validation"]
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)
    pca = PCA(n_components=PCA_DIM, random_state=0).fit(X_train_s)
    X_train_p, X_val_p = pca.transform(X_train_s), pca.transform(X_val_s)
    return X_train_p, y_train, X_val_p, y_val


def regression_framing(X_train, y_train, X_val, y_val, out_dir):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0, 100.0]}, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    search.fit(X_train, y_train)
    pred_continuous = search.predict(X_val)
    pred_rounded = np.clip(np.round(pred_continuous), 0, 5).astype(int)

    mae_continuous = float(np.mean(np.abs(pred_continuous - y_val)))
    mae_rounded = float(np.mean(np.abs(pred_rounded - y_val)))
    top1 = float(np.mean(pred_rounded == y_val))
    qwk = float(cohen_kappa_score(y_val, pred_rounded, weights="quadratic", labels=list(range(6))))

    result = {"framing": "regression", "best_alpha": search.best_params_["alpha"],
              "mae_continuous_decades": mae_continuous, "mae_rounded_decades": mae_rounded,
              "top1_after_rounding": top1, "quadratic_weighted_kappa": qwk}
    with open(os.path.join(out_dir, "regression.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[regression] MAE(raw)={mae_continuous:.3f} MAE(rounded)={mae_rounded:.3f} "
          f"top1={top1:.4f} QWK={qwk:.4f}")
    return result


def hierarchical_framing(X_train, y_train, X_val, y_val, out_dir):
    coarse_train = np.array([IDX_TO_COARSE[y] for y in y_train])
    coarse_val = np.array([IDX_TO_COARSE[y] for y in y_val])
    fine_train = np.array([IDX_TO_FINE[y] for y in y_train])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    coarse_clf = GridSearchCV(LogisticRegression(max_iter=5000), {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
                               cv=cv, scoring="accuracy", n_jobs=1)
    coarse_clf.fit(X_train, coarse_train)
    coarse_pred_val = coarse_clf.predict(X_val)
    coarse_acc = float(np.mean(coarse_pred_val == coarse_val))

    # one fine (within-group, 2-way) classifier per coarse group, trained only on that group's train rows
    fine_clfs = {}
    for g in range(3):
        mask = coarse_train == g
        fc = GridSearchCV(LogisticRegression(max_iter=5000), {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0), scoring="accuracy", n_jobs=1)
        fc.fit(X_train[mask], fine_train[mask])
        fine_clfs[g] = fc

    final_pred = np.zeros(len(y_val), dtype=int)
    for i in range(len(y_val)):
        g = coarse_pred_val[i]
        fine_pred = fine_clfs[g].predict(X_val[i:i + 1])[0]
        final_pred[i] = COARSE_GROUPS[g][fine_pred]

    top1 = float(np.mean(final_pred == y_val))
    mae = float(np.mean(np.abs(final_pred - y_val)))
    qwk = float(cohen_kappa_score(y_val, final_pred, weights="quadratic", labels=list(range(6))))

    result = {"framing": "hierarchical", "coarse_groups": COARSE_GROUPS, "coarse_accuracy": coarse_acc,
              "final_top1": top1, "final_mae_decades": mae, "quadratic_weighted_kappa": qwk}
    with open(os.path.join(out_dir, "hierarchical.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[hierarchical] coarse_acc={coarse_acc:.4f} final_top1={top1:.4f} "
          f"final_MAE={mae:.3f} QWK={qwk:.4f}")
    return result


if __name__ == "__main__":
    out_dir = os.path.join(RESULTS_DIR, "task1_ordinal_framing")
    os.makedirs(out_dir, exist_ok=True)
    X_train, y_train, X_val, y_val = prep()
    reg = regression_framing(X_train, y_train, X_val, y_val, out_dir)
    hier = hierarchical_framing(X_train, y_train, X_val, y_val, out_dir)
    print("\nFlat-classification baseline (from earlier layer-sweep deep-dive): top1=0.4621, MAE=0.947, QWK=0.621")
