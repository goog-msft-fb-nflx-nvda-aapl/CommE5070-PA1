"""Generic leakage-free OOF fusion of two frozen-probe encoders (round-5 queue item 7:
PupuM2D-Large as an OOF fusion member with our current-best encoder, MERT-v2-30s --
proposed specifically as a decorrelated-error source, distinct pretraining family (2D-patch
JEPA vs. MERT-v2's 1D masked-token prediction), regardless of PupuM2D's own standalone
accuracy). Same leakage-free methodology as every fusion this project (src/af3_stack_oof.py):
fit the fusion weight on 5-fold OOF training-set predictions from each encoder's own probe,
then apply the frozen weight once to validation -- never sweep the fusion weight on
validation directly.

Generic over two (encoder_name, layer, classifier, pca_dim) configs, so reusable for any
future two-encoder fusion check, not just this one pair.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR, CACHE_DIR
from src.train_probe import load_cached
from src.metrics import summarize
from src.significance import bootstrap_ci_top1, compare

CLASSIFIER_CV = {
    "logreg": (LogisticRegression(max_iter=5000), {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}),
    "svm": (SVC(probability=True, random_state=0, max_iter=20000),
            {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "kernel": ["rbf", "linear"]}),
    "mlp": (None, None),
}


def fit_probe(X_train, y_train, classifier="logreg", pca_dim=None):
    scaler = StandardScaler().fit(X_train)
    X_s = scaler.transform(X_train)
    pca = None
    if pca_dim:
        pca = PCA(n_components=pca_dim, random_state=0).fit(X_s)
        X_s = pca.transform(X_s)
    base, grid = CLASSIFIER_CV[classifier]
    search = GridSearchCV(base, grid, cv=5, scoring="accuracy")
    search.fit(X_s, y_train)
    return scaler, pca, search.best_estimator_


def transform(scaler, pca, X):
    X_s = scaler.transform(X)
    if pca is not None:
        X_s = pca.transform(X_s)
    return X_s


def oof_probe_probs(X_train, y_train, classifier="logreg", pca_dim=None, n_splits=5, seed=0):
    n_class = len(set(y_train))
    oof = np.zeros((len(y_train), n_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, held_idx in cv.split(X_train, y_train):
        scaler, pca, clf = fit_probe(X_train[train_idx], y_train[train_idx], classifier, pca_dim)
        oof[held_idx] = clf.predict_proba(transform(scaler, pca, X_train[held_idx]))
    return oof


def fit_fusion_weight(probs_a, probs_b, y_true):
    def nll(alpha):
        p = alpha * probs_a + (1 - alpha) * probs_b
        return -np.mean(np.log(np.clip(p[np.arange(len(y_true)), y_true], 1e-12, 1.0)))
    res = minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded")
    return float(res.x), float(res.fun)


def fit_fusion_weight_accuracy(probs_a, probs_b, y_true):
    best_acc, best_alpha = -1.0, None
    for a in np.arange(0.0, 1.001, 0.05):
        combo = a * probs_a + (1 - a) * probs_b
        acc = (combo.argmax(1) == y_true).mean()
        if acc > best_acc:
            best_acc, best_alpha = acc, a
    return float(best_alpha), float(best_acc)


def run(dataset_key, config_a, config_b, n_boot=10000):
    """config_a/config_b: dict with keys encoder_name, layer, classifier, pca_dim (optional)."""
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    n_class = len(labels)

    def load(cfg):
        cache_dir = os.path.join(CACHE_DIR, cfg["encoder_name"], dataset_key)
        data = load_cached(dataset_key, cache_dir, cfg["layer"])
        return data

    data_a, data_b = load(config_a), load(config_b)
    X_train_a, y_train, train_ids_a = data_a["train"]
    X_val_a, y_val, val_ids_a = data_a["validation"]
    X_train_b, y_train_b, train_ids_b = data_b["train"]
    X_val_b, y_val_b, val_ids_b = data_b["validation"]

    # os.listdir order can differ between the two encoders' cache dirs even for the same
    # sample_ids -- reindex b to a's order explicitly rather than assuming they match.
    def reorder(X_b, ids_b, y_b, ids_a, y_a):
        by_id = {sid: i for i, sid in enumerate(ids_b)}
        idx = [by_id[sid] for sid in ids_a]
        assert (y_b[idx] == y_a).all(), "label mismatch after reindexing -- sample_id sets differ between encoders"
        return X_b[idx]

    X_train_b = reorder(X_train_b, train_ids_b, y_train_b, train_ids_a, y_train)
    X_val_b = reorder(X_val_b, val_ids_b, y_val_b, val_ids_a, y_val)

    oof_a = oof_probe_probs(X_train_a, y_train, config_a["classifier"], config_a.get("pca_dim"))
    oof_b = oof_probe_probs(X_train_b, y_train, config_b["classifier"], config_b.get("pca_dim"))

    alpha_nll, train_nll = fit_fusion_weight(oof_a, oof_b, y_train)
    alpha_acc, train_acc = fit_fusion_weight_accuracy(oof_a, oof_b, y_train)
    print(f"[{dataset_key}] fusion weight fit on train OOF: alpha_nll(a)={alpha_nll:.3f} (train NLL={train_nll:.4f}), "
          f"alpha_acc(a)={alpha_acc:.3f} (train OOF acc={train_acc:.4f})")

    scaler_a, pca_a, clf_a = fit_probe(X_train_a, y_train, config_a["classifier"], config_a.get("pca_dim"))
    scaler_b, pca_b, clf_b = fit_probe(X_train_b, y_train, config_b["classifier"], config_b.get("pca_dim"))
    probs_val_a = clf_a.predict_proba(transform(scaler_a, pca_a, X_val_a))
    probs_val_b = clf_b.predict_proba(transform(scaler_b, pca_b, X_val_b))

    fused_nll = alpha_nll * probs_val_a + (1 - alpha_nll) * probs_val_b
    fused_acc = alpha_acc * probs_val_a + (1 - alpha_acc) * probs_val_b

    configs = {
        f"{config_a['encoder_name']}_alone": probs_val_a,
        f"{config_b['encoder_name']}_alone": probs_val_b,
        "oof_fused_nll_optimal": fused_nll,
        "oof_fused_accuracy_optimal": fused_acc,
    }
    cis = {name: bootstrap_ci_top1(p, y_val, n_boot=n_boot) for name, p in configs.items()}
    for name, ci in cis.items():
        print(f"  {name}: top1={ci['top1']:.4f} 95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

    comps = {}
    for fused_name in ("oof_fused_nll_optimal", "oof_fused_accuracy_optimal"):
        for name in (f"{config_a['encoder_name']}_alone", f"{config_b['encoder_name']}_alone"):
            comps[f"{fused_name}_vs_{name}"] = compare(fused_name, configs[fused_name], name, configs[name], y_val, n_boot=n_boot)

    out = {"dataset": dataset_key, "config_a": config_a, "config_b": config_b,
           "alpha_nll_optimal": alpha_nll, "alpha_accuracy_optimal": alpha_acc,
           "individual_ci": cis, "paired_comparisons": comps}
    out_dir = os.path.join(RESULTS_DIR, "encoder_fusion_oof")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}_{config_a['encoder_name']}_x_{config_b['encoder_name']}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{dataset_key}] saved -> {out_path}")
    return out


# MERT-v2-30s (current best) is always encoder "a" -- config per dataset.
MERTV2_CFG = {
    "A": {"encoder_name": "mertv2_30s", "layer": 10, "classifier": "logreg"},
    "B": {"encoder_name": "mertv2_30s", "layer": "mean_all", "classifier": "logreg", "pca_dim": 128},
}
# Partner encoder ("b") configs, keyed by --partner. Add a new entry here to fusion-check
# any future encoder against MERT-v2 without touching run()'s general logic.
PARTNER_CFGS = {
    "pupum2d_large": {
        "A": {"encoder_name": "pupum2d_large", "layer": 0, "classifier": "logreg", "pca_dim": 128},
        "B": {"encoder_name": "pupum2d_large", "layer": 0, "classifier": "logreg"},
    },
    "maest": {
        "A": {"encoder_name": "maest", "layer": "mean_all", "classifier": "svm"},
        "B": {"encoder_name": "maest", "layer": "concat_last4", "classifier": "logreg"},
    },
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--partner", choices=list(PARTNER_CFGS.keys()), default="pupum2d_large")
    args = p.parse_args()
    run(args.dataset, MERTV2_CFG[args.dataset], PARTNER_CFGS[args.partner][args.dataset])
