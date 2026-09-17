"""Train a frozen-embedding classifier probe (logistic regression / MLP) on cached
encoder features (e.g. MERT layer-pooled means from mert_features.py).

No artist_id is available in the manifest (anonymized), so k-fold CV for model
selection uses plain StratifiedKFold, not artist-grouped folds -- noted as a
limitation in the report per PLAN.md.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR, RESULTS_DIR
from src.metrics import summarize


def load_cached(dataset_key, cache_dir, layer):
    spec = DATASETS[dataset_key]
    label_to_int = spec["label_to_int"]
    data = {"train": ([], [], []), "validation": ([], [], []), "test": ([], [], [])}
    for fname in os.listdir(cache_dir):
        if not fname.endswith(".npz"):
            continue
        npz = np.load(os.path.join(cache_dir, fname), allow_pickle=True)
        split = str(npz["split"])
        emb = npz["embedding"]  # (n_layers, dim)
        if layer == "mean_all":
            vec = emb.mean(axis=0)
        elif layer == "concat_last4":
            vec = emb[-4:].reshape(-1)
        else:
            vec = emb[int(layer)]
        label_str = str(npz["label"])
        label_int = label_to_int[label_str] if label_str else -1
        xs, ys, ids = data[split]
        xs.append(vec)
        ys.append(label_int)
        ids.append(str(npz["sample_id"]))
    return {k: (np.stack(v[0]) if v[0] else np.empty((0,)), np.array(v[1]), v[2]) for k, v in data.items()}


def run(dataset_key, layer="mean_all", classifier="logreg", pca_dim=None,
        cache_dir=None, out_dir=None, encoder_name="mert_v1_330m", n_jobs=1):
    cache_dir = cache_dir or os.path.join(CACHE_DIR, encoder_name, dataset_key)
    out_dir = out_dir or os.path.join(RESULTS_DIR, f"{encoder_name}_probe_{dataset_key}")
    os.makedirs(out_dir, exist_ok=True)

    spec = DATASETS[dataset_key]
    n_class = len(spec["labels"])
    data = load_cached(dataset_key, cache_dir, layer)
    X_train, y_train, _ = data["train"]
    X_val, y_val, val_ids = data["validation"]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    if pca_dim:
        pca = PCA(n_components=pca_dim, random_state=0).fit(X_train_s)
        X_train_s = pca.transform(X_train_s)
        X_val_s = pca.transform(X_val_s)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    if classifier == "logreg":
        base = LogisticRegression(max_iter=5000, multi_class="multinomial")
        grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}
    elif classifier == "mlp":
        base = MLPClassifier(hidden_layer_sizes=(128,), max_iter=2000, random_state=0)
        grid = {"alpha": [1e-4, 1e-3, 1e-2, 1e-1]}
    elif classifier == "svm":
        # max_iter caps runaway convergence time on pathological (C, kernel) combos
        # at this scale/dimensionality -- hit one that ran 40+ min unbounded.
        base = SVC(probability=True, random_state=0, max_iter=20000)
        grid = {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "kernel": ["rbf", "linear"]}
    else:
        raise ValueError(classifier)

    search = GridSearchCV(base, grid, cv=cv, scoring="accuracy", n_jobs=n_jobs)
    search.fit(X_train_s, y_train)
    best = search.best_estimator_

    probs = best.predict_proba(X_val_s)
    metrics = summarize(probs, y_val, n_class, ordinal=spec["ordinal"], label_names=spec["labels"])
    metrics["best_params"] = search.best_params_
    metrics["cv_best_score"] = float(search.best_score_)
    metrics["layer"] = str(layer)
    metrics["classifier"] = classifier
    metrics["pca_dim"] = pca_dim

    with open(os.path.join(out_dir, f"metrics_layer{layer}_{classifier}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{dataset_key}] layer={layer} clf={classifier} cv={search.best_score_:.4f} "
          f"val_top1={metrics['top1']:.4f} val_top3={metrics['top3']:.4f}")
    fitted = {"scaler": scaler, "pca": pca if pca_dim else None, "clf": best}
    return metrics, fitted


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--layer", default="mean_all", help="int layer index, 'mean_all', or 'concat_last4'")
    p.add_argument("--classifier", choices=["logreg", "mlp"], default="logreg")
    p.add_argument("--pca-dim", type=int, default=None)
    p.add_argument("--encoder-name", default="mert_v1_330m")
    args = p.parse_args()
    run(args.dataset, layer=args.layer, classifier=args.classifier, pca_dim=args.pca_dim, encoder_name=args.encoder_name)
