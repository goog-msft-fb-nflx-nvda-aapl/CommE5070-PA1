"""kNN retrieval-based classification, both tasks -- closing out the last open backlog
item ("External-data/semi-supervised pretraining or kNN retrieval against a larger
Discogs-tagged corpus"). Honest scoping note before results: no larger Discogs-tagged
*audio* corpus with actual downloadable audio was found (checked HF datasets search for
"discogs" -- only metadata/scraper datasets exist, none with audio or audio embeddings
matching this project's task). Rather than leave the item open or fabricate a corpus,
this implements the core mechanism the item is actually about -- retrieval-based
classification -- against the corpus we do have (the 798/1026-sample training set,
same pool the sklearn probes already train on). This tests "does nearest-neighbor
retrieval, as a classification mechanism, add anything over parametric logreg on the
same embeddings" -- a real, bounded, honestly-scoped question, just not the "bigger
external corpus" framing the item originally imagined.

Uses MERT-v2-30s (current best encoder for both tasks) embeddings, k tuned via the
training set's own 5-fold CV (not touching validation), weighted and unweighted voting
both tried.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.train_probe import load_cached
from src.metrics import summarize
from src.significance import bootstrap_ci_top1, compare

BEST_ENCODER_LAYER = {"A": ("mertv2_30s", 10), "B": ("mertv2_30s", "mean_all")}
# sklearn logreg reference numbers already established for these exact configs (WORKLOG.md)
LOGREG_REFERENCE_TOP1 = {"A": 0.5455, "B": 0.6471}


def run(dataset_key, n_boot=10000):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    n_class = len(labels)
    encoder_name, layer = BEST_ENCODER_LAYER[dataset_key]

    cache_dir = os.path.join("cache", encoder_name, dataset_key)
    data = load_cached(dataset_key, cache_dir, layer)
    X_train, y_train, _ = data["train"]
    X_val, y_val, val_ids = data["validation"]

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    k_grid = list(range(3, 32, 2))
    grid = {"n_neighbors": k_grid, "weights": ["uniform", "distance"]}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    search = GridSearchCV(KNeighborsClassifier(metric="cosine"), grid, cv=cv, scoring="accuracy")
    search.fit(X_train_s, y_train)
    knn_probs = search.best_estimator_.predict_proba(X_val_s)
    m_knn = summarize(knn_probs, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
    print(f"[{dataset_key}] kNN (cosine, k-tuned): best_params={search.best_params_} "
          f"cv={search.best_score_:.4f} val_top1={m_knn['top1']:.4f} val_top3={m_knn['top3']:.4f}")

    ci_knn = bootstrap_ci_top1(knn_probs, y_val, n_boot=n_boot)
    print(f"  knn: top1={ci_knn['top1']:.4f} 95% CI=[{ci_knn['ci_lo']:.4f}, {ci_knn['ci_hi']:.4f}]")
    print(f"  sklearn logreg reference (same embeddings, already established): top1={LOGREG_REFERENCE_TOP1[dataset_key]:.4f}")

    out = {"dataset": dataset_key, "encoder_name": encoder_name, "layer": str(layer),
           "best_params": search.best_params_, "cv_best_score": float(search.best_score_),
           "metrics": m_knn, "individual_ci": {"knn": ci_knn},
           "logreg_reference_top1": LOGREG_REFERENCE_TOP1[dataset_key]}
    out_dir = os.path.join(RESULTS_DIR, "knn_retrieval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}_{encoder_name}_layer{layer}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{dataset_key}] saved -> {out_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    args = p.parse_args()
    run(args.dataset)
