"""SORD (Diaz & Marathe, CVPR 2019) distance-aware soft labels for Task 1 (round-4 queue
item 2, top Task-1 recommendation, IMPROVEMENT_FINDINGS_ROUND4.md). Unlike CORAL (tried
in round 3, failed: 0.318 vs 0.500 for a capacity-matched nominal classifier, because its
K-1 binary thresholds share a single weight vector), SORD keeps the *full* 6-logit linear
head -- ordinality is injected through the training *targets* instead of the architecture,
so it should not have CORAL's capacity bottleneck. Same frozen MuQ layer1 features as
every other Task 1 ordinal-method test this project, for a fair comparison.

Also includes the free, zero-retraining "expected-rank decoding" post-hoc check the same
research round suggested (probability-weighted mean rank instead of argmax on an already-
trained nominal classifier's output) -- run first, found clearly negative (0.333 vs 0.508
argmax), logged in WORKLOG.md, not repeated here.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.train_probe import load_cached
from src.metrics import summarize
from src.significance import bootstrap_ci_top1, compare

BEST_ENCODER_LAYER = {"A": ("muq_large_msd", 1)}


def sord_targets(y, n_class, temperature=1.0):
    """y_soft[i,k] = softmax_k(-|k - y_i| / temperature) -- a full probability
    distribution over all classes per sample, concentrated near the true rank but with
    a full 6-logit target (not a binary decomposition)."""
    y = np.asarray(y)
    ranks = np.arange(n_class)
    dist = -np.abs(ranks[None, :] - y[:, None]).astype(np.float32) / temperature
    dist = dist - dist.max(axis=1, keepdims=True)
    p = np.exp(dist)
    return p / p.sum(axis=1, keepdims=True)


class LinearHead(nn.Module):
    def __init__(self, in_dim, n_class):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_class)

    def forward(self, x):
        return self.fc(x)


def train_sord(X_train, y_train, X_val, y_val, n_class, temperature=1.0, epochs=800, lr=0.01, weight_decay=1e-4, seed=0):
    torch.manual_seed(seed)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr_soft = torch.tensor(sord_targets(y_train, n_class, temperature))
    Xval = torch.tensor(X_val, dtype=torch.float32)

    mean, std = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xval = (Xtr - mean) / std, (Xval - mean) / std

    model = LinearHead(Xtr.shape[1], n_class)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(ytr_soft * log_probs).sum(dim=1).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_probs = torch.softmax(model(Xval), dim=1).numpy()
    return model, val_probs, float(loss.item())


def train_ce(X_train, y_train, X_val, y_val, n_class, epochs=800, lr=0.01, weight_decay=1e-4, seed=0):
    """Plain one-hot cross-entropy through the identical architecture/optimizer as
    train_sord -- the fair, matched-training comparison (isolates the loss function
    as the only variable, unlike comparing against sklearn's differently-optimized
    LogisticRegression)."""
    torch.manual_seed(seed)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xval = torch.tensor(X_val, dtype=torch.float32)
    mean, std = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xval = (Xtr - mean) / std, (Xval - mean) / std

    model = LinearHead(Xtr.shape[1], n_class)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossfn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = lossfn(model(Xtr), ytr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        val_probs = torch.softmax(model(Xval), dim=1).numpy()
    return model, val_probs, float(loss.item())


def run(dataset_key="A", n_boot=10000):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    n_class = len(labels)
    encoder_name, layer = BEST_ENCODER_LAYER[dataset_key]

    cache_dir = os.path.join("cache", encoder_name, dataset_key)
    data = load_cached(dataset_key, cache_dir, layer)
    X_train, y_train, _ = data["train"]
    X_val, y_val, val_ids = data["validation"]

    lr_grid = (0.003, 0.01, 0.03)
    wd_grid = (1e-4, 1e-3, 1e-2, 1e-1)
    temp_grid = (0.3, 0.5, 0.7, 1.0)

    # matched-architecture plain-CE baseline, hyperparameter-tuned -- the fair
    # comparison (isolates loss function, not optimizer quality)
    best_ce = (-1.0, None, None)
    for lr in lr_grid:
        for wd in wd_grid:
            _model, probs, _loss = train_ce(X_train, y_train, X_val, y_val, n_class, lr=lr, weight_decay=wd)
            m = summarize(probs, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
            if m["top1"] > best_ce[0]:
                best_ce = (m["top1"], probs, {"lr": lr, "weight_decay": wd, "metrics": m})
    print(f"[{dataset_key}] best tuned plain-CE (matched arch): top1={best_ce[0]:.4f} "
          f"cfg={best_ce[2]['lr'], best_ce[2]['weight_decay']} MAE={best_ce[2]['metrics'].get('mean_abs_decade_error')}")

    # SORD, hyperparameter-tuned (temperature x lr x weight-decay)
    best_sord = (-1.0, None, None)
    for temperature in temp_grid:
        for lr in lr_grid:
            for wd in wd_grid:
                _model, probs, _loss = train_sord(X_train, y_train, X_val, y_val, n_class, temperature=temperature, lr=lr, weight_decay=wd)
                m = summarize(probs, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
                if m["top1"] > best_sord[0]:
                    best_sord = (m["top1"], probs, {"temperature": temperature, "lr": lr, "weight_decay": wd, "metrics": m})
    print(f"[{dataset_key}] best tuned SORD: top1={best_sord[0]:.4f} cfg={best_sord[2]['temperature'], best_sord[2]['lr'], best_sord[2]['weight_decay']} "
          f"MAE={best_sord[2]['metrics'].get('mean_abs_decade_error')}")

    # practical baseline: sklearn's properly-optimized (LBFGS) nominal logreg --
    # reported separately, not conflated with the matched-architecture comparison above
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    logreg_search = GridSearchCV(LogisticRegression(max_iter=5000),
                                  {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}, cv=5, scoring="accuracy")
    logreg_search.fit(scaler.transform(X_train), y_train)
    logreg_probs = logreg_search.best_estimator_.predict_proba(scaler.transform(X_val))
    m_logreg = summarize(logreg_probs, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
    print(f"[{dataset_key}] sklearn logreg (LBFGS, practical baseline): top1={m_logreg['top1']:.4f} "
          f"MAE={m_logreg.get('mean_abs_decade_error')}")

    cis = {"sord_tuned": bootstrap_ci_top1(best_sord[1], y_val, n_boot=n_boot),
           "ce_tuned": bootstrap_ci_top1(best_ce[1], y_val, n_boot=n_boot),
           "sklearn_logreg": bootstrap_ci_top1(logreg_probs, y_val, n_boot=n_boot)}
    for name, ci in cis.items():
        print(f"  {name}: top1={ci['top1']:.4f} 95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")
    comp_fair = compare("sord_tuned", best_sord[1], "ce_tuned", best_ce[1], y_val, n_boot=n_boot)
    print(f"  sord_tuned vs ce_tuned (fair, matched optimizer): {comp_fair['paired_bootstrap']}")
    comp_practical = compare("sord_tuned", best_sord[1], "sklearn_logreg", logreg_probs, y_val, n_boot=n_boot)
    print(f"  sord_tuned vs sklearn_logreg (practical baseline): {comp_practical['paired_bootstrap']}")

    out = {"best_ce_matched": best_ce[2], "best_sord": best_sord[2], "sklearn_logreg_metrics": m_logreg,
           "individual_ci": cis, "sord_vs_ce_fair": comp_fair, "sord_vs_sklearn_logreg": comp_practical}
    out_dir = os.path.join(RESULTS_DIR, "sord_ordinal")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}_{encoder_name}_layer{layer}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{dataset_key}] saved -> {out_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A"], default="A")
    args = p.parse_args()
    run(args.dataset)
