"""Round-3 queue item 2: CORAL ordinal head for Task 1 (Cao et al. 2019/2020,
arXiv:1901.07884, cumulative-link/"extended binary classification" reformulation of
K-way ordinal regression) -- the strongest cross-source-convergent recommendation from
IMPROVEMENT_FINDINGS_ROUND3.md for Task 1. Directly tests whether the nominal-vs-ordinal
decision surface (plain 6-way softmax) is why several of our top Task-1 configs are
statistically tied: our own confusion diagnostic already found adjacent-decade errors
dominate, which is exactly the structure CORAL is built to exploit.

Trained as a small linear head on top of the best already-cached frozen encoder layer
(no encoder fine-tuning) -- cheap, and cleanly attributable to the ordinal-vs-nominal
question alone.

Implementation note (stated plainly, not hidden): this uses CORAL's extended-binary
label encoding and shared-weight-vector-plus-per-threshold-bias architecture, but does
NOT hard-constrain the K-1 bias terms to be monotonically decreasing during optimization
(the original CORAL paper's rank-consistency guarantee assumes that constraint). This is
a common, simpler practical relaxation -- rank-consistency is empirically encouraged by
the shared weight vector and extended-binary loss, not mathematically guaranteed here.
Checked directly (see `run()`'s consistency-violation print) rather than assumed.
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

BEST_ENCODER_LAYER = {"A": ("muq_large_msd", 1), "B": ("muq_large_msd", 2)}


class CoralHead(nn.Module):
    """logit_k(x) = w^T x + b_k for k=0..n_class-2, representing P(y > k | x)."""

    def __init__(self, in_dim, n_class):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_class - 1))

    def forward(self, x):
        return self.fc(x) + self.bias


def extended_binary_targets(y, n_class):
    """t[i, k] = 1 if y[i] > k else 0, for k=0..n_class-2."""
    y = np.asarray(y)
    ranks = np.arange(n_class - 1)
    return (y[:, None] > ranks[None, :]).astype(np.float32)


def coral_logits_to_class_probs(logits):
    """P(y>k) = sigmoid(logit_k); P(y=k) = P(y>k-1) - P(y>k), with P(y>-1)=1 and
    P(y>K-1)=0. Clipped to non-negative and renormalized -- the monotonicity that
    would make this exact isn't hard-enforced (see module docstring), so this is a
    practical (not mathematically guaranteed) conversion to a proper probability
    vector, same as the coral-pytorch reference implementation's approach."""
    probs_gt = torch.sigmoid(logits)  # (n, K-1), P(y>k)
    ones = torch.ones(probs_gt.shape[0], 1, device=probs_gt.device)
    zeros = torch.zeros(probs_gt.shape[0], 1, device=probs_gt.device)
    cum = torch.cat([ones, probs_gt, zeros], dim=1)  # (n, K+1): [P(y>-1)..P(y>K-1)]
    class_probs = cum[:, :-1] - cum[:, 1:]  # (n, K)
    class_probs = class_probs.clamp(min=1e-6)
    return class_probs / class_probs.sum(dim=1, keepdim=True)


def train_coral(X_train, y_train, X_val, y_val, n_class, epochs=300, lr=0.01, weight_decay=1e-4, seed=0):
    torch.manual_seed(seed)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr_ext = torch.tensor(extended_binary_targets(y_train, n_class))
    Xval = torch.tensor(X_val, dtype=torch.float32)

    mean, std = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xval = (Xtr - mean) / std, (Xval - mean) / std

    model = CoralHead(Xtr.shape[1], n_class)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        loss = loss_fn(logits, ytr_ext)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(Xval)
        val_probs = coral_logits_to_class_probs(val_logits).numpy()
        # rank-consistency check: how often is sigmoid(logit_k) NOT monotonically
        # decreasing in k (i.e. the un-enforced constraint is actually violated)?
        val_probs_gt = torch.sigmoid(val_logits).numpy()
        violations = (np.diff(val_probs_gt, axis=1) > 0).any(axis=1).mean()
    return model, val_probs, float(loss.item()), float(violations)


def run(dataset_key="A", n_boot=10000):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    n_class = len(labels)
    encoder_name, layer = BEST_ENCODER_LAYER[dataset_key]

    cache_dir = os.path.join("cache", encoder_name, dataset_key)
    data = load_cached(dataset_key, cache_dir, layer)
    X_train, y_train, _ = data["train"]
    X_val, y_val, val_ids = data["validation"]

    model, coral_probs, final_loss, violation_rate = train_coral(X_train, y_train, X_val, y_val, n_class)
    print(f"[{dataset_key}] CORAL head on {encoder_name} layer{layer}: final_train_loss={final_loss:.4f} "
          f"rank_consistency_violation_rate={violation_rate:.3f} (fraction of val samples where "
          f"the un-enforced monotonicity constraint doesn't hold)")

    metrics = summarize(coral_probs, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
    print(f"[{dataset_key}] CORAL validation: top1={metrics['top1']:.4f} top3={metrics['top3']:.4f} "
          f"MAE={metrics.get('mean_abs_decade_error', 'n/a')} "
          f"acc_within_1_decade={metrics.get('acc_within_1_decade', 'n/a')} "
          f"QWK={metrics.get('quadratic_weighted_kappa', 'n/a')}")

    # Two nominal baselines on the exact same layer/features:
    # (a) a LINEAR nominal classifier (logreg) -- the fair, capacity-matched comparison
    #     for isolating "does the ordinal decision surface help", since CORAL's head
    #     here is also linear; SVM's RBF kernel is a different, nonlinear capacity.
    # (b) the actual established best-config classifier (SVM, per af3_stack.py's
    #     FROZEN_CONFIGS) -- kept for context, but NOT a fair ordinal-vs-nominal test
    #     on its own since it differs in capacity as well as decision surface.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    logreg_search = GridSearchCV(LogisticRegression(max_iter=5000),
                                  {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}, cv=5, scoring="accuracy")
    logreg_search.fit(scaler.transform(X_train), y_train)
    logreg_probs = logreg_search.best_estimator_.predict_proba(scaler.transform(X_val))
    logreg_metrics = summarize(logreg_probs, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
    print(f"[{dataset_key}] LINEAR nominal (logreg, same layer, capacity-matched to CORAL) validation: "
          f"top1={logreg_metrics['top1']:.4f} top3={logreg_metrics['top3']:.4f} "
          f"MAE={logreg_metrics.get('mean_abs_decade_error', 'n/a')} "
          f"acc_within_1_decade={logreg_metrics.get('acc_within_1_decade', 'n/a')} "
          f"QWK={logreg_metrics.get('quadratic_weighted_kappa', 'n/a')}")

    from src.af3_stack import frozen_probe_probs
    nominal_probs, nominal_y, nominal_ids = frozen_probe_probs(dataset_key, encoder_name)
    from src.ensemble import align
    nominal_aligned = align(coral_probs, val_ids, nominal_probs, nominal_ids)
    nominal_metrics = summarize(nominal_aligned, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
    print(f"[{dataset_key}] established-best nominal (SVM, same layer, NOT capacity-matched) validation: "
          f"top1={nominal_metrics['top1']:.4f} top3={nominal_metrics['top3']:.4f} "
          f"MAE={nominal_metrics.get('mean_abs_decade_error', 'n/a')} "
          f"acc_within_1_decade={nominal_metrics.get('acc_within_1_decade', 'n/a')} "
          f"QWK={nominal_metrics.get('quadratic_weighted_kappa', 'n/a')}")

    cis = {
        "coral": bootstrap_ci_top1(coral_probs, y_val, n_boot=n_boot),
        "linear_nominal_logreg": bootstrap_ci_top1(logreg_probs, y_val, n_boot=n_boot),
        "established_best_svm": bootstrap_ci_top1(nominal_aligned, y_val, n_boot=n_boot),
    }
    for name, ci in cis.items():
        print(f"  {name}: top1={ci['top1']:.4f} 95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")
    comp_fair = compare("coral", coral_probs, "linear_nominal_logreg", logreg_probs, y_val, n_boot=n_boot)
    comp_svm = compare("coral", coral_probs, "established_best_svm", nominal_aligned, y_val, n_boot=n_boot)

    out = {"encoder": encoder_name, "layer": layer, "final_train_loss": final_loss,
           "rank_consistency_violation_rate": violation_rate,
           "coral_metrics": metrics, "linear_nominal_logreg_metrics": logreg_metrics,
           "established_best_svm_metrics": nominal_metrics,
           "individual_ci": cis,
           "coral_vs_linear_nominal_fair_comparison": comp_fair,
           "coral_vs_established_best_svm_context_only": comp_svm}
    out_dir = os.path.join(RESULTS_DIR, "coral_ordinal")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}_{encoder_name}_layer{layer}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{dataset_key}] saved -> {out_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], default="A")
    args = p.parse_args()
    run(args.dataset)
