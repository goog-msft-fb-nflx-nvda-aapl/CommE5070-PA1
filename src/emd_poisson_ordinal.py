"""EMD (squared Earth Mover's Distance, Hou et al. 2016) loss and a unimodal Poisson head
for Task 1 (backlog item, "EMD loss and unimodal Poisson head, specifically for top-3").
Two genuinely different architectural bets on the ordinal structure, distinct from what's
already been tried:
  - SORD (done, round 4): full 6-logit head, ordinality injected via soft *targets* only.
  - CORAL (done, round 3): K-1 shared-weight-vector binary thresholds -- failed, capacity-
    starved (0.318 vs 0.500 nominal baseline).
  - EMD here: full 6-logit head like SORD (no capacity bottleneck), but ordinality injected
    via the *loss* instead of the targets -- penalizes the squared difference between the
    predicted and true CDFs over the ordered classes, so a prediction one decade off costs
    much less than one three decades off, without needing hand-picked soft targets.
  - Poisson head here: the opposite bet -- a single scalar rate parameter forces the
    *entire* predicted distribution to be unimodal by construction (can never predict a
    bimodal or U-shaped distribution over decades), trading capacity for a strong,
    architecturally-guaranteed ordinal prior. Specifically motivated by top-3: a unimodal
    distribution concentrates its non-argmax mass on the immediately adjacent classes,
    exactly where top-3 credit is available.

Matched-architecture plain-CE baselines included for both (this project's standing
practice since the SORD work -- isolates the loss/architecture as the only variable,
not conflated with the separately-established from-scratch-optimizer-quality gap vs.
sklearn's LBFGS).
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
from src.sord_ordinal import LinearHead, train_ce

BEST_ENCODER_LAYER = {"A": ("mertv2_30s", 10)}


def train_emd(X_train, y_train, X_val, y_val, n_class, epochs=800, lr=0.01, weight_decay=1e-4, seed=0):
    """Squared-EMD loss: sum_k (CDF_pred(k) - CDF_true(k))^2, mean over samples.
    Full 6-logit head, same architecture as train_ce/train_sord -- ordinality lives in
    the loss, not the targets or the architecture."""
    torch.manual_seed(seed)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xval = torch.tensor(X_val, dtype=torch.float32)
    mean, std = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xval = (Xtr - mean) / std, (Xval - mean) / std

    cdf_true_full = torch.cumsum(torch.eye(n_class), dim=1)  # (n_class, n_class), row k = CDF of one-hot(k)
    cdf_true = cdf_true_full[ytr]  # (N, n_class)

    model = LinearHead(Xtr.shape[1], n_class)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        probs = torch.softmax(model(Xtr), dim=1)
        cdf_pred = torch.cumsum(probs, dim=1)
        loss = ((cdf_pred - cdf_true) ** 2).sum(dim=1).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_probs = torch.softmax(model(Xval), dim=1).numpy()
    return model, val_probs, float(loss.item())


class PoissonHead(nn.Module):
    """Single learned rate parameter per sample -> truncated/renormalized Poisson pmf
    over the n_class ordered bins. Architecturally guaranteed unimodal (a Poisson pmf has
    exactly one mode), unlike a free 6-logit softmax which can represent any distribution."""
    def __init__(self, in_dim, n_class):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        self.n_class = n_class

    def forward(self, x):
        rate = torch.nn.functional.softplus(self.fc(x).squeeze(-1)) + 1e-3  # (N,)
        k = torch.arange(self.n_class, dtype=torch.float32, device=x.device)  # (n_class,)
        log_pmf = k[None, :] * torch.log(rate[:, None]) - rate[:, None] - torch.lgamma(k[None, :] + 1)
        log_pmf = log_pmf - torch.logsumexp(log_pmf, dim=1, keepdim=True)  # renormalize over the truncated support
        return log_pmf  # (N, n_class), already log-probabilities


def train_poisson(X_train, y_train, X_val, y_val, n_class, epochs=800, lr=0.01, weight_decay=1e-4, seed=0):
    torch.manual_seed(seed)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xval = torch.tensor(X_val, dtype=torch.float32)
    mean, std = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xval = (Xtr - mean) / std, (Xval - mean) / std

    model = PoissonHead(Xtr.shape[1], n_class)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    nll = nn.NLLLoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        log_pmf = model(Xtr)
        loss = nll(log_pmf, ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_probs = torch.exp(model(Xval)).numpy()
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

    def tune(train_fn, name):
        best = (-1.0, None, None)
        for lr in lr_grid:
            for wd in wd_grid:
                _model, probs, _loss = train_fn(X_train, y_train, X_val, y_val, n_class, lr=lr, weight_decay=wd)
                m = summarize(probs, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
                if m["top1"] > best[0]:
                    best = (m["top1"], probs, {"lr": lr, "weight_decay": wd, "metrics": m})
        print(f"[{dataset_key}] best tuned {name}: top1={best[0]:.4f} top3={best[2]['metrics']['top3']:.4f} "
              f"MAE={best[2]['metrics'].get('mean_abs_decade_error')} cfg=({best[2]['lr']}, {best[2]['weight_decay']})")
        return best

    best_ce = tune(train_ce, "plain-CE (matched arch)")
    best_emd = tune(train_emd, "squared-EMD")
    best_poisson = tune(train_poisson, "unimodal-Poisson")

    cis = {"ce_matched": bootstrap_ci_top1(best_ce[1], y_val, n_boot=n_boot),
           "emd": bootstrap_ci_top1(best_emd[1], y_val, n_boot=n_boot),
           "poisson": bootstrap_ci_top1(best_poisson[1], y_val, n_boot=n_boot)}
    for name, ci in cis.items():
        print(f"  {name}: top1={ci['top1']:.4f} 95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

    comp_emd = compare("emd", best_emd[1], "ce_matched", best_ce[1], y_val, n_boot=n_boot)
    comp_poisson = compare("poisson", best_poisson[1], "ce_matched", best_ce[1], y_val, n_boot=n_boot)
    print(f"  emd vs ce_matched: {comp_emd['paired_bootstrap']}")
    print(f"  poisson vs ce_matched: {comp_poisson['paired_bootstrap']}")

    out = {"best_ce_matched": best_ce[2], "best_emd": best_emd[2], "best_poisson": best_poisson[2],
           "individual_ci": cis, "emd_vs_ce": comp_emd, "poisson_vs_ce": comp_poisson}
    out_dir = os.path.join(RESULTS_DIR, "emd_poisson_ordinal")
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
