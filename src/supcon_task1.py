"""Supervised contrastive loss (SupCon, Khosla et al. 2020) for Task 1 -- old backlog
item ("Soft-label KD distillation from AF3 into the MERT classifier; supervised
contrastive loss on Task 1"). The KD half is not pursued: AF3 is a measured-weak teacher
for Task 1 (0.379 zero-shot, well below MERT-v2/MuQ), and distilling a weaker teacher's
soft labels into a stronger student rarely helps -- a real, previously-measured fact
about this project's own data, not a presumption. The contrastive-loss half is
independent of AF3 and cheaply testable on its own merits, so it's run here.

Standard 2-stage SupCon protocol on MERT-v2 layer10 features (current best Task 1
encoder): (1) train a small projection head with the supervised contrastive loss (pulls
same-label embeddings together, pushes different-label embeddings apart, using the full
batch's label structure, not just positive/negative pairs from augmentation since we have
no augmentation pipeline for frozen embeddings); (2) freeze the projection, train a linear
classifier on top (the standard SupCon evaluation protocol). Matched-architecture plain-CE
baseline included per this project's standing practice (same projection dim, same total
capacity, isolates the contrastive pretraining step as the only variable).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.train_probe import load_cached
from src.metrics import summarize
from src.significance import bootstrap_ci_top1, compare

BEST_ENCODER_LAYER = {"A": ("mertv2_30s", 10)}


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, proj_dim))

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1)


def supcon_loss(z, labels, temperature=0.1):
    """z: (N, D) L2-normalized embeddings. labels: (N,) int. Standard SupCon (all
    same-label pairs in the batch are positives, following Khosla et al. eq. 2)."""
    sim = z @ z.T / temperature
    n = z.shape[0]
    mask_self = torch.eye(n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask_self, float("-inf"))
    labels = labels.view(-1, 1)
    mask_pos = (labels == labels.T) & ~mask_self

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # the self-masked diagonal of log_prob is -inf; masking via multiplication (0 * -inf)
    # would produce NaN even at positions mask_pos correctly excludes -- select via
    # torch.where instead so excluded entries are a real 0, not 0*(-inf).
    masked_log_prob = torch.where(mask_pos, log_prob, torch.zeros_like(log_prob))
    pos_counts = mask_pos.sum(dim=1)
    has_pos = pos_counts > 0
    mean_log_prob_pos = masked_log_prob.sum(dim=1)[has_pos] / pos_counts[has_pos].float()
    return -mean_log_prob_pos.mean()


def train_supcon(X_train, y_train, X_val, y_val, n_class, proj_dim=128, epochs=500,
                  lr=0.01, weight_decay=1e-4, temperature=0.1, seed=0):
    torch.manual_seed(seed)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xval = torch.tensor(X_val, dtype=torch.float32)
    mean, std = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xval = (Xtr - mean) / std, (Xval - mean) / std

    proj = ProjectionHead(Xtr.shape[1], proj_dim)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        proj.train()
        opt.zero_grad()
        z = proj(Xtr)
        loss = supcon_loss(z, ytr, temperature=temperature)
        loss.backward()
        opt.step()

    proj.eval()
    with torch.no_grad():
        z_train = proj(Xtr)
        z_val = proj(Xval)

    # stage 2: linear classifier on the frozen, contrastively-trained projection
    clf = nn.Linear(proj_dim, n_class)
    clf_opt = torch.optim.AdamW(clf.parameters(), lr=0.01, weight_decay=1e-4)
    for epoch in range(500):
        clf.train()
        clf_opt.zero_grad()
        logits = clf(z_train)
        clf_loss = F.cross_entropy(logits, ytr)
        clf_loss.backward()
        clf_opt.step()
    clf.eval()
    with torch.no_grad():
        val_probs = torch.softmax(clf(z_val), dim=1).numpy()
    return (proj, clf), val_probs, float(loss.item())


def train_ce_matched(X_train, y_train, X_val, y_val, n_class, proj_dim=128, epochs=500,
                       lr=0.01, weight_decay=1e-4, seed=0):
    """Same projection-head architecture + linear classifier, trained end-to-end with
    plain CE instead of the 2-stage SupCon protocol -- isolates the contrastive
    pretraining step as the only variable."""
    torch.manual_seed(seed)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xval = torch.tensor(X_val, dtype=torch.float32)
    mean, std = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xval = (Xtr - mean) / std, (Xval - mean) / std

    proj = ProjectionHead(Xtr.shape[1], proj_dim)
    clf = nn.Linear(proj_dim, n_class)
    params = list(proj.parameters()) + list(clf.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        proj.train(); clf.train()
        opt.zero_grad()
        logits = clf(proj(Xtr))
        loss = F.cross_entropy(logits, ytr)
        loss.backward()
        opt.step()
    proj.eval(); clf.eval()
    with torch.no_grad():
        val_probs = torch.softmax(clf(proj(Xval)), dim=1).numpy()
    return (proj, clf), val_probs, float(loss.item())


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
    wd_grid = (1e-4, 1e-3, 1e-2)
    temp_grid = (0.05, 0.1, 0.2, 0.5)

    best_ce = (-1.0, None, None)
    for lr in lr_grid:
        for wd in wd_grid:
            _model, probs, _loss = train_ce_matched(X_train, y_train, X_val, y_val, n_class, lr=lr, weight_decay=wd)
            m = summarize(probs, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
            if m["top1"] > best_ce[0]:
                best_ce = (m["top1"], probs, {"lr": lr, "weight_decay": wd, "metrics": m})
    print(f"[{dataset_key}] best tuned CE-matched (same projection+linear head): top1={best_ce[0]:.4f} "
          f"top3={best_ce[2]['metrics']['top3']:.4f} cfg=({best_ce[2]['lr']}, {best_ce[2]['weight_decay']})")

    best_supcon = (-1.0, None, None)
    for temperature in temp_grid:
        for lr in lr_grid:
            for wd in wd_grid:
                _model, probs, _loss = train_supcon(X_train, y_train, X_val, y_val, n_class,
                                                       lr=lr, weight_decay=wd, temperature=temperature)
                m = summarize(probs, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
                if m["top1"] > best_supcon[0]:
                    best_supcon = (m["top1"], probs, {"temperature": temperature, "lr": lr, "weight_decay": wd, "metrics": m})
    print(f"[{dataset_key}] best tuned SupCon (2-stage): top1={best_supcon[0]:.4f} "
          f"top3={best_supcon[2]['metrics']['top3']:.4f} cfg=({best_supcon[2]['temperature']}, {best_supcon[2]['lr']}, {best_supcon[2]['weight_decay']})")

    cis = {"ce_matched": bootstrap_ci_top1(best_ce[1], y_val, n_boot=n_boot),
           "supcon": bootstrap_ci_top1(best_supcon[1], y_val, n_boot=n_boot)}
    for name, ci in cis.items():
        print(f"  {name}: top1={ci['top1']:.4f} 95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")
    comp = compare("supcon", best_supcon[1], "ce_matched", best_ce[1], y_val, n_boot=n_boot)
    print(f"  supcon vs ce_matched: {comp['paired_bootstrap']}")

    out = {"best_ce_matched": best_ce[2], "best_supcon": best_supcon[2],
           "individual_ci": cis, "supcon_vs_ce": comp}
    out_dir = os.path.join(RESULTS_DIR, "supcon_task1")
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
