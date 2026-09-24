"""Learned softmax-weighted sum of all layers as the probe input (round-5 queue item 2,
L02b slide 19's "MERT finetuning strategy"): freeze the backbone, learn
`z = sum_l softmax(w_l) * h_l` over all cached layers, then a single linear head --
instead of picking one layer by validation-score sweep. Two stated advantages: (1) can
blend information from multiple layers (e.g. mid-layer production/timbre + top-layer
style), (2) less vulnerable to the "winner's curse" of selecting the best of N layers
by validation score, which inflates the apparent accuracy of a single-layer choice.

Applied to both MuQ (13 layers incl. mean_all/concat_last4 excluded -- only true layers
used) and MERT-v2-30s (24 layers), whichever is each task's current-best encoder family,
so the comparison is apples-to-apples against that encoder's own best single-layer result.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR, CACHE_DIR
from src.metrics import summarize
from src.significance import bootstrap_ci_top1, compare


def load_all_layers(dataset_key, cache_dir):
    spec = DATASETS[dataset_key]
    label_to_int = spec["label_to_int"]
    data = {"train": ([], [], []), "validation": ([], [], []), "test": ([], [], [])}
    for fname in os.listdir(cache_dir):
        if not fname.endswith(".npz"):
            continue
        npz = np.load(os.path.join(cache_dir, fname), allow_pickle=True)
        split = str(npz["split"])
        emb = npz["embedding"]  # (n_layers, dim) -- every layer, not reduced
        label_str = str(npz["label"])
        label_int = label_to_int[label_str] if label_str else -1
        xs, ys, ids = data[split]
        xs.append(emb)
        ys.append(label_int)
        ids.append(str(npz["sample_id"]))
    return {k: (np.stack(v[0]) if v[0] else np.empty((0,)), np.array(v[1]), v[2]) for k, v in data.items()}


class WeightedLayerProbe(nn.Module):
    def __init__(self, n_layers, dim, n_class):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.head = nn.Linear(dim, n_class)

    def forward(self, x):
        # x: (batch, n_layers, dim)
        w = torch.softmax(self.layer_logits, dim=0)  # (n_layers,)
        z = (x * w[None, :, None]).sum(dim=1)  # (batch, dim)
        return self.head(z)


def train_weighted(X_train, y_train, X_val, y_val, n_class, epochs=500, lr=0.01, weight_decay=1e-3, seed=0):
    torch.manual_seed(seed)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xval = torch.tensor(X_val, dtype=torch.float32)

    # standardize per-layer-per-dim using train stats (fold the layer dim into batch
    # for standardization purposes, then reshape back)
    n_layers, dim = Xtr.shape[1], Xtr.shape[2]
    mean = Xtr.mean(dim=0, keepdim=True)
    std = Xtr.std(dim=0, keepdim=True) + 1e-6
    Xtr_n, Xval_n = (Xtr - mean) / std, (Xval - mean) / std

    model = WeightedLayerProbe(n_layers, dim, n_class)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossfn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = lossfn(model(Xtr_n), ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_probs = torch.softmax(model(Xval_n), dim=1).numpy()
        layer_weights = torch.softmax(model.layer_logits, dim=0).numpy()
    return model, val_probs, layer_weights, float(loss.item())


def run(dataset_key, encoder_name, n_boot=10000):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    n_class = len(labels)

    cache_dir = os.path.join(CACHE_DIR, encoder_name, dataset_key)
    data = load_all_layers(dataset_key, cache_dir)
    X_train, y_train, _ = data["train"]
    X_val, y_val, val_ids = data["validation"]
    print(f"[{dataset_key}/{encoder_name}] loaded {X_train.shape[1]} layers, dim={X_train.shape[2]}")

    best = (-1.0, None, None, None)
    for lr in (0.003, 0.01, 0.03):
        for wd in (1e-4, 1e-3, 1e-2):
            model, val_probs, layer_weights, _loss = train_weighted(X_train, y_train, X_val, y_val, n_class, lr=lr, weight_decay=wd)
            m = summarize(val_probs, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
            if m["top1"] > best[0]:
                best = (m["top1"], val_probs, layer_weights, {"lr": lr, "weight_decay": wd, "metrics": m})

    top1, probs, layer_weights, cfg = best
    print(f"[{dataset_key}/{encoder_name}] weighted-layer-sum best: top1={top1:.4f} top3={cfg['metrics']['top3']:.4f} "
          f"cfg=(lr={cfg['lr']}, wd={cfg['weight_decay']})")
    top_layers = np.argsort(-layer_weights)[:5]
    print(f"[{dataset_key}/{encoder_name}] top-5 weighted layers: " +
          ", ".join(f"L{int(l)}={layer_weights[l]:.3f}" for l in top_layers))

    ci = bootstrap_ci_top1(probs, y_val, n_boot=n_boot)
    print(f"  weighted_sum: top1={ci['top1']:.4f} 95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

    out = {"encoder": encoder_name, "best_config": cfg, "layer_weights": layer_weights.tolist(),
           "individual_ci": ci}
    out_dir = os.path.join(RESULTS_DIR, "weighted_layer_probe")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}_{encoder_name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{dataset_key}/{encoder_name}] saved -> {out_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--encoder-name", default="mertv2_30s")
    args = p.parse_args()
    run(args.dataset, args.encoder_name)
