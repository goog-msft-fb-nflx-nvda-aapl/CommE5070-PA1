"""Required comparison (Description.md, Baseline 2): "Compare frozen features with
fine-tuning if resources allow." Full end-to-end fine-tune of MERT-v1-330M + a
linear classification head, vs. the frozen-probe results already measured. Not
presumed to under/over-perform the frozen probe -- run and observe.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR, SAMPLE_RATE
from src.data import ClipDataset, crop_waveform
from src.mert_features import MERT_MODEL_ID
from src.metrics import summarize

CROP_SECONDS = 10.0  # shorter crop keeps memory/compute tractable for fine-tuning


class MERTClassifier(nn.Module):
    def __init__(self, n_class, model_id=MERT_MODEL_ID):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden, n_class)

    def forward(self, wav):
        out = self.encoder(wav)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.head(pooled)


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_key, split, crop_seconds=CROP_SECONDS):
        # cache_audio=True regardless of split: loudness-normalized waveforms don't
        # change between epochs, only the random crop does -- caching avoids
        # re-reading + re-normalizing all files from disk every single epoch
        # (this was the bottleneck: ~15min/epoch instead of ~1-2min once cached).
        self.base = ClipDataset(dataset_key, split, cache_audio=True)
        self.crop_seconds = crop_seconds
        self.random_crop = split == "train"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        y = crop_waveform(item["audio"], self.crop_seconds, random_crop=self.random_crop)
        return torch.from_numpy(y).float(), item["label"]


def get_probs(model, ds, device, verbose=True):
    """Raw per-sample probs (+ labels + sample_ids) -- reused by evaluate() and by
    the ensembling script, which needs the actual probability vectors, not just
    summary metrics."""
    model.eval()
    probs, labels, ids = [], [], []
    if verbose:
        print(f"  [eval] starting, {len(ds)} samples", flush=True)
    with torch.no_grad():
        for i in range(len(ds)):
            if verbose and i % 20 == 0:
                print(f"  [eval] sample {i}/{len(ds)}", flush=True)
            wav, label = ds[i]
            logits = model(wav.unsqueeze(0).to(device))
            probs.append(F.softmax(logits, dim=1).squeeze(0).cpu().numpy())
            labels.append(label)
            ids.append(ds.base.rows[i]["sample_id"])
    if verbose:
        print(f"  [eval] done", flush=True)
    return np.stack(probs), np.array(labels), ids


def evaluate(model, ds, device, n_class, ordinal, label_names):
    probs, labels, _ids = get_probs(model, ds, device)
    return summarize(probs, labels, n_class, ordinal=ordinal, label_names=label_names)


def finetune(dataset_key, epochs=15, batch_size=4, lr=2e-5, device="cuda", out_dir=None):
    spec = DATASETS[dataset_key]
    n_class = len(spec["labels"])
    out_dir = out_dir or os.path.join(RESULTS_DIR, f"mert_finetune_{dataset_key}")
    os.makedirs(out_dir, exist_ok=True)

    model = MERTClassifier(n_class).to(device)
    train_ds = CropDataset(dataset_key, "train")
    val_ds = CropDataset(dataset_key, "validation")  # built once, cached, reused every epoch
    # num_workers=0: keeps the in-memory audio cache in the main process so it
    # persists across epochs (DataLoader workers are forked fresh each epoch by
    # default, which would otherwise defeat the cache entirely).
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_top1 = -1.0
    history = []
    for epoch in range(epochs):
        print(f"[{dataset_key}] epoch {epoch:03d} starting training, {len(train_loader)} batches", flush=True)
        model.train()
        losses = []
        for bi, (wavs, labels) in enumerate(train_loader):
            if bi % 20 == 0:
                print(f"  [train] batch {bi}/{len(train_loader)}", flush=True)
            wavs, labels = wavs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(wavs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"[{dataset_key}] epoch {epoch:03d} training done, entering eval", flush=True)
        sched.step()
        # release any unused cached CUDA blocks between epochs -- defensive against
        # allocator fragmentation causing progressive slowdown over many epochs.
        torch.cuda.empty_cache()

        metrics = evaluate(model, val_ds, device, n_class, spec["ordinal"], spec["labels"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                         "val_top1": metrics["top1"], "val_top3": metrics["top3"]})
        print(f"[{dataset_key}] epoch {epoch:03d} loss={np.mean(losses):.4f} "
              f"val_top1={metrics['top1']:.4f} val_top3={metrics['top3']:.4f}", flush=True)

        if metrics["top1"] > best_val_top1:
            best_val_top1 = metrics["top1"]
            with open(os.path.join(out_dir, "best_val_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            # checkpoint the best epoch -- required for reproducibility (Description.md:
            # "uploaded code and checkpoint must reproduce the submitted predictions")
            # and to reuse this model for ensembling without retraining.
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[{dataset_key}] fine-tune done. best_val_top1={best_val_top1:.4f}")
    return best_val_top1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    finetune(args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
