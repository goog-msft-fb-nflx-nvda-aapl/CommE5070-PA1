import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR, SAMPLE_RATE
from src.data import ClipDataset, CropTrainDataset, collate_crops, multi_crop
from src.scnn import ShortChunkCNN, spec_augment, mixup
from src.metrics import summarize


def eval_multicrop(model, dataset_key, split, device, crop_seconds=3.7, n_crops=8):
    spec = DATASETS[dataset_key]
    ds = ClipDataset(dataset_key, split)
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for item in ds:
            crops = multi_crop(item["audio"], crop_seconds, n_crops)
            batch = torch.from_numpy(np.stack(crops)).float().to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(item["label"])
            all_ids.append(item["sample_id"])
    return np.stack(all_probs), np.array(all_labels), all_ids, spec["labels"]


def train(dataset_key, epochs=40, batch_size=32, crop_seconds=3.7, lr=1e-3,
          device="cuda", out_dir=None, use_mixup=True, use_specaug=True, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    spec = DATASETS[dataset_key]
    n_class = len(spec["labels"])
    out_dir = out_dir or os.path.join(RESULTS_DIR, f"scnn_{dataset_key}")
    os.makedirs(out_dir, exist_ok=True)

    train_ds = CropTrainDataset(dataset_key, "train", crop_seconds=crop_seconds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_crops, num_workers=4, drop_last=True)

    model = ShortChunkCNN(n_class=n_class, sample_rate=SAMPLE_RATE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_top1 = -1.0
    history = []

    for epoch in range(epochs):
        model.train()
        losses = []
        for waves, labels in train_loader:
            waves, labels = waves.to(device), labels.to(device)
            opt.zero_grad()
            if use_mixup:
                mixed_waves, y_soft = mixup(waves, labels, n_class, alpha=0.2)
                logits = model(mixed_waves)
                loss = -(F.log_softmax(logits, dim=1) * y_soft).sum(dim=1).mean()
            else:
                logits = model(waves)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        probs, labels_arr, ids, label_names = eval_multicrop(model, dataset_key, "validation", device, crop_seconds)
        metrics = summarize(probs, labels_arr, n_class, ordinal=spec["ordinal"], label_names=label_names)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_top1": metrics["top1"], "val_top3": metrics["top3"]})
        print(f"[{dataset_key}] epoch {epoch:03d} loss={np.mean(losses):.4f} val_top1={metrics['top1']:.4f} val_top3={metrics['top3']:.4f}", flush=True)

        if metrics["top1"] > best_val_top1:
            best_val_top1 = metrics["top1"]
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            with open(os.path.join(out_dir, "best_val_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[{dataset_key}] done. best_val_top1={best_val_top1:.4f}. saved to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--crop-seconds", type=float, default=3.7)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-mixup", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(args.dataset, epochs=args.epochs, batch_size=args.batch_size,
          crop_seconds=args.crop_seconds, lr=args.lr, device=args.device,
          use_mixup=not args.no_mixup, seed=args.seed)
