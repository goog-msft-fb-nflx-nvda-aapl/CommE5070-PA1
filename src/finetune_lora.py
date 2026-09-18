"""LoRA fine-tune of MERT-v1-330M (IMPROVEMENT_FINDINGS.md section 5 -- direct test of
Huang et al. arXiv:2411.19371's finding that PEFT beats full fine-tune on MERT at small
data scale, which already matches our own measured Task 2 result: full fine-tune lost to
the frozen probe there). Reuses finetune_mert.py's MERTClassifier/CropDataset/get_probs
machinery unmodified; only the trainable-parameter surface and the training loop change.

Two augmentations added here that full-fine-tune runs did NOT use (flagged as a real gap
by every deep-research source): waveform-domain time masking (our analogue of SpecAugment
-- MERT takes raw waveform, not a precomputed mel-spectrogram, so literal time/frequency-
bin masking on a spectrogram isn't directly wired into this pipeline; masking contiguous
waveform spans is the direct-input-domain equivalent, stated explicitly rather than
silently calling it "SpecAugment") and waveform mixup (soft-label interpolation between
two random training clips).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.finetune_mert import MERTClassifier, CropDataset, evaluate
from src.mert_features import MERT_MODEL_ID

N_ENCODER_LAYERS = 24


def build_lora_model(n_class, lora_rank=8, lora_alpha=16, top_n_layers=8, model_id=MERT_MODEL_ID):
    model = MERTClassifier(n_class, model_id=model_id)
    # freeze everything first -- LoRA adapters + the classification head are the
    # only trainable surface (head must stay fully trainable, it's randomly
    # initialized and has no pretrained weights to adapt from).
    for p in model.encoder.parameters():
        p.requires_grad = False

    target_layers = range(N_ENCODER_LAYERS - top_n_layers, N_ENCODER_LAYERS)
    target_modules = [f"encoder.layers.{i}.attention.{proj}"
                       for i in target_layers for proj in ("q_proj", "k_proj", "v_proj", "out_proj")]
    lora_cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules,
                           lora_dropout=0.05, bias="none")
    model.encoder = get_peft_model(model.encoder, lora_cfg)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: rank={lora_rank} alpha={lora_alpha} top_n_layers={top_n_layers} "
          f"trainable={n_trainable:,} / total={n_total:,} ({100 * n_trainable / n_total:.2f}%)")
    return model


def time_mask(wav, max_mask_frac=0.08, n_masks=2):
    """Waveform-domain analogue of SpecAugment time masking (see module docstring
    for why this operates on the waveform rather than a spectrogram)."""
    wav = wav.clone()
    n = wav.shape[-1]
    for _ in range(n_masks):
        mask_len = int(n * np.random.uniform(0, max_mask_frac))
        if mask_len == 0:
            continue
        start = np.random.randint(0, max(1, n - mask_len))
        wav[..., start:start + mask_len] = 0.0
    return wav


def mixup_batch(wavs, labels, n_class, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(wavs.size(0))
    mixed_wavs = lam * wavs + (1 - lam) * wavs[perm]
    y_a = F.one_hot(labels, n_class).float()
    y_b = F.one_hot(labels[perm], n_class).float()
    mixed_labels = lam * y_a + (1 - lam) * y_b
    return mixed_wavs, mixed_labels


def finetune_lora(dataset_key, epochs=15, batch_size=4, lr=1e-4, device="cuda",
                   lora_rank=8, lora_alpha=16, top_n_layers=8,
                   use_time_mask=True, use_mixup=True, out_dir=None):
    spec = DATASETS[dataset_key]
    n_class = len(spec["labels"])
    out_dir = out_dir or os.path.join(RESULTS_DIR, f"mert_lora_{dataset_key}")
    os.makedirs(out_dir, exist_ok=True)

    model = build_lora_model(n_class, lora_rank=lora_rank, lora_alpha=lora_alpha,
                              top_n_layers=top_n_layers).to(device)
    train_ds = CropDataset(dataset_key, "train")
    val_ds = CropDataset(dataset_key, "validation")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_top1 = -1.0
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for bi, (wavs, labels) in enumerate(train_loader):
            wavs, labels = wavs.to(device), labels.to(device)
            if use_time_mask:
                wavs = torch.stack([time_mask(w) for w in wavs])
            opt.zero_grad()
            if use_mixup:
                mixed_wavs, soft_labels = mixup_batch(wavs, labels, n_class)
                logits = model(mixed_wavs)
                loss = -(F.log_softmax(logits, dim=1) * soft_labels.to(device)).sum(dim=1).mean()
            else:
                logits = model(wavs)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        torch.cuda.empty_cache()

        metrics = evaluate(model, val_ds, device, n_class, spec["ordinal"], spec["labels"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                         "val_top1": metrics["top1"], "val_top3": metrics["top3"]})
        print(f"[{dataset_key}/lora] epoch {epoch:03d} loss={np.mean(losses):.4f} "
              f"val_top1={metrics['top1']:.4f} val_top3={metrics['top3']:.4f}", flush=True)

        if metrics["top1"] > best_val_top1:
            best_val_top1 = metrics["top1"]
            with open(os.path.join(out_dir, "best_val_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            torch.save({k: v for k, v in model.state_dict().items() if "lora" in k or k.startswith("head")},
                        os.path.join(out_dir, "best_adapter.pt"))

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[{dataset_key}/lora] done. best_val_top1={best_val_top1:.4f}")
    return best_val_top1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--top-n-layers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-time-mask", action="store_true")
    p.add_argument("--no-mixup", action="store_true")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    finetune_lora(args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, top_n_layers=args.top_n_layers,
                  device=args.device, use_time_mask=not args.no_time_mask, use_mixup=not args.no_mixup,
                  out_dir=args.out_dir)
