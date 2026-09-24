"""Round-5 queue item 3: LoRA fine-tune of Audio Flamingo 3 itself (Task 2 only, per the
source doc's own scope). Distinguished from our closed MERT-LoRA line (src/finetune_lora.py,
clean negative): AF3 already starts at 58.8% zero-shot, so this corrects a miscalibrated
prior rather than building task knowledge from scratch on a frozen-random-init head.

Setup, sourced directly from the loaded model's own module structure (checked via
named_children() before writing this, not assumed): AF3 = audio_tower (AudioFlamingo3Encoder,
637M, frozen -- "AF-Whisper") + language_model (Qwen2Model, 7.07B, LoRA via peft on
q/k/v/o_proj) + multi_modal_projector (17.4M, fully trainable) + lm_head (543.6M, frozen,
tied to no LoRA). Training examples reuse alm_infer.py's exact "direct" prompt template and
_prep() chat-templating so the fine-tuned model is evaluated under the same conditions as
the zero-shot baseline it's meant to improve on -- teacher-forced next-token loss on the true
label's tokens only (prefix tokens masked to -100), one clip per training step (audio length
varies, no padding-batch support attempted), gradient accumulation for an effective batch.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.data import load_manifest, load_audio_normalized
from src.metrics import summarize
from src.alm_infer import PROMPTS, DECADE_ALIASES, MARKET_ALIASES

TARGET_SR = 16000


def load_af3_for_training(device="cuda"):
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model

    model_id = "nvidia/audio-flamingo-3-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

    for p in model.parameters():
        p.requires_grad_(False)

    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model.model.language_model = get_peft_model(model.model.language_model, lora_cfg)

    for p in model.model.multi_modal_projector.parameters():
        p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
    return model, processor


def prep_inputs(processor, audio, sr, prompt_text, device):
    import librosa
    if sr != TARGET_SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    conversation = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": prompt_text}]}]
    prefix_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    return processor(text=prefix_text, audio=[audio], sampling_rate=TARGET_SR,
                      return_tensors="pt", padding=True).to(device)


def training_loss(model, processor, audio, sr, prompt_text, answer_prefix, true_label, device):
    inputs = prep_inputs(processor, audio, sr, prompt_text, device)
    prefix_ids = inputs["input_ids"]
    prefix_len = prefix_ids.shape[1]
    cand_text = " " + answer_prefix + true_label if answer_prefix else " " + true_label
    cand_ids = processor.tokenizer(cand_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    full_ids = torch.cat([prefix_ids, cand_ids], dim=1)
    labels = full_ids.clone()
    labels[:, :prefix_len] = -100
    model_inputs = dict(inputs)
    model_inputs["input_ids"] = full_ids
    model_inputs["attention_mask"] = torch.ones_like(full_ids)
    model_inputs["labels"] = labels
    out = model(**model_inputs)
    return out.loss


@torch.no_grad()
def score_candidates(model, processor, audio, sr, prompt_text, candidates, answer_prefix, device):
    inputs = prep_inputs(processor, audio, sr, prompt_text, device)
    prefix_ids = inputs["input_ids"]
    prefix_len = prefix_ids.shape[1]
    scores = []
    for cand in candidates:
        cand_text = " " + answer_prefix + cand if answer_prefix else " " + cand
        cand_ids = processor.tokenizer(cand_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        full_ids = torch.cat([prefix_ids, cand_ids], dim=1)
        model_inputs = dict(inputs)
        model_inputs["input_ids"] = full_ids
        model_inputs["attention_mask"] = torch.ones_like(full_ids)
        out = model(**model_inputs)
        logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)
        total = 0.0
        for j in range(cand_ids.shape[1]):
            pos = prefix_len - 1 + j
            tok = full_ids[0, prefix_len + j]
            total += logprobs[pos, tok].item()
        scores.append(total / max(1, cand_ids.shape[1]))
    return np.array(scores)


def evaluate(model, processor, dataset_key, prompt_name, split, device, max_samples=None):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    label_to_int = spec["label_to_int"]
    n_class = len(labels)
    axis = "decade this track was released in" if dataset_key == "A" else "release market of this track"
    prompt_spec = PROMPTS[prompt_name]
    prompt_text = prompt_spec["template"].format(axis=axis, choices=", ".join(labels))
    answer_prefix = prompt_spec["answer_prefix"]

    rows, _ = load_manifest(dataset_key, split)
    if max_samples:
        rows = rows[:max_samples]
    model.eval()
    probs, y_true = [], []
    for row in rows:
        audio_path = os.path.join(spec["dir"], row["audio_path"])
        y = load_audio_normalized(audio_path)
        scores = score_candidates(model, processor, y, 24000, prompt_text, labels, answer_prefix, device)
        probs.append(np.exp(scores - scores.max()) / np.exp(scores - scores.max()).sum())
        y_true.append(label_to_int[row["label"]])
    probs, y_true = np.stack(probs), np.array(y_true)
    metrics = summarize(probs, y_true, n_class, ordinal=spec["ordinal"], label_names=labels)
    return metrics


def run(dataset_key="B", prompt_name="direct", epochs=3, lr=1e-4, grad_accum=8, device="cuda", eval_every_epoch=True):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    axis = "decade this track was released in" if dataset_key == "A" else "release market of this track"
    prompt_spec = PROMPTS[prompt_name]
    prompt_text = prompt_spec["template"].format(axis=axis, choices=", ".join(labels))
    answer_prefix = prompt_spec["answer_prefix"]

    model, processor = load_af3_for_training(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=lr)

    rows, _ = load_manifest(dataset_key, "train")
    print(f"[{dataset_key}] training on {len(rows)} clips, {epochs} epochs, grad_accum={grad_accum}")

    history = []
    baseline_metrics = evaluate(model, processor, dataset_key, prompt_name, "validation", device)
    print(f"[{dataset_key}] epoch 0 (pre-fine-tune) val top1={baseline_metrics['top1']:.4f} top3={baseline_metrics['top3']:.4f}")
    history.append({"epoch": 0, **baseline_metrics})

    for epoch in range(1, epochs + 1):
        model.train()
        import random
        order = list(range(len(rows)))
        random.Random(epoch).shuffle(order)
        opt.zero_grad()
        total_loss = 0.0
        for step, idx in enumerate(tqdm(order, desc=f"{dataset_key} epoch {epoch}")):
            row = rows[idx]
            audio_path = os.path.join(spec["dir"], row["audio_path"])
            y = load_audio_normalized(audio_path)
            loss = training_loss(model, processor, y, 24000, prompt_text, answer_prefix, row["label"], device)
            (loss / grad_accum).backward()
            total_loss += loss.item()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        print(f"[{dataset_key}] epoch {epoch} mean train loss={total_loss / len(rows):.4f}")

        if eval_every_epoch:
            metrics = evaluate(model, processor, dataset_key, prompt_name, "validation", device)
            print(f"[{dataset_key}] epoch {epoch} val top1={metrics['top1']:.4f} top3={metrics['top3']:.4f}")
            history.append({"epoch": epoch, "train_loss": total_loss / len(rows), **metrics})

    out_dir = os.path.join(RESULTS_DIR, "af3_lora")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}_{prompt_name}_history.json")
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[{dataset_key}] history saved -> {out_path}")
    best = max(history, key=lambda h: h["top1"])
    print(f"[{dataset_key}] best epoch={best['epoch']} top1={best['top1']:.4f} top3={best['top3']:.4f}")
    return history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], default="B")
    p.add_argument("--prompt", default="direct")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(args.dataset, prompt_name=args.prompt, epochs=args.epochs, lr=args.lr,
        grad_accum=args.grad_accum, device=args.device)
