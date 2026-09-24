"""Whisper-large-v3 encoder as a frozen feature extractor (round-5 queue item 5, the
"Whisper-lineage encoder" diagnostic -- explicitly flagged as the single most useful
test this round, since it explains whether AF3's Task-2 edge over every music-SSL
encoder is because AF3's encoder (AF-Whisper) descends from Whisper-large-v3, trained
for multilingual ASR and therefore phonetic/language-identity-aware, unlike MERT/MuQ/
MusicFM/CultureMERT. Also directly satisfies round-4's B1 item (probe AF3's own encoder
instead of its output) as a close proxy: AF-Whisper is a Whisper-large-v3 derivative
finetuned for joint speech/sound/music representation -- probing the same-lineage base
model captures the hypothesis's core mechanism (multilingual-ASR-trained encoder) with
far less integration risk than extracting AF-Whisper's weights out of the full AF3
checkpoint.

Our project already has a verified, working Whisper-large-v3 load path (src/language_id.py,
used for its language-detection head) -- this reuses the same model, extracting the
encoder's hidden states instead of its language logits.
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR
from src.data import load_manifest, load_audio_normalized

WHISPER_MODEL_ID = "openai/whisper-large-v3"
WHISPER_SR = 16000
N_ENCODER_LAYERS = 32  # whisper-large-v3 encoder depth


def load_whisper_encoder(device="cuda"):
    from transformers import WhisperModel, WhisperFeatureExtractor
    model = WhisperModel.from_pretrained(WHISPER_MODEL_ID, dtype=torch.float32).to(device).eval()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_ID)
    return model.encoder, feature_extractor


def extract(dataset_key, device="cuda", out_dir=None):
    import librosa

    spec = DATASETS[dataset_key]
    out_dir = out_dir or os.path.join(CACHE_DIR, "whisper_encoder", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    encoder, feature_extractor = load_whisper_encoder(device)

    rows, _ = load_manifest(dataset_key)
    for row in tqdm(rows, desc=f"{dataset_key}/whisper_encoder"):
        sample_id = row["sample_id"]
        cache_path = os.path.join(out_dir, f"{sample_id}.npz")
        if os.path.exists(cache_path):
            continue
        audio_path = os.path.join(spec["dir"], row["audio_path"])
        y = load_audio_normalized(audio_path)
        y_16k = librosa.resample(y.astype(np.float32), orig_sr=24000, target_sr=WHISPER_SR)
        inputs = feature_extractor(y_16k, sampling_rate=WHISPER_SR, return_tensors="pt")
        input_features = inputs["input_features"].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            out = encoder(input_features, output_hidden_states=True)
        # out.hidden_states: embedding + N_ENCODER_LAYERS transformer blocks
        layer_means = torch.stack([h.mean(dim=1).squeeze(0) for h in out.hidden_states])
        np.savez(cache_path, embedding=layer_means.cpu().numpy().astype(np.float32),
                  label=row["label"], sample_id=sample_id, split=row["split"])
    print(f"[{dataset_key}] Whisper encoder features cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    extract(args.dataset, device=args.device)
