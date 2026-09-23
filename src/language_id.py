"""Whisper-large-v3 sung-language identification (round-4 queue item 1, top-ranked
recommendation, IMPROVEMENT_FINDINGS_ROUND4.md). A linguistic, not acoustic, signal --
mechanistically unlike every SSL probe or ALM label-scoring approach tried so far. Runs
on Task B's already-separated Demucs vocal stems (`src/stems.py`, `dataset/extracted/
B_stems/htdemucs/<sample_id>/vocals.wav`, 1002/1002 clips already on disk from the
earlier stem-comparison experiment -- no new separation needed).

`WhisperForConditionalGeneration.detect_language()` only returns the argmax language id,
not a probability distribution -- inspected its source directly (not assumed) and
replicated the same masked-logit computation here, keeping the full softmax over the
~100 language tokens instead of collapsing to argmax, so the market classifier gets a
real posterior (Portuguese/Spanish/Italian/German/English probabilities) rather than a
single hard label.
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR
from src.data import load_manifest

WHISPER_MODEL_ID = "openai/whisper-large-v3"
WHISPER_SR = 16000
# the market labels' plausible sung/spoken languages, kept as an explicit small feature
# vector (not the full ~100-language posterior) since only these are label-relevant --
# "other" catches everything else (instrumental tracks, code-switching, mis-ID, etc.)
RELEVANT_LANGS = ["en", "pt", "es", "de", "it"]


def load_whisper_lid(device="cuda"):
    from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID, dtype=torch.float32)
    model = model.to(device).eval()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_ID)
    gc = model.generation_config
    lang_token_ids = {lang: gc.lang_to_id[f"<|{lang}|>"] for lang in RELEVANT_LANGS}
    all_lang_ids = list(gc.lang_to_id.values())
    return model, feature_extractor, lang_token_ids, all_lang_ids


def language_posterior(model, feature_extractor, lang_token_ids, all_lang_ids, wav_16k, device):
    """Full softmax over Whisper's language tokens (replicating detect_language's own
    masked-logit computation, source-inspected -- see module docstring), reduced to
    the label-relevant subset plus an explicit "other" mass."""
    inputs = feature_extractor(wav_16k, sampling_rate=WHISPER_SR, return_tensors="pt")
    input_features = inputs["input_features"].to(device=device, dtype=next(model.parameters()).dtype)
    batch_size = input_features.shape[0]
    decoder_input_ids = torch.ones((batch_size, 1), device=device, dtype=torch.long) * model.generation_config.decoder_start_token_id
    with torch.no_grad():
        logits = model(input_features=input_features, decoder_input_ids=decoder_input_ids, use_cache=False).logits[:, -1]
    non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
    non_lang_mask[all_lang_ids] = False
    logits = logits.clone()
    logits[:, non_lang_mask] = -float("inf")
    probs = torch.softmax(logits[0].float(), dim=-1)
    relevant = {lang: probs[tok_id].item() for lang, tok_id in lang_token_ids.items()}
    other = 1.0 - sum(relevant.values())
    return relevant, max(0.0, other)


def extract(dataset_key="B", device="cuda", out_dir=None, stem="vocals"):
    import librosa
    import soundfile as sf

    spec = DATASETS[dataset_key]
    stems_dir = os.path.join(os.path.dirname(spec["dir"]), f"{dataset_key}_stems", "htdemucs")
    out_dir = out_dir or os.path.join(CACHE_DIR, f"whisper_lid_{stem}", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    model, feature_extractor, lang_token_ids, all_lang_ids = load_whisper_lid(device)

    rows, _ = load_manifest(dataset_key)
    for row in tqdm(rows, desc=f"{dataset_key}/whisper_lid/{stem}"):
        sample_id = row["sample_id"]
        cache_path = os.path.join(out_dir, f"{sample_id}.npz")
        if os.path.exists(cache_path):
            continue
        wav_path = os.path.join(stems_dir, sample_id, f"{stem}.wav")
        y, sr = sf.read(wav_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)  # stereo -> mono
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=WHISPER_SR)
        relevant, other = language_posterior(model, feature_extractor, lang_token_ids, all_lang_ids, y_16k, device)
        feat = np.array([relevant[l] for l in RELEVANT_LANGS] + [other], dtype=np.float32)
        np.savez(cache_path, feature=feat, langs=RELEVANT_LANGS + ["other"],
                 label=row["label"], sample_id=sample_id, split=row.get("split", ""))
    print(f"[{dataset_key}] Whisper language-ID features ({stem}) cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="B", choices=["A", "B"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--stem", default="vocals", choices=["vocals", "no_vocals"])
    args = p.parse_args()
    extract(args.dataset, device=args.device, stem=args.stem)
