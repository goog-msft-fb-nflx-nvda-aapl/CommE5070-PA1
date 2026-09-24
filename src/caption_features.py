"""Round-5 queue item 9 (lowest priority, both tasks): structured free-text captioning
+ sentence embedding as a probe feature. Distinguished from our existing ALM work: that
work asked the model for the label directly (teacher-forced scoring); this asks for
intermediate, verbalised evidence -- sung language, vocal style, instrumentation,
production character (reverb, drum machine, synth type), apparent era -- and lets a
*trained* classifier learn how that evidence maps to our labels. Sidesteps the label-prior
miscalibration of zero-shot scoring and (for Music Flamingo) its 38%-invalid-generation
problem on the direct-label prompt, since any caption is usable input here.

Per the source doc's own suggestion: ask each task's *strongest* ALM for captions --
Music Flamingo for Task A (new best zero-shot ALM there, 0.402 vs AF3's 0.379), Audio
Flamingo 3 for Task B (strongest ALM there). Captions embedded with a standard sentence-
transformer (all-MiniLM-L6-v2, 384-dim), then a plain logreg probe on the embedding.
"""
import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR
from src.data import load_manifest, load_audio_normalized
from src.alm_infer import load_audio_flamingo3, load_music_flamingo

CAPTION_PROMPT = (
    "Listen to this 30-second music excerpt and describe it in 2-3 sentences. Cover: "
    "the sung language (if any vocals), vocal style, main instrumentation, production "
    "character (e.g. reverb, drum machine, synth type, mastering loudness), and the "
    "apparent era or decade this recording style suggests."
)

MODEL_LOADERS = {"audioflamingo3": load_audio_flamingo3, "musicflamingo": load_music_flamingo}


def generate_captions(dataset_key, model_name, device="cuda", out_dir=None):
    spec = DATASETS[dataset_key]
    out_dir = out_dir or os.path.join(CACHE_DIR, "captions", f"{model_name}_{dataset_key}")
    os.makedirs(out_dir, exist_ok=True)

    generate_fn, _score_fn = MODEL_LOADERS[model_name](device)
    rows, _ = load_manifest(dataset_key)
    for row in tqdm(rows, desc=f"{dataset_key}/{model_name}/caption"):
        sample_id = row["sample_id"]
        cache_path = os.path.join(out_dir, f"{sample_id}.json")
        if os.path.exists(cache_path):
            continue
        audio_path = os.path.join(spec["dir"], row["audio_path"])
        y = load_audio_normalized(audio_path)
        # alm_infer.py's generate() defaults to max_new_tokens=64 (fine for a short
        # classification label) -- too short for a 2-3 sentence caption, confirmed via
        # smoke test (default cut a caption off mid-word); use a longer budget here.
        caption = generate_fn(y, 24000, CAPTION_PROMPT, max_new_tokens=150)
        with open(cache_path, "w") as f:
            json.dump({"sample_id": sample_id, "caption": caption, "label": row["label"], "split": row["split"]}, f)
    print(f"[{dataset_key}/{model_name}] captions cached to {out_dir}")
    return out_dir


def embed_captions(dataset_key, model_name, caption_dir=None, out_dir=None, device="cuda"):
    from sentence_transformers import SentenceTransformer

    spec = DATASETS[dataset_key]
    caption_dir = caption_dir or os.path.join(CACHE_DIR, "captions", f"{model_name}_{dataset_key}")
    out_dir = out_dir or os.path.join(CACHE_DIR, f"caption_embed_{model_name}", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    rows, _ = load_manifest(dataset_key)
    for row in tqdm(rows, desc=f"{dataset_key}/{model_name}/embed"):
        sample_id = row["sample_id"]
        cache_path = os.path.join(out_dir, f"{sample_id}.npz")
        if os.path.exists(cache_path):
            continue
        cap_path = os.path.join(caption_dir, f"{sample_id}.json")
        with open(cap_path) as f:
            cap = json.load(f)
        emb = embedder.encode(cap["caption"], convert_to_numpy=True).astype(np.float32)
        np.savez(cache_path, embedding=emb[None, :], label=row["label"], sample_id=sample_id, split=row["split"])
    print(f"[{dataset_key}/{model_name}] caption embeddings cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--model", choices=["audioflamingo3", "musicflamingo"], required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stage", choices=["caption", "embed", "both"], default="both")
    args = p.parse_args()
    if args.stage in ("caption", "both"):
        generate_captions(args.dataset, args.model, device=args.device)
    if args.stage in ("embed", "both"):
        embed_captions(args.dataset, args.model, device=args.device)
