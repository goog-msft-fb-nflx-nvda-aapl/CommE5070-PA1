"""Required experiment: compare mixture vs vocal-only vs accompaniment-only
inputs for Task 2, using Demucs-separated stems (src/stems.py) and the winning
MERT layer-7 probe config established in the layer-sweep deep-dive."""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR, RESULTS_DIR
from src.data import load_manifest, load_audio_normalized
from src.mert_features import MERT_MODEL_ID, _install_layer_hooks, _run_one
from src.train_probe import load_cached, run

STEM_AUDIO_PATH = {
    "mixture": None,  # use the original manifest audio_path (already cached under mert_v1_330m/B)
    "vocals": "vocals.wav",
    "accompaniment": "no_vocals.wav",
}
WINNING = {"layer": 7, "classifier": "logreg", "pca_dim": None}


def extract_stem(dataset_key, stem, stems_dir, model_id=MERT_MODEL_ID, device="cuda"):
    assert stem in ("vocals", "accompaniment")
    out_dir = os.path.join(CACHE_DIR, f"mert_v1_330m_stem_{stem}", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    captured, handles = _install_layer_hooks(model)

    rows, _ = load_manifest(dataset_key)
    for row in tqdm(rows, desc=f"{dataset_key}/{stem}"):
        sample_id = row["sample_id"]
        cache_path = os.path.join(out_dir, f"{sample_id}.npz")
        if os.path.exists(cache_path):
            continue
        stem_file = os.path.join(stems_dir, "htdemucs", sample_id, STEM_AUDIO_PATH[stem])
        if not os.path.exists(stem_file):
            print(f"WARNING: missing stem file {stem_file}, skipping {sample_id}")
            continue
        y = load_audio_normalized(stem_file)
        layer_means = _run_one(model, captured, y, device)
        np.savez(cache_path, embedding=layer_means.cpu().numpy().astype(np.float32),
                 label=row["label"], split=row["split"], sample_id=sample_id)
    for h in handles:
        h.remove()
    print(f"[{dataset_key}/{stem}] cached to {out_dir}")
    return out_dir


def evaluate_all_stems(dataset_key="B", stems_dir=None, device="cuda"):
    stems_dir = stems_dir or os.path.join(CACHE_DIR, "..", "dataset", "extracted", f"{dataset_key}_stems")
    results = {}

    # mixture: reuse the already-cached full-mix embeddings (no re-extraction needed)
    metrics, _ = run(dataset_key, layer=WINNING["layer"], classifier=WINNING["classifier"],
                      pca_dim=WINNING["pca_dim"], encoder_name="mert_v1_330m",
                      out_dir=os.path.join(RESULTS_DIR, "task2_stems", "mixture"))
    results["mixture"] = {"top1": metrics["top1"], "top3": metrics["top3"]}

    for stem in ("vocals", "accompaniment"):
        extract_stem(dataset_key, stem, stems_dir, device=device)
        metrics, _ = run(dataset_key, layer=WINNING["layer"], classifier=WINNING["classifier"],
                          pca_dim=WINNING["pca_dim"], encoder_name=f"mert_v1_330m_stem_{stem}",
                          out_dir=os.path.join(RESULTS_DIR, "task2_stems", stem))
        results[stem] = {"top1": metrics["top1"], "top3": metrics["top3"]}

    out_path = os.path.join(RESULTS_DIR, "task2_stems", "summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    for stem, m in results.items():
        print(f"[{stem}] top1={m['top1']:.4f} top3={m['top3']:.4f}")
    print(f"summary -> {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="B", choices=["A", "B"])
    p.add_argument("--stems-dir", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    evaluate_all_stems(args.dataset, stems_dir=args.stems_dir, device=args.device)
