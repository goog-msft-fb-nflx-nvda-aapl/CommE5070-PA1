"""Extract CultureMERT-95M (round-3 queue item 5, IMPROVEMENT_FINDINGS_ROUND3.md section 4)
frame-level hidden states, time-mean-pool per layer, cache to disk. `ntua-slp/CultureMERT-95M`
(arXiv:2506.17818) is MERT-v1-95M continually pretrained on 650 hours of Greek/Turkish/Indian
music -- ChatGPT's round-3 pick, specifically motivated by Task 2's cross-cultural framing
(market classification). Task 2 only, per the research recommendation.

Same custom-code MERT architecture family as m-a-p/MERT-v1-330M (confirmed via HF model
metadata: base_model m-a-p/MERT-v1-95M, same modeling_MERT.py/configuration_MERT.py pattern)
but with 12 transformer layers instead of 24 (95M vs 330M variant) and hidden_size=768 instead
of 1024 -- config.json's baked-in transformers_version (4.39.3) suggests the same version-drift
risk that hit MERT-330M and MuQ; verified directly with a smoke test before trusting the full
extraction, same as those two.
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR
from src.data import load_manifest, load_audio_normalized

CULTUREMERT_MODEL_ID = "ntua-slp/CultureMERT-95M"
N_TRANSFORMER_LAYERS = 12


def load_culturemert(device="cuda"):
    model = AutoModel.from_pretrained(CULTUREMERT_MODEL_ID, trust_remote_code=True).to(device)
    model.eval()
    return model


def _install_layer_hooks(model):
    """Same forward-hook approach used for MERT-330M and MuQ (see those modules'
    docstrings) -- captures each transformer layer's output directly rather than
    trusting output_hidden_states, which has repeatedly proven unreliable across
    this whole MERT-architecture-family + our current transformers version."""
    captured = []

    def make_hook():
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured.append(hs.detach())
        return hook

    handles = [model.feature_projection.register_forward_hook(make_hook())]
    for layer in model.encoder.layers:
        handles.append(layer.register_forward_hook(make_hook()))
    return captured, handles


def _run_one(model, captured, wav_np, device):
    wav = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
    captured.clear()
    with torch.no_grad():
        model(wav)
    expected = N_TRANSFORMER_LAYERS + 1
    assert len(captured) == expected, f"expected {expected} layer outputs (1 embed + {N_TRANSFORMER_LAYERS} transformer), got {len(captured)}"
    return torch.stack([h.mean(dim=1).squeeze(0) for h in captured])


def extract(dataset_key, device="cuda", out_dir=None, splits=("train", "validation", "test")):
    out_dir = out_dir or os.path.join(CACHE_DIR, "culturemert_95m", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    model = load_culturemert(device)
    captured, handles = _install_layer_hooks(model)

    spec = DATASETS[dataset_key]
    for split in splits:
        rows, _ = load_manifest(dataset_key, split)
        for row in tqdm(rows, desc=f"{dataset_key}/{split}"):
            sample_id = row["sample_id"]
            cache_path = os.path.join(out_dir, f"{sample_id}.npz")
            if os.path.exists(cache_path):
                continue
            audio_path = os.path.join(spec["dir"], row["audio_path"])
            y = load_audio_normalized(audio_path)
            layer_means = _run_one(model, captured, y, device)
            np.savez(cache_path, embedding=layer_means.cpu().numpy().astype(np.float32),
                     label=row["label"], split=split, sample_id=sample_id)
    for h in handles:
        h.remove()
    print(f"[{dataset_key}] CultureMERT features cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], default="B")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    extract(args.dataset, device=args.device)
