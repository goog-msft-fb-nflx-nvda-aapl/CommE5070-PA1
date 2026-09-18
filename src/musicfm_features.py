"""Extract MusicFM (round-3 queue item 8, already planned pre-round-3 in PLAN.md as a
differentiator, never done until now) frame-level hidden states, time-mean-pool per
layer, cache to disk. Won et al., "A Foundation Model for Music Informatics", ICASSP 2024,
arXiv:2311.03318 (NOT the arXiv IDs some round-3 research sources cited for "MusicFM" --
2408.01124/2505.16306 don't match this repo's own paper reference, caught by checking the
actual GitHub README rather than trusting the research summary's citation).

Reference implementation cloned from github.com/minzwon/musicfm into `musicfm/` at the
project root (not pip-installable) -- weights: pretrained_msd.pt + msd_stats.json,
downloaded from HF-hosted files into `musicfm/data/`. 24kHz native (matches this dataset,
no resampling), 13-layer output (post-conv embedding + 12 Wav2Vec2Conformer layers).

Same underlying `Wav2Vec2ConformerEncoder` (from
transformers.models.wav2vec2_conformer) that MuQ uses -- MusicFM's own `encoder()` method
does `out = self.conformer(x, output_hidden_states=True); hidden_emb = out["hidden_states"]`,
the exact code shape that broke for MuQ (KeyError, transformers-version drift). Applied the
same forward-hook fix proactively rather than waiting to hit the crash, then verified with
a smoke test before trusting it (this session's `reference_alm_pipeline_verification`
practice) -- unlike MuQ, this call site did NOT need the extra `_attn_implementation`
monkey-patch (MusicFM's config comes from a live `Wav2Vec2ConformerConfig.from_pretrained`
call, not a stale baked-in EasyDict like MuQ's checkpoint), verified directly rather than
assumed.
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.config import DATASETS, CACHE_DIR
from src.data import load_manifest, load_audio_normalized

MUSICFM_STAT_PATH = os.path.join(PROJECT_ROOT, "musicfm", "data", "msd_stats.json")
MUSICFM_MODEL_PATH = os.path.join(PROJECT_ROOT, "musicfm", "data", "pretrained_msd.pt")


def load_musicfm(device="cuda"):
    from musicfm.model.musicfm_25hz import MusicFM25Hz
    model = MusicFM25Hz(is_flash=False, stat_path=MUSICFM_STAT_PATH, model_path=MUSICFM_MODEL_PATH)
    model = model.to(device).eval()
    return model


def _install_layer_hooks(model):
    """Hook the post-conv embedding stage (analogous to MERT/MuQ's embedding layer,
    "layer 0") plus each of the 12 conformer layers -- 13 total, matching
    get_latent()'s own layer_ix=0..12 indexing convention."""
    captured = []

    def make_hook():
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured.append(hs.detach())
        return hook

    handles = [model.conv.register_forward_hook(make_hook())]
    for layer in model.conformer.layers:
        handles.append(layer.register_forward_hook(make_hook()))
    return captured, handles


def cache_is_complete(dataset_key, cache_dir):
    if not os.path.isdir(cache_dir):
        return False
    spec = DATASETS[dataset_key]
    total_expected = sum(1 for _ in open(os.path.join(spec["dir"], "manifest.csv"))) - 1
    n_cached = len([f for f in os.listdir(cache_dir) if f.endswith(".npz")])
    return n_cached >= total_expected


def extract(dataset_key, device="cuda", out_dir=None, splits=("train", "validation", "test")):
    out_dir = out_dir or os.path.join(CACHE_DIR, "musicfm_msd", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    model = load_musicfm(device)
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
            wav = torch.from_numpy(y).float().unsqueeze(0).to(device)
            captured.clear()
            with torch.no_grad():
                try:
                    model.get_predictions(wav)
                except KeyError as e:
                    if str(e) != "'hidden_states'" or len(captured) == 0:
                        raise
            assert len(captured) == 13, f"expected 13 layer outputs (conv + 12 conformer), got {len(captured)}"
            layer_means = torch.stack([h.mean(dim=1).squeeze(0) for h in captured])
            np.savez(cache_path, embedding=layer_means.cpu().numpy().astype(np.float32),
                     label=row["label"], split=split, sample_id=sample_id)
    for h in handles:
        h.remove()
    print(f"[{dataset_key}] MusicFM features cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    extract(args.dataset, device=args.device)
