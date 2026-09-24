"""PupuM2D-Large frozen probe (round-5 queue item 7, referred to there as
"PupuJEPA-Large" -- the source doc itself flagged a naming ambiguity between
github.com/sizigi/PupuM2D and github.com/sizigi/PupuJEPA; verified the real repo is
sizigi/PupuM2D, arXiv:2606.25713, and its own model module is literally named
model/pupujepa.py, so both names point at the same thing). A JEPA-family (predict
latent embeddings of masked 2D spectrogram patches, EMA target encoder) SSL model --
genuinely different pretraining family from every encoder tried this project (MERT/MuQ/
MusicFM/CultureMERT are all 1D masked-token prediction), proposed specifically as a
decorrelated-error source for fusion.

No official inference/feature-extraction script ships with the repo (training-code-only
release) -- this reimplements the minimum needed to get a frozen embedding, sourced
directly from the authors' own training code (train_pupum2d.py's extract_mel_features,
including its hardcoded normalization constants, and model/pupujepa.py's patch_embed +
teacher encoder), not guessed. At inference: run patch_embed + the EMA teacher encoder
over the FULL unmasked patch sequence (no masking, no predictor -- those are pretraining-
only), then mean-pool over patches for a global embedding -- the standard JEPA-family
linear-probe protocol (matches I-JEPA/A-JEPA eval convention).

Input contract (from the authors' own config, not assumed): 24kHz mono waveform,
10.24s (245760-sample) chunks -- one center crop per clip, matching the model's own
pretraining crop length (cut_mel_frame=1024 @ hop=240).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR
from src.data import load_manifest, load_audio_normalized

PUPUM2D_REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "PupuM2D")
sys.path.insert(0, PUPUM2D_REPO)

PUPUM2D_SR = 24000
CHUNK_SAMPLES = 1024 * 240  # cut_mel_frame * hop_size = 245760 (10.24s)

# From train_pupum2d.py's extract_mel_features -- hardcoded by the authors, computed
# once via calc_stat.py over their own training set. Not something we recompute.
MEL_MEAN = -4.089994845986366
MEL_STD = 2.0242277159094813


class Config:
    """Minimal attribute-access dict, matching the authors' own cfg.model.X / cfg.preprocess.X access pattern."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)


def load_args_json(variant="large"):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("spellbrush/PupuM2D", f"pupum2dV2_25hz_{variant}/args.json")
    with open(path) as f:
        raw = f.read()
    # args.json has trailing commas (non-standard JSON), authors' own loader (ruamel.yaml) tolerates it
    import re
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
    return json.loads(raw)


def load_pupum2d(variant="large", device="cuda"):
    # NOTE: external/PupuM2D/model/__init__.py ships with `from .pupum2d import *`, a stale
    # module name (upstream repo bug -- the actual file is model/pupujepa.py). Patched our
    # local clone's __init__.py to `from .pupujepa import *` before this import works; see
    # REPRODUCE.md.
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from model.pupujepa import PupuM2D

    cfg_dict = load_args_json(variant)
    cfg = Config(cfg_dict)
    # args.json declares qk_norm=true, but the released checkpoint has no q_norm/k_norm
    # weights at all (verified directly against the safetensors key list) -- a real
    # config/checkpoint mismatch, likely from a timm version difference at their training
    # time vs. today's EvaBlock defaults. Trust the checkpoint's actual saved parameters.
    cfg.model.qk_norm = False

    ckpt_name = {"large": "step-0500000_loss-0.176985"}[variant]
    ckpt_path = hf_hub_download("spellbrush/PupuM2D",
                                 f"pupum2dV2_25hz_{variant}/checkpoint/{ckpt_name}/model.safetensors")
    state_dict = load_file(ckpt_path)

    model = PupuM2D(cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    model = model.to(device).eval()
    return model, cfg


mel_basis_cache = {}
hann_window_cache = {}


@torch.no_grad()
def extract_mel_features(y, cfg, device):
    """Verbatim port of train_pupum2d.py's extract_mel_features (center=False)."""
    from librosa.filters import mel as librosa_mel_fn
    key = f"{cfg.preprocess.fmax}_{device}"
    if key not in mel_basis_cache:
        mel = librosa_mel_fn(sr=cfg.preprocess.sample_rate, n_fft=cfg.preprocess.n_fft,
                              n_mels=cfg.preprocess.n_mels, fmin=cfg.preprocess.fmin, fmax=cfg.preprocess.fmax)
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[device] = torch.hann_window(cfg.preprocess.win_size).to(device)

    pad = int((cfg.preprocess.n_fft - cfg.preprocess.hop_size) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

    spec = torch.stft(y, cfg.preprocess.n_fft, hop_length=cfg.preprocess.hop_size,
                       win_length=cfg.preprocess.win_size, window=hann_window_cache[device],
                       center=False, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(mel_basis_cache[key], spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))

    if cfg.preprocess.normalize:
        spec = (spec - MEL_MEAN) / (MEL_STD + 1e-8)
    if cfg.preprocess.flip_ft:
        spec = spec.transpose(-2, -1)
    return spec.unsqueeze(1)  # (B, 1, H=time, W=freq)


@torch.no_grad()
def embed_batch(model, cfg, imgs, device):
    """patch_embed + full-sequence teacher encoder (no masking, no predictor) -> mean-pooled global embedding.
    The original forward() always narrows rope_encoder.get_embed(grid_size) via fancy-indexing
    with a (B, N_subset) mask-index tensor before passing it to the encoder (batches the rope
    embedding per-sample) -- mirror that with a "full" index (arange over all patches) since
    we use every patch, not a masked subset."""
    B = imgs.shape[0]
    grid_size = model.get_dynamic_grid_size(imgs)
    x_all = model.patch_embed(imgs)
    num_patches = grid_size[0] * grid_size[1]
    full_idx = torch.arange(num_patches, device=device).unsqueeze(0).expand(B, -1)
    rope_full_enc = model.rope_encoder.get_embed(grid_size)[full_idx]
    h = model.teacher(x_all, rope=rope_full_enc.unsqueeze(1))
    return h.mean(dim=1)  # (B, embed_dim)


def center_crop_waveform(y, chunk_samples):
    if len(y) <= chunk_samples:
        return np.pad(y, (0, chunk_samples - len(y)))
    start = (len(y) - chunk_samples) // 2
    return y[start:start + chunk_samples]


def extract(dataset_key, variant="large", device="cuda", out_dir=None):
    spec = DATASETS[dataset_key]
    out_dir = out_dir or os.path.join(CACHE_DIR, f"pupum2d_{variant}", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    model, cfg = load_pupum2d(variant, device)
    embed_dim = cfg.model.embed_dim

    rows, _ = load_manifest(dataset_key)
    for row in tqdm(rows, desc=f"{dataset_key}/pupum2d_{variant}"):
        sample_id = row["sample_id"]
        cache_path = os.path.join(out_dir, f"{sample_id}.npz")
        if os.path.exists(cache_path):
            continue
        audio_path = os.path.join(spec["dir"], row["audio_path"])
        y = load_audio_normalized(audio_path)
        y_crop = center_crop_waveform(y, CHUNK_SAMPLES)
        y_t = torch.from_numpy(y_crop).unsqueeze(0).to(device)
        imgs = extract_mel_features(y_t, cfg, device)
        emb = embed_batch(model, cfg, imgs, device)
        emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
        assert emb.shape == (embed_dim,), f"unexpected embedding shape {emb.shape}"
        np.savez(cache_path, embedding=emb[None, :], label=row["label"], sample_id=sample_id, split=row["split"])

    print(f"[{dataset_key}] PupuM2D-{variant} features cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--variant", default="large")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    extract(args.dataset, variant=args.variant, device=args.device)
