"""Round-5 queue item 8: full GTZAN-style hand-crafted timbre/rhythm/pitch feature set
for Task 1 (decade), extending round-4's loudness-only descriptors (src/production_features.py,
clean negative when concatenated with MuQ) to the full ~100-dim set the source doc specifies
(L02 slide 47's "wide & deep" architecture): spectral centroid/bandwidth/rolloff/contrast/
flatness/flux, ZCR, 20 MFCCs, 12 chroma, RMS/crest factor, tempo, beat-histogram peak ratio,
beat strength -- mean+std over frames where applicable.

Caveat from the source doc, applied here: our audio is 24kHz, so everything above 12kHz is
gone -- rolloff uses 0.85 as the primary brightness feature (0.99 is also kept for
completeness but not over-interpreted, per the source's own warning). RMS/crest factor
computed on the already loudness-normalized audio -- per round-4's finding (see
production_features.py), absolute level is uninformative post-normalization, but crest
factor (peak/RMS ratio) and RMS *variation* across frames remain gain-invariant/informative.
"""
import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR, RESULTS_DIR, SAMPLE_RATE
from src.data import load_manifest, load_audio_normalized

# (name, group) in the exact order extract_one() appends them
FEATURE_SPEC = (
    [("spectral_centroid_mean", "timbre"), ("spectral_centroid_std", "timbre")]
    + [("spectral_bandwidth_mean", "timbre"), ("spectral_bandwidth_std", "timbre")]
    + [("rolloff85_mean", "timbre"), ("rolloff85_std", "timbre")]
    + [("rolloff99_mean", "timbre"), ("rolloff99_std", "timbre")]
    + [(f"contrast{i}_mean", "timbre") for i in range(7)]
    + [(f"contrast{i}_std", "timbre") for i in range(7)]
    + [("flatness_mean", "timbre"), ("flatness_std", "timbre")]
    + [("flux_mean", "timbre"), ("flux_std", "timbre")]
    + [("zcr_mean", "timbre"), ("zcr_std", "timbre")]
    + [(f"mfcc{i}_mean", "timbre") for i in range(20)]
    + [(f"mfcc{i}_std", "timbre") for i in range(20)]
    + [(f"chroma{i}_mean", "pitch") for i in range(12)]
    + [(f"chroma{i}_std", "pitch") for i in range(12)]
    + [("rms_mean", "dynamics"), ("rms_std", "dynamics")]
    + [("crest_factor_db", "dynamics")]
    + [("tempo_bpm", "rhythm"), ("beat_strength", "rhythm"), ("beat_hist_peak_ratio", "rhythm")]
)
FEATURE_NAMES = [n for n, _ in FEATURE_SPEC]
FEATURE_GROUPS = {g: [i for i, (_, gg) in enumerate(FEATURE_SPEC) if gg == g]
                   for g in ("timbre", "pitch", "dynamics", "rhythm")}


def extract_one(y, sr=SAMPLE_RATE):
    import librosa

    feats = []
    ms = lambda x: [float(np.mean(x)), float(np.std(x))]

    feats += ms(librosa.feature.spectral_centroid(y=y, sr=sr))
    feats += ms(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    feats += ms(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
    feats += ms(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # (7, T)
    feats += [float(x) for x in contrast.mean(axis=1)]
    feats += [float(x) for x in contrast.std(axis=1)]
    feats += ms(librosa.feature.spectral_flatness(y=y))
    feats += ms(librosa.onset.onset_strength(y=y, sr=sr))  # spectral-flux-based onset envelope
    feats += ms(librosa.feature.zero_crossing_rate(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feats += [float(x) for x in mfcc.mean(axis=1)]
    feats += [float(x) for x in mfcc.std(axis=1)]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats += [float(x) for x in chroma.mean(axis=1)]
    feats += [float(x) for x in chroma.std(axis=1)]

    rms = librosa.feature.rms(y=y)
    feats += ms(rms)
    peak = np.abs(y).max() + 1e-12
    rms_overall = np.sqrt(np.mean(y ** 2)) + 1e-12
    feats.append(float(20 * np.log10(peak / rms_overall)))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(np.asarray(tempo).item()) if np.ndim(tempo) else float(tempo)
    beat_strength = float(onset_env[beat_frames].mean()) if len(beat_frames) > 0 else 0.0
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    ac = tempogram.mean(axis=1)
    peaks = np.sort(ac)[::-1]
    beat_hist_peak_ratio = float(peaks[1] / (peaks[0] + 1e-12)) if len(peaks) > 1 else 0.0
    feats += [tempo, beat_strength, beat_hist_peak_ratio]

    arr = np.array(feats, dtype=np.float32)
    assert len(arr) == len(FEATURE_NAMES), f"{len(arr)} != {len(FEATURE_NAMES)}"
    return arr


def extract(dataset_key="A", out_dir=None):
    spec = DATASETS[dataset_key]
    out_dir = out_dir or os.path.join(CACHE_DIR, "wide_deep_features", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    rows, _ = load_manifest(dataset_key)
    for row in tqdm(rows, desc=f"{dataset_key}/wide_deep_features"):
        sample_id = row["sample_id"]
        cache_path = os.path.join(out_dir, f"{sample_id}.npz")
        if os.path.exists(cache_path):
            continue
        audio_path = os.path.join(spec["dir"], row["audio_path"])
        y = load_audio_normalized(audio_path)
        feat = extract_one(y)
        np.savez(cache_path, embedding=feat[None, :], label=row["label"], sample_id=sample_id, split=row["split"])
    print(f"[{dataset_key}] wide/deep hand-crafted features cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="A", choices=["A", "B"])
    args = p.parse_args()
    extract(args.dataset)
