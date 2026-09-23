"""Production/mastering descriptors for Task 1 (round-4 queue item 4) -- the "loudness
war" is a documented, roughly monotonic 1980s->2000s trend (declining dynamic range,
rising loudness). Confirmed our own preprocessing (`load_audio_normalized`, TARGET_LUFS
= -23) loudness-normalizes every clip via a single per-clip gain -- absolute LUFS is
therefore uninformative (every clip is forced to the same integrated loudness), exactly
the confound the round-4 research flagged. Used **gain-invariant** descriptors instead
(crest factor, a percentile-based dynamic-range proxy, spectral centroid, spectral tilt)
-- all ratios/shape measures unaffected by a uniform per-clip gain, so computing them on
the already-normalized audio (rather than needing the pre-normalization original) is
valid and was verified to be the right call before writing this, not assumed.
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR, SAMPLE_RATE
from src.data import load_manifest, load_audio_normalized

FEATURE_NAMES = ["crest_factor_db", "dynamic_range_db", "spectral_centroid_hz", "spectral_tilt"]


def extract_one(y, sr=SAMPLE_RATE):
    import librosa

    # crest factor: peak / RMS, in dB -- gain-invariant (a uniform scale cancels in
    # the ratio), directly tracks the loudness-war's compression trend
    peak = np.abs(y).max() + 1e-12
    rms = np.sqrt(np.mean(y ** 2)) + 1e-12
    crest_factor_db = 20 * np.log10(peak / rms)

    # dynamic range proxy: spread of short-term (400ms) RMS level in dB, 95th minus
    # 10th percentile -- a percentile-based proxy for loudness range (EBU R128-style
    # LRA logic simplified to avoid a full standards implementation), still
    # gain-invariant since it's a difference of two log-scaled quantities computed
    # from the same signal (a uniform gain shifts both percentiles equally, the
    # difference is unchanged)
    win = int(0.4 * sr)
    hop = win // 2
    n_frames = max(1, (len(y) - win) // hop + 1)
    frame_rms_db = []
    for i in range(n_frames):
        seg = y[i * hop: i * hop + win]
        if len(seg) == 0:
            continue
        r = np.sqrt(np.mean(seg ** 2)) + 1e-12
        frame_rms_db.append(20 * np.log10(r))
    frame_rms_db = np.array(frame_rms_db)
    dynamic_range_db = float(np.percentile(frame_rms_db, 95) - np.percentile(frame_rms_db, 10)) if len(frame_rms_db) > 1 else 0.0

    # spectral centroid (Hz) -- brightness/timbre, indirectly tracks production era
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

    # spectral tilt: slope of a linear fit to log-magnitude vs log-frequency on the
    # long-term average spectrum -- a coarse "how much energy rolls off toward high
    # frequencies" measure, another production/mastering-era proxy
    S = np.abs(librosa.stft(y, n_fft=2048))
    avg_mag = S.mean(axis=1) + 1e-12
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = freqs > 20  # skip DC/near-DC bins for the log-log fit
    slope = np.polyfit(np.log(freqs[mask]), np.log(avg_mag[mask]), 1)[0]

    return np.array([crest_factor_db, dynamic_range_db, float(centroid), float(slope)], dtype=np.float32)


def extract(dataset_key="A", out_dir=None):
    spec = DATASETS[dataset_key]
    out_dir = out_dir or os.path.join(CACHE_DIR, "production_features", dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    rows, _ = load_manifest(dataset_key)
    for row in tqdm(rows, desc=f"{dataset_key}/production_features"):
        sample_id = row["sample_id"]
        cache_path = os.path.join(out_dir, f"{sample_id}.npz")
        if os.path.exists(cache_path):
            continue
        audio_path = os.path.join(spec["dir"], row["audio_path"])
        y = load_audio_normalized(audio_path)
        feat = extract_one(y)
        np.savez(cache_path, feature=feat, names=FEATURE_NAMES, label=row["label"],
                  sample_id=sample_id, split=row["split"])
    print(f"[{dataset_key}] production features cached to {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="A", choices=["A", "B"])
    args = p.parse_args()
    extract(args.dataset)
