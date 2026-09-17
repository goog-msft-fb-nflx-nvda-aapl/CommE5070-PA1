import csv
import os
import random

import librosa
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import torch
from torch.utils.data import Dataset

from src.config import DATASETS, SAMPLE_RATE, TARGET_LUFS


def load_manifest(dataset_key, split=None):
    spec = DATASETS[dataset_key]
    path = os.path.join(spec["dir"], "manifest.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if split is not None:
        rows = [r for r in rows if r["split"] == split]
    return rows, spec


_meter = pyln.Meter(SAMPLE_RATE)


def load_audio_normalized(path, target_lufs=TARGET_LUFS):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:  # some stem-separation tools (e.g. Demucs) output stereo
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        # dataset audio is natively 24kHz already (no-op here); Demucs stems come
        # out at their model's native rate (44.1kHz for htdemucs) and need resampling.
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    loudness = _meter.integrated_loudness(y)
    if np.isfinite(loudness):
        y = pyln.normalize.loudness(y, loudness, target_lufs)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return y


def crop_waveform(y, crop_seconds, sr=SAMPLE_RATE, random_crop=True, rng=None):
    crop_len = int(crop_seconds * sr)
    if len(y) <= crop_len:
        pad = crop_len - len(y)
        return np.pad(y, (0, pad))
    if random_crop:
        r = rng or random
        start = r.randint(0, len(y) - crop_len)
    else:
        start = (len(y) - crop_len) // 2
    return y[start:start + crop_len]


def multi_crop(y, crop_seconds, n_crops, sr=SAMPLE_RATE, overlap=0.5):
    crop_len = int(crop_seconds * sr)
    if len(y) <= crop_len:
        return [np.pad(y, (0, crop_len - len(y)))]
    hop = max(1, int(crop_len * (1 - overlap)))
    starts = list(range(0, len(y) - crop_len + 1, hop))
    if len(starts) > n_crops:
        idx = np.linspace(0, len(starts) - 1, n_crops).astype(int)
        starts = [starts[i] for i in idx]
    elif len(starts) < n_crops:
        starts = starts + [starts[-1]] * (n_crops - len(starts))
    return [y[s:s + crop_len] for s in starts]


class ClipDataset(Dataset):
    """Loads full 30s clips (loudness-normalized), label as int. Cropping done by caller/collate."""

    def __init__(self, dataset_key, split, cache_audio=False):
        self.rows, self.spec = load_manifest(dataset_key, split)
        self.dataset_key = dataset_key
        self.split = split
        self.cache_audio = cache_audio
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = os.path.join(self.spec["dir"], row["audio_path"])
        if self.cache_audio and path in self._cache:
            y = self._cache[path]
        else:
            y = load_audio_normalized(path)
            if self.cache_audio:
                self._cache[path] = y
        label = row["label"]
        label_int = self.spec["label_to_int"][label] if label else -1
        return {
            "sample_id": row["sample_id"],
            "audio": y,
            "label": label_int,
            "label_str": label,
        }


class CropTrainDataset(Dataset):
    """Random-crop training dataset for Short-Chunk CNN."""

    def __init__(self, dataset_key, split, crop_seconds=3.7, cache_audio=True):
        self.base = ClipDataset(dataset_key, split, cache_audio=cache_audio)
        self.crop_seconds = crop_seconds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        y = crop_waveform(item["audio"], self.crop_seconds, random_crop=True)
        return torch.from_numpy(y), item["label"]


def collate_crops(batch):
    waves = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return waves, labels
