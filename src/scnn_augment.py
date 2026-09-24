"""Round-5 queue item 10: label-aware waveform augmentation for the from-scratch
Short-Chunk CNN, Task 1 only. The source doc's own hypothesis: many standard SSL-style
augmentations (filtering, reverb, noise/degradation) alter the very production cues that
define a decade label, so a 3-way ablation -- (A) none, (B) "label-preserving"
(polarity inversion, time-stretch +-5%, pitch-shift +-1 semitone -- random crop is
already the existing default, not repeated here), (C) "production-altering" (lowpass/
highpass filter, synthetic reverb, additive noise, bit-depth degradation) -- is
"informative whether it wins or loses": the predicted ordering is B >= A > C, and if C
specifically hurts, that's direct evidence production cues carry decade information.

Applied post-crop (on the model's fixed-length input), not pre-crop, to keep the existing
CropTrainDataset/ShortChunkCNN pipeline unchanged -- time-stretch is length-corrected
(re-cropped/padded back to the original crop length) so the network's fixed input size
is preserved.
"""
import numpy as np


def polarity_inversion(y, p=0.5, rng=None):
    rng = rng or np.random
    return -y if rng.random() < p else y


def time_stretch_fixed_length(y, sr, rate_range=(0.95, 1.05), rng=None):
    import librosa
    rng = rng or np.random
    rate = rng.uniform(*rate_range)
    y_s = librosa.effects.time_stretch(y, rate=rate)
    n = len(y)
    if len(y_s) >= n:
        start = (len(y_s) - n) // 2
        return y_s[start:start + n]
    return np.pad(y_s, (0, n - len(y_s)))


def pitch_shift(y, sr, semitone_range=(-1.0, 1.0), rng=None):
    import librosa
    rng = rng or np.random
    n_steps = rng.uniform(*semitone_range)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def random_filter(y, sr, rng=None):
    from scipy.signal import butter, sosfilt
    rng = rng or np.random
    kind = rng.choice(["lowpass", "highpass"])
    cutoff = rng.uniform(1000, 6000) if kind == "lowpass" else rng.uniform(100, 1000)
    sos = butter(4, cutoff, btype=kind, fs=sr, output="sos")
    return sosfilt(sos, y).astype(np.float32)


def synthetic_reverb(y, sr, rng=None):
    rng = rng or np.random
    decay = rng.uniform(0.15, 0.5)
    ir_len = int(decay * sr)
    t = np.arange(ir_len)
    ir = (rng.uniform(0.3, 1.0) ** (t / sr)) * rng.standard_normal(ir_len).astype(np.float32)
    ir /= (np.abs(ir).sum() + 1e-8)
    wet = np.convolve(y, ir, mode="full")[:len(y)]
    mix = rng.uniform(0.2, 0.5)
    out = (1 - mix) * y + mix * wet
    peak = np.abs(out).max() + 1e-8
    return (out / peak * np.abs(y).max()).astype(np.float32) if peak > 0 else out.astype(np.float32)


def additive_noise(y, snr_db_range=(5, 20), rng=None):
    rng = rng or np.random
    snr_db = rng.uniform(*snr_db_range)
    sig_power = np.mean(y ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(y)).astype(np.float32) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)


def bit_degradation(y, rng=None):
    rng = rng or np.random
    bits = rng.randint(4, 9)
    levels = 2 ** bits
    return (np.round(y * levels) / levels).astype(np.float32)


def augment_label_preserving(y, sr, rng=None):
    """Condition B: random crop is already applied upstream; this adds polarity
    inversion, time-stretch +-5%, pitch-shift +-1 semitone."""
    rng = rng or np.random
    y = polarity_inversion(y, rng=rng)
    if rng.random() < 0.5:
        y = time_stretch_fixed_length(y, sr, rng=rng)
    if rng.random() < 0.5:
        y = pitch_shift(y, sr, rng=rng)
    return y.astype(np.float32)


def augment_production_altering(y, sr, rng=None):
    """Condition C: lowpass/highpass filter, synthetic reverb, additive noise, bit
    degradation -- alters the production cues the source doc predicts carry decade
    information, applied one-at-a-time (uniformly chosen) rather than stacked, to keep
    each augmented clip recognizable rather than destroyed."""
    rng = rng or np.random
    op = rng.choice(["filter", "reverb", "noise", "degrade"])
    if op == "filter":
        y = random_filter(y, sr, rng=rng)
    elif op == "reverb":
        y = synthetic_reverb(y, sr, rng=rng)
    elif op == "noise":
        y = additive_noise(y, rng=rng)
    else:
        y = bit_degradation(y, rng=rng)
    return y.astype(np.float32)


AUGMENT_FNS = {
    "none": None,
    "label_preserving": augment_label_preserving,
    "production_altering": augment_production_altering,
}


def apply_augment_batch(waves_np, sr, mode, rng=None):
    """waves_np: (B, T) numpy array (post-crop, pre-model). Returns augmented copy."""
    fn = AUGMENT_FNS[mode]
    if fn is None:
        return waves_np
    rng = rng or np.random
    out = np.empty_like(waves_np)
    for i in range(waves_np.shape[0]):
        out[i] = fn(waves_np[i], sr, rng=rng)
    return out
