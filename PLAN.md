# Implementation Plan (reconciled from 4-way deep research, 2026-09-15)

Sources: SURVEY.md (own WebSearch fork) + survey_response_1/{gemini,kimi,perplexity,qwen}.md (external deep research) + 2 direct verifications (WebFetch on HF/GitHub). Kimi and Perplexity were the strongest independent passes (self-flagged confidence, caught that MSD era-recognition numbers aren't comparable to our artist-disjoint splits); Gemini/Qwen were near-duplicate/templated, used as corroboration only.

**Corrections from direct verification**: Qwen2-Audio-7B-Instruct license = **Apache 2.0** (confirmed via HF page; perplexity.md's "Qwen Research License" was wrong). MusicFM native input = **24kHz confirmed** (GitHub code sample); exact param count still unconfirmed on the repo page, will read off the actual checkpoint when loaded.

## Encoders (baseline matrix)

1. **Short-Chunk CNN** (Baseline 1, explicit) — trained from scratch, log-mel input, ~3.7s crops, multi-segment-averaged inference.
2. **MERT-v1-330M** (Baseline 2, explicit) — `m-a-p/MERT-v1-330M`, frozen, 24kHz native (no resample), CC BY-NC 4.0. Sweep a few layers (e.g. 3/4/5/6/23) + learned weighted-sum-across-layers as an ablation — don't hardcode last layer.
3. **MusicFM** (3rd encoder, A/A+ differentiator) — `github.com/minzwon/musicfm`, frozen, 24kHz native, MIT. Gives a second embedding family for the t-SNE/UMAP comparison.
4. Optional 4th: hand-crafted librosa/torchaudio MFCC+spectral-stats (mean+std pooled) → cheapest interpretable floor for the report discussion.

LP-first for all pretrained encoders (frozen linear probe before any fine-tuning attempt) — consistent across all 4 research passes, and MusicFM's own ablation reportedly shows fine-tuning *hurts* tagging via overfitting at small scale.

## Classifier heads

- Primary: L2-regularized logistic regression, `C` via CV.
- Secondary: 1-hidden-layer MLP (1024→128→6), dropout 0.3-0.5, weight decay.
- Ablation: PCA to 64-128 dims before classifier.
- **CV caveat**: manifest has no artist_id (anonymized), so GroupKFold-by-artist (recommended by 2 of the 4 sources) isn't implementable — splits are already artist-disjoint per dataset provider, but within-train k-fold must fall back to plain StratifiedKFold(k=5). Note this limitation explicitly in the report rather than silently doing plain CV.
- Standardize embedding dims (z-score, fit on train) before any classifier — required by Description.md.

## Preprocessing

- Per-clip loudness normalization (target LUFS via `pyloudnorm`, ITU-R BS.1770-4) before feature extraction for **both** the CNN log-mel path and encoder path, to prevent the confirmed loudness-war confound (Serrà et al. 2012, cited by 3/4 sources) from being a shortcut — while still reporting the raw per-decade RMS trend (WORKLOG.md) as a real discussion point, not purely noise to remove.
- Resample only where the encoder needs it (not MERT/MusicFM — both 24kHz native; Music2Vec/CLAP would need it but aren't in the primary 3).

## Augmentation / TTA

- SpecAugment + mixup (α=0.2) for Short-Chunk CNN training.
- **Avoid pitch-shift** (unanimous caution across all 4 sources — risks removing real era/market cues: tuning standards, mastering pitch drift, vocal formants). Time-stretch allowed as a secondary/cautious ablation only.
- Multi-crop TTA: extract overlapping crops (5/10/15/30s per the required-experiment sweep) per clip, mean-pool predictions/embeddings at inference — doubles as the "multiple excerpts" required experiment.

## Task 2 stem comparison

- Demucs (`pip install demucs`, `demucs --two-stems=vocals <file>.wav` → `vocals.wav` + `no_vocals.wav`), run as 3 parallel labeled pipelines (mixture / vocals / accompaniment), not blended augmentation. `htdemucs_ft` if time allows (better quality, ~4x slower); `htdemucs` default otherwise.

## Audio Language Model (required experiment)

- **Qwen2-Audio-7B-Instruct** (Apache 2.0, confirmed) — note: internally resamples to 16kHz (Whisper encoder), so our 24kHz clips get resampled at inference time; ~17-20GB VRAM, trivial on H200.
- **NVIDIA Audio Flamingo 3** (NVIDIA OneWay Noncommercial + Qwen Research License restrictions on weights, code MIT — confirmed by 3/4 sources, fine for coursework, cite it).
- Closed-set-constrained prompt (explicit label list, "answer with exactly one of: ..."), regex/fuzzy-match parsing against canonical labels, log+report invalid-output rate as its own metric (assignment explicitly requires this). ≥2 prompt designs compared per Description.md.

## Metrics

- Standard: confusion matrix (raw + row-normalized), top-1, top-3 for both tasks.
- Task 1 ordinal add-ons (decades have a natural order; markets don't): Mean Absolute Decade Error (map decade→0-5, MAE), adjacent-vs-nonadjacent error rate, quadratic weighted kappa. These directly operationalize the assignment's "discuss whether errors occur mainly between neighboring decades" as numbers, not just qualitative confusion-matrix reading.
- Task 2: confusion matrix + qualitative discussion only (no natural ordinal structure among markets).

## Execution order

1. Short-Chunk CNN baseline, both tasks (A, B) — fastest path to a real, submittable result.
2. MERT-v1-330M frozen probe (logistic regression), both tasks.
3. Required-experiment layer: segment-length sweep, multi-excerpt TTA (already built into 1-2's design), year-regression/hierarchical framing for Task 1, MusicFM 3rd-encoder ablation, t-SNE/UMAP, ordinal/confusion analysis.
4. Task 2 stem comparison (Demucs).
5. ALM track (Qwen2-Audio + Audio Flamingo 3) — independent, can run in parallel on a separate GPU once its own env is set up.

Next: write code (data loading, preprocessing, Short-Chunk CNN, training loop) and set up a dedicated conda env on gsm-gpu2.
