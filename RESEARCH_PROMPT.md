# Deep Research Prompt — Music Era & Release-Market Classification (CommE5070 PA1)

Paste everything below into a deep-research-capable LLM (e.g. Gemini Deep Research, OpenAI Deep Research). It should browse the web and return a sourced survey — not rely on memorized knowledge alone, since model/leaderboard state changes fast.

---

## Context

I'm building a graduate coursework project with two single-label audio classification tasks, both on 30-second music excerpts (WAV, mono, PCM16, 24kHz), sourced from Discogs-VI (a cover-song-identification dataset repurposed here using its Discogs editorial metadata):

- **Task 1 — Release-decade classification**: 1,290 US-release recordings, 6 balanced classes (1960s/1970s/1980s/1990s/2000s/2010s), split 1026/132/132 train/val/test, artist-disjoint across splits.
- **Task 2 — Release-market classification**: 1,002 recordings all released in the 1980s, 6 balanced classes (US/UK/Brazil/Spain/Germany/Italy), split 798/102/102, artist-disjoint. Note: "release market" is explicitly *not* artist nationality, language, ethnicity, or recording location — it's a commercial/production category.

Two baseline methods are required by the assignment and must be implemented regardless of what you find:
1. **Short-Chunk CNN**: log-mel spectrogram → small CNN trained from scratch on short (~3.7s) random crops, multi-segment-averaged at inference (github.com/minzwon/sota-music-tagging-models).
2. **Pretrained audio encoder + classifier**: MERT (huggingface.co/m-a-p/MERT-v1-330M) or another pretrained encoder → pooled embedding → logistic regression / SVM / MLP, frozen features vs. fine-tuning compared if resources allow.

Required experiments beyond the two baselines: compare 5/10/15/30s input lengths; use multiple excerpts per recording during training; treat Task 1 as year-regression or hierarchical prediction instead of flat 6-way classification; compare mixture vs. vocal-only vs. accompaniment-only inputs for Task 2 (via source separation); visualize learned embeddings with t-SNE/UMAP; analyze confusion patterns and dataset bias; and run an audio language model (Qwen2-Audio-7B-Instruct or Audio Flamingo 3) on both datasets with closed-set-constrained prompting, reporting invalid-output handling.

Compute: 4x H200 NVL GPUs (one currently malfunctioning, 3 usable), no sudo, conda+tmux workflow. Training data is small (~170-230 clips/class for Task 1, 133/class for Task 2).

## What I need you to research (with links/citations for every claim)

1. **Encoders for song-level classification of ~30s clips**: current (2025-2026) best open-source pretrained audio/music encoders suited to timbral/production-style classification (not just genre tagging) — MERT (confirm current version/checkpoints), MusicFM, Music2Vec, CLAP variants (LAION-CLAP, MS-CLAP), Jukebox-derived encoders, and anything newer. For each: exact HuggingFace model ID or GitHub repo, parameter count, native sample rate (does it need resampling from 24kHz?), embedding dimension, how the community typically pools frame-level output to a single clip vector, and whether frozen linear-probing or fine-tuning is standard practice at this data scale (~1000-1300 total clips).

2. **Prior published work specifically on release-decade/release-year classification or release-country/market classification from audio** (as distinct from genre or mood classification) — search for "music era classification", "release year prediction audio", "decade classification music", "country of origin music classification audio", "geographic origin classification music". Report methods used and any accuracy/MAE numbers, and whether errors were reported as concentrated on adjacent decades.

3. **Discogs-VI dataset**: find its original paper/GitHub (it's normally used for cover-song/version identification, not decade/market classification) — confirm provenance, size, and whether anyone has repurposed its editorial metadata (decade, country) for classification before.

4. **Classifier heads and training recipes for small, high-dimensional embedding datasets** (~150-230 samples/class, ~768-1024 dim pretrained embeddings): what regularization, cross-validation strategy, and dimensionality reduction (e.g. PCA) approaches are recommended to avoid overfitting at this scale.

5. **Augmentation and test-time-augmentation (TTA) tricks for song-level audio classification**: SpecAugment, mixup on spectrograms, pitch/time-stretch (note: pitch-shift risks changing era/market-relevant cues like mastering pitch, so flag if literature agrees this should be avoided or used cautiously), and multi-crop/multi-segment averaging at inference.

6. **Source separation for the required mixture/vocal/accompaniment comparison**: current best-in-class, easy-to-install open-source stem separator as of 2026 (Demucs htdemucs vs. Open-Unmix vs. Spleeter vs. anything newer) — exact pip install and usage command to get a 2-stem (vocals / accompaniment) split.

7. **Audio language models for closed-set classification via generation**: current availability, VRAM requirements, and license for Qwen2-Audio-7B-Instruct and NVIDIA Audio Flamingo 3, plus whether any newer open ALM (released after mid-2025) is reported to outperform both specifically on audio understanding/classification benchmarks. Also research best practices for constraining a generative ALM to answer from a fixed 6-way label set reliably (prompt design, regex/fuzzy output parsing, handling of non-matching generations) and any papers reporting invalid-output rates for this kind of setup.

8. **Known "loudness war" / mastering-loudness trends across release decades** — is there published evidence (beyond general knowledge) quantifying how mean loudness/dynamic range of commercial music masters changed from the 1960s through the 2010s? This is relevant because I've measured rising RMS loudness by decade in my own training data and want to cite whether this is an established, documented phenomenon and how prior work has handled it as a potential classification confound (e.g. loudness-normalizing audio before feature extraction).

9. **Ordinal/ranked classification metrics** for decade classification specifically — since decades have a natural order, is there established practice (beyond the generic ML literature on ordinal classification) for evaluating "how far off" an error is (e.g. mean absolute decade error, weighted kappa) in an MIR/audio context, with any concrete examples from published MIR papers doing decade/year prediction.

## Output format requested

Organize your findings under the 9 numbered headings above. For every model, dataset, or tool you recommend, give the exact link (HuggingFace model ID or GitHub URL) and note its license if findable. Flag anything you're not confident about rather than stating it as fact. Do not pad with generic ML background — assume I already know standard deep learning / audio classification fundamentals (log-mel spectrograms, CNNs, softmax classifiers, confusion matrices, top-k accuracy) and only tell me what's specific, current, and non-obvious.
