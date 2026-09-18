# Deep Research Prompt — Improving Music Era & Release-Market Classification (CommE5070 PA1)

Paste everything below into a deep-research-capable LLM (Gemini Deep Research, OpenAI Deep Research, etc.). It should browse the web and return sourced, current suggestions — not rely on memorized knowledge alone.

---

## 1. Problem statement

Graduate coursework assignment. Two single-label 6-way audio classification tasks on 30-second music excerpts (WAV, mono, PCM16, 24kHz), sourced from **Discogs-VI** (Araz et al., "Discogs-VI: A Musical Version Identification Dataset Based on Public Editorial Metadata," ISMIR 2024, [arXiv:2410.17400](https://arxiv.org/abs/2410.17400)) — a cover-song/version-identification dataset repurposed here via its Discogs editorial metadata (release decade, release country).

- **Task 1 — release-decade classification**: 1,290 US-release recordings, 6 balanced classes (1960s/1970s/1980s/1990s/2000s/2010s). Split 1026/132/132 train/val/test, **artist-disjoint** across splits.
- **Task 2 — release-market classification**: 1,002 recordings, all released in the 1980s, 6 balanced classes (US/UK/Brazil/Spain/Germany/Italy). Split 798/102/102, artist-disjoint. Note: "release market" is explicitly *not* artist nationality, language, ethnicity, or recording location — it's a commercial/production category.
- Random baseline both tasks: Top-1 = 16.7%, Top-3 = 50%.
- No artist IDs are provided in the manifest (anonymized) — official splits are artist-disjoint, but we can't do artist-grouped k-fold *within* the training split for model selection; using plain StratifiedKFold instead.
- Compute: 4x H200 NVL GPUs (one physically malfunctioning, one more that's invisible to the CUDA runtime despite `nvidia-smi` showing it idle — effectively 2 usable GPUs), no sudo.

## 2. Methods already tried, with results (validation set, Top-1 / Top-3)

### Baseline 1 — Short-Chunk CNN (trained from scratch)
Won et al., "Evaluation of CNN-based automatic music tagging models," SMC 2020 ([github.com/minzwon/sota-music-tagging-models](https://github.com/minzwon/sota-music-tagging-models)). Log-mel spectrogram input, 3.7s training crops, multi-segment-averaged inference.
- Task 1: **44.7% / ~78%**
- Task 2: **37.3% / ~72%**

### Baseline 2 — MERT-v1-330M, frozen probe
[`m-a-p/MERT-v1-330M`](https://huggingface.co/m-a-p/MERT-v1-330M) (Li et al., "MERT: Acoustic Music Understanding Model with Large-Scale Self-Supervised Training," [arXiv:2306.00107](https://arxiv.org/abs/2306.00107)), 330M params, 24 transformer layers, native 24kHz (no resampling needed). Swept all 25 layer outputs (embedding + 24 transformer layers), all 3 named classifier options (logistic regression, SVM, MLP), and a PCA-dimensionality ablation.
- Task 1 best: **layer 4 + PCA(64) + SVM(RBF) → 47.0% / 84.1%**
- Task 2 best: **layer 7 + logistic regression, no PCA → 47.1% / 75.5%**
- Different optimal layer per task — matches the general finding in music-SSL literature that lower/intermediate layers carry timbral/production cues (era-relevant) while deeper layers drift toward higher-level semantics.

### MERT-v1-330M, full end-to-end fine-tune
All 330M params trainable (not frozen, not LoRA/partial), linear head on mean-pooled final hidden state, 10s random crops, AdamW (lr=2e-5, cosine schedule), 15 epochs, batch size 4.
- Task 1: best epoch **50.0% / 85.6%** — clean win over every frozen-probe config, but overfits noticeably past the best epoch (train loss keeps falling toward ~0.03 while val accuracy declines/wobbles), and shows real run-to-run variance (~1-2 points of Top-1 between identical reruns, since batch order isn't seeded).
- Task 2: best epoch **44.1% / 77.4%** — did **not** beat the frozen probe.

### Probability-weighted ensembling
Motivated by a prior-semester coursework report (different task, singer identification) whose graded-best result was a weighted ensemble of 7 from-scratch models rather than any single model.
- Task 1: 3-way ensemble (fine-tuned MERT weight 0.20 + Short-Chunk CNN weight 0.40 + frozen-MERT-probe weight 0.40) → **52.3% / 85.6% — current overall best for Task 1.**
- Task 2: no weighted combination beat the single frozen probe on Top-1 (stayed ≤ 47.1%); ensembling only helped Top-3 (up to 79.4%).

### Required-experiment findings (secondary, for context)
- **Segment length (5/10/15/30s)**: Task 1 Top-1 improves monotonically with length (33%→41%→42%→46%), no plateau. Task 2 also favors the full 30s clip on Top-1, but Top-3 is non-monotonic (small val set, 102 samples).
- **Multi-excerpt/TTA** (3-8 overlapping 10s crops, embeddings mean-pooled): for Task 1, averaging 3-5 crops of 10s **exceeds** the 30s single-clip result (48.5% vs 46.2% at the time); for Task 2, no crop count beats the single 30s clip. This asymmetry (technique helps Task 1, not Task 2) recurs across several of the experiments above — ensembling, fine-tuning, classifier choice all show the same pattern.
- **Task 1 as year-regression**: Ridge regression on the decade index, rounded — clearly worse than classification (27.3% Top-1 after rounding).
- **Task 1 as hierarchical (coarse 3-way decade-pair → fine 2-way)**: Top-1 slightly below flat classification (43.2% vs 46.2% at the time) but better ordinal agreement (quadratic weighted kappa 0.665 vs 0.621) — smoother errors, not higher raw accuracy.
- **Task 2 stem separation** (Demucs htdemucs, [github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)): mixture (47.1%) clearly beats vocals-only (40.2%) and accompaniment-only (37.3%).
- **Zero-shot audio language models** (teacher-forced label-probability scoring for a fair ranked Top-1/Top-3, not just free-form generation):
  - [`Qwen/Qwen2-Audio-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) (Apache 2.0): Task 1 ~33% both prompts (below our baselines). Task 2: 32.4%-44.1% depending on prompt design, with the higher-scoring prompt also showing a 27.5% invalid-output rate in free-form generation (model drifts to genre-labeling instead of market).
  - [`nvidia/audio-flamingo-3-hf`](https://huggingface.co/nvidia/audio-flamingo-3-hf) (NVIDIA OneWay Noncommercial license): Task 1 best 37.9% (below baselines; also **never** predicts "2010s" as its top choice in either prompt, a measured bias). Task 2: **both prompts beat every one of our trained models — 58.8% / 76.5% (direct prompt)** — the single best Task 2 result found, but from a zero-shot generalist model, not our task-specific trained pipeline.
- **t-SNE/UMAP** on the winning MERT layer's embeddings: no clean visual class separation in 2D for either task, despite the classifiers' real above-chance performance — noted as high-dim linear separability not implying 2D visual separability, not a contradiction.

## 3. Where we're stuck / what we want deep research on

Current best measured results: **Task 1 = 52.3% Top-1 / 85.6% Top-3** (3-way ensemble). **Task 2 = 47.1% Top-1 / 75.5% Top-3 within our own trained pipeline** (58.8%/76.5% if a zero-shot off-the-shelf ALM is allowed to count, which feels like a different category of result). Both are well above the 16.7%/50% random baseline but we don't have external validation that these numbers are "good" for this task, and Task 2 in particular has resisted every technique that helped Task 1 (fine-tuning, ensembling, multi-crop TTA, SVM-over-logreg all won for Task 1 and lost or were neutral for Task 2).

Please research and suggest, with citations/links wherever possible:

1. Any published work specifically on **release-decade or release-country/market classification from audio** (as opposed to genre/mood/tagging) that we may have missed, with reported accuracy numbers we could sanity-check against — especially anything using Discogs-VI or similar editorially-labeled datasets, or anything training/evaluating on artist-disjoint splits specifically (most benchmarks we found are not artist-disjoint, which makes their numbers not directly comparable to ours).
2. **Why might Task 2 (market) be structurally harder to improve than Task 1 (decade)** for our architecture family (MERT-based), given the six recurring instances above where a technique helped Task 1 and not Task 2? Is this a known phenomenon for geographic/regional classification tasks vs. temporal ones in the MIR literature, or does it suggest something specific we should investigate (e.g. class overlap between markets, a different useful frequency/temporal resolution, a different pretrained encoder)?
3. **Newer or alternative pretrained audio/music encoders** (post-2024, ideally with HuggingFace/GitHub availability) worth trying instead of or alongside MERT-v1-330M — we are specifically interested in whether a differently-pretrained encoder might close Task 2's gap, given the zero-shot Audio Flamingo 3 result above beat our frozen-MERT probe by 10+ points on Task 2 despite worse Task 1 performance.
4. Concrete methods to **narrow the gap between our own trained pipeline and the zero-shot Audio Flamingo 3 result on Task 2** — e.g. distilling from the ALM's outputs, using the ALM's predictions as a pseudo-label/feature input to our classifier, prompt-based feature extraction from the ALM instead of generation, or anything else in current practice for combining a generalist ALM's judgment with a small task-specific classifier.
5. Any **overfitting-mitigation techniques for fine-tuning a 330M-parameter encoder on ~800-1000 training clips** more effective than what we tried (full fine-tune, early-stop-at-best-epoch) — e.g. LoRA/adapter fine-tuning, layer-wise learning-rate decay, stronger augmentation during fine-tuning (we used none), or a partial-unfreeze schedule — and whether any of these specifically tend to help exactly the kind of task (Task 2) that outright full fine-tuning failed to help here.
6. Anything else in current (2025-2026) practice for small-data (~150-230 samples/class), artist-disjoint, 6-way audio classification that we should consider and haven't.

Please flag confidence level on anything you're not fully sure about rather than stating it as fact, and prioritize concrete, testable suggestions (exact model IDs, exact technique names, exact papers) over general advice.
