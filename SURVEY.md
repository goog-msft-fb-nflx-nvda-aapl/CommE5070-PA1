# SOTA / Method Survey — CommE5070 PA1 (Music Era & Release-Market Classification)

Scope: Task 1 = decade classification (Dataset A, 6 classes, US releases). Task 2 = market classification (Dataset B, 6 classes, 1980s only). 30s clips, mono/PCM16/24kHz.

---

## 1. Encoders / Feature Extractors

| Model | Params | Pretrain data/objective | Native SR | Output | Pool→clip | HF/GitHub | Fit for this task |
|---|---|---|---|---|---|---|---|
| **MERT-v1-330M** | 330M | 160K hrs, MLM (masked acoustic + musical pseudo-labels: CQT + K-means) | **24kHz** (matches dataset exactly — no resample needed) | 24 transformer layers, 1024-dim, 75 Hz frame rate | mean-pool over time; often weighted-sum over layers (paper does per-layer probing — layer choice matters, don't just take last layer) | [m-a-p/MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M), [m-a-p/MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) (lighter, try if 330M is slow), [GitHub yizhilll/MERT](https://github.com/yizhilll/MERT) | Named baseline in Description.md. **Gotcha**: MERT-v1 was trained at 24kHz — this dataset is *already* 24kHz, so no resampling needed (unlike most other pretrained audio encoders below which expect 16 or 48kHz and *do* need a resample step — get this wrong and results silently degrade). |
| **MusicFM** | — | BEST-RQ (masked, random-projection quantized targets) | check repo (commonly 24kHz) | layer_ix=12 recommended for embeddings | mean-pool | [GitHub minzwon/musicfm](https://github.com/minzwon/musicfm) | Same author as Short-Chunk CNN baseline; good second pretrained-encoder option beyond MERT for the ablation matrix. |
| **Music2Vec (m-a-p/music2vec-v1)** | 90M | data2vec-style SSL, teacher/student, 1000 hrs | 16kHz (needs resample from 24kHz) | avg-pool top-K layers | mean-pool | [HF m-a-p/music2vec-v1](https://huggingface.co/m-a-p/music2vec-v1), [arXiv 2212.02508](https://arxiv.org/abs/2212.02508) | Lightweight alt to MERT; comparable to Jukebox at <2% params. Good if compute-constrained. |
| **LAION-CLAP (music)** | ~190M audio tower | Contrastive audio-text, LAION-Audio-630K | 48kHz (resample needed) | joint embed space | built-in pooling in HF `ClapModel` | [laion/larger_clap_music](https://huggingface.co/laion/larger_clap_music), [GitHub LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) | Useful for zero-shot sanity check (e.g. prompt "a song from the 1990s") and as a 3rd embedding family for ablation/t-SNE comparison. |
| **AST / HTS-AT / PANNs** | varies | AudioSet (event detection) | 16/32kHz | frame posteriors | mean-pool | in lecture notes already | Event-detection-oriented, weaker fit for era/market (timbre-heavy, not event-heavy) — mention in report as "considered but not primary" rather than implement fully. |
| **Short-Chunk CNN** | small (few M) | supervised on MagnaTagATune/MSD tags (or trained from scratch here) | flexible (log-mel input) | conv stack on log-mel, avg over chunks at inference | multi-segment average (explicit in Description.md Baseline 1) | [GitHub minzwon/sota-music-tagging-models](https://github.com/minzwon/sota-music-tagging-models) | Explicit Baseline Method 1 — train from scratch on Dataset A/B (small enough to fit; no pretrained checkpoint needed, or optionally init from MTAT tagging checkpoint in the repo for transfer). |

**Frozen-probe vs fine-tune**: LP-FT (linear-probe first, then full fine-tune init'd from the probe) is the empirically best recipe — beats both pure linear-probing (better OOD, worse ID) and pure fine-tuning (better ID, worse OOD, more overfit risk) on both axes. Given ~170-230 train clips/class here, **do LP first always** (fast, cheap ablation surface across all encoders), then fine-tune only the winning 1-2 encoder/classifier combos if time allows, since full fine-tuning on this little data risks overfitting the 330M-param MERT badly.

---

## 2. Prior Work on Decade / Market Classification

- **"Music Era Recognition Using Supervised Contrastive Learning and Artist Information"** ([arXiv 2407.05368](https://arxiv.org/abs/2407.05368), 2024) — closest prior work to Task 1. Uses Million Song Dataset. Audio-only model: **54% acc at ±3yr tolerance**; adding artist-ID via multimodal contrastive (MMC) fusion: **+9%**. Confirms this is a genuinely hard task from audio alone and that decade-adjacent confusion is expected/reported in the literature — supports the assignment's request to analyze neighboring-decade errors. No artist-split detail available from the abstract; Dataset A here is explicitly artist-disjoint across splits (per Description.md), which is a *harder*, more honest setup — expect noticeably lower than the paper's number, especially early on.
- **YearPredictionMSD (UCI/MSD)** — the classic release-year regression benchmark (1922–2011, timbre-only features), worth citing as the canonical version of "treat Task 1 as year regression" (a Required Experiment). Standard baseline there is linear regression / kNN on MSD timbre features, MAE typically ~7-9 years — gives a sanity floor to compare a from-scratch regression head against.
- **Discogs-VI** ([MTG/discogs-vi-dataset](https://github.com/MTG/discogs-vi-dataset), [arXiv 2410.17400](https://arxiv.org/abs/2410.17400), ISMIR 2024) — this dataset's actual source. Built for cover-song/version identification (~1.9M versions, 348K cliques) from Discogs editorial metadata, *not* originally for decade/market classification — Dataset A/B here are a repurposing (decade and market are Discogs editorial fields, not the VI task itself). Cite the paper for dataset provenance in the report; note in Rules/limitations that release-decade and release-market are editorial metadata, not perceptual ground truth, so some label noise (misattributed reissues, compilations) should be expected and can explain some errors beyond model weakness.
- **Geographic-origin-from-audio** literature ([Predicting the Geographical Origin of Music](https://www.researchgate.net/publication/282176865_Predicting_the_Geographical_Origin_of_Music), MARSYAS descriptors, 1142 pieces/73 countries; [From Sound to Map (ICCS 2024)](https://link.springer.com/chapter/10.1007/978-3-031-63751-3_12) for traditional/folk music by region) — relevant precedent for Task 2 framing, but note importantly these predict *provenance of the music tradition* (folk/ethnomusicology), not *commercial release market* of pop/rock recordings from a fixed decade — Task 2 here is closer to a production/mastering-style and stylistic-trend classification problem, so treat these as loosely related rather than directly transferable baselines. Emphasize this distinction in the report per Description.md's explicit note: "Release market is not the artist's nationality, language, ethnicity, or recording location."

---

## 3. Classifier Heads & Training Recipes

- **Given ~170-230 samples/class (A) or 133/class (B), 1024-d MERT embeddings**: prefer **logistic regression / linear SVM** or a small **1-hidden-layer MLP (e.g. 1024→128→6) with dropout 0.3-0.5 and weight decay** over anything deeper — high risk of overfitting a 1024-d input with <1000 total train rows.
- **Standardization**: z-score each embedding dimension using train-set mean/std (fit on train only, apply to val/test) before the classifier — this is explicitly required by Description.md and matters more for MLP/SVM (scale-sensitive) than for tree-based methods.
- **Validation given small val sets (132/102)**: single held-out val is noisy at this size (132 samples ÷ 6 classes ≈ 22/class → ±1 correct prediction ≈ ±0.8pp). Recommend **k-fold CV on train (k=5) for model/hyperparameter selection**, then report final numbers on the official val split (and eventually the hidden test) as the assignment specifies — CV protects against overfitting the tiny official val set during model selection while still following the "train for fitting, validation for model selection" instruction.
- **Regularization for small-data high-dim probes**: L2 (ridge) logistic regression with C tuned via CV is a strong, fast default; also try **PCA to ~64-128 dims before the classifier** as an ablation (may help SVM/MLP, rarely hurts logistic regression much, cheap to test).

---

## 4. Augmentation & TTA

- **SpecAugment** (time+freq masking on log-mel) — standard, easy with `torchaudio.transforms.TimeMasking/FrequencyMasking`, applies directly to the Short-Chunk CNN path.
- **Mixup** on log-mel + soft labels — commonly paired with SpecAugment; straightforward for the from-scratch CNN, less standard/necessary when just training a linear/MLP head on frozen pretrained embeddings.
- **pyrubberband** pitch/time-stretch — mentioned in lecture slides; usable as waveform-level augmentation before feature extraction for either baseline. Time-stretch is more defensible than pitch-shift here since pitch-shift could plausibly *remove* real era/market cues (e.g. tuning/mastering pitch drift across pressing eras) — Description.md explicitly warns "avoid transformations that change the label meaning," so lean towards time-stretch/SpecAugment/mixup over pitch-shift, and justify the choice in the report.
- **Random-crop segment sampling during training + multi-segment average at inference** — this *is* Baseline Method 1's design (Short-Chunk CNN trains on short crops, evaluates by averaging chunk predictions) and doubles as the natural implementation of the Required Experiment "use multiple excerpts from the same recording during training." Use the same trick for the MERT path: extract embeddings from multiple overlapping 10-30s crops per clip and average (mean-pool) before the classifier — simple, effective TTA with no extra training cost.
- **Stem separation as an analysis axis, not just augmentation**: Description.md requires comparing mixture vs. vocal vs. accompaniment stems for Task 2 specifically — run this as a *separate labeled ablation arm* (train/eval 3 parallel pipelines: mixture, vocals-only, accompaniment-only), not blended into a single augmented training set.

---

## 5. Source Separation Tool

**Recommendation: Demucs (htdemucs / htdemucs_ft, Meta AI)** — `pip install demucs`, run via `demucs --two-stems=vocals <file>.wav` to get `vocals.wav` + `no_vocals.wav` (= accompaniment) directly, which maps cleanly onto Description.md's mixture/vocal/accompaniment comparison. Preferred over Open-Unmix (older, lower quality) and Spleeter (TF1-era, more install friction, effectively unmaintained relative to Demucs' hybrid-transformer models). Needs Python ≥3.10; GPU strongly recommended given ~2300 clips × 30s to process — run on gsm-gpu2.

---

## 6. Audio Language Model (Required Experiment)

| Model | HF ID | Params | Notes |
|---|---|---|---|
| **Qwen2-Audio-7B-Instruct** | [Qwen/Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) | ~8B total (Whisper-large-v3 audio encoder + 7B LLM) | Named in Description.md. Whisper encoder adds ~1.5GB VRAM on top of base LLM; full-precision ~16GB+, Q4 quant ~5GB — comfortably fits on any of gsm-gpu2's H200 NVL (143GB) GPUs unquantized. |
| **Audio Flamingo 3** | [nvidia/audio-flamingo-3](https://huggingface.co/nvidia/audio-flamingo-3) | Qwen2.5-7B backbone + AF-Whisper encoder | Named in Description.md. Handles up to 10 min audio (plenty for 30s clips). **License: NVIDIA OneWay Noncommercial** — fine for coursework, note in report citations. Reported to beat Qwen2-Audio and Qwen2.5-Omni on several understanding/reasoning benchmarks — worth running both since Description.md asks to compare ≥2 prompt designs and this gives a natural 2-model × N-prompt matrix. |

**Prompting for closed-set output**: literature on Qwen2-Audio genre classification (FMA) and regional-style classification (OverClocked ReMix) uses direct natural-language questions like *"What genre does this piece of music fall under?"* / *"What regional style would you say this music belongs to?"* — adapt directly: e.g. *"Listen to this 30-second music excerpt. Which decade was it most likely released in: 1960s, 1970s, 1980s, 1990s, 2000s, or 2010s? Answer with only the decade."* Constraining prompts explicitly to the closed label set (rather than open-ended) measurably improves accuracy and reduces affirmative/verbosity bias in comparable audio-QA evaluations. For **invalid-output handling**: (a) regex/fuzzy-match the generated text against the 6 canonical labels (handle paraphrases like "the 1990s" or "nineties"), (b) if no match, either resample with a stricter re-prompt ("Answer with exactly one of: ...") or fall back to a fixed default class, and (c) report the invalid-output rate itself as a metric per Description.md's explicit requirement.

---

## 7. t-SNE / UMAP Practice Notes

- Standard: run on the pooled clip-level embeddings (not raw frames), color by class, one plot per task per encoder. With only ~1000-1300 points this is cheap and fast.
- Always fit t-SNE/UMAP separately per split-purpose (e.g. train+val together for one visualization) rather than mixing with test (test labels are hidden anyway).
- Report perplexity (t-SNE) / n_neighbors (UMAP) used — results are sensitive to these at this sample size; try 2-3 settings, pick the most stable/interpretable, mention this isn't cherry-picking for a "nicer" plot but for stability.
- Use this plot directly to support (or complicate) the confusion-matrix discussion of neighboring-decade/market confusion — a natural place to show whether decades form a visible gradient/manifold (supports the "ordinal" framing) or discrete clusters.

---

## 8. Class-Imbalance / Confusion-Matrix Analysis (beyond lecture notes)

Dataset is already perfectly balanced (no imbalance handling needed for training), but for **error analysis**, since Description.md explicitly asks to discuss neighboring-decade confusion:

- **Row-normalize the confusion matrix** (already in lecture notes) — report both raw counts and row-normalized version.
- **Ordinal-aware metrics for Task 1** (decades have a natural order; Task 2/market classes do not): report **Mean Absolute Decade Error** (map decades to integers 0-5, compute mean |pred-true|) alongside top-1/top-3 — directly operationalizes "are errors mainly between neighboring decades" as one number instead of only qualitative confusion-matrix reading. Also consider **weighted kappa** (linear or quadratic weights) as a single ordinal-agreement statistic.
- **Adjacent-vs-nonadjacent error rate**: of all misclassifications, what fraction land on an immediately adjacent decade vs. further away — directly answers the required discussion question quantitatively. Not meaningful for Task 2 (markets are categorical, no natural adjacency) — for Task 2 instead look at raw confusion pairs and discuss plausible cultural/stylistic proximity (e.g. does the model conflate UK/US, or Spain/Italy?) qualitatively.

---

## 9. Baseline-Specific Gotchas

- **librosa/torchaudio hand-crafted features (GTZAN-style, Baseline pairing for Task classifiers)**: use per-frame MFCC/spectral-centroid/rolloff/contrast/chroma, then **temporal pooling (mean + std, per lecture slide 36)** to get clip-level vectors — do NOT feed raw frame-level features into sklearn without pooling. Keep hop/window sizes consistent with lecture guidance (smaller windows for timbre/rhythm features used here, since neither task is pitch/harmony-centric — though chroma stats are cheap to include for the market task in case market correlates with any harmonic/tonal convention).
- **MERT**: (a) native 24kHz — no resample needed for this dataset (see §1), a genuine advantage over CLAP/Music2Vec/AST which all need resampling; (b) MERT is multi-layer — the original paper finds different layers best for different downstream tasks, so **sweep over a few layers (or a learned weighted sum across layers) rather than hardcoding the last layer** as part of the ablation; (c) 330M model at 75Hz frame rate on 30s clips = ~2250 frames — batch size will be VRAM-limited even on H200s if processing many clips at once; chunk if needed.
- **General**: dataset's loudness increases with decade (see WORKLOG.md EDA) — always apply per-clip loudness/RMS normalization before feature extraction for both the hand-crafted and CNN baselines so the classifier isn't trivially keying on mastering loudness alone; still report the raw finding as a discussion point since it's real added signal for Task 1, not purely a confound to remove.

---

## Recommended Plan

**(a) Ablation matrix — implement first, in this order:**
1. **Baseline Method 1**: Short-Chunk CNN trained from scratch on log-mel (per Description.md's explicit spec) — one model per task (A, B).
2. **Baseline Method 2**: MERT-v1-330M (frozen, linear-probe first) → logistic regression / small MLP — one per task. This + #1 satisfies the two explicitly-named baselines.
3. **Third encoder for comparison** (pick one): MusicFM (same lineage as Short-Chunk CNN author, different pretraining objective) or Music2Vec (cheap, fast, good contrast point on model size) — feeds the "goes beyond basic requirements" (A/A+) bar and gives the t-SNE/UMAP comparison more than one embedding space to contrast.
4. Optional 4th: hand-crafted librosa/torchaudio features (MFCC+spectral stats, temporally pooled) → same classifier heads, as the cheapest possible baseline floor — good for the report's "what did the model learn" discussion (interpretable features vs. black-box embeddings).

**(b) Preprocessing/standardization**: per-clip loudness normalization (peak or RMS-based) → resample only where the encoder needs it (not MERT) → z-score embedding dims (fit on train) before any classifier head → for the CNN path, log-mel with standard normalization (per-band mean/std from train).

**(c) Augmentation/TTA plan**: SpecAugment + mixup for the Short-Chunk CNN; multi-crop mean-pooled embeddings (10-30s overlapping windows) as TTA for all embedding-based paths — this also implements the "multiple excerpts" required experiment for free at inference time. Reserve pitch-shift/time-stretch as a secondary ablation only, with an explicit note in the report about the "don't change label meaning" constraint.

**(d) Required-experiments execution order** (each reuses the pipeline built above, minimal extra engineering):
1. Segment-length sweep (5/10/15/30s) — reuse MERT+classifier pipeline, just change crop length pre-pooling.
2. Multi-excerpt training/inference — covered by (c)'s TTA design; also try training the CNN directly on multiple crops/clip.
3. Task 1 as year-regression / hierarchical (predict decade via ordinal regression head, or coarse 60s-80s/90s-10s → fine decade hierarchy) — cheap add-on to the MERT MLP head (swap softmax-6 for a regression or 2-stage head), directly ties into §8's ordinal-MAE metric.
4. Task 2 stem comparison (mixture/vocals/accompaniment via Demucs, §5) — run the winning Task-2 pipeline (from step 1-2 results) three times, once per stem.
5. t-SNE/UMAP — run once winning embeddings are chosen from steps 1-2, per §7.
6. Confusion/bias analysis — ordinal MAE + adjacent-error-rate (Task 1) and qualitative market-confusion discussion (Task 2), per §8, done alongside every model's val evaluation, not as a separate late step.
7. ALM run (Qwen2-Audio + Audio Flamingo 3, ≥2 prompts each) — independent track, can run in parallel with 1-6 on a separate GPU once environment is set up (§6).

**Top-line recommendation**: get Short-Chunk CNN and frozen-MERT+logistic-regression working end-to-end on both tasks first (satisfies the two named baselines + gives a real top-1/top-3/confusion-matrix result fast), then layer in the required experiments incrementally on top of whichever of the two baselines is winning per task, rather than building all 4 ablation arms to completion before any required experiment — this front-loads a submittable A-minus-level result early and leaves the rest of the timeline for the A/A+ differentiators (3rd encoder, stem comparison, ALM, ordinal framing).
