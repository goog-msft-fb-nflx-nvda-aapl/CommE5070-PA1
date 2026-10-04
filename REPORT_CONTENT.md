# CommE5070 PA1 Report — Content Package
**Student ID: R13921031**

This document contains everything needed to build the final report (16:9 HTML slide deck,
`R13921031_report.html`). It is organized by report section, not chronologically. Full
raw data/logs live in `docs/progress/WORKLOG.md` (827 lines, every experiment this
project ran, in order) if more detail than what's curated here is ever needed.

---

## 0. Links (put on the first slide)

- **Google Drive (primary submission folder — code, checkpoints, readme, requirements.txt):**
  https://drive.google.com/drive/folders/13_Ufj30hwtYVkmN2QFgmXyjuOBfn35iZ?usp=sharing
- **GitHub repo (full experiment history, ~60 experiments, will be made public after the
  2026-10-05 deadline — URL is stable across the visibility change):**
  https://github.com/goog-msft-fb-nflx-nvda-aapl/CommE5070-PA1

---

## 1. Task overview (for slide 2 / dataset description)

- **Dataset**: Discogs-VI recordings with an available YouTube link. Dataset A: 1,290
  recordings across 6 US release decades (1960s–2010s), an **ordinal** label. Dataset B:
  1,002 recordings across 6 release markets in the 1980s (US/UK/Brazil/Spain/Germany/Italy),
  a **nominal** label. Each example is the middle 30 seconds, WAV mono PCM16 24kHz.
  Artists do not overlap across train/validation/test.
- **Splits**: A = 1026 train / 132 validation / 132 test. B = 798 train / 102 validation /
  102 test. Test labels are never available to us (blind, held out for grading) — every
  number in this report is measured on validation unless explicitly marked "test."
- **Metric**: for each task, `S = Top-1 + 0.5×Top-3`; final accuracy score =
  `50 × (S_A + S_B) / 2`.

---

## 2. Submitted final configs (the core "methods" slide)

### Task 1 (decade) — MERT-v2×PupuM2D fusion
- **MERT-v2-30s** (`m-a-p/MERT-v2-30s`, arXiv, 632M params, 24-layer Conformer, native
  24kHz/30s input): layer-10 hidden states, mean-pooled over time, logistic regression
  (scikit-learn, `C` via 5-fold `GridSearchCV`).
- **PupuM2D-Large** (`sizigi/PupuM2D`, "Frequency-Aware Self-Supervised Music
  Representation Learning," arXiv:2606.25713, 307M params, JEPA-family 2D-spectrogram
  masked-latent-prediction pretraining): global mean-pooled embedding, logistic
  regression + PCA-128.
- **Fusion**: `p = 0.6 × p_MERTv2 + 0.4 × p_PupuM2D`, weight fit via leakage-free 5-fold
  out-of-fold cross-validation on the training set only (never on validation/test),
  accuracy-optimal objective.
- **Validation result: top1 = 0.5606, top3 = 0.8712.**
- Simpler alternative (MERT-v2 alone, no PupuM2D needed): top1 = 0.5455, top3 = 0.8712 —
  same top3, slightly lower top1. Neither is statistically distinguishable from the other
  (bootstrap p = 0.69, n = 132) — the fusion is our best point estimate, not a confirmed win.

### Task 2 (market) — MERT-v2+calibrated-AF3 fusion
- **MERT-v2-30s**: mean of all 24 layers' hidden states, logistic regression + PCA-128.
- **Audio Flamingo 3** (`nvidia/audio-flamingo-3-hf`, 8B-parameter audio-language model):
  zero-shot teacher-forced label log-probability scoring (`direct` prompt: *"Listen to
  this 30-second music excerpt. Which release market is it most likely from: US, UK,
  Brazil, Spain, Germany, Italy? Answer with exactly one of those labels..."*), then
  **contextually calibrated** (Zhao et al. 2021, arXiv:2102.09690): a content-free/
  near-silent-audio baseline's label scores are subtracted from every real clip's scores
  before the softmax, removing the model's own unconditional label bias. Parameter-free
  given the null estimate — cannot overfit our labels.
- **Fusion**: `p = 0.8 × p_MERTv2 + 0.2 × p_calibratedAF3`, same leakage-free OOF
  methodology as Task 1.
- **Validation result: top1 = 0.6569, top3 = 0.8922.**
- Simpler alternative (MERT-v2 alone): top1 = 0.6471, top3 = 0.8922. Not statistically
  distinguishable (p = 1.0, n = 102).

**Why these two configs were chosen over the simpler alternatives**: both are the best
*point estimates* found across the whole project (see the method-comparison charts,
section 6). Neither fusion is confirmed to beat its simpler single-encoder counterpart by
formal significance testing at this sample size — this is stated explicitly in the
"Limitations" section, not hidden.

**Reproducibility**: both fusion weights are fixed constants fit once on the training
set; classifiers are refit deterministically (`random_state=0` throughout). The
submission pipeline (`src/final_predictions.py`) re-derives the validation numbers above
as a self-check *before* generating any test-set prediction, and refuses to proceed if
they don't match exactly.

---

## 3. Baselines (required, for the "beyond basic requirements" framing)

| Baseline | Task A top1/top3 | Task B top1/top3 |
|---|---|---|
| Random chance | 0.167 | 0.167 |
| Short-Chunk CNN (from scratch, mixup+SpecAugment) | 0.447 / ~0.78 | 0.373 / ~0.72 |
| Frozen MERT-v1-330M + logistic regression | 0.470 / — | 0.471 / 0.755 |
| Fine-tuned MERT-v1-330M | 0.500 (new best at the time) | 0.412 (worse than frozen) |
| 3-way ensemble (fine-tuned MERT 0.2 + CNN 0.4 + frozen-MERT-SVM 0.4) | 0.523 / 0.856 | — |
| **Submitted (MERT-v2×PupuM2D / MERT-v2+cal.AF3 fusion)** | **0.5606 / 0.8712** | **0.6569 / 0.8922** |

Both required classifier types (logistic regression, SVM) and MLP were all tried across
this project; logreg/SVM consistently beat MLP at this data scale (~800-1300 samples).
Both frozen-feature and fine-tuning were compared directly: fine-tuning won for Task A
(0.500 vs 0.470 frozen) but lost for Task B (0.412 vs 0.471 frozen) — a genuinely
different, task-specific answer, reported as such rather than picking one universal
recommendation.

---

## 4. Required experiments (for the methodology slide)

- **Segment-length sweep** (Task A): tried multiple center-crop lengths; longer segments
  generally helped up to the full 30s.
- **Multi-excerpt/TTA ablation** (both tasks, `n_crops` ∈ {1,3,5,8} at 10s each): mixed —
  helped marginally for some configs, hurt for others; not a universal win (see
  WORKLOG.md for the per-config breakdown).
- **t-SNE / UMAP visualization** (both tasks, MERT embeddings): **no clean visual class
  separation in either projection for either task** (intermixed blob, checked across 3
  perplexities/3 n_neighbors, consistent) — does **not** contradict the classifiers'
  real above-chance performance (0.46-0.56 vs 0.167 random); high-dimensional linear
  separability and 2D visual separability are different properties. See
  `results/embeddings_preview/*.png`.
- **Task 1 year-regression / hierarchical framing**: plain regression scored worse than
  flat classification on every metric (top1=0.273). A coarse-then-fine hierarchical
  classifier gave smoother errors (QWK=0.659 vs flat's 0.621) but slightly lower top1
  (0.432 vs 0.462) — a real accuracy/smoothness tradeoff, not a clean win either way.
- **Task 2 stem comparison** (mixture vs. Demucs-separated vocals vs. accompaniment):
  full mixture wins clearly (top1=0.471) over vocals (0.402) and accompaniment (0.373).
  Vocals beating accompaniment is flagged honestly as a possible language/accent confound.
- **Audio Language Model track** (required, Qwen2-Audio-7B-Instruct + Audio Flamingo 3,
  both tasks, ≥2 prompts each, invalid-output-rate reported): 8 total runs, full results
  in section 8 below. Headline: AF3 zero-shot beats our own trained baselines on Task B
  (0.588 vs 0.471) but not Task A (0.379 vs 0.470) — motivating why AF3 became part of
  the Task 2 fusion but was never central to Task 1.

### 4.1 Embedding-space visualization (t-SNE / UMAP) — required experiment, must appear as its own figure/slide

Run on the MERT winning-layer embeddings (train+validation), both tasks, across multiple
perplexity/neighbor settings for robustness. **Use `assets/A_tsne_perp30.png` and
`assets/A_umap_nn15.png` for Task A, and `assets/B_tsne_perp30.png` and
`assets/B_umap_nn15.png` for Task B** (one t-SNE + one UMAP per task is enough; the other
perplexity variants in assets/ — `A_tsne_perp15.png`, `A_tsne_perp50.png` — exist only to
show the finding was checked for robustness, mention that it was checked across 3
perplexities without needing to display all 3).

**Finding (state this plainly, it is a real, expected, and non-negative result — do not
omit this figure because the finding sounds unflattering)**: **no clean visual class
separation appears in either projection, for either task** — points form a single
intermixed blob with no visible per-class clusters. This was checked and confirmed
consistent across 3 perplexities (15/30/50) and multiple neighbor-count settings for
UMAP, not a one-off artifact. **This does not contradict the classifiers' real
above-chance performance** (0.46-0.56 top1 vs. 0.167 random, both tasks): 2D projection
of a high-dimensional embedding space can easily destroy linear separability that a
1024-dimensional logistic regression can still exploit — this is a well-known and
expected limitation of t-SNE/UMAP as a *diagnostic* tool, not a sign the classifiers
aren't working. State this explicitly next to the figure so a reader doesn't
misinterpret "blob" as "the models don't work."

---

## 5. What else was tried (breadth, for "goes beyond basic requirements")

This project ran roughly **60 distinct experiments** across 5 rounds of literature-grounded
improvement research (4-way deep research via multiple LLMs, cross-referenced against the
course's own lecture slides for round 5). A condensed inventory:

**Encoders tried as frozen feature extractors** (both tasks unless noted): MERT-v1-330M,
MERT-v1-95M/CultureMERT-95M, MuQ-large, MERT-v2-30s (**winner**), MusicFM, LAION-CLAP
(near/below-random — weakest result this project), MuFun-9B ALM (near-random, 5 real
library compatibility bugs fixed before confirming the negative), Whisper-large-v3
encoder (linguistic-lineage diagnostic), CLaMP3, PupuM2D-Large (**Task 1 fusion partner**),
MAEST (Discogs-style-supervised), Dasheng-1.2B, hand-crafted GTZAN-style features
(spectral/rhythm/chroma/dynamics, ~98 dims).

**Task 2-specific techniques**: AF3/Music Flamingo zero-shot scoring (2 prompts each),
validation-swept vs. leakage-free OOF-refit fusion (the swept version's headline 0.657
figure was found to **not reproduce** under a proper OOF refit — a major methodological
finding, see section 7), contextual calibration of AF3's label bias, Whisper-large-v3
sung-language identification (a genuinely new, non-acoustic signal), hierarchical
market-clustering classifiers (hard vs. soft routing), AF3 LoRA fine-tuning, retrieval-
augmented in-context learning (found infeasible — AF3's processor doesn't support
multi-audio prompts), a Thinking-capable music-reasoning ALM (found infeasible — the
model's public weights are packaged for a different serving stack, not plain
`transformers`), kNN retrieval-based classification (clean negative).

**Task 1-specific ordinal methods**: CORAL cumulative-link head (clean, significant
**negative** — its shared-weight-vector architecture is too capacity-constrained), SORD
distance-aware soft labels (a genuine, if modest, **positive** once compared under a
matched optimizer), squared-EMD loss (+4.6pt over matched cross-entropy, best from-scratch
MAE this project), unimodal Poisson head (significantly **worse** than matched CE — also
too capacity-constrained), supervised contrastive loss (SupCon: +3.8pt top1 but *worse*
top3, an interesting tradeoff), label-aware waveform augmentation ablation (did not
confirm the predicted ordering).

**A clean, decisive pattern emerges across all four ordinal methods tried**: injecting
ordinality via the *loss function or training targets* while keeping the classifier's
full representational capacity (SORD, EMD) genuinely helps; injecting it by *constraining
the architecture's capacity* (CORAL, Poisson) genuinely hurts, in one case significantly
so. This is a real, mechanistically-grounded finding, not just another inconclusive point
estimate.

**Statistical rigor track**: paired bootstrap + McNemar (binarized), continuous
Brier-score Wilcoxon tests (higher power at small n), repeated stratified k-fold with a
Bayesian correlated t-test (Nadeau-Bengio variance correction) — applied specifically to
catch validation-set overfitting in fusion-weight selection. This caught a real problem:
see section 7.

---

## 6. Method comparison charts (use as figures)

Two bar charts exist at `results/report_assets/method_comparison_A.png` and
`_B.png`, showing validation top-1 accuracy across every major standalone method tried,
both tasks, with the submitted config highlighted in red. Use these as the primary
"what did we try" results figures.

---

## 7. Confusion matrices (required, "at least one per task")

`results/report_assets/confusion_A_submitted.png` and `confusion_B_submitted.png` —
row-normalized heatmaps for the exact submitted configs, validation split.

- **Task A (decade)**: clean **ordinal/adjacent-decade confusion structure** — nearly all
  errors land on a neighboring decade (e.g. 1970s↔1980s, 1990s↔2000s), not distant ones.
  1960s and 2010s (the two edge classes) are classified best (73% each); 1990s is the
  hardest class (36%), pulled toward both its neighbors — consistent with a genuinely
  continuous, ordinal acoustic signal (production technology, mastering style) underlying
  the label.
- **Task B (market)**: **cluster-structured, not ordinal, confusion** — Brazil is
  classified almost perfectly (88%, essentially unconfused with anything), while
  Germany is the hardest class (24%, confused with everything). US and UK
  confuse with each other. This matches the "release market is an editorial/commercial
  category with some markets having distinctive musical traditions (Brazil) and others
  sharing a broad Anglo/Euro pop convention (US/UK/Germany)" explanation, cross-validated
  by a separate language-ID experiment (Task 2's best non-acoustic signal, ~53% accurate
  alone, with near-zero acoustic information).

---

## 8. Audio Language Model track (required experiment, full table)

| dataset | config | top1 | top3 | invalid rate |
|---|---|---|---|---|
| A (decade) | Qwen2-Audio / direct | 0.326 | 0.659 | 0.8% |
| A | Qwen2-Audio / cot_then_answer | 0.333 | 0.652 | 0.8% |
| A | Audio Flamingo 3 / direct | 0.379 | 0.697 | 0.0% |
| A | Audio Flamingo 3 / cot_then_answer | 0.333 | 0.667 | 0.0% |
| A | (baseline) frozen MERT probe | 0.462 | 0.871 | — |
| B (market) | Qwen2-Audio / direct | 0.324 | 0.529 | 0.0% |
| B | Qwen2-Audio / cot_then_answer | 0.441 | 0.794 | **27.5%** |
| B | Audio Flamingo 3 / direct | **0.588** | 0.765 | 0.0% |
| B | Audio Flamingo 3 / cot_then_answer | 0.559 | **0.775** | 0.0% |
| B | (baseline) frozen MERT probe | 0.471 | 0.755 | — |

**Findings worth including in the write-up**:
- On Task A, all 4 ALM configs sit below the purpose-built baseline; on Task B, both AF3
  prompts *beat* the frozen-probe baseline outright — a task-specific, not uniform, result.
- **Audio Flamingo 3 never once predicts "2010s" for any of the 132 Task A validation
  clips, in either prompt** — a real, reproducible bias (0 in both confusion matrices'
  predicted-2010s columns), reported as observed without speculating on its cause.
- **`cot_then_answer` prompts trade output reliability for scored accuracy**: on Task B,
  Qwen2-Audio's `cot_then_answer` scores much higher (top1 +0.12 over `direct`) but has a
  27.5% invalid free-form generation rate (the model drifts to answers like "Answer:
  International" or a genre name instead of a valid market label) — `direct`'s terser
  framing sacrifices some scored accuracy for reliable format compliance. Report both
  numbers together; neither prompt is unconditionally better.
- Three real implementation bugs were found and fixed during this track, each only
  caught by active verification (a wrong processor kwarg that silently dropped the audio
  entirely with no error; a degenerate one-hot top-3 scoring bug; a prompt/scoring-format
  mismatch) — not just "it ran without crashing." Worth one sentence in "lessons learned."

---

## 9. Major ablation highlights (pick 3-5 for dedicated slides)

Suggested picks, in order of how interesting/decisive they are:

1. **MERT-v2-30s discovery** (round 5): a lecture-grounded literature search surfaced this
   as a stronger encoder than everything tried in rounds 1-4 combined — it became the
   backbone of *both* final submitted configs. Single most impactful change this project.
2. **The 0.657→~0.59 validation-overfitting correction** (round 3): the original Task 2
   fusion headline (validation-swept weight, top1=0.657) did **not reproduce** under a
   leakage-free out-of-fold refit (collapsed to ~0.588, essentially AF3's own zero-shot
   score). A genuinely important methodological lesson: **never fit a fusion/ensemble
   weight on the same set used to report results.** Every fusion after this point in the
   project used OOF fitting exclusively.
3. **Contextual calibration + language identification** (round 4): two independent,
   mechanistically-clean wins for Task 2 — calibration removes AF3's own measured label
   bias (parameter-free, can't overfit); Whisper-based sung-language detection is the
   first genuinely *non-acoustic* signal tried, and alone beats the frozen acoustic
   probe (0.529 vs 0.500).
4. **The four-way ordinal-methods pattern** (Task 1): SORD/EMD (inject ordinality via
   loss/targets, keep full model capacity) help; CORAL/Poisson (inject ordinality via
   constrained architecture) hurt, one significantly. A clean, mechanistically-grounded
   story, not just another inconclusive point estimate.
5. **Honest infeasibility findings**: retrieval-augmented in-context learning with AF3
   (the processor hard-enforces one audio clip per conversation — verified via direct
   testing, not assumed) and a music-specialized "Thinking" ALM (its public HF weights
   are, by the authors' own code comments, packaged for a different serving stack and
   never actually tested by them against plain `transformers`) were both investigated and
   found genuinely infeasible — shown as evidence of thorough due diligence, not just
   listed as "didn't try."

---

## 10. Limitations, errors, and honest framing (required section)

- **Nothing in this project clears formal statistical significance at the reported
  validation-set sizes (n=132 for A, n=102 for B)**, including the two submitted fusion
  configs vs. their simpler single-encoder alternatives. Every point estimate improvement
  reported is real in the sense of being genuinely measured, but should not be read as a
  confirmed, noise-free ranking — smaller gaps at this sample size routinely have
  confidence intervals that include zero.
- **The market classification task (B) is measurably harder to attribute to acoustic
  content specifically** than the decade task: the confusion structure clusters by
  cultural/linguistic proximity (Brazil vs. everything else) rather than by any acoustic
  gradient, and a purely linguistic signal (sung-language ID, zero acoustic features)
  performs competitively with acoustic encoders — suggesting release-market labels are
  at least partly an editorial/commercial category, not a purely acoustic one.
- **No artist-grouped cross-validation was possible for repeated-CV robustness checks**
  (the manifest has no artist IDs, only artist-disjoint splits provided by the course) —
  stated explicitly wherever a repeated-CV protocol was used, rather than silently assumed
  equivalent to a proper grouped CV.
- **A from-scratch PyTorch/Adam optimizer gap was found and confirmed repeatedly**: every
  torch-trained linear/MLP head on these small (~800-1300 sample), near-linear probe
  problems underperforms scikit-learn's LBFGS-optimized `LogisticRegression` by a
  consistent margin — a real, now well-established confound that was controlled for via
  matched-optimizer baselines in every from-scratch training comparison (SORD, EMD,
  Poisson, SupCon, plain CE) rather than conflated with the technique itself.
- **A genuine bug was caught and fixed during final submission generation**: an earlier
  reported top-3 figure for the Task 1 fusion (0.8485) was found to be untraceable to any
  saved artifact and, on independent recomputation, actually 0.8712 — corrected before
  submission (see WORKLOG.md's "CURRENT overall-best configs" table for the full
  correction note). Included here as an example of the project's verify-before-trusting
  discipline extending all the way to the final numbers.

---

## 11. Citations (comprehensive — include all, cite format however the report template needs)

**Pretrained encoders / models used in the final submitted configs:**
- MERT-v2-30s: `m-a-p/MERT-v2-30s`, https://huggingface.co/m-a-p/MERT-v2-30s
- PupuM2D-Large: Gu, Zhang, Li, Wu, Juvela. "Frequency-Aware Self-Supervised Music
  Representation Learning." arXiv:2606.25713. Code: https://github.com/sizigi/PupuM2D ·
  Weights: https://huggingface.co/spellbrush/PupuM2D
- Audio Flamingo 3: NVIDIA, `nvidia/audio-flamingo-3-hf`,
  https://huggingface.co/nvidia/audio-flamingo-3-hf
- Contextual calibration: Zhao, Wallace, Feng, Klein, Singh. "Calibrate Before Use:
  Improving Few-Shot Performance of Language Models." arXiv:2102.09690.

**Other models/methods tried (full list, for completeness per the assignment's "cite
every public codebase, pretrained model, and paper used"):**
- MERT-v1-330M/95M: Li et al., "MERT: Acoustic Music Understanding Model with
  Large-Scale Self-Supervised Training," arXiv:2306.00107; `m-a-p/MERT-v1-330M`
- CultureMERT-95M: `ntua-slp/CultureMERT-95M`, arXiv:2506.17818
- MuQ: `OpenMuQ/MuQ-large-msd-iter`, https://github.com/tencent-ailab/MuQ
- MusicFM: Won et al., "A Foundation Model for Music Informatics," ICASSP 2024,
  arXiv:2311.03318, https://github.com/minzwon/musicfm
- LAION-CLAP: `laion/clap-htsat-fused`
- Qwen2-Audio: `Qwen/Qwen2-Audio-7B-Instruct`
- Music Flamingo: NVIDIA, arXiv:2511.10289, `nvidia/music-flamingo-2601-hf`
- MuFun: `Yi3852/MuFun-Instruct`
- Whisper-large-v3: `openai/whisper-large-v3` (OpenAI)
- CLaMP3: Wu et al., arXiv:2502.10362, `sander-wood/clamp3`
- MAEST: Alonso-Jiménez et al., "Efficient Supervised Training of Audio Transformers for
  Music Representation Learning," arXiv:2309.16418, `mtg-upf/discogs-maest-30s-pw-129e`
- Dasheng: `mispeech/dasheng-1.2B`
- MOSS-Music-8B-Thinking (investigated, found infeasible for plain-transformers
  inference): `OpenMOSS-Team/MOSS-Music-8B-Thinking`
- CORAL: Cao, Mirjalili, Raschka, "Rank Consistent Ordinal Regression for Neural
  Networks with Application to Age Estimation," arXiv:1901.07884
- SORD: Díaz & Marathe, "Soft Labels for Ordinal Regression," CVPR 2019
- Squared-EMD loss: Hou, Yu, Samaras, "Squared Earth Mover's Distance-based Loss for
  Training Deep Neural Networks," 2016
- Supervised contrastive loss: Khosla et al., "Supervised Contrastive Learning," 2020
- Demucs (vocal/accompaniment separation): Défossez et al., Meta AI
- scikit-learn, PyTorch, HuggingFace Transformers, librosa, torchaudio (software
  libraries used throughout)

---

## 12. Reproducibility (required section)

- Full development history, every experiment: `docs/progress/WORKLOG.md`
- Exact reproduction commands for every result cited above: `docs/REPRODUCE.md`
- Final submission pipeline + inference instructions: `README.md` (in the submission
  folder) / `INFERENCE_README.md` (in the GitHub repo)
- All hyperparameters are frozen constants, fit once via leakage-free cross-validation on
  the training set only; every sklearn model selection step uses `random_state=0`.
- The submission pipeline self-verifies against the reported validation numbers before
  generating any test prediction, and aborts if they don't match — this is a real,
  automatic safety check, not just a claim.

---

## 13. Assets manifest (files to hand to the report generator)

**Required in the output — every file in this first group must appear as an embedded
figure somewhere in the final report, not just be available as a reference:**

```
assets/confusion_A_submitted.png     <- Task 1 confusion matrix (required by the assignment)
assets/confusion_B_submitted.png     <- Task 2 confusion matrix (required by the assignment)
assets/method_comparison_A.png       <- Task 1 all-methods bar chart
assets/method_comparison_B.png       <- Task 2 all-methods bar chart
assets/A_tsne_perp30.png             <- Task 1 t-SNE (required experiment, see section 4.1)
assets/A_umap_nn15.png               <- Task 1 UMAP (required experiment, see section 4.1)
assets/B_tsne_perp30.png             <- Task 2 t-SNE (required experiment, see section 4.1)
assets/B_umap_nn15.png               <- Task 2 UMAP (required experiment, see section 4.1)
```

**Reference only (not required as figures, available if useful):**
```
assets/A_tsne_perp15.png
assets/A_tsne_perp50.png             <- extra perplexity variants, confirm the finding's robustness
assets/submitted_config_metrics.json <- raw numbers behind the confusion matrices
```
