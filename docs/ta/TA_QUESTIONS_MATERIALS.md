# Materials for Drafting TA Questions — CommE5070 PA1 (Music Era & Release-Market Classification)

## Task

Using the background below, draft a short, polite set of questions to send to the teaching assistant. Specifically:
1. Ask whether there is an expected/target **Top-1 and Top-3 accuracy range on the validation set** for Task 1 (decade) and Task 2 (market) — we want to sanity-check whether our current numbers are in a reasonable ballpark for this assignment, or whether they suggest something is off in our implementation.
2. Ask if the TA has **any suggestions for further improving accuracy** given everything we've already tried (listed below) — i.e., are we missing an approach that's expected to matter for this task.
3. (Optional, include if it reads naturally) Ask whether there's a **recommended cross-validation strategy** given the manifest doesn't include artist IDs (see "known limitation" below) — we're currently using plain StratifiedKFold on the training split since artist-grouped folds aren't possible without that field.

Also include the additional clarification questions listed in "Questions we (the implementers) would also like clarified" below — pick whichever of those are genuinely useful to ask alongside 1-3 above, without making the final list too long; merge/cut as needed for a clean, short set.

Keep the final question set short (a handful of specific questions, not a wall of text), professional, and easy for a TA to answer quickly. Don't pad with information they already know (they wrote the assignment) — just enough context so the two specific config descriptions below make sense.

**Output format**: render the final question set as a **self-contained HTML snippet** (not a full `<html>`/`<head>`/`<body>` document — just the inner markup, e.g. a wrapping `<div>` with inline or `<style>`-block CSS) with a clean, professional color scheme (readable on both light and dark forum themes if possible — avoid pure white backgrounds/pure black text that could look jarring, prefer a neutral/muted palette), good typographic hierarchy (a short intro line, then a numbered or bulleted question list, maybe light use of `<strong>`/color to highlight the two key numbers — our Top-1/Top-3 results — and the specific model names). This needs to be **directly copy-pasteable into a forum discussion's HTML/rich-text editor**, so keep the markup simple and avoid anything that depends on external stylesheets, fonts, or JS.

---

## Assignment context (brief)

Graduate MIR coursework assignment: 30-second music excerpts (WAV, mono, 24kHz), two single-label 6-way classification tasks derived from Discogs-VI editorial metadata:
- **Task 1 (Dataset A)**: release-decade classification — 1960s/1970s/1980s/1990s/2000s/2010s. 1026 train / 132 validation / 132 test, artist-disjoint splits, perfectly class-balanced.
- **Task 2 (Dataset B)**: release-market classification — US/UK/Brazil/Spain/Germany/Italy, all releases from the 1980s. 798 train / 102 validation / 102 test, artist-disjoint, balanced.

Random baseline for both: Top-1 = 16.7%, Top-3 = 50%.

## Our best configuration per task

**Task 1 (decade) — best result: Top-1 = 52.27%, Top-3 = 85.61% (validation)**

A 3-way weighted probability ensemble of:
1. **Fine-tuned MERT-v1-330M** (weight 0.20) — HuggingFace model [`m-a-p/MERT-v1-330M`](https://huggingface.co/m-a-p/MERT-v1-330M), a self-supervised music foundation model (330M params, 24 transformer layers, native 24kHz). We fully fine-tuned all parameters end-to-end (not frozen) with a linear classification head on top of the mean-pooled final hidden state, on 10-second random crops, using AdamW (lr=2e-5, cosine schedule), 15 epochs.
2. **Short-Chunk CNN** (weight 0.40) — a small CNN trained entirely from scratch on log-mel spectrograms (per Won et al., "Evaluation of CNN-based automatic music tagging models," SMC 2020), using 3.7-second training crops and multi-segment-averaged inference.
3. **Frozen MERT-v1-330M** (weight 0.40) — same base model as #1 but with weights frozen, using layer 4's (of 24) time-mean-pooled hidden states, PCA-reduced to 64 dimensions, classified with an SVM (RBF kernel).

**Task 2 (market) — best result: Top-1 = 47.06%, Top-3 = 75.49% (validation)**

A single model: **frozen MERT-v1-330M**, layer 7 (of 24), time-mean-pooled hidden states (no PCA), classified with logistic regression (L2-regularized). Notably, for this task neither fine-tuning MERT nor ensembling with the CNN improved on this frozen-probe result — every combination we tried (weighted ensembles, fine-tuned MERT alone, SVM instead of logistic regression) either matched or underperformed this single configuration on Top-1.

## What we've already tried (so the TA can gauge what's left)

- Both explicitly-required baselines: Short-Chunk CNN (from scratch) and MERT + classifier (frozen probe), with all three named classifier options (logistic regression, SVM, MLP) and a PCA-dimensionality ablation.
- Frozen-probe vs. full fine-tuning comparison for MERT (fine-tuning won for Task 1, did not win for Task 2).
- A full MERT layer-sweep (all 24 transformer layers + embedding layer) to find the best-performing layer per task — found different optimal layers per task (layer 4 for decade, layer 7 for market).
- Required experiments: segment-length comparison (5/10/15/30s), multi-excerpt/test-time-augmentation ablation, Task 1 reframed as year-regression and as a hierarchical (coarse-then-fine) classifier, Task 2 mixture/vocals/accompaniment stem comparison via Demucs source separation, t-SNE/UMAP embedding visualization.
- Zero-shot audio language models: Qwen2-Audio-7B-Instruct and NVIDIA Audio Flamingo 3, with two different prompt designs each, evaluated via teacher-forced label scoring for a fair Top-1/Top-3 comparison. Both stayed below the frozen-probe baseline on Task 1; Audio Flamingo 3 actually *beat* the frozen MERT probe on Task 2 (Top-1 58.8%) — interesting but a separate track from our main "trained classifier" pipeline above.
- 2-way and 3-way probability-ensemble sweeps combining the above.

## Known limitation we're not sure how to handle

The provided manifest does **not** include artist IDs (they're stripped for anonymity, only `sample_id`/`split`/`label`/`audio_path` are given). The official train/val/test splits are already artist-disjoint per the assignment description, but this means we can't do artist-grouped k-fold cross-validation *within* the training split for model/hyperparameter selection — we're using plain (non-grouped) StratifiedKFold instead. Not sure if this is expected/fine or if there's a recommended alternative.

## Questions we (the implementers) would also like clarified

- **Scope of "fine-tuning" in Baseline 2**: Description.md says "Compare frozen features with fine-tuning if resources allow." We did a full end-to-end fine-tune of all 330M MERT parameters (not partial/last-layers-only, not LoRA), which overfits within a handful of epochs on this small dataset (1026/798 training clips) and shows meaningful run-to-run variance (different runs peak at different epochs, ~1-2 points of Top-1 apart) even with the same hyperparameters. Is full fine-tuning what's intended, or would a lighter-touch approach (partial unfreezing, LoRA, lower LR, stronger regularization) be more in the spirit of the assignment at this data scale?
- **Is combining/ensembling multiple trained models an acceptable "final" configuration**, or is a single, clearly-identified model expected for the submitted prediction JSON and report headline number? Our best Task 1 result is a 3-way weighted-probability ensemble (fine-tuned MERT + a from-scratch CNN + a frozen-MERT+SVM probe) rather than any single model — want to confirm this is a reasonable thing to submit as our primary result, versus reporting it as a secondary "beyond the baseline" experiment alongside a single-model primary submission.
- **Which split to use for the required Audio Language Model experiment**: we ran Qwen2-Audio-7B-Instruct and Audio Flamingo 3 on the **validation** split (rather than train or test) so we could compute accuracy against ground-truth labels, since test labels are hidden. Is validation the intended split for this experiment, or did you have train (larger, but used for model fitting elsewhere) or test (matches the eventual grading setup, but we can't score it ourselves) in mind?
- **Label-noise caveat**: since the decade/market labels come from Discogs editorial metadata rather than perceptual annotation, is some irreducible label noise (e.g. misattributed reissues/compilations) expected and an acceptable point to raise when discussing our confusion-matrix/error analysis, or should we treat the labels as ground truth without qualification in the report?
