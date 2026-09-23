# CommE5070 PA1 — Music Era & Release-Market Classification

Discogs-VI-derived 30s music excerpts, two tasks: Task 1 (Dataset A) = 6-way release-decade classification, Task 2 (Dataset B) = 6-way release-market classification. Deadline 2026-10-05.

Code, experiment log, and TODO tracking live here. GPU experiments run on the lab server (gsm-gpu2); this repo is the source of truth for code and progress.

## Layout

```
docs/
  spec/       Description.md, lecture02*.md          — assignment spec + course lecture notes
  research/   SURVEY.md, RESEARCH_PROMPT.md, PLAN.md,
              IMPROVEMENT_FINDINGS.md, IMPROVEMENT_FINDINGS_ROUND3.md, IMPROVEMENT_RESEARCH_PROMPT.md,
              survey_response_1/, survey_response_2/, survey_response_3/  — SOTA survey + 2 rounds of improvement deep research (own + 4-way each) + reconciled plans
  progress/   WORKLOG.md, TODO.md                     — full experiment log + current status/queue
  ta/         TA_QUESTIONS_MATERIALS.md, TA_QUESTIONS_MATERIALS_2.md  — briefing material for drafting TA questions (2 rounds)
  REPRODUCE.md                                        — exact commands to reproduce every best-config number below
src/          all pipeline code (data loading, baselines, MERT/MuQ probe/fine-tune, LoRA, ALM eval, ensembling, AF3 fusion (validation-swept + OOF-refit), significance testing, etc.)
results/       lightweight result summaries (json/csv) for every logged experiment, incl. results/alm/ (raw AF3 zero-shot predictions, validation + train splits); embeddings_preview/ has a few representative t-SNE/UMAP plots
```

Start with `docs/progress/WORKLOG.md` for the full story, or `docs/progress/TODO.md` for current status — every other doc's path is listed above. **To reproduce any number below, see `docs/REPRODUCE.md`.**

**Not in this repo** (too large for git / lives on the GPU server): the dataset itself, cached embeddings, and model checkpoints. Checkpoints will be packaged separately for the assignment's required cloud-drive submission.

## Current best results (validation set; see `docs/progress/WORKLOG.md` "CURRENT overall-best configs" for full detail + statistical-significance caveats)

**Important, learned the hard way (2026-09-18/19)**: our first-pass fusion numbers were picked by sweeping the weight directly on the 102/132-sample validation set — the same set used to report the result. A leakage-free out-of-fold (OOF) refit on the training set showed Task 2's fusion number was substantially inflated by this (0.657 did not reproduce, landing at ≈0.588 instead); Task 1's held up much better (0.553 → 0.538, a small drop). **Always prefer the OOF-refit numbers below over the original validation-swept ones.**

| task | config | top1 | top3 | notes |
|---|---|---|---|---|
| Task 1 (decade) | MuQ(layer1,SVM) probe + Audio Flamingo 3 (`direct` prompt), OOF-refit fusion weight (fit on 1026 training-set out-of-fold predictions, not validation) | 0.538 | 0.864 | significantly beats AF3-alone (p=0.0055); not significantly different from the probe alone (p=0.29) or the row below |
| Task 1 (decade), statistically defensible alternative | 3-way ensemble: fine-tuned MERT-v1-330M (0.2) + Short-Chunk CNN (0.4) + frozen MERT-v1-330M+SVM (0.4) | 0.523 | 0.856 | confirmed-best if the simplest, most conservative single number is needed |
| Task 2 (market), **recommended primary number** | Contextually-calibrated Audio Flamingo 3 zero-shot alone (`direct` prompt) | **0.6275** | 0.7941 | a real, mechanistically-explained gain over raw AF3 (0.588) — tracks a measured ~2-nat label bias in this specific (model,prompt)'s content-free null distribution; negligible for the other prompt/task combos tested, as expected |
| Task 2 (market), unconfirmed higher point estimate | MuQ(layer2,logreg) + Whisper-large-v3 sung-language-ID (6-dim, Demucs vocal stem) concat probe, OOF-fitted fusion with the calibrated AF3 above | 0.647 | 0.863 | higher on the single validation split, but a repeated-CV Bayesian correlated t-test (Nadeau-Bengio corrected) does **not** confirm this beats calibrated-AF3-alone — it actually leans the other way (P(AF3-alone better)=0.78 vs P(fused better)=0.01). Not recommended as the primary number. |

## Workflow

- Mac: Claude Code session only — no local-only canonical copies of code/docs, everything durable lives here on GitHub.
- GPU server (gsm-gpu2): experiments only — code synced from this repo, results/logs written back up here.
