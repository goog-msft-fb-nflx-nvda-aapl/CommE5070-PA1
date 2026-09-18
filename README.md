# CommE5070 PA1 — Music Era & Release-Market Classification

Discogs-VI-derived 30s music excerpts, two tasks: Task 1 (Dataset A) = 6-way release-decade classification, Task 2 (Dataset B) = 6-way release-market classification. Deadline 2026-10-05.

Code, experiment log, and TODO tracking live here. GPU experiments run on the lab server (gsm-gpu2); this repo is the source of truth for code and progress.

## Layout

```
docs/
  spec/       Description.md, lecture02*.md          — assignment spec + course lecture notes
  research/   SURVEY.md, RESEARCH_PROMPT.md, PLAN.md,
              IMPROVEMENT_FINDINGS.md, IMPROVEMENT_RESEARCH_PROMPT.md,
              survey_response_1/, survey_response_2/  — SOTA survey + improvement-round deep research (own + 4-way each) + reconciled plans
  progress/   WORKLOG.md, TODO.md                     — full experiment log + current status/queue
  ta/         TA_QUESTIONS_MATERIALS.md, TA_QUESTIONS_MATERIALS_2.md  — briefing material for drafting TA questions (2 rounds)
  REPRODUCE.md                                        — exact commands to reproduce every best-config number below
src/          all pipeline code (data loading, baselines, MERT/MuQ probe/fine-tune, LoRA, ALM eval, ensembling, AF3 fusion, significance testing, etc.)
results/       lightweight result summaries (json/csv) for every logged experiment; embeddings_preview/ has a few representative t-SNE/UMAP plots
```

Start with `docs/progress/WORKLOG.md` for the full story, or `docs/progress/TODO.md` for current status — every other doc's path is listed above. **To reproduce any number below, see `docs/REPRODUCE.md`.**

**Not in this repo** (too large for git / lives on the GPU server): the dataset itself, cached embeddings, and model checkpoints. Checkpoints will be packaged separately for the assignment's required cloud-drive submission.

## Current best results (validation set; see `docs/progress/WORKLOG.md` "CURRENT overall-best configs" for full detail + statistical-significance caveats)

| task | config | top1 | top3 | notes |
|---|---|---|---|---|
| Task 1 (decade) | MuQ(layer1,SVM) probe + Audio Flamingo 3 (`direct` prompt) label-probability fusion, w_af3=0.5 | 0.553 | 0.849 | best point estimate; **not** statistically distinguishable from the row below at n=132 (bootstrap/McNemar) |
| Task 1 (decade), statistically defensible | 3-way ensemble: fine-tuned MERT-v1-330M (0.2) + Short-Chunk CNN (0.4) + frozen MERT-v1-330M+SVM (0.4) | 0.523 | 0.856 | confirmed-best if a single conservative number is needed |
| Task 2 (market) | MuQ(layer2,logreg) probe + Audio Flamingo 3 (`cot_then_answer` prompt) label-probability fusion, w_af3=0.4 | 0.657 | 0.882 | significantly beats every single-model config (p<0.01); not confirmed to beat AF3 alone (p=0.21) at n=102 |

## Workflow

- Mac: Claude Code session only — no local-only canonical copies of code/docs, everything durable lives here on GitHub.
- GPU server (gsm-gpu2): experiments only — code synced from this repo, results/logs written back up here.
