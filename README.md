# CommE5070 PA1 — Music Era & Release-Market Classification

Discogs-VI-derived 30s music excerpts, two tasks: Task 1 (Dataset A) = 6-way release-decade classification, Task 2 (Dataset B) = 6-way release-market classification. Deadline 2026-10-05.

Code, experiment log, and TODO tracking live here. GPU experiments run on the lab server (gsm-gpu2); this repo is the source of truth for code and progress.

## Layout

```
docs/
  spec/       Description.md, lecture02*.md          — assignment spec + course lecture notes
  research/   SURVEY.md, RESEARCH_PROMPT.md, PLAN.md,
              survey_response_1/                      — SOTA survey (own + 4-way deep research) + reconciled plan
  progress/   WORKLOG.md, TODO.md                     — full experiment log + current status/queue
  ta/         TA_QUESTIONS_MATERIALS.md                — briefing material for drafting TA questions
src/          all pipeline code (data loading, baselines, MERT probe/fine-tune, ALM eval, ensembling, etc.)
results/embeddings_preview/   a few representative t-SNE/UMAP plots
```

Start with `docs/progress/WORKLOG.md` for the full story, or `docs/progress/TODO.md` for current status — every other doc's path is listed above.

**Not in this repo** (too large for git / lives on the GPU server): the dataset itself, cached embeddings, and model checkpoints. Checkpoints will be packaged separately for the assignment's required cloud-drive submission.

## Current best results (validation set; see `docs/progress/WORKLOG.md` "Final overall-best configs" for full detail)

| task | config | top1 | top3 |
|---|---|---|---|
| Task 1 (decade) | 3-way ensemble: fine-tuned MERT-v1-330M (0.2) + Short-Chunk CNN (0.4) + frozen MERT-v1-330M+SVM (0.4) | 0.523 | 0.856 |
| Task 2 (market) | frozen MERT-v1-330M (layer 7) + logistic regression | 0.471 | 0.755 |

## Workflow

- Mac: Claude Code session only — no local-only canonical copies of code/docs, everything durable lives here on GitHub.
- GPU server (gsm-gpu2): experiments only — code synced from this repo, results/logs written back up here.
