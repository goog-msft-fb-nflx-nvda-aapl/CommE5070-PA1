# CommE5070 PA1 — Music Era & Release-Market Classification

Discogs-VI-derived 30s music excerpts, two tasks: Task 1 (Dataset A) = 6-way release-decade classification, Task 2 (Dataset B) = 6-way release-market classification. Deadline 2026-10-05.

Code, experiment log, and TODO tracking live here. GPU experiments run on the lab server (gsm-gpu2); this repo is the source of truth for code and progress.

## Layout

- `Description.md` — assignment spec.
- `SURVEY.md`, `RESEARCH_PROMPT.md`, `survey_response_1/` — SOTA survey (own + 4-way deep research) that informed method choices.
- `PLAN.md` — reconciled implementation plan from that survey.
- `WORKLOG.md` — full experiment log: every config tried, every bug found/fixed, all results, in the order they happened.
- `TODO.md` — current status / queue, with a "Final overall-best configs" summary.
- `TA_QUESTIONS_MATERIALS.md` — briefing material for drafting TA questions.
- `src/` — all pipeline code (data loading, baselines, MERT probe/fine-tune, ALM eval, ensembling, etc.).
- `results/embeddings_preview/` — a few representative t-SNE/UMAP plots.
- `lecture02*.md` — course lecture notes (fundamentals reference).

**Not in this repo** (too large for git / lives on the GPU server): the dataset itself, cached embeddings, and model checkpoints. Checkpoints will be packaged separately for the assignment's required cloud-drive submission.

## Current best results (validation set; see WORKLOG.md "Final overall-best configs" for full detail)

| task | config | top1 | top3 |
|---|---|---|---|
| Task 1 (decade) | 3-way ensemble: fine-tuned MERT-v1-330M (0.2) + Short-Chunk CNN (0.4) + frozen MERT-v1-330M+SVM (0.4) | 0.523 | 0.856 |
| Task 2 (market) | frozen MERT-v1-330M (layer 7) + logistic regression | 0.471 | 0.755 |

## Workflow

- Mac: Claude Code session only — no local-only canonical copies of code/docs, everything durable lives here on GitHub.
- GPU server (gsm-gpu2): experiments only — code synced from this repo, results/logs written back up here.
