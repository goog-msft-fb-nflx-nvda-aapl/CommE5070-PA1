# CommE5070 PA1 — Experiment Tracker

Working one item at a time, deep-dive ablations before moving on (per standing instruction). Status: `[ ]` not started, `[~]` in progress, `[x]` done. Full detail/results live in WORKLOG.md; this file is just the queue + pointer.

## Baselines (fully done, both tasks — Baseline 1 + Baseline 2 incl. all 3 classifiers + frozen-vs-fine-tuned)
- [x] Baseline 1 — Short-Chunk CNN, both tasks. A: top1=0.4470/top3~0.78. B: top1=0.3725/top3~0.72.
- [x] Baseline 2 — MERT-v1-330M. Layer sweep + logreg + MLP + SVM + PCA (frozen probe) + fine-tuning, done both tasks. **Final winning config per task**: A = MERT **fine-tuned** (epoch-6 checkpoint), top1=**0.500**/top3=0.856 — beats every frozen-probe config (SVM+PCA64 top1=0.470) and Short-Chunk CNN (0.447). B = **frozen** logreg no-PCA (layer7), top1=0.471/top3=0.755 — fine-tuning tested but did NOT beat it here (best fine-tuned top1=0.412). Task-dependent result, reported as measured, not smoothed into one story.

## Required experiments (Description.md)
- [x] Segment-length sweep: 5s/10s/15s/30s, both tasks done. A: top1 monotonic 0.333→0.409→0.424→0.462, full clip wins clearly. B: top1 also favors 30s (0.471 best) but top3 non-monotonic, peaks at 10s/15s (0.794) above 30s (0.755) -- noisy small val set (102 samples), flagged as such not smoothed over. See WORKLOG.md.
- [x] Multi-excerpt/TTA ablation (10s crops, n_crops 1/3/5/8), both tasks done. A: n_crops=3/5 EXCEED the 30s baseline (0.485/0.477 vs 0.462). B: does NOT replicate -- 30s baseline (0.471) beats every multi-crop config (best 0.441). Task-dependent effect, not universal. See WORKLOG.md.
- [x] Task 1 year-regression / hierarchical framing — done. Regression: worse than flat on every metric (top1=0.273). Hierarchical (coarse pairs → fine 2-way): top1=0.432 (slightly below flat's 0.462) but QWK=0.659 beats flat's 0.621 and MAE=0.939 ties/slightly beats flat's 0.947 -- smoother errors, not higher accuracy. See WORKLOG.md.
- [x] Task 2 mixture/vocal/accompaniment stem comparison — done. Mixture wins clearly (top1=0.471) over vocals (0.402) and accompaniment (0.373). Vocals>accompaniment on top1 flagged as a possible language/accent confound to discuss honestly. See WORKLOG.md.
- [x] t-SNE/UMAP visualization — both tasks done. No clean visual class separation in either projection for either task (intermixed blob, checked across 3 perplexities/3 n_neighbors, consistent). Doesn't contradict the classifier's real above-chance performance (0.46-0.48 vs 0.167 random) -- high-dim separability ≠ 2D visual separability. See WORKLOG.md. Will revisit with MusicFM once that encoder is added for a 2nd embedding-family comparison.
- [x] Audio Language Model track — FULLY DONE: both models (Qwen2-Audio-7B-Instruct, Audio Flamingo 3), both datasets, 2 prompts each, invalid-output-rate reported = 8 runs total.
  - Qwen2-Audio: A (prompts equivalent, ~0.33 top1, below baselines). B (cot_then_answer top1=0.441/27.5% invalid vs direct top1=0.324/0% invalid; cot's top3=0.794 beats the MERT probe's 0.755).
  - Audio Flamingo 3: A (direct top1=0.379 best of the 4 ALM configs on A but lower QWK than Qwen2-Audio; both prompts NEVER predict "2010s" top-1). B (**both prompts beat the MERT probe baseline on top1** -- direct 0.588, cot 0.559, vs baseline 0.471 -- strong US-prediction skew, ~50-55% of all predictions).
  - See WORKLOG.md for the full 8-config comparison table and all bugs found/fixed along the way.

- [x] SVM classifier on the winning MERT layer, both tasks, pca∈{None,64,128} each (matching logreg/MLP ablation depth). **SVM+PCA64 is the new best config for A** (top1=0.470, beats logreg+PCA128's 0.462). For B, logreg remains best on top1 (0.471 vs SVM's best 0.431) but SVM+PCA64 has the best top3 for B (0.784). Baseline 2 now covers all 3 named classifiers. See WORKLOG.md.

## A/A+ differentiators (not required, but planned per SURVEY.md/PLAN.md)
- [x] Ensembling (inspired by prior-semester HW1's graded best = weighted ensemble of 7 models). Both tasks, 2-way and 3-way, fully done. **A: new overall best config, top1=0.5227** (3-way: ft=0.2/cnn=0.4/mert=0.4). **B: ensembling does NOT beat the pure frozen-MERT probe on top1** (stays at 0.471); helps top3 only (0.794 vs 0.755). Same A-benefits/B-doesn't pattern seen in TTA, SVM, and fine-tuning too. See WORKLOG.md "Final overall-best configs" table.
- [ ] MusicFM 3rd encoder — same layer-sweep + classifier/PCA ablation treatment as MERT, both tasks.
- [ ] Hand-crafted librosa/torchaudio features (MFCC+spectral stats, mean+std pooled) — cheapest interpretable baseline, useful for "what did the model learn" report discussion.
- [x] Fine-tuning the encoder (vs. frozen-probe-only) — both tasks done. A: fine-tuning wins outright (new best overall, top1=0.500). B: fine-tuning tested, does NOT beat the frozen probe (best 0.412 vs 0.471). See Baseline 2 line above and WORKLOG.md.

## Known limitations to state in the report (not experiments, just carry forward)
- No artist_id in manifest → can't do artist-grouped k-fold CV, used plain StratifiedKFold instead.
- GPU3 on gsm-gpu2 unusable (`torch.cuda.is_available()` False despite looking idle) — only GPU1/GPU2 used, on top of the assignment's own documented GPU0 malfunction.
- 2026-09-15 ~20:40-21:20: gsm-gpu2 had a system-wide load/process-count incident (unrelated to our work, traced to another user) that interrupted one run; no impact on reported results, logged in WORKLOG.md for completeness/reproducibility notes.

---
## Resume point (updated 2026-09-17 ~00:22)

**Baseline 2 is now fully complete for both tasks** — all 3 named classifiers (logreg/MLP/SVM) via frozen probing, plus the frozen-vs-fine-tuned comparison, both A and B. This closes every required item from Description.md's baseline spec. Final winning configs: A = fine-tuned MERT (epoch-6 checkpoint, top1=0.500/top3=0.856); B = frozen MERT layer7+logreg (top1=0.471/top3=0.755).

**Everything required by Description.md is now done**: both baselines (with all named sub-variants), segment-length sweep, multi-excerpt/TTA, Task 1 year-regression/hierarchical framing, Task 2 stem comparison, t-SNE/UMAP, and the full ALM track (2 models × 2 datasets × 2 prompts). Remaining items are optional A/A+ differentiators only, not required: MusicFM 3rd encoder, hand-crafted librosa/torchaudio features. Next natural step is either one of those, or moving toward the report/writeup (16:9 HTML slide deck per Description.md's Submission section) and the prediction JSON for the hidden test set.

Useful reusable fix from this session: `conda run -n ENV python ...` can cause a long-running single-process job to stall for real (not just hide log output) — prefer invoking the env's Python binary directly (`/home/jtan/miniconda3/envs/pa1_env/bin/python3 -u script.py ...`) for any multi-epoch training job on gsm-gpu2. See reference_gsmgpu_condarun_buffering.md memory.

Note: gsm-gpu2's `uptime` load reading has periodically shown a suspicious frozen-looking value (stuck at exactly the same number across all three time windows) that didn't match actual system responsiveness (a real `time sleep 1` command still returned in ~1s) — if this recurs, don't trust the raw load figure alone, verify with a real timed command before concluding the system is unusable.
