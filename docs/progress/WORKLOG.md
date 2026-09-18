# CommE5070 PA1 — Work Log

Assignment: Music Era and Release-Market Classification (Discogs-VI, 30s excerpts)
- Task 1: Dataset A, 6 US release decades (1960s-2010s), 1026/132/132 train/val/test
- Task 2: Dataset B, 6 release markets in 1980s (US/UK/Brazil/Spain/Germany/Italy), 798/102/102
Deadline: 10/05 23:59. GPU: gsm-gpu2 (H200 NVL x4, GPU 0 malfunctioning per assignment env notes, GPU 3 also empirically fails `torch.cuda.is_available()` despite looking idle — use GPU1/GPU2 only), no sudo, tmux+conda, scope strictly to /home/jtan/comme5070_pa1.

## 2026-09-15 — Setup + EDA

- Read Description.md, lecture02_classification.md, lecture02b_transcription.md.
- Extracted dataset_A.zip / dataset_B.zip to dataset/extracted/ (Mac, for inspection only).
- Manifest check: both datasets perfectly class-balanced in train/val. A: train 171/class x6, val 22/class x6, test 132. B: train 133/class x6, val 17/class x6, test 102.
- Audio uniform: mono, PCM16, 24kHz, exactly 30.000s. No near-silent clips.
- **Loudness-war confound (Task 1)**: per-decade mean RMS rises 1960s 0.208 → 2010s 0.292 (1970s dip 0.196). Confirmed real (Serrà et al. 2012, cross-checked by deep research).
- No artist_id in manifest — artist-grouped CV not implementable, plain StratifiedKFold used, noted as limitation.

### Research phase
- Own WebSearch survey → SURVEY.md. User relayed 4-way deep research (Gemini/Kimi/Perplexity/Qwen) → survey_response_1/*.md. Kimi/Perplexity strongest independent passes. Reconciled + 2 direct verifications → PLAN.md. Corrections: Qwen2-Audio-7B-Instruct = Apache 2.0 (perplexity.md wrong); MusicFM native = 24kHz confirmed.

### Implementation
- All execution on gsm-gpu2 only (never locally on Mac, even smoke tests — standing rule). Code: src/{config,data,scnn,train_scnn,metrics,mert_features,train_probe,ablate_mert_layers,predict,stems,visualize_embeddings,alm_infer}.py.
- GPU env: `pa1_env` cloned from `hw1_singer_env`, + pyloudnorm/umap-learn/pandas. Had to upgrade torch 2.5.1→2.6.0+cu124 in pa1_env only (transformers 5.16.1 blocks `torch.load` below torch 2.6 for non-safetensors checkpoints, CVE-2025-32434).

### Baseline 1 — Short-Chunk CNN (60 epochs, mixup+SpecAugment, 3.7s crops)
- Note: GPU3 looks idle in nvidia-smi but `torch.cuda.is_available()` is False on it — unusable, unlike GPU0 (assignment's documented malfunctioning one). Only GPU1/GPU2 confirmed working.
- Dataset A (decade): best val top1=0.4470, top3~0.77-0.79 (random baseline 0.167/0.50).
- Dataset B (market): best val top1=0.3725, top3~0.71-0.72 (random baseline 0.167/0.50).

### Baseline 2 — MERT-v1-330M probe, Dataset A deep-dive (per "one experiment at a time" — worked A fully before starting B)
- **Bug found+fixed**: transformers 5.16.1's Hubert internals drifted from what MERT's pinned trust_remote_code expects — `output_hidden_states=True` silently returns `hidden_states=None` even with `return_dict=True`/`config.output_hidden_states=True` forced. Fixed via forward hooks on `encoder.layers` (24) + `feature_projection` (embedding layer) = 25 layer outputs, version-independent.
- All 1290 Dataset A clips cached (cache/mert_v1_330m/A/*.npz, 25×1024 each, ~12 clips/s on GPU1).
- **Layer sweep (logreg, 25 layers + mean_all + concat_last4)** — results/mert_layer_sweep_A/layer_sweep_summary.csv: **layer 4 wins** (top1=0.4545, top3=0.8561, QWK=0.6545, MAE=0.924 decades), beats mean_all (0.4394) and last layer 24 (0.4015) — matches literature that intermediate layers carry timbral/production cues most relevant to era.
- **Classifier/PCA ablation on layer 4** — results/mert_layer4_ablation_A/summary.json: **logreg+PCA(128) best** (top1=0.4621, top3=0.8712); PCA(64) too aggressive (0.4318); MLP worse than logreg at every PCA setting (0.379-0.402), confirms small-data-favors-linear-probe consensus from deep research.
- **Current best A config**: MERT layer4 + PCA128 + logreg → top1=0.4621, top3=0.8712, MAE=0.947, acc-within-1-decade=0.780, **adjacent_fraction_of_errors=0.592** (59% of misclassifications land on an immediately neighboring decade — directly answers the assignment's required discussion), QWK=0.621.
- Row-normalized confusion: 1960s 64%, 1970s 45%, 1980s 59%, 1990s 32%, **2000s weakest at 18%** (confused most with 2010s at 32%), 2010s 59%. 2000s being the weak point matches the loudness-war EDA finding (2000s/2010s both "loud mastering" era) — ties EDA → layer-sweep → confusion-matrix into one coherent story.
- Dataset B MERT features cached (1002/1002 clips) but the B layer sweep was interrupted mid-run by a dropped SSH connection (see incident below) — not yet re-run.

### Incident (2026-09-15 ~20:40-20:56) — gsm-gpu2 extreme system load, unrelated to our work
- SSH connectivity to gsm-gpu2 became intermittently hanging/dropping (some commands timing out or failing with exit 255) around the time the Dataset B MERT layer sweep was running via bare `ssh` (not tmux) — the dropped connection likely killed that process via SIGHUP.
- Diagnosed `uptime`: **load average ~6467-6470 sustained across all three 1/5/15-min windows** (should be low single/double digits) — elevated for 15+ min before we even checked, not caused by our own lightweight jobs (a finished CNN training run + a small sklearn grid search). 44 users on the box.
- Confirmed via `pgrep -u jtan -fa python3` that **jtan's own PA1-related processes were fully clean** — no orphaned `ablate_mert_layers.py`/`train_probe.py`/`mert_features.py`/`train_scnn.py` processes survived the dropped connection.
- Found and killed 3 pairs of **stray, hung `ps aux | grep python3` diagnostic one-liners** (PIDs 930477/930480, 2759309/2759312, 2762988/2762990) left over from earlier unscoped `ps aux` checks (mine, likely stuck because unscoped `ps aux` itself hangs under this load) — these were harmless but pointless, cleaned up per user request. Load average unchanged after killing them (confirms they weren't the cause, just victims of the same slowness).
- **Left untouched**: two legitimate long-running jtan training jobs from the unrelated [[project_tohoku_wind]] project (`scripts/train_model.py --split hour3_matched` on GPU2, `--split chronological` on GPU0, both nohup+disown'd — clearly intentional background training, not orphaned). Did not kill these without the user's explicit confirmation, since they're real ongoing work in another project's conda env (rl_hw3), not something this PA1 session launched.
- No `pa1_*` tmux sessions were stale/leftover (each auto-exits cleanly when its command finishes or crashes) — nothing to clean up there.
- Root cause of the load-6467 spike itself not investigated further (per standing rule: never enumerate/touch other users' processes on this shared box) — likely another user's runaway job. Flagged to user; not our issue to fix.

### Process-cleanup incident, continued
- User confirmed the two tohoku_wind jobs were jtan's own → killed. Parent bash wrappers (3865963/3865969) died; python3 child 3865988 became a zombie (harmless, awaiting reap); python3 child 3865986 stayed stuck in R state, unresponsive to repeated `kill -9` — genuinely wedged at the kernel level (SigQ limit read as 6189326, implying millions of processes/threads system-wide — likely another user's fork bomb, not ours). Swap also found 100%/8GB full; jtan's own processes show ~0 swap contribution (checked via /proc, not other users' data). Drafted a polite JP message + exact `sudo kill -9 3865986` / process-group fallback commands for the user to send to their manager (has sudo). This is now out of our hands — needs admin intervention, not something fixable as jtan.
- Despite the system distress, GPU1/GPU2 CUDA and lightweight CPU-only sklearn jobs (train_probe.py) continued to work fine when launched inside tmux — the earlier failure was specifically bare-ssh connections dying, not GPU/compute failure. User said to continue the experiment; relaunched Dataset B MERT layer sweep in tmux (survives dropped connections) and it completed normally.

### Baseline 2 — MERT-v1-330M probe, Dataset B deep-dive (mirrors the Dataset A sequence)
- **Layer sweep (logreg, 25 layers + mean_all + concat_last4)** — results/mert_layer_sweep_B/layer_sweep_summary.csv: **layer 7 wins** (top1=0.4706, top3=0.7549, cv=0.3346) — a different optimal layer than Dataset A's (layer 4), consistent with literature that optimal layer is task-dependent, not a fixed property of the encoder. Beats mean_all (top1=0.3529) and every other single layer.
- Layer 7 already clearly beats the Short-Chunk CNN Dataset B baseline (top1=0.3725, top3~0.72) — top1 +0.098, top3 +0.035.
- **Classifier/PCA ablation on layer 7** — results/mert_layer7_ablation_B/summary.json: **logreg, no PCA is best here** (top1=0.4706, top3=0.7549) — unlike Dataset A, where PCA(128) gave a small boost, for B any PCA reduction *hurts* (PCA64: 0.382, PCA128: 0.412) — so the "PCA helps" finding from A does not generalize uniformly; it's embedding/task-dependent, not applied blindly. MLP no-PCA (0.4412) is close second but still below logreg, again confirming linear-probe-beats-MLP at this data scale.
- **Current best config for Dataset B**: MERT-v1-330M layer 7 + logistic regression (no PCA) → val top1=0.4706, top3=0.7549.
- Row-normalized confusion (results/mert_layer7_B_final/): US 65%, UK 29%, **Brazil best-separated at 71%**, **Spain weakest at 18%** (confused with UK 29% and Brazil 29%), Germany 59%, Italy 41%. UK also weak (29%, confused with US 47%). No natural ordinal structure for markets (per Description.md), so this is reported qualitatively rather than via MAE/QWK — plausible reading: Brazil's Portuguese-language/distinct production tradition separates cleanly, while UK/US/Spain share more overlapping Anglo/Euro-pop production conventions in the 1980s, which is exactly the kind of "release market ≠ language/nationality but still correlates with regional production style" nuance the assignment flags.

### Baseline 2 summary (both tasks)
| | Short-Chunk CNN | MERT probe (best layer+clf) |
|---|---|---|
| A (decade) | top1=0.4470, top3~0.78 | layer4+PCA128+logreg: top1=0.4621, top3=0.8712 |
| B (market) | top1=0.3725, top3~0.72 | layer7+logreg: top1=0.4706, top3=0.7549 |

MERT frozen-probe baseline beats the from-scratch Short-Chunk CNN on both tasks at this data scale, on both top1 and top3 — consistent with the deep-research consensus that pretrained SSL audio encoders transfer well even to a small, out-of-domain classification task, and are the stronger of the two required baselines here.

### Required experiment — segment-length sweep, Dataset A (single center-ish crop per length, n_crops=1; multi-excerpt/TTA is a separate axis via --n-crops, not yet run)
- src/mert_features.py extended with crop_seconds/n_crops params (cache keyed by length so it never collides with the full-clip cache); src/sweep_segment_length.py drives extraction+probe (winning per-dataset config: A=layer4+PCA128+logreg) at each length.
- results/seglen_sweep_A/summary_1crop.csv:

| length | top1 | top3 | MAE (decades) | QWK |
|---|---|---|---|---|
| 5s | 0.333 | 0.795 | 1.083 | 0.610 |
| 10s | 0.409 | 0.826 | 1.091 | 0.558 |
| 15s | 0.424 | 0.826 | 1.068 | 0.549 |
| 30s | 0.462 | 0.871 | 0.947 | 0.621 |

- **Top1 improves monotonically with length, no plateau below 30s** — full clip is still meaningfully better than any truncation tried (30s beats 15s by +0.038 top1, 15s beats 5s by +0.091). Top3 saturates earlier (10s≈15s≈0.826, only jumps again at 30s to 0.871). QWK/MAE are non-monotonic in the middle (10s/15s dip below even 5s's QWK) — likely noise at this val-set size (132 samples), not a real regression; top1 trend is the reliable signal. **Conclusion for report: more audio helps, use the full 30s when available; there's no free lunch from truncating to save compute on this task.**
- results/seglen_sweep_B/summary_1crop.csv (winning config: B=layer7+logreg, no PCA; no ordinal columns, market has no natural order):

| length | top1 | top3 |
|---|---|---|
| 5s | 0.382 | 0.765 |
| 10s | 0.353 | 0.794 |
| 15s | 0.392 | 0.794 |
| 30s | 0.471 | 0.755 |

- **Different (non-monotonic) trend vs Task A**: top1 still favors the full 30s clip (best at 0.471, consistent with A's "more audio helps" finding for top1), but top3 actually *peaks* at 10s/15s (0.794) and dips at 30s (0.755) — the reverse of A's pattern. This is measured, not adjusted: val set here is only 102 samples (~17/class), so this dip could be noise rather than a real effect (report as observed, flagged as noisy-small-n, not as a claimed ceiling either way). Worth noting in the report alongside A's cleaner monotonic trend rather than smoothing it into one universal "length helps" claim — the two tasks don't necessarily benefit from audio length identically.

### Required experiment — multi-excerpt/TTA ablation, Dataset A (10s crops, n_crops in {1,3,5,8}, mean-pooled embeddings)
- src/sweep_multicrop.py added (reuses mert_features.py's crop_seconds/n_crops extraction).
- **Bug hunt along the way** (worth recording for reproducibility): first attempt looked "stuck" (0 cache files, 0% GPU util, ~10 min with no progress) — misdiagnosed as `GridSearchCV(n_jobs=-1)` deadlocking under the ongoing system-load incident and killed prematurely. Added an `n_jobs` param (default 1) to train_probe.run() as a genuinely good fix regardless (avoids multiprocess spawn contention on a shared/loaded box), but the real bug was elsewhere: after the kill, relaunching hit a **stale-cache bug** — the sweep scripts' "cache dir exists and is non-empty → skip re-extraction" check doesn't verify completeness, so the killed job's partial n_crops=8 cache (777/1290 clips, train-split only, before it ever reached validation) was wrongly treated as finished, and `train_probe.run()` crashed on an empty validation array. **Fixed** with `cache_is_complete()` in src/mert_features.py (compares cached file count against the full manifest row count) used in both sweep_segment_length.py and sweep_multicrop.py. Also fixed a `--crop-seconds` int/float type inconsistency between the two sweep scripts that was silently creating duplicate cache dirs (`seg10.0s_*` vs `seg10s_*`) — normalized to int. Orphaned float-named cache dirs deleted after confirming the int-named rerun reproduced the same numbers.
- results/multicrop_sweep_A/summary_10s.csv (verified via clean rerun after the fix, not the pre-fix numbers):

| n_crops | top1 | top3 | MAE |
|---|---|---|---|
| 1 (single 10s crop) | 0.409 | 0.826 | 1.091 |
| 3 | **0.485** | 0.871 | 0.917 |
| 5 | 0.477 | **0.879** | 0.939 |
| 8 | 0.462 | 0.879 | 0.939 |

- **Notable finding**: multi-crop TTA at 10s with n_crops=3 or 5 doesn't just close the gap to the 30s full-clip baseline (top1=0.462, from the segment-length sweep) — it **exceeds it** (0.485 and 0.477 vs 0.462). n_crops=8 comes back down to about parity with 30s (0.462). So there's a sweet spot around 3-5 overlapping 10s crops that beats processing the raw 30s clip as one long segment outright — plausibly because mean-pooling several independently-encoded short segments acts as an implicit ensembling/denoising step that a single long forward pass doesn't get. Genuinely useful, non-obvious result for the report; worth repeating on Dataset B to see if it generalizes.

### Required experiment — multi-excerpt/TTA ablation, Dataset B (10s crops, n_crops in {1,3,5,8})
- results/multicrop_sweep_B/summary_10s.csv:

| n_crops | top1 | top3 |
|---|---|---|
| 1 (single 10s crop) | 0.353 | 0.794 |
| 3 | **0.441** | 0.775 |
| 5 | 0.412 | 0.775 |
| 8 | 0.412 | 0.745 |

- **Does NOT replicate Dataset A's finding**: B's 30s full-clip baseline (top1=0.471, from the segment-length sweep) beats every multi-crop-of-10s configuration tried here (best multi-crop top1 is 0.441 at n_crops=3, still below 0.471). Top-3 at n_crops=1 (0.794) is close to/slightly above 30s's top3 (0.755), but top1 clearly favors the full clip for B. **Conclusion: "multi-crop TTA can beat the full clip" is a real, measured effect on Task A, not a universal property of the MERT embedding space — don't generalize it across tasks.** Worth discussing in the report as a task-dependent finding rather than a general trick.
- Required "5/10/15/30s comparison" and "multiple excerpts" experiments now complete for both tasks.

### Required experiment — t-SNE/UMAP visualization (MERT winning-layer embeddings, train+val, both tasks)
- **Artifact configuration** (src/visualize_embeddings.py): encoder=MERT-v1-330M (`m-a-p/MERT-v1-330M`), layer=4 for A / layer=7 for B (winning layers from the earlier layer-sweep deep-dive), pooling=per-layer temporal mean (from mert_features.py's hook-based extraction). **Embeddings are the raw pooled 1024-dim MERT vectors — NOT standardized or PCA-reduced** (unlike the classifier pipeline, which applies StandardScaler + PCA(128) before logreg; the plot shows the encoder's native geometry, not exactly what the classifier sees post-preprocessing — worth flagging in the report). Data = train+val combined per task (658/908 points for B/A respectively... actually A=1026+132=1158, B=798+102=900), test excluded (labels hidden). t-SNE: perplexity ∈ {15,30,50}, init=pca, random_state=0. UMAP: n_neighbors ∈ {10,15,30}, min_dist=0.1, random_state=0. Color = 6 ground-truth classes per task (legend in each plot). Output: 6 PNGs/task at results/embeddings/mert_v1_330m_{A,B}/{tsne_perp{15,30,50},umap_nn{10,15,30}}.png; representative pair (perplexity=30, n_neighbors=15) pulled to Mac at results/embeddings_preview/{A,B}_{tsne_perp30,umap_nn15}.png.
- **Measured finding, reported as-is**: neither task shows clean visual class separation in either projection — both A (decades) and B (markets) look like one large intermixed blob with classes scattered throughout, no obvious ordinal gradient visible for A's decades despite them having a natural order. A few small local pockets exist at the plot edges (e.g. a small isolated cluster top-left of A's UMAP, top-left of B's UMAP) but they're not cleanly single-class either.
- **This does not contradict the classifier results** — it means the discriminative signal MERT layer4/layer7 carries for era/market lives in directions across the full 1024-dim (or 128-dim-PCA) embedding space that a single 2D nonlinear projection doesn't collapse into visually separable clusters, not that the signal is absent (the frozen-probe classifiers hit top1≈0.46-0.48 vs a 0.167 random baseline on both tasks, a real and substantial lift). Report both facts side by side rather than either overclaiming visible structure or dismissing the classifier because the plot looks messy — linear/near-linear separability in high dimensions and visual separability in a 2D t-SNE/UMAP projection of that same space are genuinely different properties.
- Perplexity/n_neighbors sweep was checked for stability per the plan (not cherry-picked for a nicer-looking single setting) — all 3 settings per method showed the same intermixed pattern, so perplexity=30/n_neighbors=15 are representative, not an outlier choice.

### Required experiment — Task 1 year-regression / hierarchical framing (src/task1_ordinal.py)
- **Config**: reuses the already-cached MERT-v1-330M layer-4 embeddings (train/validation splits of Dataset A only, test excluded), same preprocessing as the winning flat-classifier config — StandardScaler (fit on train) → PCA(128, fit on train). No new extraction.
- **(a) Regression framing**: Ridge regression (GridSearchCV over alpha∈{0.1,1,10,100}, 5-fold StratifiedKFold CV on train, n_jobs=1) predicting decade-index 0-5 as a continuous target, prediction rounded to nearest int (clipped to [0,5]) for top1/accuracy comparison. Result: MAE(raw prediction)=1.060 decades, MAE(rounded)=1.030, **top1(after rounding)=0.273**, QWK=0.605 — **worse than flat 6-way classification on every metric** (flat: top1=0.462, MAE=0.947, QWK=0.621). A linear regression head doesn't capture the era-discriminative structure as well as a direct classification decision boundary here.
- **(b) Hierarchical framing**: coarse 3-way classifier grouping adjacent decade pairs ({1960s,1970s}/{1980s,1990s}/{2000s,2010s}, logreg + GridSearchCV C, 5-fold CV) → predicted-coarse-group-conditional fine 2-way classifier (separate logreg per group, trained only on that group's train rows) picks the specific decade within the group. Result: coarse-stage accuracy=0.689, **final top1=0.432** (slightly below flat's 0.462), **final MAE=0.939** (marginally better than flat's 0.947), **QWK=0.659 (better than flat's 0.621)**.
- **Takeaway for report**: flat 6-way classification remains the best top1 config, but the hierarchical framing makes measurably "smoother" errors — better ordinal agreement (QWK) and MAE despite slightly lower raw accuracy — meaning when hierarchical gets it wrong, it tends to land closer to the true decade (e.g. confusing within-group neighbors) rather than jumping across groups. Regression framing is the weakest of the three approaches on every axis tested; don't recommend it. Worth reporting all three side by side rather than picking only the "best" one, since they answer slightly different questions (top1 vs ordinal-closeness-of-errors).

### Required experiment — Task 2 mixture/vocal/accompaniment stem comparison (src/stems.py + src/task2_stems.py)
- **Config**: source separation = Demucs `htdemucs` (`pip install demucs`, `python -m demucs --two-stems vocals`, batched 32 files/invocation), run on all 1002 Dataset B clips on GPU1 → `dataset/extracted/B_stems/htdemucs/<sample_id>/{vocals,no_vocals}.wav`. Demucs's native output rate is 44.1kHz (not the dataset's 24kHz) — `src/data.py`'s `load_audio_normalized` extended with librosa resampling (44.1kHz→24kHz) + mono-downmix (stems can be stereo) before feeding MERT; this crashed once on the strict-24kHz assert before the fix (see below). Encoder = MERT-v1-330M, layer 7 (Dataset B's winning layer, from the earlier layer-sweep deep-dive), classifier = logistic regression, no PCA (Dataset B's winning classifier config). Evaluated on Dataset B validation split. Mixture arm reuses the already-cached full-mix layer-7 embeddings (no re-extraction); vocals/accompaniment newly extracted to `cache/mert_v1_330m_stem_{vocals,accompaniment}/B/`.
- **Bug hit + fixed**: first attempt crashed immediately (`AssertionError: unexpected sample rate 44100`) because `load_audio_normalized` hard-asserted the dataset's native 24kHz — didn't anticipate Demucs stems having a different native rate. Fixed generally (resample whenever `sr != SAMPLE_RATE`, mono-downmix if stereo), not just special-cased for this experiment.
- results/task2_stems/summary.json:

| input | top1 | top3 |
|---|---|---|
| **mixture** | **0.471** | **0.755** |
| vocals | 0.402 | 0.706 |
| accompaniment | 0.373 | 0.755 |

- **Mixture wins clearly** on top1 (0.471 vs 0.402/0.373) — separating into stems loses information relative to the full mix, both because real cross-stem interaction/mastering cues are discarded and because Demucs separation itself introduces artifacts MERT wasn't trained on. Vocals mildly beats accompaniment on top1 (0.402 vs 0.373) but accompaniment ties mixture on top3 (0.755 both) while vocals is lower there (0.706) — a mixed picture, not a clean "vocals > accompaniment" story.
- **Worth flagging honestly in the report**: Description.md explicitly states release market is *not* defined by language/nationality, yet vocals carries more top1 signal than accompaniment here. This doesn't prove the model is using language as a shortcut (MERT's classifier never sees a language label, and vocal *production style* — not just linguistic content — is a legitimate market-correlated cue), but it's a plausible confound worth naming as a limitation rather than glossing over: some of what discriminates 1980s US/UK/Brazil/Spain/Germany/Italy vocals may correlate with language/accent even though that's not the intended signal. Mixture still being the overall best input is the headline result; this vocal/accompaniment nuance is secondary discussion.

### Baseline 2 gap-fill — SVM classifier (Description.md names logreg/SVM/MLP as the three options; only logreg+MLP had been tested)
- **Bug found+fixed**: `SVC()` had no `max_iter` cap. First attempt ran 43+ minutes without completing even the second of 6 planned configs (A/B × pca∈{None,64,128}), stuck on some (C, kernel) combination inside the grid search that never converged — confirmed genuinely slow-not-stuck via `ps` (sustained ~101% CPU) before concluding it was actually pathological, not just slow. Killed it, added `max_iter=20000` to `SVC()` in src/train_probe.py, relaunched (in tmux this time) — completed all 6 configs quickly with the cap in place.
- results/svm_ablation_summary.json — same layer/classifier grid depth as the earlier logreg/MLP ablation:

| dataset | pca_dim | SVM top1 | SVM top3 | (for reference) best logreg | best MLP |
|---|---|---|---|---|---|
| A (layer 4) | None | 0.439 | 0.848 | 0.455 (no-PCA) | 0.394 (no-PCA) |
| A (layer 4) | 64 | **0.470** | 0.841 | 0.432 (PCA64) | 0.379 (PCA64) |
| A (layer 4) | 128 | 0.432 | 0.833 | **0.462 (PCA128)** | 0.402 (PCA128) |
| B (layer 7) | None | 0.431 | 0.755 | **0.471 (no-PCA)** | 0.441 (no-PCA) |
| B (layer 7) | 64 | 0.431 | **0.784** | 0.382 (PCA64) | 0.422 (PCA64) |
| B (layer 7) | 128 | 0.402 | 0.755 | 0.412 (PCA128) | 0.402 (PCA128) |

- **Reported factually, no presumption**: **SVM+PCA64 is now the new best classifier config for Dataset A** (top1=0.470, beating logreg+PCA128's previous best of 0.462) — SVM wins here. For Dataset B, SVM does NOT beat logreg on top1 (best SVM=0.431 vs logreg's 0.471) — logreg remains the top1 winner for B — but SVM+PCA64 does post the best top3 seen yet for B (0.784, beating logreg-no-PCA's 0.755). Classifier choice is task-dependent, same as every other "which config wins" finding in this project — no universal answer.
- **Updated winning configs**: A = MERT layer4 + PCA64 + **SVM** (top1=0.470, top3=0.841) supersedes the earlier logreg+PCA128 pick. B = MERT layer7 + logreg, no PCA (top1=0.471, top3=0.755) remains best on top1, though SVM+PCA64 (top1=0.431, top3=0.784) is worth citing as the best top3 config for B in the report.

### Baseline 2 gap-fill — fine-tuning vs. frozen (Description.md: "Compare frozen features with fine-tuning if resources allow")
- **Config**: full end-to-end fine-tune of MERT-v1-330M (all params trainable) + a linear classification head on mean-pooled `last_hidden_state`. 10s random crops (train) / fixed crop (val) per clip, audio loudness-normalized + cached in memory for both splits (loaded once, reused every epoch). AdamW lr=2e-5, weight_decay=1e-4, cosine LR schedule, 15 epochs, batch_size=4, `num_workers=0` (keeps the audio cache alive in the main process). `torch.cuda.empty_cache()` between epochs. Code: src/finetune_mert.py.
- **Debugging trail** (three real issues, in order):
  1. `CropDataset` didn't cache audio for the `train` split → every epoch re-read+re-normalized all 1026 files from disk from scratch (~15min/epoch). Fixed: cache always, `num_workers=0` so the cache persists across epochs.
  2. After that fix, training reliably stalled hard at epoch 2+ (sustained high CPU, 0% GPU, zero progress for 17+ minutes) — reproduced twice, including once after a session pause with the box's `uptime` load reading still stuck at a suspicious frozen ~6480 value (though a real timed test command ran normally, suggesting the load figure itself was stale). Root cause turned out to be **`conda run`'s output buffering interacting badly with the long-running job** — not a training-logic bug at all. Fixed by invoking the target env's Python directly (`/home/jtan/miniconda3/envs/pa1_env/bin/python3 -u src/finetune_mert.py ...`, bypassing `conda run` entirely) plus adding fine-grained flush-based progress logging (batch/eval-sample counters, not just per-epoch, and not relying on tqdm which also buffers under this wrapper). Once bypassed, training ran cleanly through all 15 epochs with no further stalls.
  3. (Minor, not a bug) `history.json` is only written once at the very end of training — checking it mid-run while debugging misleadingly looked stale/unchanged; the live log (with the new fine-grained prints) is the correct source of truth during an in-progress run, not `history.json`.
- **Full 15-epoch trend** (results/mert_finetune_A/, `grep` of finetune_A.log):

| epoch | train loss | val top1 | val top3 |
|---|---|---|---|
| 0 | 1.705 | 0.356 | 0.818 |
| 1 | 1.347 | 0.364 | 0.795 |
| 2 | 1.151 | 0.492 | 0.864 |
| 3 | 0.955 | 0.470 | 0.864 |
| 4 | 0.768 | 0.470 | 0.848 |
| 5 | 0.603 | 0.462 | 0.833 |
| **6** | **0.399** | **0.500** | **0.856** |
| 7 | 0.324 | 0.477 | 0.773 |
| 8 | 0.156 | 0.432 | 0.833 |
| 9 | 0.116 | 0.455 | 0.848 |
| 10 | 0.077 | 0.455 | 0.856 |
| 11 | 0.055 | 0.455 | 0.826 |
| 12 | 0.044 | 0.447 | 0.811 |
| 13 | 0.043 | 0.447 | 0.826 |
| 14 | 0.034 | 0.447 | 0.841 |

- **Clear, textbook overfitting**: train loss decreases monotonically and smoothly from 1.705 to 0.034 across all 15 epochs, while val top1 peaks at epoch 6 (0.500) then wobbles down and roughly plateaus around 0.44-0.46 for the remaining epochs — exactly the "keep training until it overfits, then look back for the best checkpoint" pattern from the lecture notes' training tips.
- **Best checkpoint (epoch 6) full metrics**: top1=0.500, top3=0.856, MAE=0.871 decades, acc-within-1-decade=0.803, adjacent-fraction-of-errors=0.606, QWK=0.665.
- **Reported factually, no presumption (this reverses the untested guess in PLAN.md that fine-tuning "risks overfitting" and should be deprioritized)**: **fine-tuning is now the new overall best config for Dataset A**, beating every frozen-probe result tested — SVM+PCA64 (top1=0.470, top3=0.841), logreg+PCA128 (top1=0.462, MAE=0.947, QWK=0.621), and the Short-Chunk CNN baseline (top1=0.447) — across top1, top3, MAE, and QWK simultaneously. Fine-tuning does overfit past epoch 6 as predicted, but the *peak* checkpoint before that overfitting sets in is a clean win over every frozen-feature approach tried. **Updated winning config for A: MERT-v1-330M fine-tuned (epoch-6 checkpoint) — top1=0.500, top3=0.856.**
### Baseline 2 gap-fill — fine-tuning, Dataset B (release market)
- **Config**: identical to Dataset A's fine-tune (full end-to-end MERT-v1-330M fine-tune, linear head, 10s crops, AdamW lr=2e-5, cosine schedule, 15 epochs, batch_size=4), Dataset B / market labels. Launched directly with the fixed invocation (`/home/jtan/miniconda3/envs/pa1_env/bin/python3 -u`, bypassing `conda run`) from the start — ran cleanly through all 15 epochs with no stalls, confirming the Dataset A stall really was the `conda run` buffering issue and not something dataset-specific.
- **Full 15-epoch trend** (finetune_B.log):

| epoch | train loss | val top1 | val top3 |
|---|---|---|---|
| 0 | 1.765 | 0.294 | 0.706 |
| 1 | 1.579 | 0.343 | 0.784 |
| 2 | 1.360 | 0.373 | 0.775 |
| 3 | 1.104 | **0.412** | 0.765 |
| 4 | 0.916 | 0.324 | 0.735 |
| 5 | 0.661 | 0.363 | 0.765 |
| 6 | 0.443 | **0.412** | **0.794** |
| 7 | 0.299 | 0.353 | 0.726 |
| 8 | 0.211 | 0.392 | 0.765 |
| 9 | 0.152 | 0.402 | 0.726 |
| 10 | 0.122 | 0.382 | 0.745 |
| 11 | 0.066 | 0.402 | 0.735 |
| 12 | 0.057 | **0.412** | 0.735 |
| 13 | 0.053 | **0.412** | 0.726 |
| 14 | 0.050 | 0.402 | 0.726 |

- Same overfitting shape as A (train loss falls monotonically 1.765→0.050), but **val top1 never exceeds 0.412** across all 15 epochs (best tied 4 times: epochs 3, 6, 12, 13) — best full metrics: top1=0.412, top3=0.765.
- **Reported factually, opposite result from Dataset A**: fine-tuning does **NOT** beat frozen probing for Dataset B — best fine-tuned top1 (0.412) stays clearly below the frozen logreg-no-PCA baseline (top1=0.471, top3=0.755), though fine-tuned top3 at epoch 6 (0.794) does edge out the frozen probe's top3 (0.755). This is a genuine task-dependent finding, not a disappointment to explain away: A and B have shown opposite results on more than one axis this project (multi-crop TTA helped A, not B; SVM won for A, logreg stayed best for B) — fine-tuning vs. frozen-probing is one more instance of the same pattern. **Winning config for B remains the frozen MERT-v1-330M layer7 + logistic regression (no PCA), top1=0.471/top3=0.755** — fine-tuning was tested, not presumed, and simply didn't win here.
- Baseline 2 (all 3 named classifiers + frozen-vs-fine-tuned comparison) is now fully complete for both tasks.

### Ensembling — prior-semester-inspired, CNN + frozen MERT probe (src/ensemble.py)
- **Motivation**: checked the prior semester's HW1 report (singer classification, different task — GitHub `goog-msft-fb-nflx-nvda-aapl/CommE5070_HW1`, `R13921031_report.html`). Their graded best (86.1% top1/91.3% top3) was a **weighted ensemble of 7 from-scratch models**, not any single model; a frozen speaker-verification encoder alone hit 95.2% (excluded from grading only because that assignment required training from scratch). Structurally different task (singer ID = stable per-artist vocal fingerprint across all their tracks, closer to speaker verification; PA1's decade/market labels are diffuse across hundreds of different artists per class) — a large absolute-accuracy gap vs. that report is expected on those grounds alone, not just a methods gap. But **ensembling itself was untried here** and is cheap to test with already-trained models — no presumption either way, just run it.
- **Config**: probability-weighted average of (1) Short-Chunk CNN (`results/scnn_{A,B}/best.pt`, multicrop-averaged) and (2) the winning frozen-MERT-probe config per task (A: layer4+PCA64+SVM; B: layer7+logreg no-PCA — refit via `train_probe.run()`'s actual GridSearchCV-selected classifier, not a hand-guessed hyperparameter — an earlier draft hardcoded a guessed SVM hyperparameter and silently reproduced the wrong pure-MERT number, 0.417 instead of 0.470, caught by comparing against the already-logged frozen-probe result rather than trusting a plausible-looking number). Predictions aligned by `sample_id` before combining (models don't necessarily iterate the manifest in the same order). Weight sweep over w_cnn ∈ {0, 0.3, 0.5, 0.7, 1.0}, evaluated on validation.
- **A** (results/ensemble_A/cnn_mert_sweep.json): pure MERT (w=0) top1=0.470/top3=0.841; **w=0.5 gives the best top3 seen for any 2-model combo, 0.871** (beats pure MERT's 0.841 and pure CNN's 0.773) while top1 stays flat at 0.470 up to w=0.5, dropping beyond that. **Reported factually**: ensembling helps top3 here, not top1 — and doesn't reach the fine-tuned-MERT single-model result (top1=0.500, top3=0.856, still the overall best for A). A 3-way ensemble adding the fine-tuned checkpoint (once its checkpoint-saving rerun finishes) is the natural next step.
- **B** (results/ensemble_B/cnn_mert_sweep.json): no benefit at any weight — every CNN-MERT blend is ≤ pure MERT's top1=0.470 (best blend 0.461 at w=0.3, declining further as CNN weight increases; CNN alone is a comparatively weak 0.373 for B). **Reported factually**: ensembling does not help B, same A-helps/B-doesn't asymmetry seen throughout this project (multi-crop TTA, SVM vs logreg, fine-tuning vs frozen).
- **Side-effect fix**: `finetune_mert.py` never saved the fine-tuned model's weights (only metrics) — a real gap against Description.md's own requirement that the uploaded checkpoint reproduce submitted predictions, independent of ensembling. Fixed: best-epoch `state_dict()` now saved to `results/mert_finetune_{A,B}/best.pt`. Re-ran both fine-tunes to produce the checkpoints.
- **Run-to-run variance noted**: Dataset A's rerun (with checkpoint saving) peaked at a different epoch than the original run — epoch 3 (top1=0.492) vs. the original's epoch 6 (top1=0.500) — expected since `DataLoader(shuffle=True)` has no fixed seed, so batch order (and therefore the SGD trajectory) differs between runs. Both are genuine, reproducible-in-spirit results; the saved checkpoint corresponds to this rerun's own peak (0.492), not the original run's higher number. Worth flagging as a reproducibility caveat in the report — this fine-tuning setup has real run-to-run variance at this data scale, not a single fixed number.

### Ensembling — 3-way (CNN + frozen MERT probe + fine-tuned MERT), Dataset A
- Extended src/ensemble.py with `finetuned_mert_probs()` (loads the now-saved fine-tune checkpoint via `MERTClassifier` + `get_probs()`) and a 3-way weight sweep (`--mode 3way`): grid over the fine-tuned-model's weight, then splits the remainder between CNN and frozen-MERT-probe.
- results/ensemble_A/3way_sweep.json — **best: ft=0.20, cnn=0.40, mert=0.40 → top1=0.5227, top3=0.8561** (best top3 across all combos tried: ft=0.40/cnn=0.15/mert=0.45 → top3=0.8864).
- **Reported factually**: this genuinely beats every single-model result on A tried this session — the fine-tuned checkpoint alone (this rerun: top1=0.4924), the frozen MERT probe alone (0.4697), the CNN alone (0.4470), and the earlier 2-way CNN+frozen-MERT ensemble ceiling (0.4697/top3=0.8712). **New overall best config for A: 3-way ensemble (ft=0.20/cnn=0.40/mert=0.40), top1=0.5227/top3=0.8561.** Confirms the prior-semester-HW1-inspired hypothesis that ensembling architecturally-diverse already-trained models helps, at least for Task A — measured, not presumed.
### Ensembling — 3-way, Dataset B
- Regenerated B's fine-tune checkpoint (rerun, same pre-fix gap as A) — peak epoch 2, top1=0.4412, top3=0.7745 (again a different peak epoch than any prior run — same run-to-run variance noted for A).
- results/ensemble_B/3way_sweep.json — **best 3-way top1 is the pure frozen-MERT-probe config itself (ft=0/cnn=0/mert=1.0), top1=0.4706** — no weighted combination of any of the three models beats the single frozen probe on top1 for B. Best top3 across all combos tried is 0.7941 (several ties: ft0.20/cnn0.60/mert0.20, ft0.40/cnn0.30/mert0.30, ft0.60/cnn0.20/mert0.20, ft0.80/cnn0.20/mert0.00) — better than pure MERT's top3=0.7549, so ensembling does help B's top3 (consistent with the 2-way finding), just never its top1.
- **Ensembling investigation complete, both tasks. Final answer, reported factually**: ensembling helps A substantially (new overall best: 3-way, top1=0.5227) but does not help B's top1 at all (frozen MERT probe alone remains the winner, top1=0.4706) — though it does modestly help B's top3. This is the same A-benefits/B-doesn't-on-top1 pattern that recurred all project (multi-crop TTA, SVM vs logreg, fine-tuning vs frozen) — worth stating in the report as a genuine, repeated, task-dependent finding rather than one-off noise, since it shows up across five independent experiments now (TTA, classifier choice, fine-tuning, and now ensembling, all favor A over B on the primary top1 metric specifically).

### Final overall-best configs (after baselines + all required experiments + ensembling)
**Superseded 2026-09-18 by the MuQ + AF3-probability-fusion results — see that section below for the current best.** Kept here as the pre-improvement-round baseline for comparison.
| task | config | top1 | top3 |
|---|---|---|---|
| A (decade) | 3-way ensemble: fine-tuned MERT (0.20) + Short-Chunk CNN (0.40) + frozen MERT-SVM probe (0.40) | 0.523 | 0.856 |
| B (market) | frozen MERT-v1-330M layer7 + logistic regression, no PCA | 0.471 | 0.755 |

### CURRENT overall-best configs (updated 2026-09-18, after round-3 OOF-refit — see full sections below)
| task | config | top1 | top3 | statistically confirmed? |
|---|---|---|---|---|
| A (decade) | MuQ(layer1,SVM,noPCA) probe + Audio Flamingo 3 `direct`-prompt label probs | **0.538** (OOF-refit, NLL-optimal; original validation-swept point estimate was 0.553) | 0.864 | **Partially — largely survives leakage-free refit**, unlike Task 2. OOF-refit (0.538) is close to the original validation-swept estimate (0.553), a much smaller drop than Task 2 saw. Significantly beats AF3-alone (p=0.0055); not significantly different from the probe alone or the 3-way ensemble (p=0.29/overlapping CIs) — real but modest, not a large confirmed jump. Report 0.538 (OOF-refit) as the more defensible number if a single figure is needed, ahead of the un-refit 0.553. |
| B (market) | Audio Flamingo 3 zero-shot alone (`direct` prompt), or equivalently an OOF-refit fusion | **≈0.588** | 0.765-0.872 | **This is now the honest number.** The earlier-reported 0.657 (validation-set-swept fusion weight) was **not reproduced** by a leakage-free out-of-fold refit — both NLL-optimal (0.569) and accuracy-optimal (0.588) OOF-fit fusion weights land at or barely above AF3-alone, not near 0.657. Conclusion: our fusion approach has confirmed evidence of beating our own trained probes (p<0.01) but **no confirmed evidence of beating AF3's own zero-shot score** — the 0.657 figure was validation-overfitting, not a real effect. See "Round 3" section below for the full refit results. |

**Why Task 1 held up better than Task 2**: AF3 performs poorly on Task 1 alone (0.379), so every fitting procedure (validation sweep or OOF refit, NLL- or accuracy-optimal) converges to a broadly similar, probe-dominated weight (alpha=0.55-0.84) — leaving little room for a small validation set to overfit the weight choice. Task 2's fusion leans much more heavily and evenly on both AF3 and the probe, which is exactly the regime where validation-set weight-fitting is most exploitable by noise. Full details in the "Round 3" section below.

Full bootstrap/McNemar tables in the "Statistical significance" section below (`src/significance.py`, `results/significance_{A,B}.json`) — read before citing either number in the report.

### Checkpoint (2026-09-16)
Baselines done for both tasks (Short-Chunk CNN + MERT frozen probe). Required experiments now mostly complete: segment-length sweep ✓, multi-excerpt/TTA ✓, Task 1 year-regression/hierarchical framing ✓, t-SNE/UMAP ✓, Task 2 stem comparison ✓. Remaining substantial items: ALM track (Qwen2-Audio + Audio Flamingo 3, required) and MusicFM as a 3rd encoder (not required, A/A+ differentiator). See TODO.md for full state.

### Required experiment — Audio Language Model track, Qwen2-Audio-7B-Instruct (src/alm_infer.py)
- **Critical bug caught during smoke-testing, before trusting any results**: the first end-to-end test produced a clean, plausible-looking output ("1970s") with NO error — but a stray warning ("Keyword argument `audios` is not a valid argument for this processor and will be ignored") turned out to mean the audio was silently never passed to the model at all (wrong kwarg name — Qwen2AudioProcessor expects `audio=`, singular, not `audios=`). Caught by inspecting the processor's actual call signature rather than trusting a clean-looking single output. Fixed the kwarg name, re-ran the same test — and it then correctly *errored* demanding 16kHz input (Whisper's feature extractor, Qwen2-Audio's audio tower, requires 16kHz; our dataset is natively 24kHz) — this error, once audio was genuinely flowing through, is itself confirmation the original silent-no-audio bug was real (an all-text-no-audio call would never have hit this validation). Added librosa 24kHz→16kHz resampling before the processor call. **Verification**: re-ran on 3 validation clips with known labels (1960s/1970s/1970s) — model correctly predicted all 3 exactly. Only trusted the pipeline after this positive-control check, not just absence-of-error.
- This is a strong argument for always including a small known-label sanity check before trusting any ALM/generative-eval pipeline's output, even when it runs without crashing — a silently-ignored kwarg produced a perfectly well-formed, plausible single-token answer with zero indication anything was wrong.
- **Second bug, caught after a full "successful" run**: the first complete pass used a one-hot pseudo-probability built from a single free-form generated answer as the "probs" array for top1/top3 scoring. Traced through `np.argsort` on a one-hot vector and confirmed this is degenerate for top-3: ties among the zero-probability classes break by array index, so top-3 always credited the predicted class plus whichever *lowest-index* classes happened to be tied at zero (for Dataset A, that systematically means 1960s/1970s get free credit whenever the true label lands there, regardless of what the model actually thinks). **Fixed** by adding teacher-forced label log-prob scoring (`score_candidates` in src/alm_infer.py): for each of the 6 candidate labels, concatenate the prompt + candidate tokens and read off the model's own token log-probabilities (single forward pass per candidate, no autoregressive generation needed) — this gives a genuine ranked distribution over all 6 classes. Verified on 3 known-label clips: sensible calibrated scores (e.g. close scores between adjacent, easily-confusable decades) before trusting it. Free-form generation is still used, separately, purely to measure the invalid-output rate (how often the model's natural-language answer fails to parse) — that part of the original design was fine.
- **Third bug, caught by a suspicious result**: with the scoring fix, `direct` scored top1=0.326/QWK=0.534 but `cot_then_answer` scored top1=0.227/QWK=0.109 — a gap large enough to be suspicious rather than just "the prompts differ." Root cause: `score_candidates` always scored a *bare* label immediately after the prompt, but `cot_then_answer` explicitly asks the model to reason first and answer in the format `Answer: <label>` — scoring a bare label right after that prompt asks the model something it was never asked to do (blurt the label with no reasoning), which isn't a fair test of what the prompt is actually eliciting. **Fixed** by giving each prompt its own `answer_prefix` (`""` for `direct`, `"Answer: "` for `cot_then_answer`) so teacher-forced scoring matches each prompt's own declared output format.
- **Config** (final): encoder=Qwen2-Audio-7B-Instruct (`Qwen/Qwen2-Audio-7B-Instruct`, Apache 2.0), audio resampled 24kHz→16kHz (Whisper audio-tower requirement) before the processor call, top1/top3/confusion computed from teacher-forced label log-prob scoring (format-matched per prompt), invalid_output_rate computed separately from free-form generation + regex/alias parsing (`parse_label`), evaluated on Dataset A validation split (132 clips), 2 prompt designs.
- **Final results** (results/alm/qwen2audio_A/{direct,cot_then_answer}_metrics.json), now a fair comparison:

| prompt | top1 | top3 | MAE (decades) | QWK | invalid rate |
|---|---|---|---|---|---|
| direct | 0.326 | 0.659 | 1.220 | 0.534 | 0.008 |
| cot_then_answer | 0.333 | 0.652 | 1.265 | 0.512 | 0.008 |

- **The two prompts perform essentially identically** once scoring is fair (top1 0.326 vs 0.333, top3 0.659 vs 0.652, QWK 0.534 vs 0.512) — confirms the earlier huge gap (0.326/0.534 vs 0.227/0.109) really was a pure scoring-methodology artifact, not a genuine prompt-design effect. Report this comparison as "no meaningful difference between the two prompt designs" — a legitimate, if less dramatic, required-experiment finding (Description.md asks to compare ≥2 prompt designs; "they don't differ much once measured fairly" is itself the answer here).
- **Vs. the purpose-built baselines**: zero-shot Qwen2-Audio (top1≈0.33, top3≈0.65) is clearly below both Short-Chunk CNN (top1=0.447) and the MERT frozen probe (top1=0.462, top3=0.871) — expected, since the ALM has zero task-specific training/probing on this exact 6-way decade distinction, unlike the other two. Still well above the 0.167 random baseline, showing genuine (if limited) audio-grounded era knowledge from a generalist 7-8B model with no fine-tuning.
- **Meta-lesson**: three real bugs surfaced in sequence in this one pipeline, each only caught by active verification (reading warnings, tracing argsort behavior, noticing a suspicious result gap) rather than trusting "it ran without crashing." See [[reference_alm_pipeline_verification]].

### ALM track — Qwen2-Audio-7B-Instruct, Dataset B (release market)
- **Config**: identical pipeline/fixes as Dataset A above, evaluated on Dataset B validation split (102 clips), labels = US/UK/Brazil/Spain/Germany/Italy (`MARKET_ALIASES` in src/alm_infer.py).
- results/alm/qwen2audio_B/{direct,cot_then_answer}_metrics.json:

| prompt | top1 | top3 | invalid rate |
|---|---|---|---|
| direct | 0.324 | 0.529 | **0.0%** |
| cot_then_answer | **0.441** | **0.794** | **27.5%** |

- **This time the gap is real, not a scoring artifact** (already fixed for A) — checked by hand: inspected the 28/102 invalid `cot_then_answer` free-form outputs and found the model frequently answers *"Answer: International"* or *"Answer: rock"* instead of one of the 6 canonical market labels — i.e. it sometimes drifts off-task entirely (answering with a genre, or a non-label cop-out word) despite the prompt explicitly listing the 6 choices and demanding the `Answer: <label>` format. The terser `direct` prompt has zero invalid outputs.
- **Trade-off worth reporting as-is**: `cot_then_answer`'s teacher-forced label *scores* are meaningfully better (top1 +0.12, top3 +0.27 over `direct`) — when forced to choose among just the 6 valid labels, its underlying preferences are more accurate — but its unconstrained free-form generation is far less reliable in practice (27.5% invalid) because reasoning-style prompts give the model room to drift toward genre-labeling or vague non-answers instead of the requested closed-set choice. `direct`'s terser framing sacrifices a little scored accuracy for much more reliable output-format compliance. Report both numbers together; neither prompt is unconditionally "better" — it depends whether the deployment constrains outputs to the closed set (as our label-scoring does) or has to parse free text (as a real invalid-output-rate metric measures).
- **Vs. purpose-built baselines**: even `cot_then_answer`'s top1=0.441 is close to but still below the MERT frozen probe's top1=0.471 (top3 0.794 vs 0.755 — ALM actually wins on top3 here); `direct`'s top1=0.324 is below the Short-Chunk CNN baseline (0.373) too. So the zero-shot ALM is competitive-but-not-better on B, closer than it was on A (where it was clearly behind both baselines) — plausibly because Task 2's market distinctions may lean more on cues a generalist audio-language model already has some exposure to (production era, language of promotional/liner metadata seen in pretraining, etc.) than the finer-grained decade distinctions in Task 1.
- Qwen2-Audio finished for both tasks. Audio Flamingo 3 not yet run — do not call the overall ALM track done until it (and Dataset B for it) are actually tested.

### ALM track — Audio Flamingo 3, Dataset A (release decade)
- **Config**: `nvidia/audio-flamingo-3-hf` via `AudioFlamingo3ForConditionalGeneration` + `AutoProcessor` (verified real class name/model id via the HF model card before writing any code, and inspected the processor's actual `__call__` signature directly rather than assuming — same discipline as the Qwen2-Audio fixes). **Bug hit + fixed**: loading with `torch_dtype=torch.bfloat16` (as done for Qwen2-Audio) crashed with `RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same` inside the audio tower's `conv1` layer — the processor's audio features come out float32 and aren't auto-cast. Fixed by loading the model in float32 instead (plenty of VRAM headroom on the H200s). Verified on the same 3 known-label clips as Qwen2-Audio before trusting it (all 3 correct, sensible calibrated scores) — same shared `generate`/`score_candidates` pipeline (16kHz resampling, teacher-forced label log-prob scoring for top1/top3/confusion, separate free-form generation + regex/alias parsing for invalid-output rate) as Qwen2-Audio, reused via a common `_build_generate_and_score` helper. Dataset A validation split (132 clips), both prompt designs.
- results/alm/audioflamingo3_A/{direct,cot_then_answer}_metrics.json:

| model | prompt | top1 | top3 | QWK | invalid rate |
|---|---|---|---|---|---|
| Qwen2-Audio | direct | 0.326 | 0.659 | 0.534 | 0.8% |
| Qwen2-Audio | cot_then_answer | 0.333 | 0.652 | 0.512 | 0.8% |
| Audio Flamingo 3 | direct | **0.379** | **0.697** | 0.412 | **0.0%** |
| Audio Flamingo 3 | cot_then_answer | 0.333 | 0.667 | 0.325 | 0.0% |
| MERT frozen probe (baseline) | — | 0.462 | 0.871 | 0.621 | — |

- **Reported factually, no presumption**: Audio Flamingo 3's `direct` prompt has the best top1/top3 of the four ALM configs (0.379/0.697) and zero invalid outputs on both prompts, but its QWK is lower than Qwen2-Audio's on both prompts (0.412/0.325 vs 0.534/0.512) — higher raw accuracy, less ordinally-consistent errors. Neither model is a clean winner across every metric; both remain below the MERT frozen probe on top1/top3/QWK.
- **Striking finding, verified in both confusion matrices, not just one**: Audio Flamingo 3 **never predicts "2010s" as its top choice for any of the 132 validation clips, in either prompt** — the predicted-2010s column sums to exactly 0 in both `direct` and `cot_then_answer`'s confusion matrices, despite 22 true 2010s samples in the validation set. This is a real, measured, reproducible bias (checked directly, not inferred), not a bug in this pipeline (Qwen2-Audio's confusion matrices don't show this pattern) — worth reporting as-is in the write-up without speculating on cause beyond noting it's consistent with either a pretraining-data-recency bias or an underrepresented/discouraged label in AF3's own training, since we can't inspect AF3's training data to confirm either way.

### ALM track — Audio Flamingo 3, Dataset B (release market)
- **Config**: identical pipeline as Dataset A above, Dataset B validation split (102 clips), market labels.
- results/alm/audioflamingo3_B/{direct,cot_then_answer}_metrics.json:

| prompt | top1 | top3 | invalid rate |
|---|---|---|---|
| direct | **0.588** | 0.765 | 0.0% |
| cot_then_answer | **0.559** | **0.775** | 0.0% |

- **Both AF3 prompts beat the MERT frozen-probe baseline for B (top1=0.4706, top3=0.7549) on top1** — the first ALM configs, out of all 8 runs (2 models × 2 datasets × 2 prompts), to beat a purpose-built baseline on any primary metric. `cot_then_answer` also edges out the MERT probe on top3 (0.775 vs 0.755).
- **No zero-prediction class here** (unlike Dataset A's complete "2010s" absence) — but both prompts show a strong skew toward predicting **US** (50/102 for `direct`, 56/102 for `cot_then_answer` — roughly half of all predictions), with Germany barely predicted at all (2/102 both prompts). Reported as observed; not yet clear whether this reflects genuine US-production-style distinctiveness the model has learned, a generic "default to the most globally-represented market" prior from pretraining, or something else — flagged as an open question rather than resolved.

### ALM track — full comparison (both models, both datasets, both prompts, vs. purpose-built baselines)

| dataset | config | top1 | top3 | vs. MERT probe top1 |
|---|---|---|---|---|
| A (decade) | MERT probe (baseline) | 0.462 | 0.871 | — |
| A | Short-Chunk CNN (baseline) | 0.447 | ~0.78 | — |
| A | Qwen2-Audio / direct | 0.326 | 0.659 | below |
| A | Qwen2-Audio / cot | 0.333 | 0.652 | below |
| A | Audio Flamingo 3 / direct | 0.379 | 0.697 | below |
| A | Audio Flamingo 3 / cot | 0.333 | 0.667 | below |
| B (market) | MERT probe (baseline) | 0.471 | 0.755 | — |
| B | Short-Chunk CNN (baseline) | 0.373 | ~0.72 | — |
| B | Qwen2-Audio / direct | 0.324 | 0.529 | below |
| B | Qwen2-Audio / cot | 0.441 | 0.794 | below (top3 above) |
| B | Audio Flamingo 3 / direct | **0.588** | 0.765 | **above** |
| B | Audio Flamingo 3 / cot | **0.559** | **0.775** | **above** |

- **Full ALM track (Description.md's required experiment) is now complete**: both named models (Qwen2-Audio-7B-Instruct, Audio Flamingo 3), both tasks, ≥2 prompt designs each, invalid-output-rate reported throughout — 8 evaluation runs total, all logged above with full config, bugs found/fixed, and factual (non-presumptive) comparisons.
- **Headline pattern, reported factually**: on Task 1 (decade), all 4 ALM configs sit below both purpose-built baselines. On Task 2 (market), Audio Flamingo 3 (both prompts) beats the MERT frozen probe on top1, while Qwen2-Audio stays below on top1 but its `cot_then_answer` beats the MERT probe on top3. This is not a uniform "ALMs are worse/better" story — it's task- and model-specific, consistent with the standing "no false ceilings, architecture-specific findings" principle: don't generalize from one task's result to the other.

---

## Session note (2026-09-17): file ordering

This log grew via appended edits at different anchor points, so entries below this line aren't in strict chronological order relative to everything above (the ALM track content above was actually finalized *before* the Baseline-2-fine-tuning/SVM/ensembling sections earlier in this file, despite appearing after them here). Each section is self-contained and dated, so this is a navigation note, not a correctness issue. **The authoritative current-best-config summary is the "Final overall-best configs" table** (search this file for that heading) — always check that table over any earlier per-section number if there's ever a discrepancy.

### TA outreach materials
- Wrote `TA_QUESTIONS_MATERIALS.md` (project root) — a briefing document (best configs for both tasks with exact HF model IDs/hyperparameters, full list of what's been tried, known CV/artist-ID limitation) plus a task instruction for an external Claude session to draft a short, polite question set for the TA covering: expected Top-1/Top-3 range on validation, further-improvement suggestions, CV strategy given no artist IDs, fine-tuning scope (full vs. partial/LoRA — ours showed real overfitting + run-to-run variance), whether an ensembled result is an acceptable primary submission, which split the ALM experiment should target, and the Discogs-editorial label-noise caveat. Output format requested: a self-contained HTML snippet (clean color scheme, no external deps) suitable for pasting directly into the course forum's rich-text editor.

### Reproducibility backup
- Per Description.md's own requirement ("uploaded code and checkpoint must reproduce the submitted predictions") and general good practice given how much the code has evolved (many bugs found/fixed mid-session), snapshotted the current `src/` — the exact versions of every script that produced the final best-config results above — to `backups/2026-09-17_src_snapshot/` on Mac. See that directory's own README for what's included.

---

## 2026-09-18 — Improvement round, based on 4-way deep research (see IMPROVEMENT_FINDINGS.md for the full reconciled synthesis)

### MuQ (OpenMuQ/MuQ-large-msd-iter) — new encoder, frozen probe
- **Config**: `pip install muq`, `MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")`, 24kHz native (no resampling), 12-layer Wav2Vec2-Conformer architecture, hidden_size=1024. Code: `src/muq_features.py`.
- **Two real library bugs found and fixed, same class as the earlier MERT issue** (transformers-version drift breaking third-party model wrappers pinned to an old transformers API):
  1. MuQ's checkpoint bakes in a stale EasyDict config (`transformers_version: '4.19.0.dev0'`) missing `_attn_implementation`, which current transformers (5.16.1) requires on the very first forward call (`AttributeError`). Fixed by patching `muq.model.conformer.config._attn_implementation = "eager"` after loading.
  2. Even after that fix, `output_hidden_states=True` silently fails downstream — MuQ's own wrapper code crashes with `KeyError: 'hidden_states'` trying to read the conformer's output dict, *after* the conformer's actual forward pass (and our layer hooks) already completed successfully. Fixed with the same forward-hook approach used for MERT (hook each of the 12 conformer layers directly, bypass `output_hidden_states` entirely), plus catching that specific downstream `KeyError` since it happens after the data we need is already captured.
  3. Verification: ran a 132-clip validation-split extraction successfully (~27 clips/sec, faster than MERT's ~12/sec) before trusting it for full extraction — caught both bugs this way rather than assuming the library "just works" because the README says so.
- Full extraction: 1290/1290 (A) and 1002/1002 (B) clips cached to `cache/muq_large_msd/{A,B}/`.
- **Layer sweep (all 12 layers + mean_all + concat_last4, logreg)** — results/muq_layer_sweep_{A,B}/summary.csv:
  - **A: layer 1 best, top1=0.500/top3=0.841** — already beats MERT's best frozen-probe result (SVM+PCA64, top1=0.470) at the *default logreg, no ablation* stage.
  - **B: layer 2 best, top1=0.500/top3=0.853** — a large jump over MERT's best frozen probe for B (logreg no-PCA, top1=0.471/top3=0.755): **+2.9pt top1, +9.8pt top3**, on exactly the task that resisted every other technique tried this project (fine-tuning, ensembling, SVM, multi-crop TTA, more MERT ablation).
- **Classifier/PCA ablation on the winning layers** — results/muq_ablation_summary.json:
  - **A best: layer1 + SVM, no PCA → top1=0.5076, top3=0.8712** — beats MERT's frozen-probe best (0.470) *and* slightly beats the single fine-tuned-MERT checkpoint's top1 (0.500), while being a frozen probe (far cheaper than fine-tuning, no overfitting risk).
  - **B best: layer2 + logreg, no PCA → top1=0.500, top3=0.853** — same as the raw layer-sweep number (no PCA/other-classifier combo beat plain logreg here); still clearly the new best *trained-pipeline* result for B, though Audio Flamingo 3's zero-shot 58.8% remains higher.
  - MLP underperforms logreg/SVM at this data scale for MuQ too, consistent with every other encoder tested this project.
- **Reported factually, no presumption**: MuQ is a straightforward win over MERT as a frozen-probe encoder for *both* tasks, and specifically closes much of Task B's gap — direct, measured support for the deep-research hypothesis that a different pretraining objective (Mel-RVQ vs. MERT's EnCodec+CQT teachers) captures different, and here more useful, information. **New standalone-model best for B: MuQ layer2+logreg, top1=0.500/top3=0.853** (up from MERT's 0.471/0.755). A's ensemble (0.523) still leads overall for A, but MuQ alone (0.508) already beats every non-ensembled result tried for A.

### AF3-probability late fusion (`src/af3_stack.py`) — weighted-sum sweep, reusing already-computed teacher-forced label scores
- **Config**: for each dataset, each frozen-probe backbone's `predict_proba` on validation is combined with Audio Flamingo 3's per-sample teacher-forced label log-prob scores (already computed by `alm_infer.py`, re-softmaxed here), via `probs = w*af3_probs + (1-w)*backbone_probs`, `w` swept over {0.0, 0.1, ..., 1.0}. No new AF3 inference run needed — reused `results/alm/audioflamingo3_{A,B}/{direct,cot_then_answer}_raw.json`. Backbones tried: MERT frozen (A: layer4+SVM+PCA64; B: layer7+logreg), MuQ frozen (A: layer1+SVM; B: layer2+logreg), and for A only, fine-tuned MERT (epoch-6 checkpoint) — all backbone configs identical to their own already-logged bests, refit via `train_probe.run()` to get `predict_proba` (not re-searched). Both AF3 prompts (`direct`, `cot_then_answer`) tried against every backbone. Full per-weight tables: `results/af3_stack/{A,B}_{backbone}_af3{prompt}.json`.
- **Task B result — new overall best by a wide margin**: **MuQ(layer2,logreg) + AF3(cot_then_answer) probs, w_af3=0.4 → top1=0.6569, top3=0.8824.** (Runner-up: MuQ + AF3-direct, w_af3=0.2 → top1=0.6471/top3=0.8627.) This beats every single-model result (MuQ alone 0.500, AF3 alone 0.588) and every earlier fusion attempt this project (MERT-probe ensembling never beat pure MERT-probe on B). MERT-probe + AF3 also improves over MERT-alone (best 0.5882 at w_af3=0.6-1.0 with `direct`, though note w=1.0 for MERT+AF3-direct is just AF3 alone) but tops out well below the MuQ-backbone combination — MuQ's higher-quality base probabilities apparently combine better with AF3's signal, not just "any probe + AF3" being equivalent.
- **Task A result — new overall best, modest gain**: **MuQ(layer1,svm) + AF3(direct) probs, w_af3=0.5 → top1=0.5530, top3=0.8485** — beats the prior overall-A-best (3-way ensemble, 0.5227) by +3pt top1, though top3 is slightly below the ensemble's 0.8561. Fine-tuned-MERT+AF3 also improves over fine-tuned-MERT-alone (0.5076 at w_af3=0.5-0.6, vs 0.4924 alone) but doesn't beat the MuQ-backbone combination.
- **Important caveat, stated plainly per measured-only-reporting rule**: these are the best points on a fine weight grid *swept and selected on the validation set itself* (n=132 for A, n=102 for B) — the same set used to report every other config's "best." This carries real risk of overfitting the weight choice to validation-set noise (the IMPROVEMENT_FINDINGS.md statistical caveat applies directly here, arguably more than anywhere else in this project, since we're now selecting 1 of 11 grid points instead of just picking a fixed config). Reported as measured, not as a guaranteed test-set result; a wide range of weights around the optimum (e.g. B: w=0.2-0.6 with either AF3 prompt) all clear 0.55-0.65, so the improvement isn't a knife-edge single point, which is at least weak evidence it's a real effect and not pure noise — but this should be validated (bootstrap CI per the still-open statistical-significance queue item) before being treated as a confident final number.
- **Updated overall-best configs**: **A: MuQ(layer1,svm) + AF3-direct (w=0.5), top1=0.5530/top3=0.8485.** **B: MuQ(layer2,logreg) + AF3-cot_then_answer (w=0.4), top1=0.6569/top3=0.8824.**

### Statistical significance (bootstrap CI + McNemar), `src/significance.py` — n_boot=10000, seeded RNG
Directly tests the caveat flagged above: was the fusion result a real improvement or just the weight-sweep finding a noise peak on a small validation set? 10000-resample paired bootstrap (same resampled indices used for both configs each draw) + exact McNemar test on the discordant-pairs count, both computed against the fused-best config, for every backbone/AF3-alone config it was built from.

**Task B (n=102) — the improvement over frozen probes is real, but the improvement over AF3 alone is not established at this sample size:**
| comparison | diff (fused − other) | 95% CI | McNemar p |
|---|---|---|---|
| fused (0.657) vs muq_alone (0.500) | +0.156 | [+0.068, +0.255] | **0.0025** |
| fused (0.657) vs mert_alone (0.471) | +0.186 | [+0.069, +0.304] | **0.0034** |
| fused (0.657) vs af3_cot_alone (0.559) | +0.098 | [+0.010, +0.186] | 0.0525 (borderline) |
| fused (0.657) vs af3_direct_alone (0.588) | +0.068 | [−0.020, +0.167] | 0.2100 (not significant) |

Honest read: fusing MuQ with AF3 clearly and significantly beats either frozen probe alone (p<0.01) — the fusion mechanism itself is doing real work, not noise. But the fused config is *not* statistically distinguishable from Audio Flamingo 3's own zero-shot score alone (p=0.21) at n=102 — the point estimate is higher (0.657 vs 0.588) but the CI on that specific gap includes zero. **Correct framing for the report: "combining our own MuQ probe with AF3's zero-shot signal significantly outperforms either probe alone, and numerically exceeds AF3's zero-shot score, though the AF3-alone comparison specifically isn't resolved at this validation-set size."** Not claiming a confirmed win over AF3 alone; claiming a confirmed win over our own prior best (MERT/MuQ probes).

**Task A (n=132) — the "new best" claim is NOT statistically supported; treat as noise until shown otherwise:**
| comparison | diff (fused − other) | 95% CI | McNemar p |
|---|---|---|---|
| fused (0.553) vs muq_alone (0.508) | +0.045 | [0.000, +0.091] | 0.1094 |
| fused (0.553) vs mert_finetuned_alone (0.492) | +0.060 | [−0.030, +0.152] | 0.2430 |
| fused (0.553) vs old_best_3way_ensemble (0.523) | +0.030 | [−0.053, +0.114] | 0.5966 |
| fused (0.553) vs af3_direct_alone (0.379) | +0.174 | [+0.076, +0.273] | **0.0018** |

Honest read: none of A's top 4 configs (fused/MuQ-alone/fine-tuned-MERT/3-way-ensemble, spanning 0.492-0.553 top1) are statistically distinguishable from each other at n=132 -- every CI against the fused config includes zero except the AF3-direct-alone comparison (AF3 alone is genuinely worse at scoring Task A, confirming the earlier ALM-track finding that AF3 underperforms on decade classification specifically). **Do not report the 0.553 fused number as "the new best for A" without this caveat** — per measured-only-reporting: it is the best point estimate found, but the evidence that it's a real improvement over the already-known-good configs (MuQ alone, the ensemble) is weak. Fair framing for the report: "several Task-1 configs cluster at 0.49-0.55 top1 with overlapping confidence intervals; we report the highest point estimate but flag that the small validation set (n=132) cannot statistically separate these configs from each other."

**Queue item closed** (statistical significance task from IMPROVEMENT_FINDINGS.md). Full JSON: `results/significance_{A,B}.json`.

### Confusion-matrix / pairwise-AUC diagnostic, `src/confusion_diagnostic.py` — analysis only, no training
Run on the current best config per task (fused MuQ+AF3) and on MuQ-alone for comparison, both tasks. Full confusion matrices, per-class P/R/F1, and all 15 pairwise one-vs-one AUCs (renormalized over just the two candidate classes) in `results/confusion_diagnostic/{A,B}_{tag}.json`.

**Task B — confirms the market-clustering hypothesis from the earlier stem-comparison section, in both configs (fusion and MuQ-alone)**: the worst-separated pairs are consistently US/UK (AUC 0.664-0.758) and anything-vs-Germany (Germany is the single worst class by F1 in both configs, 0.40-0.41) — all Anglo/Euro markets that plausibly share production conventions, matching the deep-research literature's "release country is editorial/commercial, not acoustic" explanation. Brazil is the cleanest class by far (near-perfect AUC 0.97-1.0 vs. US/UK) — the one market with a genuinely distinct language/production tradition in this label set, consistent with geography-from-audio literature (cited in IMPROVEMENT_FINDINGS.md) finding *some* signal exists, just not evenly distributed across all class pairs. Fusion improves Germany's recall (0.353 vs 0.412 alone — actually roughly flat) and clearly improves US's recall (0.765 vs 0.412) and Spain's precision (0.800 vs 0.500), i.e. the AF3 signal it's adding doesn't fix every class evenly, it disproportionately helps some (US, Spain) more than others (Germany stays the hard class in both configs).

**Task A — clean ordinal/adjacent-decade confusion structure, as expected and already suggested by the earlier hierarchical-framing experiment**: pairwise AUC is close to monotonic in decade-distance — adjacent decades are hardest (2000s vs 2010s: 0.688-0.725; 1990s vs 2000s: 0.655-0.709) and far-apart decades are trivially separated (1960s vs 2010s: 0.961-0.979). 2000s is the single worst class by F1 (0.298-0.333 in both configs) — it's the "middle" class getting pulled toward both its neighbors, not confused with anything acoustically distant. This is a genuinely different error structure from Task B's (topical/market-cluster confusion vs. Task A's purely ordinal-adjacency confusion), consistent with the decade-has-real-continuous-acoustic-correlates vs. market-is-editorial-category explanation reconciled in IMPROVEMENT_FINDINGS.md section 2.

**Queue item closed** (confusion-matrix diagnostic from IMPROVEMENT_FINDINGS.md).

---

## 2026-09-18 (later) — Round 3 improvement queue, from a second 4-way deep research pass (see IMPROVEMENT_FINDINGS_ROUND3.md for full reconciliation)

### Task 2: OOF-fitted, temperature-calibrated fusion — `src/af3_stack_oof.py`
Round 3's top cross-source-convergent recommendation (3-4 of 4 independent deep-research sources): our original fusion weight (w_af3=0.4, top1=0.657) was picked by sweeping the **102-sample validation set directly** — the same set used to report the result. This experiment refits the fusion using **out-of-fold predictions on the 798-sample training set** instead (5-fold CV for the probe, so every training row's "prediction" comes from a fold that never saw it; AF3's own zero-shot scores need no OOF machinery since AF3 is never fit to our data, but its probabilities are temperature-calibrated via a scalar T fit on the training set via NLL, since zero-shot ALM scores are known to be overconfident). Both `T` and the fusion weight `alpha` are frozen after being learned purely from the training set, then applied **once** to validation — no further validation-set tuning.

- **AF3 temperature**: T=0.577 (mild sharpening, not a large recalibration).
- **NLL-optimal fusion weight** (minimize log-loss on OOF-train): alpha(probe)=0.268, i.e. 73% weight on AF3 — **validation result: top1=0.5686/top3=0.8824**. Barely above AF3-alone (0.5588) — paired bootstrap CI is [+0.000, +0.029], McNemar p=1.0 (n01=1, n10=0) — **not meaningfully different from AF3 alone at all**.
- **Accuracy-optimal fusion weight** (maximize accuracy on OOF-train instead of minimizing NLL, as a secondary check since the two objectives can disagree): alpha(probe)=0.55, roughly balanced — **validation result: top1=0.5882/top3=0.8725** — **exactly ties AF3-alone's top1 (0.5882)**, only top3 is better (0.8725 vs AF3-alone's 0.7647).
- **Honest conclusion, stated plainly**: neither leakage-free (OOF-fit) version of the fusion reproduces anywhere near the original 0.657 point estimate — both land at or barely above AF3's own zero-shot score. This strongly suggests **the original 0.657 result was substantially inflated by fitting the fusion weight on the same 102-sample validation set used to report it** — exactly the overfitting risk already flagged (but not yet confirmed) when that result was first logged. The already-run significance test (fused vs AF3-alone, p=0.21, not confirmed) and this new OOF-refit result are now two independent pieces of evidence pointing the same direction: **our fusion approach does not have confirmed evidence of beating Audio Flamingo 3's own zero-shot score on Task 2**, only of beating our own trained probes alone (which remains true and confirmed, p<0.01).
- **Revised honest framing for Task 2**: the best-supported, non-overfit number for Task 2 is **top1≈0.588 (matching AF3's own zero-shot score, whether reported as AF3 alone or as an OOF-fit fusion that converges to nearly the same place)** — not 0.657. The 0.657 figure should be reported, if at all, explicitly labeled as "validation-set-optimized point estimate, not reproduced by a leakage-free refit" rather than as our headline Task 2 number.
- Full details: `results/af3_stack_oof/B_muq_large_msd_af3cot_then_answer.json`.

**Updated CURRENT overall-best config for Task 2**: given this finding, the defensible reported number for B is now **top1≈0.588** (Audio Flamingo 3 zero-shot, `direct` prompt, or equivalently the OOF-refit fusion) rather than the previously-reported 0.657 — see the updated table below.

### Task 1: same OOF-refit check, run as a follow-up given what just happened to Task 2
Task 1's 0.553 point estimate was selected the identical way (sweeping the validation set directly), so it needed the same scrutiny. Ran `src/af3_stack_oof.py --dataset A` with the **correct** probe config this time (layer=1, classifier=SVM, matching `af3_stack.py`'s `FROZEN_CONFIGS["A"]` — an initial run mistakenly reused Task B's layer2/logreg config, caught before trusting the result, and rerun).

- **AF3 temperature**: T=0.395.
- **NLL-optimal fusion weight**: alpha(probe)=0.837 (i.e. only 16% weight on AF3, since AF3 alone is weak on Task 1 at 0.379) — **validation result: top1=0.5379/top3=0.8636**. Compare to probe-alone (0.5076): +3.0pt, not significant (McNemar p=0.29, CI touches zero). Compare to AF3-alone (0.379): +15.9pt, **significant** (p=0.0055).
- **Accuracy-optimal fusion weight**: alpha(probe)=0.550 — **validation result: top1=0.5303/top3=0.8409**. Similar pattern: not significantly different from probe-alone (p=0.69), significantly beats AF3-alone (p=0.0017).
- **Conclusion, in clear contrast to Task 2**: Task 1's fusion result **does largely survive the leakage-free refit** — 0.538 (OOF, NLL-optimal) vs. the original validation-swept 0.553 is only a 1.5pt drop, much smaller than Task 2's collapse (0.657 → 0.569-0.588, an 8-9pt drop). The likely reason: AF3 performs poorly on Task 1 (0.379) so the fitted fusion weight naturally puts most weight on the probe regardless of which set it's fit on (alpha=0.55-0.84 across all 4 fitting variants tried across both tasks for A), leaving much less room for validation-set overfitting to matter — whereas Task 2's fusion leans much more heavily on AF3 (a harder-fought, more evenly-balanced combination), which is exactly the regime where a small validation set can overfit the weight choice.
- **Still not a fully confirmed "beats the probe" result for A** (p=0.29 vs 0.69 for the two variants, both non-significant against probe-alone) — but it IS confirmed to beat AF3-alone, and the point estimate itself replicates reasonably well under the stricter fitting procedure, unlike Task 2's.
- Full details: `results/af3_stack_oof/A_muq_large_msd_af3direct.json`.

### Task 1: CORAL ordinal head — `src/coral_ordinal.py`
Round 3's strongest cross-source-convergent recommendation (all 4 sources) for Task 1: a CORAL (Cao et al. 2019, arXiv:1901.07884) cumulative-link ordinal head — shared weight vector `w^T x`, per-threshold biases `b_k`, extended-binary targets, `P(y>k)=sigmoid(w^T x + b_k)` — trained on the same MuQ layer1 embeddings that are Task 1's established best frozen-probe features. Rank-consistency (monotonic `P(y>k)` in `k`) is **not hard-constrained** here, same practical relaxation the reference `coral-pytorch` implementation uses — checked directly rather than assumed: 0% violation rate on validation, so this wasn't the issue.

- **First comparison caught a design flaw before trusting the result**: initially compared CORAL (a linear head) against the established best-config classifier, which is an **SVM with an RBF kernel** — a capacity/nonlinearity mismatch, not a fair "ordinal vs nominal decision surface" test. Added a proper **capacity-matched comparison**: plain logistic regression (also linear) on the identical layer1 features.
- **Result, both comparisons — CORAL loses clearly and consistently**:
  - CORAL: top1=0.3182, top3=0.7424, MAE=1.18 decades, QWK=0.630
  - Linear nominal (logreg, capacity-matched): top1=0.5000, top3=0.8409, MAE=0.79, QWK=0.751 — **CORAL loses on every metric**, including the ordinal-specific ones (MAE, QWK) CORAL is supposed to be better at.
  - Established best (SVM): top1=0.5076 — same story.
  - Both comparisons significant: CORAL vs logreg p=0.0005, CORAL vs SVM p=0.0002 (McNemar), CIs exclude zero, all favoring the nominal classifier.
- **Verified this is a real effect, not a training bug**: swept 9 (epochs × lr) configurations (300/1000/3000 epochs × lr 0.001/0.01/0.1) — val accuracy stayed in a narrow 0.28-0.34 band throughout, never approaching logreg's 0.50, while train accuracy at the default config (0.565) already exceeds val accuracy (0.318) by 25pt — a genuine train/val generalization gap, not undertraining. More epochs/higher LR only pushed train loss lower (more overfit) without moving val accuracy.
- **Honest interpretation**: CORAL's single shared-direction weight vector (one linear projection reused across all K-1 thresholds) is a much more constrained decision surface than logistic regression's per-class weight vectors — here that constraint costs more in raw discriminative flexibility than it gains from encoding the ordinal structure. This directly contradicts all 4 round-3 research sources' top recommendation for Task 1 — reported as a genuine, verified negative result, not dismissed as a training artifact. Given how large and hyperparameter-robust the gap is (never closer than ~16pt across the whole sweep), **not pursuing further CORAL tuning or a CORN variant** — treating this queue item as closed with a clear negative finding.
- Full details: `results/coral_ordinal/A_muq_large_msd_layer1.json`.

### Task 2 statistical reporting upgrade: continuous scoring + one-sided McNemar — `src/significance_continuous.py`
Round-3 recommendation from 2 of 4 sources (Gemini, Qwen): binarized Top-1 McNemar collapses each prediction to right/wrong, discarding information the full probability vector carries — a paired Wilcoxon signed-rank test on per-sample **Brier score** (continuous, lower=better) uses that information and should have more power at small n. Also added a **one-sided McNemar** matching our actual directional hypothesis ("does fusion beat AF3", not "is it merely different"). Applied to both OOF-fusion variants (NLL-optimal, accuracy-optimal) from the round-3 refit above, both tasks.

**Task B**: confirms the OOF-refit finding with a more powerful test, doesn't reverse it — **fused vs. AF3-alone is still not significant** (Wilcoxon p=0.78 NLL-variant / p=0.62 accuracy-variant, one-sided McNemar p=0.50/0.25) even with continuous scoring. But the higher-power test **does** newly resolve fused-vs-probe-alone as significant, which the binarized McNemar test (p=0.38/0.20) had missed: **Wilcoxon p=0.0108 (NLL) / p=0.0011 (accuracy)** — the fusion's real, well-confirmed benefit over Task 2's frozen probe is now clearer with the right test, even though it still doesn't beat AF3 alone.

**Task A**: fused **clearly and robustly beats AF3-alone** on the continuous test (Wilcoxon p=0.0009 NLL / p<0.0001 accuracy-variant, one-sided McNemar p=0.0027/0.0008) — strongly confirms the earlier binarized result. But fused-vs-probe-alone stays non-significant even with the more powerful test (Wilcoxon p=0.92/0.40) — the fusion's real confirmed benefit for A is specifically "beats AF3 alone" (which AF3 alone is bad at on this task anyway), not a confirmed lift over the probe by itself.

**Clean summary of what's actually confirmed, both tasks, using the best available test in each case**:
- **A**: fusion beats AF3-alone (confirmed, p<0.01). Fusion vs. probe-alone: not confirmed either way.
- **B**: fusion beats the frozen probe alone (confirmed, p<0.05 via Brier/Wilcoxon — the binarized test alone had missed this). Fusion vs. AF3-alone: not confirmed, consistent with the OOF-refit collapsing toward AF3's own score.

**Queue item closed.** Full tables: `results/significance_continuous/{A,B}_af3*.json`.

### Music Flamingo as AF3 replacement — `src/alm_infer.py` (`--model musicflamingo`)
Round-3 queue item 4, best-supported new-model candidate (3 of 4 sources: ChatGPT, Perplexity, Gemini) — `nvidia/music-flamingo-2601-hf` (arXiv:2511.10289), NVIDIA's music-specialized successor built on the Audio Flamingo 3 backbone. Verified the model id and `MusicFlamingoForConditionalGeneration`/`MusicFlamingoProcessor` classes actually exist in the installed transformers (5.16.1) before writing any code, then verified with a 3-clip known-label smoke test (all 3 correctly predicted, sensible score ordering) before trusting a full run — same pattern as every other new pipeline this project. Added as a new `model_name` option in `alm_infer.py`, zero changes to the existing teacher-forced scoring / invalid-rate infrastructure.

**Zero-shot results, both tasks, validation split, both prompts:**
| task | prompt | top1 | top3 | invalid_rate | vs. AF3's own number |
|---|---|---|---|---|---|
| A | direct | 0.3485 | 0.6591 | 0.000 | worse (AF3 direct: 0.379) |
| A | cot_then_answer | **0.4015** | 0.7348 | 0.053 | **better** (AF3's best-on-A was 0.379, this is a new best zero-shot ALM number for Task A, +2.3pt) |
| B | direct | 0.4804 | 0.6471 | 0.000 | worse (AF3 direct: 0.588) |
| B | cot_then_answer | 0.5588 | 0.7941 | 0.382 | ties on top1 (AF3 cot: 0.559), slightly better top3 (0.794 vs 0.775) -- but 38% invalid free-form generation rate on this prompt is a real quality flag, even though it doesn't affect the teacher-forced top1/top3 numbers per the established methodology |

**Honest conclusion**: mixed, not a clean win. Music Flamingo's `direct` prompt underperforms AF3's `direct` prompt on both tasks. Its `cot_then_answer` prompt is a genuine, measured improvement for **Task A specifically** (new best zero-shot ALM result there, beating both AF3 and Qwen2-Audio's earlier numbers) but only ties AF3 on Task B, with a concerning invalid-generation-rate signal. Since Music Flamingo shares AF3's backbone (per the round-3 research), its errors are expected to correlate with AF3's — **not fusing the two together** (per ChatGPT's explicit warning that this wouldn't add independent signal the way MuQ+AF3 did); using it as a standalone alternative zero-shot number for Task A is the more defensible next step if pursued further, though Task A's zero-shot ALM track was never competitive with the trained/frozen-probe pipeline (0.40 vs. 0.50+ for our probes) regardless of which ALM is used. **Queue item closed** -- results recorded, no further Music Flamingo fusion planned given the correlated-backbone concern.

Full metrics: `results/alm/musicflamingo_{A,B}/{direct,cot_then_answer}_metrics.json`.

### LoRA fine-tune of MERT (`src/finetune_lora.py`) — testing IMPROVEMENT_FINDINGS.md's PETL-paper-informed recipe
- **Config**: MERT-v1-330M encoder frozen except LoRA adapters (peft 0.21.0) on `q_proj/k_proj/v_proj/out_proj` of the **top 8 of 24 encoder layers** (layers 16-23), rank=8, alpha=16, dropout=0.05 — 530,438 trainable params out of 315,959,430 total (0.17%), verified via a forward+backward smoke test before committing to a full run. Classification head fully trainable (randomly initialized, no pretrained weights). lr=1e-4 (higher than the 2e-5 used for full fine-tune, appropriate for a much smaller trainable surface), AdamW, cosine schedule, 15 epochs, batch=4, 10s crops (same as full fine-tune). Two augmentations added that the full-fine-tune runs never used: waveform-domain time masking (our stated analogue of SpecAugment — MERT takes raw waveform, not a precomputed spectrogram, so this masks contiguous waveform spans, up to 8% length, 2 masks/clip, rather than literal time/frequency spectrogram bins) and waveform mixup (beta(0.2,0.2) interpolation + soft-label KL loss).
- **Task B, first run (LoRA + time-mask + mixup)**: loss decreased monotonically every epoch (1.798→1.566) but **never got close to competitive** — best_val_top1=**0.3235** (epoch 8+, plateaued), well below every other config tried for B (frozen MERT 0.471, full fine-tuned MERT 0.412, MuQ 0.500, MuQ+AF3 fusion 0.657). Loss curve was still decreasing at epoch 14 with no plateau in the loss itself (though val_top1 did plateau from epoch 8 on) — ambiguous whether this needs more epochs or whether the combination of a very small trainable surface (0.17%) with aggressive augmentation (mixup soft-labels + waveform masking) is simply too hard a training signal to converge quickly. Not concluding "LoRA doesn't work" from this alone — ablating the augmentation next, per standing no-premature-conclusions practice, before drawing that conclusion.
- **Ablation 1 (no-aug, 15 epochs)**: `--no-time-mask --no-mixup`, otherwise identical — `results/mert_lora_B_noaug/`. best_val_top1=**0.3627** — removing augmentation helped (+3.9pt over the augmented run), confirming augmentation was part of the problem, but still far below every competing config.
- **Ablation 2 (no-aug, 40 epochs, checking convergence)**: same no-aug config, 40 epochs instead of 15 — `results/mert_lora_B_noaug_40ep/`. Loss kept decreasing the whole way (1.59→0.90) but val_top1 clearly decelerated and plateaued in the 0.33-0.38 range from epoch ~20 onward (epoch 20: 0.353, epoch 30: 0.343, epoch 39: 0.333, best overall 0.3824 at one epoch in between) — **not a still-converging run cut short, a genuine plateau**, confirmed by running 2.7x longer than the original 15 epochs and gaining only +2pt top1 for it while train loss nearly halved (classic overfitting-to-train-loss-without-val-gain signature).
- **Conclusion (measured, not presumed)**: LoRA fine-tuning of MERT (rank 8, top-8-layers, q/k/v/out_proj, 0.17% trainable params) **does not close the gap for Task B at this data scale** — best achieved 0.382, well below frozen MERT probe (0.471), frozen MuQ probe (0.500), and nowhere near the MuQ+AF3 fusion result (0.657). This is a real, measured discrepancy with Huang et al.'s (arXiv:2411.19371) GTZAN finding that LoRA(rank 2) beat full fine-tune and nearly matched the frozen probe (74.7% vs 75.6%) — worth reporting honestly in the write-up as a task/dataset-regime difference rather than a contradiction of their result: GTZAN is a much larger, acoustically-grounded genre task, while Task B is our smallest-n, most editorial/non-acoustic label (per the confusion-diagnostic and literature discussion above) — plausibly the hardest possible case for any encoder-adaptation method, LoRA included, consistent with this project's now-repeated finding that Task B specifically resists nearly every technique except a differently-pretrained encoder (MuQ) and probability-level fusion (AF3). Not testing further LoRA hyperparameters (rank/layer-count/lr grid) given this plateau evidence and the much larger gap to already-established strong configs — deprioritizing further LoRA tuning in favor of the remaining lower-priority queue items.
- **Task A**: not run — given Task B's clear negative/plateaued result and the added evidence that Task A's configs are already statistically indistinguishable from each other (see significance section above), a LoRA run for A is deprioritized rather than skipped on presumption; can be picked up if time allows.
