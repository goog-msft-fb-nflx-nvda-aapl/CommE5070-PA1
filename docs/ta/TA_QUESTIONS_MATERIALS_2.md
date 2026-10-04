# Materials for Drafting TA Questions (Round 2) — CommE5070 PA1 (Music Era & Release-Market Classification)

## Task

Using the background below, draft a short, polite, **direction-focused** set of questions to send to the teaching assistant. This is a follow-up round after a large amount of additional experimentation — the core ask is: **our performance is still not where we'd like it, what should we try next?**

Specifically:
1. State our current best validation results plainly (below) and ask directly: **given everything we've already tried (full list below), does the TA see an approach or angle we're missing that's expected to move the needle further** — especially for Task 2 (market classification), which has resisted most standard improvement techniques.
2. Ask whether our current numbers are simply **near the practical ceiling for this task/data scale** given prior literature we found (cited below), or whether the TA believes meaningfully higher accuracy is achievable and we're missing something.
3. Ask about the **validity of reporting a confidence-interval-qualified result** rather than a single point estimate — our best-observed configuration for Task 1 is not statistically distinguishable from a simpler prior configuration at our validation-set size (n=132); is it acceptable/expected to report both the best point estimate and a more conservative, statistically-supported number, or does the TA want a single headline number only?
4. (Include if it reads naturally, don't overload the list) Ask about the **acceptable scope of external components** — our best Task 2 result depends partly on Audio Flamingo 3 (a pretrained audio-language model's zero-shot output) fused with our own trained classifier's probabilities. Is combining a zero-shot foundation model's output with our own trained model an acceptable technique for this assignment, or should the "final" submitted result be restricted to models we trained/fine-tuned ourselves?

Keep the final question set short (4-5 questions max), professional, and easy for a TA to answer quickly. Don't pad with information they already know. Lead with the numbers so the TA immediately sees we're asking from a position of "we've done a lot and are still short," not "we haven't tried anything."

**Output format**: same as last time — a self-contained HTML snippet (not a full document, just inner markup + inline/`<style>`-block CSS), clean professional muted color scheme readable on light and dark forum themes, good typographic hierarchy, light `<strong>`/color highlighting on the key numbers and model names, directly copy-pasteable into a forum's rich-text/HTML editor, no external dependencies.

---

## Assignment context (brief, unchanged from round 1)

Graduate MIR coursework assignment: 30-second music excerpts (WAV, mono, 24kHz), two single-label 6-way classification tasks derived from Discogs-VI editorial metadata:
- **Task 1 (Dataset A)**: release-decade — 1960s/1970s/1980s/1990s/2000s/2010s. 1026 train / 132 validation / 132 test, artist-disjoint, balanced.
- **Task 2 (Dataset B)**: release-market — US/UK/Brazil/Spain/Germany/Italy, all 1980s releases. 798 train / 102 validation / 102 test, artist-disjoint, balanced.

Random baseline both tasks: Top-1 = 16.7%, Top-3 = 50%.

## Current best results (validation) — this is a big jump from what we last asked the TA about, and performance is *still* what we'd consider modest for a 6-way task

**Task 1 (decade)**: two numbers worth giving the TA, with the statistical caveat stated honestly:
- Best point estimate: **Top-1 = 55.30%, Top-3 = 84.85%** — a frozen MuQ-encoder probe fused with Audio Flamingo 3's zero-shot label probabilities.
- More conservative, statistically-supported number: **Top-1 = 52.27%, Top-3 = 85.61%** — our earlier 3-way ensemble (fine-tuned MERT + from-scratch CNN + frozen MERT probe). A 10,000-sample paired bootstrap shows these two configs (and two others we tried) are **not statistically distinguishable from each other at our validation-set size (n=132)** — the confidence intervals overlap substantially.

**Task 2 (market)**: **Top-1 = 65.69%, Top-3 = 88.24%** — a frozen MuQ-encoder probe (layer 2, logistic regression) fused with Audio Flamingo 3's zero-shot label probabilities (chain-of-thought prompt), weighted-sum combination. This is a real, statistically confirmed improvement over any single model we've built (p<0.01 vs. either component alone via paired bootstrap + McNemar test), though it is not statistically distinguishable from Audio Flamingo 3's own zero-shot score alone (58.8%) at n=102 — the point estimate is higher but the gap itself isn't confirmed at this sample size.

Even with these gains, **Task 1 sits ~30 points above random but well below what full-scale literature achieves** (see next section), and **Task 2's error analysis shows the remaining confusion concentrates in specific market clusters** (US/UK/Germany all get confused with each other; Brazil is cleanly separated) rather than spreading evenly — suggesting a structural ceiling on some class pairs rather than a fixable modeling gap, but we're not confident in that read and would value the TA's opinion.

## External benchmark context we found (relevant to question 2 above)

A closely comparable published task — *"Measuring Cross-Cultural Style Diffusion Through Era Classification"* (arXiv:2608.10980, ISMIR 2026) — does 6-way decade classification (1960s-2010s), artist-disjoint, 30s clips, and reports 67-71% macro accuracy from CNN architectures similar to our Short-Chunk CNN baseline — but on **~22,000 tracks, roughly 20x our training set size**. We have no comparable published benchmark for release-market classification specifically; adjacent geography-from-audio literature (UCI Geographical Origin of Music dataset, a Técnico Lisboa MSD thesis) consistently reports that geographic/origin prediction from audio is a much weaker signal than temporal prediction, with one thesis explicitly concluding there "may be no significant relationship between audio content and the place... the song was released" for a meaningful fraction of cases.

## Everything we've tried since our last question round (for the TA's context — long list, but shows we're not asking without having exhausted the obvious options)

- All required baselines (Short-Chunk CNN from scratch; MERT-v1-330M frozen probe with all 3 named classifiers + PCA ablation; frozen-vs-fine-tuned comparison) and all required experiments (segment-length sweep, multi-crop TTA, Task 1 year-regression/hierarchical framing, Task 2 stem separation, t-SNE/UMAP, zero-shot ALM evaluation with Qwen2-Audio-7B-Instruct and Audio Flamingo 3).
- 2-way/3-way probability ensembling across our trained models.
- A second self-supervised encoder, **MuQ** (`OpenMuQ/MuQ-large-msd-iter`, Mel-RVQ pretraining objective, distinct from MERT's EnCodec+CQT teachers) — full layer-sweep + classifier/PCA ablation, both tasks. Clearly beat MERT as a frozen probe on both tasks.
- **Late fusion of Audio Flamingo 3's zero-shot probabilities with our own trained classifiers** (weighted-sum sweep) — the single biggest improvement found this round, especially for Task 2.
- **Statistical significance testing** (10,000-sample paired bootstrap + exact McNemar test) on our top configs for both tasks, specifically to avoid over-claiming small point-estimate differences as real improvements — see the caveats stated in the results above.
- **Confusion-matrix / pairwise-AUC diagnostic** on our best configs — found Task 2's remaining errors concentrate in acoustically/culturally-similar market clusters (US/UK/Germany), not spread evenly; found Task 1's errors are cleanly ordinal (adjacent decades confused, distant decades not).
- **LoRA fine-tuning of MERT** (rank 8, attention projections of the top 8 of 24 layers, ~0.17% trainable params) with waveform-domain time-masking and mixup augmentation, both with and without augmentation, and at 15 vs. 40 epochs — this **did not work**: best result (Task 2) was 38.2% Top-1, well below our frozen-probe baseline of 47.1%, and confirmed plateaued (not under-converged) rather than just needing more training time. This directly contradicts a paper we'd found suggesting LoRA should beat full fine-tuning on MERT at small data scale (their result was on a larger, more acoustically-grounded genre-classification dataset, not our smaller/more editorial market-classification task) — genuinely unclear to us whether this means LoRA needs different hyperparameters, or whether Task 2 in particular is just resistant to any form of gradient-based encoder adaptation at this data scale (frozen probing has now beaten every fine-tuning variant we've tried on Task 2: full fine-tune, LoRA, and ensembling-with-a-fine-tuned-model).

## Known limitation, still unresolved from round 1

No artist IDs in the manifest (splits are already artist-disjoint per the assignment, but we can't do artist-grouped k-fold *within* the training split for model selection) — using plain StratifiedKFold instead. Not re-asking this explicitly unless it fits naturally with the other questions, since we didn't get a direct answer last round and don't want to repeat a question that may already be answered generically in course materials we haven't checked recently.
