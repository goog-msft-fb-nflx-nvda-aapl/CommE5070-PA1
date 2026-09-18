# Deep Research Synthesis — Improvement Directions (2026-09-18)

Reconciled from 4 independent deep-research passes (`survey_response_2/{chatgpt,kimi,perplexity,qwen}.md`) on `IMPROVEMENT_RESEARCH_PROMPT.md`. ChatGPT and Perplexity gave the deepest, most independently-verified passes (each fetched and quoted specific papers with numbers); Kimi and Qwen were shorter but converged on the same core recommendations. Cross-checked for agreement before trusting any single source.

## 1. External benchmark found for Task 1 (not previously known to us)

**3 of 4 sources independently found**: *"Measuring Cross-Cultural Style Diffusion Through Era Classification: US and Korean Popular Music"* (arXiv:2608.10980, ISMIR 2026) — the closest comparable benchmark that exists. Same task shape as our Task 1: 6-way decade classification (1960s-2010s), 30s crops, **artist-disjoint** split (via collaboration-graph partitioning). Results: 6 from-scratch CNN architectures (incl. Short-Chunk CNN), macro accuracy 67.0-71.2%, pooled 69.0±2.0% — but on **~22,000 Billboard tracks**, ~20x our 1026 training clips. Our 52.3% is a reasonable result given the dataset-size gap, not evidence of a weak pipeline. Also notable: that paper found **2010s was their hardest class too** (28.9-50.9% depending on architecture) — matches our own Audio Flamingo 3 finding that it never predicts "2010s" at all; may be a genuine, cross-study property of this decade, not just our pipeline's quirk.

Kimi did not surface this paper (likely a search-coverage gap, not a real disagreement) — flagging so we don't treat its absence in Kimi's response as contradicting evidence.

No source found any published benchmark for **release-market** classification specifically (all agree this is a genuinely novel task framing — Discogs-VI's own paper only uses country/decade as dataset-construction metadata, never as a classification target). Adjacent geographic-origin-from-audio literature (UCI Geographical Origin of Music, Técnico Lisboa MSD thesis) consistently finds geography is a *much* weaker audio signal than time: one thesis reports year prediction "reasonably well" (7.02yr MAE) vs. geographic prediction "poor performance in most cases" (3,203km mean error), even concluding "there may be no significant relationship between audio content and the place... the song was released." This is independent, citable support for why Task 2 is structurally harder — not just our models being weak.

## 2. Why Task 2 resists every technique that helps Task 1 (convergent across all 4 sources)

Six of our own experiments now show the same asymmetry (fine-tuning, multi-crop TTA, SVM-over-logreg, ensembling all help A and not B). The reconciled explanation, cited independently by multiple sources:

- **Decade has real, distributed, technology-driven acoustic correlates** (tape saturation, gated reverb, loudness-war compression, synth/drum-machine timbres) that are genuinely present throughout a clip — explaining why more audio (longer segments, more crops, more fine-tuning) keeps helping Task 1.
- **Release market is a commercial/editorial category, not necessarily an acoustic one** — Discogs defines "release country" as where a release was *distributed*, not recorded or mixed; the same master can legitimately appear under multiple market labels. There may be no strong, consistent, learnable acoustic mapping for large parts of this task, which would explain why techniques that add capacity or averaging don't move the needle.
- **Artist-disjoint splitting removes a shortcut for Task 2 specifically**: if certain artists correlate strongly with certain markets, a non-disjoint model could partly be learning "artist," but artist-disjoint splitting (which we already do, correctly) removes that, leaving a genuinely harder residual problem.
- **Statistical caveat, worth taking seriously**: at n=102 validation samples / 6 classes, binomial standard error on a ~47% accuracy is roughly ±5-10 points. Several of our "X beats Y by 2-3 points" comparisons for Task 2 (e.g. frozen probe 47.1% vs. fine-tuned 44.1%) are not statistically distinguishable at this sample size. **Recommendation: report bootstrap or McNemar-style confidence intervals for Task 2 comparisons going forward, and be more cautious about "doesn't help" framing where the gap is a few points.**

## 3. New encoders to try — convergent recommendation

All 4 sources independently recommend **MuQ** as the top next experiment:

| model | HF ID | notes |
|---|---|---|
| **MuQ** (top pick) | [`OpenMuQ/MuQ-large-msd-iter`](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) | ~300M, Mel-RVQ self-supervised (different objective from MERT's EnCodec+CQT), **24kHz native — drop-in replacement for MERT in our exact existing pipeline** (same layer-sweep + classifier + PCA ablation code, just swap the encoder). [arXiv:2501.01108](https://arxiv.org/abs/2501.01108), [GitHub](https://github.com/tencent-ailab/MuQ), license CC-BY-NC 4.0. |
| MuQ-MuLan | [`OpenMuQ/MuQ-MuLan-large`](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) | ~700M music-text contrastive; zero-shot class-description scoring, natural fit for Task 2's more semantic/cultural signal. |
| Music Flamingo | `nvidia/music-flamingo-2601-hf` | AF3 successor, music-specialized, **exposes audio hidden states via a feature-extraction path**, not just generation — worth checking `get_audio_features()`. NVIDIA OneWay Noncommercial license (same as AF3, already cited). |
| MusicFM-MSD | [`minzwon/musicfm`](https://github.com/minzwon/musicfm) | already in our original PLAN.md as a planned-but-not-yet-run differentiator; documented discriminative-LR fine-tuning recipe (1e-5 encoder / 1e-4 head). |
| CLaMP3 (Task 2 specifically) | `sander-wood/clamp3` | multilingual (194 countries/27 languages) music-text contrastive; audio path is MERT-derived (not fully independent) but the multilingual alignment is conceptually on-target for market classification. |

**Recommendation: MuQ first** — cleanest, cheapest, isolates "does a different pretraining objective help" with zero pipeline changes beyond swapping the encoder.

## 4. Closing the Task 2 gap to zero-shot Audio Flamingo 3 (58.8% vs. our 47.1%)

Convergent, ranked by effort:

1. **Stack AF3's probability vectors as features** (cheapest, uses infrastructure we already have — our `alm_infer.py` already computes exactly this via teacher-forced scoring). Concatenate AF3's 6-class probability vector with MERT embeddings (or with our classifier's own output probabilities) and train a small logistic/SVM stacker. All 4 sources rate this the highest-value-per-effort move.
2. **Validation-tuned late fusion**: same idea, simpler — grid-search a weight between AF3's probabilities and our MERT probe's probabilities directly (we already have `ensemble.py` set up for exactly this kind of weighted-probability sweep).
3. **Soft-label KL distillation** from AF3 into our MERT classifier (temperature-scaled KL term added to the training loss) — medium effort, well-precedented in the audio-classification KD literature (CMKD and related work cited across sources).
4. **AF-Whisper encoder as a frozen feature extractor** (not generation) — higher effort/risk (may need model-internals access), but conceptually the "purest" version of the idea, since AF3's *encoder* is what's doing the work, not its language-generation head.

## 5. Overfitting mitigation for fine-tuning — now backed by a directly-relevant paper

Perplexity found **"Parameter-Efficient Transfer Learning for Music Foundation Models"** (Huang et al., [arXiv:2411.19371](https://arxiv.org/html/2411.19371v1)) — a study that tested full fine-tuning vs. frozen probing vs. 6 PEFT methods **on MERT specifically**. Key numbers (GTZAN genre classification): full fine-tune 63.8%, frozen probe 75.6%, LoRA(rank 2) 74.7%, Adapter(bottleneck 16) 72.8%, BitFit 73.8%, SSF 74.8%. Their finding "fine-tuning often leads to overfitting" and "increasing dataset size reduces the [fine-tune vs. probe] gap" is a direct, independent, MERT-specific confirmation of our own measured result (frozen probe beat full fine-tune on Task 2; only won on Task 1 at one specific early-stopped checkpoint before the well-documented overfitting set in). **This paper is worth citing directly in our report as external validation of our own finding, not just as a source of new ideas.**

Concrete next-step recipe (convergent across sources):
- **LoRA rank 4-8** on attention projections (q/k/v/out) of the top 6-8 layers only, not the whole encoder — cheapest PEFT option, directly informed by the paper above.
- **Layer-wise learning-rate decay (LLRD)** if doing any broader fine-tuning — top-layer LR ~2e-5 decaying toward ~1e-6 at the bottom, preserving the lower/mid layers we already found are the informative ones (layer 4 for A, layer 7 for B).
- **Augmentation** (we used none during fine-tuning — flagged as a real gap by every source): SpecAugment (time/frequency masking) + mixup, both considered safe. **Avoid pitch-shift and aggressive time-stretch** — every source independently flags this as risky since it could destroy the exact production/tuning cues the tasks depend on, consistent with our own original SURVEY.md caution from before any fine-tuning was attempted.
- **Partial-unfreeze schedule** (freeze-then-gradually-unfreeze top layers) as a lower-effort alternative to full LoRA plumbing.

## 6. Other concrete, cheap next steps worth doing before more training runs

- **Confusion-matrix / pairwise-AUC diagnostic on Task 2** (no training needed, just analysis of embeddings we already have): check whether Task 2's difficulty concentrates in specific market pairs (e.g. US/UK/Germany/Italy sharing 1980s Anglo/Euro-pop production conventions, per our own earlier WORKLOG finding) vs. spread evenly across all 6 classes.
- **Statistical significance reporting** for Task 2 comparisons specifically (bootstrap CI or McNemar), given the n=102 sample-size caveat above — apply retroactively to existing results before drawing more "X doesn't help Task 2" conclusions.
- **Supervised contrastive loss** on top of MERT embeddings (from the "Music Era Recognition" paper we already cited in SURVEY.md, arXiv:2407.05368) — proposed as directly testable on our Task 1 pipeline, not yet tried.
- (Perplexity only, flagged as speculative) — try to determine whether any near-duplicate masters appear under multiple market labels in the underlying Discogs-VI data (would set a real ceiling on Task 2's achievable accuracy) — interesting but lower priority given we don't have the original artist/master IDs to check this ourselves.

## Prioritized queue for next session

1. MuQ frozen-probe sweep, both tasks (same pipeline as MERT, cheap, cleanest "different encoder" test).
2. AF3-probability-stacking / late-fusion on Task 2 (cheap, reuses existing `alm_infer.py` + `ensemble.py` infrastructure).
3. LoRA(rank 4-8, top layers only) fine-tune of MERT with SpecAugment+mixup, both tasks — direct test of the PETL paper's recipe against our own full-fine-tune numbers.
4. Confusion-matrix / pairwise-AUC diagnostic on Task 2 (analysis only, no new training).
5. If time allows: soft-label KD from AF3 into the MERT classifier; supervised contrastive loss on Task 1.
