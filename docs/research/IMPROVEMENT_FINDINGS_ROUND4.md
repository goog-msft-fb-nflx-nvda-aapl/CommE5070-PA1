# Deep Research Synthesis, Round 4 (2026-09-23)

Single-source this round (AI-assisted deep research, `survey_response_4/compass_artifact_wf-6adfbabb-1956-5139-8f3a-e0f2123684ad_text_markdown.md`, on `research_prompt_round4.txt`) — no reconciliation-across-sources step needed, but every claim in it was already flagged by the source itself with a citation-confidence level, which is preserved below rather than silently treated as uniform.

## Key diagnosis: CORAL's failure is expected and mechanistically explained, not mysterious
CORAL (Cao et al. 2019) forces all K-1 binary thresholds to share **one** weight vector, differing only in bias — a severe capacity bottleneck on frozen features, since a plain 6-way logistic regression gets six independent weight vectors. This directly explains our measured 31.8% (CORAL) vs 50.0% (capacity-matched logreg) result. CORN (Shi/Cao/Raschka, arXiv:2111.08851) exists specifically to remove this constraint. **Implication: don't write off ordinal methods generally — the specific architecture we tried was capacity-starved, not "ordinality doesn't help here."**

## Top recommendation, Task 2 (market): sung-language ID as an explicit feature
Mechanistically unlike everything tried (SSL embeddings, ALM label scoring, contrastive zero-shot) — a **linguistic**, not acoustic, signal. Whisper-large-v3's `detect_language` gives a 99-language posterior; run on the mixture and/or Demucs vocal stem. Directly targets our own confusion-diagnostic finding: Brazil/Spain/Italy have near-deterministic language cues (Portuguese/Spanish/Italian), while US/UK/Germany's confusion is explained by shared English — language ID cleanly resolves the Romance cluster and leaves capacity for the harder Anglo/Germanic sub-problem.

Paired with a **hierarchical/grouped classifier** (Stage 1: Romance {BR,ES,IT} vs Anglo-Germanic {US,UK,DE} using the language posterior; Stage 2: resolve within-cluster) — directly exploits the market-cluster structure our own pairwise-AUC diagnostic already found (Brazil near-perfectly separable, US/UK/Germany mutually confusable).

## Top recommendation, Task 1 (decade): SORD soft labels, not another binary-decomposition ordinal head
Díaz & Marathe (CVPR 2019): replace one-hot targets with a softmax over `-|rank_i - rank_true|` (temperature-tuned) — keeps the full 6-logit head (no CORAL-style capacity loss), injects ordinality through the *targets* instead of the architecture. ~5-line change to the existing frozen-probe training loop. Also cited: EMD/squared-EMD loss (Hou et al., arXiv:1611.05916, good for top-3 since it concentrates mass near the true class); unimodal Poisson/Binomial output (Beckham & Pal, arXiv:1705.05278, explicitly improves top-k); a **zero-retraining post-hoc option** (probability-weighted expected-rank decoding of our already-trained nominal probabilities).

## Task 2 gap-to-AF3: probe AF3's internal representations instead of its output text
AF3 uses AF-Whisper (Whisper-large-v3-derived) as its audio encoder in a LLaVA-style setup. SonicBench (arXiv:2601.11039) reports frozen-encoder linear probes beat end-to-end ALM output scoring across 8 LALMs (≥0.60 vs ~0.50 accuracy) — **but explicitly does NOT find this holds for Qwen-Omni models**, so the source flags this must be verified empirically for AF3, not assumed. Also recommended: **contextual calibration** (Zhao et al., arXiv:2102.09690) — a parameter-free (or OOF-fit) prior-correction wrapper on any ALM's label scores using a content-free/silence input, cited as unable to overfit the way our earlier validation-weight-sweep did.

## Statistical protocol critique — our current approach is flagged as underpowered
Single 102/132-item split + bootstrap + McNemar is criticized directly: recommends repeated **artist-grouped k-fold** on pooled train+val with the **Nadeau-Bengio corrected resampled t-test** (ordinary paired t-tests on overlapping CV folds underestimate variance — Dietterich 1998), a **Bayesian correlated t-test with a region-of-practical-equivalence** (`baycomp`, directly answers "are these configs practically equivalent" rather than a bare p-value), log-loss/Brier as the *primary* metric (already computed, not yet primary), and **pre-registering exactly one final test-set comparison** — explicitly named as "the only real defense against the weight-sweep overfitting that produced your spurious 65.7%," i.e. validating our own already-learned lesson from a different angle.

## External benchmark reconfirmed, same conclusion as round 3
arXiv:2608.10980 (67-71% macro accuracy, ~20x our Task 1 data) reconfirmed and interpreted as "your 52.3% is data-limited, not method-limited." MuQ is independently noted as MARBLE's best frozen music-SSL encoder, consistent with our own finding — "you are already using near-SOTA representations, so encoder swaps alone are unlikely to move the needle much," reinforcing why CultureMERT/MusicFM/CLAP all underperformed MuQ. Per this project's standing "no false ceilings" practice, this is recorded as the source's interpretation for calibration, not adopted as a reason to stop measuring.

## New models mentioned, explicitly flagged by the source itself as lower-confidence or lower-value
- MAEST (`mtg-upf/discogs-maest-30s-pw-129e`) / discogs-effnet (`mtg/effnet-discogs`) — Discogs-*supervised* embeddings (not self-supervised), same label universe as our task; source's #2 pick after language-ID.
- MiDashengLM / Dasheng (`mispeech/dasheng-0.6B`/`-1.2B`, arXiv:2508.03983) — general-audio (not music-specialized) masked encoder; medium priority.
- Qwen3-Omni / Kimi-Audio / Step-Audio — explicitly flagged as "still generative ALMs, the paradigm that already underperformed" unless paired with a changed readout (probing or CoT+self-consistency), not plain label scoring. Step-Audio's arXiv citation flagged by the source itself as "medium confidence — verify before citing."
- Explicitly told NOT to re-attempt: another SSL encoder as a plain frozen probe (MuQ already best), CLAP-style contrastive zero-shot (already far below random), plain ALM teacher-forced scoring on yet another model without a changed readout.

## Prioritized queue for round 4 (source's own staging, adopted as-is — ranked by expected value ÷ effort)

**Stage 1 — cheap, high-probability wins:**
1. Task 2: Whisper-large-v3 sung-language posterior (on Demucs vocal stem) as a feature; build the Romance-vs-Anglo hierarchical classifier.
2. Task 1: SORD soft labels on the best frozen features; also the zero-retrain ordinal-decoding post-hoc check.
3. Both: wrap ALM label scoring (AF3) in contextual calibration.
4. Task 1: production/loudness descriptors (crest factor/dynamic range/spectral tilt; LUFS only if our preprocessing didn't already loudness-normalize — needs checking, source flags this explicitly as a possible confound).
5. Stats: Brier as primary metric; artist-grouped repeated CV + Nadeau-Bengio correction + `baycomp`; pre-register the single final test-set comparison.

**Stage 2 — medium effort:**
6. Task 2: linear probe on AF-Whisper features (and AF3 LLM hidden states) vs AF3 output scoring — verify empirically, source flags this may not win (Qwen-Omni caveat).
7. Both: MAEST / discogs-effnet as fusion members.
8. Task 1: EMD loss and unimodal Poisson head, specifically for top-3.
9. Task 2: CoT + self-consistency on a Thinking-capable ALM with constrained output format.

**Stage 3 — higher effort:**
10. MiDashengLM/Dasheng as a diverse frozen extractor.
11. External-data / semi-supervised pretraining or kNN retrieval against a larger Discogs-tagged corpus (source's highest-ceiling, highest-effort item).

**Source's own stated decision rules** (kept verbatim since they're falsifiable, actionable checkpoints): if language ID pushes Romance-class recall near-ceiling but US/UK/DE stays ~chance, redirect remaining Task 2 effort only to the Anglo/Germanic sub-classifier. If AF-Whisper probing doesn't beat AF3 output scoring by more than the ROPE (±1%) under the Bayesian test on OOF folds, keep AF3-alone. If SORD/ordinal decoding doesn't beat plain CE on Brier under Nadeau-Bengio CV, abandon ordinal methods for this data size (capacity, not ordinality, is the binding constraint) — matching what we already found for CORAL specifically.
