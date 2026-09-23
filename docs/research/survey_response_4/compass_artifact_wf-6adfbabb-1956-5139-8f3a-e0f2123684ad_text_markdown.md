# Round-4 Research Report: Mechanistically-New Directions for Discogs-VI Decade & Market Classification

## TL;DR
- **Task 2 (market): the single highest-value genuinely new lever is sung-language identification** via Whisper (`openai/whisper-large-v3`, arXiv:2212.04356) language detection or a dedicated singing-LID phonotactic pipeline, used as an explicit categorical feature — this attacks the exact structure your confusion diagnostics show (Brazil/Portuguese separable; US/UK/Germany confuse because they largely share English), and it is mechanistically unlike anything tried (ASR/phonotactics, not SSL embeddings or ALM label scoring).
- **Task 1 (decade): CORAL failed because of its single-shared-weight-vector rank-consistency constraint; switch to distance-aware soft labels (SORD, Díaz & Marathe CVPR 2019) / unimodal or EMD losses on your existing best frozen features** (and, as a "correct baseline," CORN, arXiv:2111.08851, `coral-pytorch`) — these keep the full per-class discriminative capacity that CORAL throws away.
- **The biggest measurement problem is statistical power at n=102/132**: move off single-split accuracy to artist-grouped repeated k-fold on pooled train+val with the **Nadeau–Bengio corrected resampled t-test** and **Bayesian correlated t-test (`baycomp`)**, pre-register one final test-set comparison, and score with **log-loss/Brier** (higher power than accuracy). Your ~52%/~59% numbers are plausibly near the achievable ceiling for ~800–1000 clips (the closest published benchmark hits 67–71% with ~20× the data), so the remaining gains are most likely in new *feature types* and better statistics, not new SSL encoders.

---

## Key Findings
1. There is real but bounded, mechanistically-distinct headroom, concentrated in **feature sources you have not exploited (sung language, production/loudness descriptors, Discogs-native supervised embeddings)** rather than in new SSL encoders or new ALMs, most of which are variants of what you already tried.
2. **CORAL's failure is expected and well-documented**: its weight-sharing constraint is exactly what CORN was designed to remove. For frozen foundation-model features, soft-label/loss-based ordinal methods are lower-risk than any binary-decomposition head.
3. For **Task 2 ALM extraction**, the literature now supports **probing the frozen audio encoder over reading the generated answer** (SonicBench, arXiv:2601.11039) — with the important caveat that this advantage does *not* hold for Qwen-Omni models, so it must be verified for AF3, not assumed.
4. Your statistical protocol (single 132/102-item split + bootstrap + McNemar) is underpowered; the correct fix is repeated artist-grouped CV with variance-corrected and Bayesian tests, plus proper scoring rules.
5. The only closely comparable published benchmark (arXiv:2608.10980, verified) reaches 67.0–71.2% decade macro-accuracy with ~20× your data, consistent with your numbers being data-limited rather than method-limited.

---

## Details, organized by your five research questions

### Q1 — Recent (2025–2026) models/methods that are MECHANISTICALLY different
*Ranked by expected value ÷ effort.*

**A1 (HIGH value / LOW–MEDIUM effort) — Sung-language identification as an explicit feature (Task 2).**
Mechanism: automatic language ID of the vocals via ASR/phonotactics — a *linguistic* signal, orthogonal to every acoustic SSL/ALM approach you tried. Two concrete pipelines:
- **Whisper language detection:** `openai/whisper-large-v3` (arXiv:2212.04356) exposes `detect_language`, producing a distribution over **exactly 99 languages**; per the HF model card the 1.55B-parameter v3 model was "trained on 1 million hours of weakly labeled audio and 4 million hours of pseudo-labeled audio," with Cantonese added as a new v3 language token. Run it on the mixture and/or on Demucs vocal stems, aggregate the language posterior over crops, and feed it (Portuguese/Spanish/Italian/German/English…) into your existing probe.
- **Dedicated singing-LID:** the deep phonotactic CTC approach (arXiv:2105.15014) and multimodal SLID (arXiv:2103.01893) are purpose-built for polyphonic music and reported to outperform metadata baselines.

Why different: your labels are *market* (US/UK/BR/ES/DE/IT), and sung language is a near-deterministic cue for BR (Portuguese), ES (Spanish), IT (Italian) and a strong prior for DE vs the Anglo cluster. This directly targets the US/UK/DE confusion (all frequently English) with a complementary within-cluster signal.
Cost: one Whisper pass; small. Overlap note: you used Demucs stems for *acoustic* classification (mixture won), but never for *language ID*, a different objective — worth re-attempting specifically for LID on the vocal stem.

**A2 (HIGH value / LOW effort) — Discogs-native supervised embeddings (MAEST / discogs-effnet).**
Mechanism: supervised music representations trained on Discogs editorial metadata — the *same label universe* your task is drawn from — versus the self-supervised (MERT/MuQ/MusicFM) objectives you swept.
- **MAEST:** `mtg-upf/discogs-maest-30s-pw-129e` (also 10s/5s/fs variants), GitHub `palonso/MAEST`, ISMIR 2023. The paper reports best features come from *intermediate* layers — do a layer sweep as you did for MERT.
- **discogs-effnet (Essentia):** an EfficientNet-B0 (`mtg/effnet-discogs`) "trained in more than two million music recordings" annotated by Discogs metadata to predict 400 styles, producing a **1,280-dimension embedding** (essentia.upf.edu). Newer MAEST-based `genre_discogs400-discogs-maest-*` heads were added Jan 2025.

Why different: MAEST/EffNet are Discogs-*supervised*, giving representations aligned to editorial taxonomy; genre/style is strongly decade- and market-correlated. Not "another SSL encoder" — a supervised, in-domain extractor.
Cost: frozen-probe extraction, same harness as MuQ. Overlap: still a frozen probe, so if MuQ already saturates the linear-probe ceiling it may only help in fusion — but it is the most promising *new* frozen encoder because of the label match.

**A3 (MEDIUM value / LOW effort) — Production / loudness-war descriptors for decade (Task 1).**
Mechanism: hand-designed mastering descriptors (integrated LUFS, crest factor/dynamic range, spectral tilt/centroid, true-peak) — physical production markers, not learned embeddings. The loudness war is a documented, monotonic 1980s→2000s trend: the average RMS level of a typical 1980s rock song was "around −16.8 dBFS" (Bob Katz, cited in Wikipedia "Loudness war"), while integrated loudness of peak-era masters fell to roughly −7 to −6 LUFS around 2008–2010, with dynamic-range databases showing a decline from ~12–15 dB in 1980s rock to under ~7 dB in mid-2000s masters — an ordinal, decade-discriminative signal.
Why different: none of your encoders explicitly represent absolute loudness/dynamic range (SSL models are largely level-normalized). A handful of scalar descriptors concatenated to your features is cheap.
Cost: trivial (pyloudnorm / Essentia). **Caveat:** mastering/remastering and streaming normalization confound absolute loudness *if your WAVs were loudness-normalized in preprocessing* — check whether your 24 kHz mono pipeline preserved level. If normalized, use normalization-invariant descriptors (dynamic range/crest factor/spectral tilt) rather than absolute LUFS.

**A4 (MEDIUM value / MEDIUM effort) — MiDashengLM / Dasheng encoder as a new frozen extractor + ALM.**
Mechanism: MiDashengLM (arXiv:2508.03983; HF `mispeech/midashenglm-7b`; GitHub `xiaomi-research/dasheng-lm`) is caption-aligned (general-audio captions) — a training objective distinct from AF3/Qwen instruction tuning. Its encoder **Dasheng** is separately available (`mispeech/dasheng-0.6B`, `mispeech/dasheng-1.2B`; `mispeech/dasheng-base` at 85.4M) and explicitly supports frozen linear-probe use (an ESC-50 finetune notebook freezes the encoder). Dasheng-1.2B was trained on **272,356 hours of diverse audio with 1.2B parameters** (primarily VGGSound, AudioSet, MTG-Jamendo and ACAV100M; Interspeech 2024, "Scaling up masked audio encoder learning for general audio classification," arXiv:2406.06992).
Why different: Dasheng is a general-audio masked encoder (not music-specialized), capturing production/timbre cues MERT/MuQ may miss; MiDashengLM's caption objective differs from AF3 for zero-shot use.
Cost: medium (7B inference or encoder extraction). Overlap: as a frozen probe it's the same *family* as MuQ/MERT; value is as a *diverse* fusion member (OOF-fit) or via caption-based prompting.

**A5 (LOW–MEDIUM value / MEDIUM effort) — Newer omni ALMs: Qwen3-Omni, Kimi-Audio, Step-Audio.**
- **Qwen3-Omni** (arXiv:2509.17765; GitHub `QwenLM/Qwen3-Omni`) replaces Whisper with a from-scratch **0.6B Audio Transformer (AuT) "trained from scratch on 20 million hours of supervised audio data (80% Chinese/English pseudo-labeled ASR, 10% other languages ASR, 10% audio understanding)"** at a 12.5 Hz token rate, and adds a Thinking mode — a materially different backbone from Qwen2-Audio (which you found weak).
- **Kimi-Audio** (arXiv:2504.18425; GitHub `MoonshotAI/Kimi-Audio`), **Step-Audio** (arXiv:2502.11946).

Why possibly different: new encoder (AuT) and explicit reasoning could change zero-shot behavior — but these are still generative-ALM label scoring, the paradigm that already underperformed for you on Task 1. Recommend only the *Thinking/CoT + self-consistency* variant (Q2/B3), not plain label scoring.
Overlap: plain zero-shot scoring overlaps Qwen2-Audio/AF3/Music-Flamingo — only worth it with a different readout.

**Explicitly NOT worth re-attempting** (overlap with what you did): another SSL encoder used as a plain frozen probe with the same LogReg/SVM head (MuQ already best); LAION-CLAP-style contrastive zero-shot (already far below random); plain ALM teacher-forced label scoring on yet another model without a changed readout.

---

### Q2 — Extracting more signal from AF3 (Task 2)

**B1 (HIGHEST value / MEDIUM effort) — Probe AF3's frozen audio encoder (AF-Whisper) instead of reading its generated answer.**
AF3 (arXiv:2507.08128; HF `nvidia/audio-flamingo-3` and `nvidia/audio-flamingo-3-hf`) uses **AF-Whisper**: confirmed in Sung et al. (arXiv:2511.05550), AF3 "consists of a unified audio encoder (AF-Whisper, derived from Whisper-large-v3 and finetuned for joint speech/sound/music representation), an MLP-based audio adaptor, and a Qwen2.5-7B language model backbone in a LLaVA-style configuration." The strongest current evidence that internal probing beats output scoring for audio is **SonicBench (arXiv:2601.11039; GitHub EIT-NLP/SonicBench)**: across 8 LALMs (incl. SALMONN, Qwen2-Audio, MiDashengLM) frozen-encoder linear probes reach ≥0.60 accuracy vs ~0.50 for end-to-end outputs. Complementary evidence: a hidden-state readout head on a Qwen-Omni backbone beats generative decoding for sentiment (arXiv:2606.05713, multimodal sentiment regression); CoAT (arXiv:2606.18273) trains linear probes on Qwen2.5-Omni/AF3 hidden states; and for text LLMs the canonical "internal probe beats the answer under a miscalibrated readout" result is arXiv:2609.04582.
Why different: you scored AF3's *output* label probabilities; you never trained a probe on AF-Whisper (loadable standalone) or on AF3 LLM hidden states — a distinct information path.
Cost: extract AF-Whisper features, train your existing LogReg/SVM. **Critical caveat:** SonicBench found the probe>output advantage does NOT hold for Qwen2.5/Qwen3-Omni — so verify empirically for AF3. Probe both AF-Whisper *and* mid-LLM layers.

**B2 (HIGH value / LOW effort) — Calibrate AF3's zero-shot label scores.**
Contextual calibration (Zhao et al., arXiv:2102.09690) uses a content-free input (silence/"N/A") to estimate and divide out the model's label prior; Batch Calibration (OpenReview `L3FHMoKZcS`) and domain-context calibration (arXiv:2305.19148) generalize it. Your 65.7%→58.8% collapse was validation overfitting from a weight sweep; contextual calibration is parameter-free (or fit leakage-free on OOF folds) and specifically reduces the majority/common-token label bias that plagues 6-way ALM scoring.
Why different: prior-correction of the *same* scores, not a new fusion weight — it cannot overfit the way your sweep did if you use the content-free estimate.
Cost: trivial. Recommended as a default wrapper on any ALM label scoring.

**B3 (MEDIUM value / MEDIUM effort) — CoT + self-consistency / majority voting on a Thinking-capable ALM.**
Use Qwen3-Omni's Thinking mode or AF3-Chat to generate reasoning, sample multiple chains, majority-vote the label — different from single teacher-forced scoring.
Cost: several samples per clip. Caveat: Music-Flamingo already had 38% invalid generations; enforce a constrained final-answer format and discard invalids.

**B4 (HIGH value / LOW effort) — Exploit the market-cluster structure with a hierarchical/grouped classifier.**
Your confusion diagnostics (BR nearly separable; US/UK/DE mutually confusable) motivate a two-stage design: Stage 1 classifies BR/Romance {BR/ES/IT} vs Anglo-Germanic {US/UK/DE} (or BR-vs-rest, since BR is cleanest); Stage 2 resolves within-cluster. Feed the sung-language posterior (A1) especially into Stage 1.
Why different: you tried flat 6-way and coarse-to-fine on *decade*; you have not applied grouped hierarchy to *market* using the empirically-observed cluster geometry plus a language feature. Language ID makes the Romance split near-trivial, concentrating capacity on the hard US/UK/DE sub-problem.
Cost: low. Likely the best single *architecture* change for Task 2.

---

### Q3 — Why CORAL failed, and ordinal methods that keep discriminative capacity (Task 1)

**Why CORAL underperformed (mechanistic explanation).** CORAL (Cao, Mirjalili & Raschka 2020, arXiv:1901.07884) guarantees rank-monotonic thresholds by forcing **all K−1 binary classifiers to share a single weight vector**, differing only in bias. On *frozen* features this is a severe capacity bottleneck: one shared direction must linearly order all six decades, whereas a plain 6-way LogReg gets six independent weight vectors. That is exactly why you saw 31.8% (CORAL) vs 50.0% (capacity-matched LogReg). CORN (Shi, Cao & Raschka, arXiv:2111.08851; `coral-pytorch`, `corn-ordinal-neuralnet`) was created to remove this: it models conditional probabilities with **independent** per-threshold weights while preserving rank consistency, and the paper states that removing the weight-sharing restriction "improves the performance substantially compared to the CORAL reference approach." So your CORAL result *confirms* the known CORAL weakness — it does not show that ordinal structure is useless.

**Ranked ordinal recommendations that keep per-class capacity:**

**C1 (HIGHEST value / LOW effort) — SORD distance-aware soft labels (Díaz & Marathe, CVPR 2019).** Replace one-hot targets with a softmax over −|rank_i − rank_t| (temperature-tuned). Keeps a full 6-logit discriminative head (no capacity loss), injects ordinality through the *targets*, and is a ~5-line change to your existing cross-entropy probe. Widely reported to beat both CE and CORAL/CORN and to reduce MAE. The single ordinal method most likely to help here.

**C2 (HIGH value / LOW effort) — Ordinal label smoothing / Gaussian soft targets.** A Gaussian bump centered on the true decade (a special case of SORD); trivially added. See the ordinal-calibration study (OpenReview `v27yHgKtMv`) comparing SORD/CDW-CE/CO2/POE.

**C3 (MEDIUM value / LOW effort) — EMD / squared-EMD loss (Hou et al., arXiv:1611.05916).** Penalizes probability mass by ordinal distance; keeps the full softmax head. Good when top-3 accuracy matters (your reported metric) because it concentrates mass near the true class.

**C4 (MEDIUM value / MEDIUM effort) — Unimodal Poisson/Binomial output (Beckham & Pal, arXiv:1705.05278).** Constrains the output distribution to be unimodal; the paper explicitly shows improved **top-k** accuracy — directly relevant to your top-3 metric. It reduces capacity somewhat (parametric), so treat as regularization and use the learned-temperature variant.

**C5 (LOWER priority) — CORN.** Worth one run as the "correct" rank-consistent baseline to contrast with CORAL, but soft-label losses (C1–C3) are lower-risk on frozen features. 2025–2026 ordinal work (D3O, arXiv:2607.23575; DisMix, arXiv:2608.04652) confirms SORD/soft-label and dynamic-distillation methods dominate CORAL/CORN on accuracy+calibration.

**Post-hoc option (cheapest possible test):** because your errors are already cleanly adjacent-decade, a simple **ordinal decoding** of your existing nominal probabilities — take the probability-weighted mean decade (expected rank), or smooth then argmax — may recover top-1 with zero retraining.

---

### Q4 — Small-sample statistical power (n=102/132)

Your current design (one fixed split, bootstrap + McNemar) is underpowered and vulnerable to the overfitting you already saw (65.7%). Recommendations:

**D1 (ESSENTIAL) — Repeated artist-grouped k-fold on pooled train+val, with the Nadeau–Bengio corrected resampled t-test.** The ordinary paired t-test on overlapping CV folds badly underestimates variance (inflated Type-I error; Dietterich 1998). Nadeau & Bengio (2003) correct this by inflating the variance by (1/n + n_test/n_train). Implementations: `correctR` (R) and `correctipy` (Python). Use `GroupKFold` on artist IDs to preserve your artist-disjoint constraint. This yields far more (correlated-but-corrected) measurements than a single 132-item split.

**D2 (ESSENTIAL) — Score with log-loss / Brier, not accuracy.** Proper scoring rules use the full probability vector and have substantially higher statistical power than 0/1 accuracy at these n — you already compute Brier; make it the *primary* comparison metric, accuracy secondary.

**D3 (HIGH) — Bayesian correlated t-test with a ROPE.** `baycomp` (`two_on_single`, `CorrelatedTTest`; Benavoli, Corani, Demšar & Zaffalon tutorial arXiv:1606.04316; Corani & Benavoli 2015) reports P(A>B), P(rope), P(B>A) with a region of practical equivalence (e.g. ±1% accuracy). This directly answers "are these configs practically equivalent?" — your actual question about the 49–55% cluster — far better than a null-hypothesis p-value.

**D4 (HIGH) — Pre-register ONE final test-set comparison.** Decide everything else (encoder, ordinal loss, fusion weights) on OOF CV; touch the held-out 132/102 test set exactly once for the single pre-registered comparison. This is the only real defense against the weight-sweep overfitting that produced your spurious 65.7%.

**D5 (MEDIUM) — Multiple-comparison control + power analysis.** Apply Holm–Bonferroni or Benjamini–Hochberg FDR across the family of configs. Run a McNemar power analysis: at n=132 with baseline p≈0.5, detecting a ~3-point difference requires a discordant-pair count you likely do not have — quantify this so you stop over-interpreting 52.3% vs 50.0%.

---

### Q5 — Sanity check on achievable accuracy at ~800–1000 clips

**Verified external benchmark (arXiv:2608.10980, "Measuring Cross-Cultural Style Diffusion Through Era Classification: US and Korean Popular Music," Lee, Lee, Ju, Kim, Lee & Jeong; submitted 11 Aug 2026).** Confirmed: CNNs trained *from scratch* on Billboard Hot 100 audio, six decades (1960s–2010s), artist-disjoint (collaboration-graph connected components), class-balanced undersampling, decade-level **macro accuracy 67.0–71.2%** across six architectures over 18 runs — on a dataset ~20× your 1026 training clips, with cleanly ordinal (adjacent-decade) confusion that matches your Task 1 diagnostics. This strongly implies your 52.3% is **data-limited**, not method-limited: the same architecture family with ~20× data gains ~15–19 points.

**Context from MARBLE (NeurIPS 2023) and MuQ (arXiv:2501.01108).** MuQ is the current best frozen music-SSL encoder on MARBLE (beats MERT/MusicFM), consistent with your finding that MuQ is your best frozen probe — you are already using near-SOTA representations, so encoder swaps alone are unlikely to move the needle much. This reinforces that the remaining gains are in (a) more data / external Discogs pretraining, (b) new *feature types* (language, production), and (c) better statistics — not new SSL encoders.

**Implication for headroom.** Real but bounded. Expect single-digit gains from ordinal soft-labels + production features (Task 1) and from language ID + hierarchical structure + encoder probing (Task 2). Closing the gap toward ~67% likely requires more labelled data (e.g., mining additional Discogs-VI audio for semi-supervised pretraining, or kNN retrieval against a large Discogs-tagged corpus), the highest-ceiling but highest-effort direction.

---

## Recommendations (staged, cross-task, ranked by expected value ÷ effort)

**Stage 1 — cheap, high-probability wins (do first):**
1. **Task 2:** Add Whisper-large-v3 sung-language posterior (on Demucs vocal stem) as features; build the BR/Romance-vs-Anglo hierarchical classifier. *(A1 + B4)*
2. **Task 1:** Swap cross-entropy for SORD soft labels on your best frozen features; also try the zero-retrain ordinal decoding of existing probabilities. *(C1 + post-hoc)*
3. **Both:** Wrap all ALM label scoring in contextual calibration. *(B2)*
4. **Task 1:** Add production descriptors (crest factor/DR, spectral tilt; LUFS only if levels preserved). *(A3)*
5. **Stats:** Switch primary metric to Brier/log-loss; move to artist-grouped repeated CV with Nadeau–Bengio correction and `baycomp`; pre-register the single final test. *(D1–D4)*

**Stage 2 — medium effort, diversify:**
6. **Task 2:** Train a linear probe on AF-Whisper features (and AF3 LLM hidden states); compare head-to-head against AF3 output scoring — but expect it may not beat output (Qwen-Omni caveat). *(B1)*
7. **Both:** Add MAEST / discogs-effnet Discogs-supervised embeddings as fusion members (layer sweep). *(A2)*
8. **Task 1:** Try EMD loss and unimodal Poisson head for top-3. *(C3, C4)*
9. **Task 2:** CoT + self-consistency on Qwen3-Omni Thinking / AF3-Chat with constrained output. *(B3)*

**Stage 3 — higher effort, higher ceiling:**
10. MiDashengLM/Dasheng as a diverse frozen extractor and caption-based ALM. *(A4)*
11. External-data / semi-supervised pretraining or kNN retrieval against a large Discogs-tagged corpus to attack the data-limited ceiling implied by arXiv:2608.10980. *(Q5)*

**Benchmarks/thresholds that change the plan:**
- If language ID alone pushes Task 2 Romance-class recall to near-ceiling but US/UK/DE stays ~chance, invest remaining effort only on the Anglo/Germanic sub-classifier.
- If AF-Whisper probing beats AF3 output scoring on OOF folds by more than the ROPE (±1%) under the Bayesian test, adopt it; otherwise keep AF3-alone (as your leakage-free analysis already concluded).
- If SORD/ordinal decoding does not beat plain CE on Brier under Nadeau–Bengio at CV, abandon ordinal methods for this data size (capacity, not ordinality, is the binding constraint).

---

## Caveats & citation-confidence flags
- **arXiv:2606.18273** is "Continuous Audio Thinking (CoAT)," which trains linear probes on Qwen2.5-Omni/AF3 hidden states; it is *not* a dedicated probe-vs-output paper. The dedicated Qwen-Omni "hidden-state readout beats generative decoding" paper is **arXiv:2606.05713** (task is multimodal sentiment regression, not pure audio classification). Treat both as supporting, not decisive, for AF3.
- **SonicBench (arXiv:2601.11039)** is the best *audio-specific* probe>output evidence, but explicitly finds the advantage does **not** hold for Qwen2.5/Qwen3-Omni — so B1's benefit for AF3 must be verified empirically, not assumed.
- **Dasheng model IDs:** `mispeech/dasheng-0.6B` and `mispeech/dasheng-1.2B` and their frozen-probe usage are fully verified; `mispeech/dasheng-base` (85.4M) is medium-high confidence (seen in the mispeech collection listing). MiDashengLM = arXiv:2508.03983 (verified).
- **AF3:** HF IDs `nvidia/audio-flamingo-3` and `nvidia/audio-flamingo-3-hf` both exist; arXiv:2507.08128; encoder is AF-Whisper (Whisper-large-v3-derived, finetuned for joint speech/sound/music), verified via the paper, HF card, and independent corroboration (arXiv:2511.05550).
- **Ordinal citations** verified: CORAL arXiv:1901.07884; CORN arXiv:2111.08851 + `Raschka-research-group/coral-pytorch`; SORD = Díaz & Marathe, CVPR 2019 (openaccess.thecvf.com); EMD arXiv:1611.05916; unimodal arXiv:1705.05278. 2025–2026 corroboration: D3O arXiv:2607.23575, DisMix arXiv:2608.04652, ordinal-calibration OpenReview `v27yHgKtMv`.
- **Statistics citations** verified: Nadeau & Bengio corrected resampled t-test (Machine Learning 2003; `correctR`/`correctipy`); Bayesian correlated t-test via `baycomp` (janezd/baycomp; tutorial arXiv:1606.04316); contextual calibration arXiv:2102.09690; Batch Calibration OpenReview `L3FHMoKZcS`.
- **Loudness/production features (A3)** are confounded if your preprocessing loudness-normalized the WAVs; use normalization-invariant descriptors if so. Absolute-LUFS trend figures come from industry/Wikipedia loudness-war sources, not a peer-reviewed audio-forensics paper — treat the magnitude as indicative.
- The **67–71%** figure from arXiv:2608.10980 is on Billboard/Melon chart pop (US/Korea), a different distribution than Discogs-VI; treat as an order-of-magnitude sanity check, not a directly comparable target.
- **Qwen3-Omni/Kimi-Audio/Step-Audio** benefits are speculative for your tasks — new backbones, but still generative ALMs; recommended only with a changed readout (probing or CoT), since plain label scoring already underperformed. Whisper = arXiv:2212.04356 (verified); Qwen3-Omni = arXiv:2509.17765 (verified); Kimi-Audio = arXiv:2504.18425 (verified); Step-Audio = arXiv:2502.11946 (cited secondhand, medium confidence — verify before citing).