# Deep Research Report: Improving Music Era & Release-Market Classification (CommE5070 PA1)

## Bottom Line — Prioritized Actions

1. **Try MuQ and MuQ-MuLan as alternative encoders** — MuQ uses Mel-RVQ pretraining (different from MERT's EnCodec+CQT teachers), is 24kHz-native, and has a 700M music-text contrastive model (MuQ-MuLan) that may capture market-relevant stylistic cues MERT misses. Low effort: frozen-probe sweep like you did for MERT.
2. **Try MusicFM-MSD** — Conformer-based, pretrained on the full Million Song Dataset, with a documented fine-tuning recipe that uses discriminative learning rates (1e-5 encoder / 1e-4 head). Different architecture from MERT's transformer encoder.
3. **Use the PETL paper's findings as your PEFT recipe** — On MERT specifically, probing outperforms full fine-tuning on small classification data; LoRA and Adapters are between the two and reduce overfitting. Use LoRA (rank 2-8) or Adapters (bottleneck 16) as a more regularized alternative to full fine-tuning.
4. **Exploit Audio Flamingo 3's probability vectors as stacker features** — Use AF3's constrained class-probability outputs (not free-form generation) as 6 numerical features input to a logistic/SVM stacker alongside your MERT embeddings. This is the lowest-effort path to closing the Task 2 gap.
5. **Add SpecAugment and mixup during fine-tuning** — You used no augmentation; time-frequency masking and mixup are standard, cheap, and specifically help small-data audio classification.
6. **Apply layer-wise learning rate decay (LLRD)** — Give lower layers exponentially smaller learning rates than upper layers during fine-tuning, preserving pretrained features while allowing task-specific adaptation.
7. **Investigate class confusion structure on Task 2** — Build a normalized confusion matrix on the validation set. The recurring pattern (techniques help Task 1, not Task 2) may stem from high inter-class acoustic overlap between 1980s markets (US/UK/Brazil/Spain/Germany/Italy), not from model capacity limitations.

---

## 1. Published Work on Release-Decade / Release-Country Classification from Audio

### Directly comparable: Era (decade) classification

**Cross-Cultural Style Diffusion Through Era Classification (arXiv:2608.10980, 2026)** — This is the most directly comparable benchmark found. It performs exactly your Task 1: 6-way decade classification (1960s–2010s) on 30-second audio crops using from-scratch CNNs, with a 16.7% random baseline — identical to your setup. Critically, the split IS artist-disjoint (constructed via a collaboration graph, with connected components assigned wholly to train/val/test at ~8:1:1).

| Model | Macro Accuracy | Micro Accuracy |
|---|---|---|
| CNN (baseline) | 67.0% | 75.0% |
| FCN | 69.6% | 75.1% |
| ShortChunkCNN | 71.2% | 77.5% |
| ShortChunkCNN_Res | 68.1% | 74.6% |
| Musicnn | 70.1% | 75.3% |
| CRNN | 69.0% | 73.7% |

Across 18 runs (3 seeds × 6 architectures), macro accuracy was 69.0% ± 2.0 and micro accuracy was 75.3% ± 1.5 ([arXiv:2608.10980](https://arxiv.org/pdf/2608.10980)). These are from-scratch CNNs on ~22,000 Billboard Hot 100 tracks — a much larger dataset than your 1,026 training clips. Your 52.3% Top-1 on 1,026 clips is in a reasonable range given the ~20× smaller training set and the difficulty of the artist-disjoint constraint. The paper also found that the 2010s class had the lowest accuracy across all architectures (28.9%–50.9%), attributed to rapid genre diversification — a pattern that may parallel your Task 1 challenges.

**Music Era Recognition Using Supervised Contrastive Learning (arXiv:2407.05368, 2024)** — The first paper to explicitly formulate "music era recognition" as a task. Uses MSD (85,475 tracks) and an in-house dataset (~800K songs). Evaluated at two granularities: 1-year-per-class (64 classes) and 10-years-per-class (8 classes including 1950s and 2020s). Uses 10-fold cross-validation with a 10% validation holdout. The split is NOT artist-disjoint. The paper proposes supervised contrastive learning (Audio-SUC) and a multimodal variant (AudioArt-MMC) that adds artist biography text. No raw accuracy numbers were extracted from the page, but the key methodological contribution — using contrastive learning to pull same-era representations together — is directly testable on your setup. The paper is motivated by the same scenario as your task: songs whose release year metadata is unavailable (remixes, covers, TikTok excerpts) ([arXiv:2407.05368](https://arxiv.org/html/2407.05368v1)).

**Million Song Dataset year prediction (Bertin-Mahieux et al., 2011)** — The foundational benchmark for release-year prediction from audio, but formulated as regression (mean absolute error), not classification. Best result: 6.14 years MAE using linear regression on 515,576 songs. NOT artist-disjoint. Uses Echo Nest timbre features, not deep learning. Your ridge-regression result (27.3% after rounding) is consistent with the general finding that regression-to-decade is harder than direct classification ([UCI MSD Year Prediction](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd), [Columbia MSD paper](https://www.ee.columbia.edu/~dpwe/pubs/BertEWL11-msd.pdf)).

### Adjacent: Geographic / cultural-origin classification

**Automatic Geocoding and Dating of Music (Técnico Lisboa thesis)** — The most relevant study for understanding why Task 2 (market) is harder. It jointly addresses release-year prediction and geographic-origin prediction from MSD audio features. Key finding: **temporal prediction performed "reasonably well" (7.02 years MAE) while geographic prediction had "poor performance in most cases" (3,203 km mean error)**. The thesis explicitly concludes that "there may be no significant relationship between audio content and the place where the song was released" — a finding directly relevant to your Task 2 ([Técnico Lisboa thesis](https://fenix.tecnico.ulisboa.pt/downloadFile/1689244997257667/resumo.pdf)).

**Cultural differences in music features across Taiwanese, Japanese and American markets (PeerJ Computer Science, 2022)** — Directly addresses market classification from audio features. Uses Spotify audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration) to classify songs into 3 cultural markets: Chinese, Japanese, English. Uses GBDT and MLP classifiers. Dataset: ~1M songs after balancing, from 10,259 artists. No accuracy numbers extracted from the page, but the key insight is that **market membership is classifiable from audio features**, though with only 3 classes and Spotify-level features (not raw audio) ([PeerJ CS](https://peerj.com/articles/cs-642.pdf)).

**Other geographic-origin work** — Zhou et al. (2014) and Schedl & Zhou (2016) used the UCI "Geographical Origin of Music" dataset (1,059 songs) with random forest regression, achieving ~3,113 km and ~2,191 km mean error respectively. These are much smaller collections and use handcrafted features, making them not directly comparable to your deep-learning setup ([UCI Geographical Origin of Music](http://archive.ics.uci.edu/ml/datasets/geographical+original+of+music)).

### Key caveat

No published work was found that specifically uses Discogs-VI metadata for release-decade or release-market classification from audio. The Discogs-VI paper itself (Araz et al., ISMIR 2024) focuses on version identification, not era/market classification — the editorial metadata is used for dataset construction, not as classification targets ([arXiv:2410.17400](https://arxiv.org/abs/2410.17400), [GitHub: MTG/discogs-vi-dataset](https://github.com/MTG/discogs-vi-dataset)). Your task appears to be a novel repurposing of this metadata.

---

## 2. Why Task 2 (Market) Is Structurally Harder Than Task 1 (Decade)

The recurring asymmetry — where fine-tuning, ensembling, multi-crop TTA, and classifier choice all help Task 1 but not Task 2 — is consistent with findings in the MIR literature, though it has not been formally documented as a single "known phenomenon." Here is the evidence base and the most likely explanations:

### Evidence from the literature

The Técnico Lisboa thesis provides the most direct evidence. It found that audio descriptors (timbre, pitch, loudness) provided "useful information for estimating release year" but "did not provide a strong, reliable geographic signal." The thesis identified two possible explanations: (1) the features may not be suitable for the geographic task, or (2) "there may be no significant relationship between audio content and the place with which the artist is associated" ([Técnico Lisboa thesis](https://fenix.tecnico.ulisboa.pt/downloadFile/1689244997257667/resumo.pdf)).

### Hypotheses for your specific setup (ranked by testability)

**H1: Decade labels correspond to acoustically causal production changes; market labels do not.** Recording technology, mastering practices, instrumentation trends, and production aesthetics change systematically across decades — these create measurable spectral differences (dynamic range, distortion characteristics, stereo panning, compression) that audio models can detect. The "production effect" literature (Tardieu et al., DAFX 2011) explicitly proposed features for recording-era classification based on dynamic range, energy difference, and phase spread ([Tardieu 2011](http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Tardieu_2011_DAFX_Production.pdf)). Market labels (US/UK/Brazil/Spain/Germany/Italy), by contrast, are commercial/editorial categories that may not map to a single acoustic dimension — a US release and a UK release of the same 1980s pop song may be acoustically near-identical. **Confidence: high that this is the primary cause.**

**H2: Inter-class acoustic overlap is higher for 1980s markets than for decades.** In the 1980s specifically, popular music across the US, UK, Germany, and Italy shared substantial production aesthetics (synth-driven pop, gated reverb drums, FM synthesis). Brazilian 1980s music may have more distinctive features (e.g., MPB influences), but the overlap between US/UK/Germany/Italy could be very high. Test this directly: build a normalized confusion matrix on your validation set and check which class pairs are most confused. **Confidence: medium-high.**

**H3: The artist-disjoint split removes the strongest market-correlated feature (artist identity).** If certain artists are strongly associated with specific markets (e.g., Brazilian artists predominantly release in Brazil), artist-disjoint splitting removes this shortcut. The model must then rely on acoustic features alone — which, per H1, may be weak. This is analogous to the cross-cultural era paper's finding that artist-disjoint splitting prevents voice recognition shortcuts ([arXiv:2608.10980](https://arxiv.org/pdf/2608.10980)). **Confidence: high that this contributes.**

**H4: MERT's pretraining data may bias toward temporal rather than geographic features.** MERT was pretrained on 160K hours of music using EnCodec and CQT teachers. The "What Makes a Good Layer?" paper (arXiv:2608.14819) found that in masked encoders, tonal tasks peak at ~30% depth, semantic tasks at ~53% depth, and rhythm tasks at ~64% depth — but no geographic/market task was evaluated. The layer-depth pattern that works for era (your layer 4 for Task 1) may not correspond to any layer that encodes market-relevant information, because market cues may not be systematically present in MERT's pretraining targets ([arXiv:2608.14819](https://arxiv.org/pdf/2608.14819)). **Confidence: medium — this is speculative but testable by sweeping MuQ layers.**

**H5: The 102-sample validation set is too small to detect the improvements that do exist.** With 102 validation samples and 6 classes (~17 per class), a 2-percentage-point improvement (2 samples) is within noise. The techniques that "helped Task 1" may also help Task 2, but the effect is too small to measure reliably. Your observation of run-to-run variance of 1-2 Top-1 points on Task 1 supports this. **Confidence: medium — this is a statistical artifact, not a structural explanation, but it complicates interpretation.**

---

## 3. Newer / Alternative Pretrained Audio Encoders (Post-2024)

### Directly comparable music encoders

| Model | Params | Pretraining | Sample Rate | HF/GitHub | License | Why it might help Task 2 |
|---|---|---|---|---|---|---|
| **MuQ** (Mel-RVQ) | ~300M | Mel-RVQ self-supervised on MSD | 24kHz | [OpenMuQ/MuQ-large-msd-iter](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) | CC-BY-NC 4.0 | Different pretraining target (Mel-RVQ vs. MERT's EnCodec+CQT) may encode different spectral/temporal features. Trained on 0.9K hours open-source, scalable to 160K. ([arXiv:2501.01108](https://arxiv.org/abs/2501.01108), [GitHub](https://github.com/tencent-ailab/muq)) |
| **MuQ-MuLan** | ~700M | Music-text contrastive | 24kHz | [OpenMuQ/MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) | CC-BY-NC 4.0 | Music-text joint embedding; could extract text-conditioned features (e.g., "1980s Brazilian pop") and measure similarity to audio embeddings. Zero-shot capability. 98K+ downloads/month. |
| **MusicFM-MSD** | ~330M | BEST-RQ masked modeling on full MSD | 24kHz | [minzwon/musicfm](https://github.com/minzwon/musicfm) | Open | Conformer encoder (not transformer); pretrained on the entire Million Song Dataset. Documented fine-tuning recipe with discriminative LR (1e-5 encoder / 1e-4 head). ([ICASSP 2024](https://github.com/minzwon/musicfm)) |
| **Dasheng** | 0.6B-1.2B | Self-supervised on 272K hours of diverse audio | 16kHz* | [mispeech/dasheng-0.6B](https://huggingface.co/mispeech/dasheng-0.6B) | Open | General-purpose (speech + music + environmental); trained on 272K hours — far more diverse than MERT's music-only pretraining. Note: native 16kHz, requires resampling from 24kHz. |

*Dasheng's feature extractor expects 16kHz; your audio is 24kHz, so you would need to resample or check if the model handles 24kHz input. The HF page shows 64 mel-filterbanks and 101 frames for 1 second of 16kHz audio.

### Contrastive / cross-modal encoders

| Model | Params | Pretraining | HF | License | Why it might help Task 2 |
|---|---|---|---|---|---|
| **LAION-CLAP (music)** | ~73M | Contrastive audio-text on music pairs | [laion/larger_clap_music](https://huggingface.co/laion/larger_clap_music) | Open | Music-specific CLAP checkpoint. Could provide complementary embeddings via `model.get_audio_features()`. SWIN-transformer audio encoder, different architecture from MERT. |
| **Myna-Base** | 22M | Contrastive masked-view learning | Not independently confirmed | — | ViT-S/16 architecture; treats paired masked views of the same track as positives. Small model — low overfitting risk on small data. Listed in the layer-analysis paper. |

### Audio language models (as feature extractors, not generators)

| Model | Params | Notes |
|---|---|---|
| **Audio Flamingo 3** | 7B | Uses AF-Whisper encoder (Whisper-based, trained on audio-caption pairs). Already your best Task 2 result (58.8% zero-shot). The AF-Whisper encoder itself could be used as a frozen feature extractor. ([arXiv:2507.08128](https://arxiv.org/html/2507.08128), [GitHub](https://github.com/NVIDIA/audio-flamingo)) |
| **Qwen2-Audio-7B** | 7B | Audio encoder + LLM. You already tested it (33% Task 1). The audio encoder (128-channel mel-spectrogram at 16kHz) could potentially be extracted separately, but this requires custom model surgery. ([arXiv:2407.10759](https://arxiv.org/html/2407.10759v1)) |

### Recommendation

**Prioritize MuQ and MuQ-MuLan** — they are the most directly comparable to MERT (music-specific, 24kHz native, similar parameter count) but use a fundamentally different pretraining objective (Mel-RVQ vs. EnCodec+CQT). The layer-analysis paper found that MuQ and MERT have different layer-depth profiles, suggesting they encode different information. If MuQ's frozen-probe performance on Task 2 is comparable to or better than MERT's, ensembling the two could capture complementary features.

**Second priority: MusicFM-MSD** — Conformer architecture and documented fine-tuning recipe with discriminative LR.

**Third: LAION-CLAP-music** — Different modality (contrastive, not masked), small model, low overfitting risk, but only 73M params.

---

## 4. Narrowing the Gap Between Your Trained Pipeline and Audio Flamingo 3 on Task 2

Your AF3 zero-shot result (58.8% Top-1) beats your best trained pipeline (47.1%) by ~12 points. Here are concrete methods to close this gap, ranked by effort:

### Low effort: AF3 probability vectors as stacker features

**Method**: Run AF3 on your training set using constrained class-probability scoring (teacher-forced label-probability, as you already do for evaluation). Extract the 6-element probability vector for each training clip. Concatenate these 6 features with your MERT embeddings (or use them as standalone features). Train a logistic regression or SVM on the combined feature set.

**Why this works**: AF3's probability vector encodes its "opinion" about market membership, which is apparently well-calibrated (58.8% accuracy). Even if AF3's individual predictions are noisy, the probability vector as a soft feature can provide complementary signal to MERT's acoustic embeddings. This is a form of model stacking / late fusion at the feature level.

**Expected gain**: If AF3 and MERT make uncorrelated errors, stacking typically yields 2-5 points improvement over the best single model. Your Task 1 ensemble already demonstrates this principle (52.3% vs. 50.0% single-model best).

**Critical detail**: Use constrained scoring (force the model to assign probabilities to your 6 market labels), NOT free-form generation. You already noted that free-form generation has a 27.5% invalid-output rate on Task 2 with Qwen2-Audio. The same risk applies to AF3.

### Low effort: Validation-tuned late fusion

**Method**: Compute AF3's 6-class probability vector and your MERT probe's 6-class probability vector for each validation sample. Learn a weight vector (w_af3, w_mert) that maximizes validation Top-1 accuracy via grid search or logistic regression on the concatenated probabilities.

**Expected gain**: Similar to stacking, but simpler. If AF3's accuracy is 58.8% and MERT's is 47.1%, and their errors are partially uncorrelated, optimal fusion could reach ~60-63%.

### Medium effort: Soft-label distillation from AF3

**Method**: Use AF3's class-probability outputs on the training set as soft labels. Train your MERT-based classifier with a combined loss: `L = α * CE(y_true, p) + (1-α) * KL(p_af3 || p)` where `p` is your model's output, `y_true` is the gold label, `p_af3` is AF3's probability vector, and `α` controls the balance (try 0.5-0.8). Use temperature scaling (T=2-4) on both the AF3 soft labels and your model's softmax to soften the distributions.

**Why this works**: AF3 has "knowledge" about market membership that MERT lacks (perhaps from its large-scale audio-language pretraining). Distillation transfers this knowledge into your smaller, faster model. The KL divergence term encourages your model to match AF3's probability distribution, not just the hard label — which is especially valuable when the gold labels are noisy or when classes are confusable.

**Reference**: Multi-representation knowledge distillation for audio classification (Gao et al., Multimedia Tools and Applications, 2021) demonstrates KD for audio classification, though not specifically for music market classification ([ACM DL](https://dl.acm.org/doi/10.1007/s11042-021-11610-8)). The DAGA 2025 paper on distilling efficient audio models with data pruning is also relevant ([DAGA 2025](https://pub.dega-akustik.de/DAS-DAGA_2025/files/upload/paper/149.pdf)).

### Medium effort: AF-Whisper encoder as feature extractor

**Method**: Audio Flamingo 3 uses AF-Whisper, a unified audio encoder pretrained on large-scale audio-caption pairs using a novel strategy. If the AF-Whisper encoder weights are accessible (the model is open-weight on [HuggingFace](https://huggingface.co/nvidia/audio-flamingo-3-hf)), extract its hidden states as features and probe them the same way you probed MERT.

**Why this works**: AF-Whisper was trained on speech, sounds, AND music with a general-purpose pretraining objective — potentially capturing a broader range of acoustic features than MERT's music-only pretraining. This is the encoder that produced the 58.8% result; its internal representations may be even more discriminative than its output probabilities.

**Risk**: AF-Whisper is part of a 7B-parameter model; loading just the encoder may require custom model surgery. The HF model page should have the encoder accessible as a submodule.

### Higher effort: Prompt-based feature extraction from AF3

**Method**: Instead of asking AF3 to classify directly, use it to generate structured descriptions of each audio clip (e.g., "Describe the production style, instrumentation, and likely era of this music"). Encode these descriptions using a text encoder (e.g., sentence-transformers) and use the resulting text embeddings as additional features for your classifier.

**Why this works**: AF3's natural-language descriptions may capture market-relevant cues (e.g., "Latin percussion," "European synth-pop production") that are difficult to extract from raw audio alone. This is a form of "ALM as a feature extractor via language."

**Risk**: Reproducibility — free-form generation is non-deterministic and may drift (as you observed with Qwen2-Audio). Use constrained or structured prompting to mitigate this.

---

## 5. Overfitting Mitigation for Fine-Tuning a 330M Encoder on ~800-1000 Clips

### Direct evidence from the PETL paper (arXiv:2411.19371)

The most directly relevant study is "Parameter-Efficient Transfer Learning for Music Foundation Models" (Huang et al., 2024), which systematically compared full fine-tuning, probing, and 6 PEFT methods on MERT and MusicFM across music classification tasks. Key findings:

**On MERT specifically:**
- Probing (frozen encoder + MLP head) **outperformed full fine-tuning** on music classification tasks. The stated reason: "fine-tuning often leads to overfitting."
- Full fine-tuning of MERT on GTZAN genre classification: 63.8% accuracy (vs. probing: 75.6%).
- LoRA (rank 2): 74.7% on GTZAN genre — close to probing, with only 165K trainable parameters (0.22% of MERT).
- Adapter (bottleneck 16): 72.8% on GTZAN genre.
- BitFit (bias-only): 73.8% on GTZAN genre.
- SSF (scale-and-shift): 74.8% on GTZAN genre.
- All PEFT methods used MLP heads with **dropout 0.5**.

**Key insight**: "Increasing dataset size reduces the performance gap between fine-tuning and probing." On your ~800-1000 clip training set, this strongly suggests that **probing should outperform full fine-tuning** — which is exactly what you observed on Task 2 (frozen probe: 47.1% vs. full fine-tune: 44.1%).

([arXiv:2411.19371](https://arxiv.org/html/2411.19371v1))

### Recommended PEFT recipe (in order of priority)

**1. LoRA (rank 4-8) with frozen base weights** — The PETL paper used rank 2; for your 6-class task with ~800 training samples, rank 4-8 gives slightly more capacity. LoRA introduces no inference overhead (weights are merged). On AST-based audio classification, LoRA achieved 84.97 average accuracy vs. 87.48 for full fine-tuning — a smaller gap than on MERT, and with 221K vs. 85.5M trainable parameters ([OpenReview PEFT paper](https://openreview.net/pdf?id=6x8ULeOAMS)).

**2. Adapters (bottleneck 16, Pfeiffer config)** — Inserted parallel to MHA layers. The PETL paper found adapters slightly outperformed LoRA on auto-tagging. The MoA paper found Soft-MoA (mixture of adapters) outperforms single adapters with limited parameters, but this adds complexity ([OpenReview](https://openreview.net/pdf?id=6x8ULeOAMS)).

**3. BitFit (bias-only fine-tuning)** — Only ~50K trainable parameters on MERT. Simplest PEFT method; surprisingly competitive (73.8% on GTZAN genre). Good as a quick baseline.

**4. SSF (Scaling and Shifting Features)** — Trainable scale/shift after each layer; ~83K parameters. Best GTZAN result among PEFT methods (74.8%). Merges into pretrained weights for inference.

### Layer-wise learning rate decay (LLRD)

Apply exponentially decaying learning rates across transformer layers: the top layer gets the full LR, each lower layer gets LR × decay_factor^(depth_from_top). This preserves pretrained features in lower layers (which you've already identified as optimal for your tasks — layer 4 for Task 1, layer 7 for Task 2) while allowing higher layers to adapt.

**Concrete recipe**: lr_head = 1e-4, lr_layer_i = 1e-4 × 0.95^(24-i) for MERT-v1-330M's 24 layers. The MusicFM repository recommends 1e-5 for the foundation model and 1e-4 for probing layers — a 10× ratio that effectively implements LLRD between encoder and head ([GitHub: minzwon/musicfm](https://github.com/minzwon/musicfm)). The general LLRD concept is well-documented for ViT fine-tuning ([Layer-wise LR Decay reference](https://aicassindra.com/blogs/transformer_math/tm_layer_decay.html)).

### Augmentation (you used none — this is a clear gap)

**SpecAugment** — Time masking and frequency masking on the log-mel spectrogram. Standard in speech and audio classification; the PETL paper's data augmentation section used pitch shifts (±4 semitones) for key detection and time-stretching (0.95–1.05) for tempo estimation. For your era/market tasks, pitch shifting and time-stretching are risky — they may distort the very production cues (pitch, tempo) that define the era/market. Stick to:
- **Time masking** (mask 5-10% of time steps)
- **Frequency masking** (mask 5-10% of mel bins)
- **Mixup** (α=0.2-0.4, blend two samples and their labels) — SpecMix is an audio-specific variant that preserves spectral correlation better than vanilla mixup ([SpecMix, Interspeech 2021](https://www.isca-archive.org/interspeech_2021/kim21c_interspeech.pdf))
- **Random gain** (±6 dB)
- **Additive noise** (low-level white or environmental noise)

**Caution for Task 2**: Pitch shifting and time-stretching should be used with care, as they may alter market-relevant production characteristics. For era classification (Task 1), the cross-cultural paper found that mel-spectrogram CNNs respond to "tape noise, compression, mastering, harmony, instrumentation" — augmentations that preserve these cues (time/frequency masking, gain) are safer than those that alter them (pitch, tempo) ([arXiv:2608.10980](https://arxiv.org/pdf/2608.10980)).

### Partial unfreeze schedule

Instead of full fine-tuning or full freezing, gradually unfreeze layers during training:
1. **Phase 1** (epochs 1-3): Freeze all encoder layers, train only the classification head.
2. **Phase 2** (epochs 4-6): Unfreeze the top 4-6 transformer layers.
3. **Phase 3** (epochs 7-10): Unfreeze all layers with LLRD.

This is a standard transfer-learning warmup that reduces overfitting by letting the head converge before the encoder starts adapting. The PETL paper's finding that probing > fine-tuning on small data suggests Phase 1 alone may be sufficient for your dataset size.

### Additional regularization

- **EMA (Exponential Moving Average)** of model weights — maintains a slowly-updated copy of weights for evaluation, smoothing out training noise.
- **SWA (Stochastic Weight Averaging)** — averages weights from multiple epochs, reducing variance.
- **Stronger weight decay** (0.01-0.1) — the PETL paper used dropout 0.5 on the MLP head; consider also weight decay on the encoder.
- **Early stopping on macro Top-1/Top-3** — you already do this; continue.

---

## 6. Current (2025-2026) Practice for Small-Data Artist-Disjoint Audio Classification

### Supervised contrastive learning for era classification

The Music Era Recognition paper (arXiv:2407.05368) proposes supervised contrastive learning (Audio-SUC) with an "era contrastive (EC) loss" that pulls same-era representations together and pushes different-era representations apart. This is directly applicable to your Task 1: add a contrastive loss term to your MERT fine-tuning objective. The paper also found that adding artist biography text (AudioArt-MMC) improved results — if you have access to artist metadata (even anonymized), text features could complement audio features ([arXiv:2407.05368](https://arxiv.org/html/2407.05368v1)).

### Layer selection without exhaustive sweeps

The "What Makes a Good Layer?" paper (arXiv:2608.14819) provides proxy metrics for selecting the best MERT layer without training probes on all 25 layers:
- **Intrinsic dimension (ID)**: highest mean correlation for non-tonal layer quality (ρ = 0.76).
- **LiDAR × ID**: strongest semantic proxy in masked models (ρ = 0.78).
- **Curvature**: negatively correlated with downstream performance for rhythm tasks.
- **PTE (Pitch Transposition Equivariance)**: consistent tonal-task proxy (ρ = 0.80-0.87).

Top-3 proxy-ranked layer selection matches the oracle on 58% of model-task pairs, with a mean gap of only 0.4 percentage points. This could save you the 25-layer sweep you performed. The paper also found that for tasks with <1,000 training clips, every trainable fusion baseline has a mean gap of at least 5.6 percentage points — meaning simple layer selection is better than complex fusion for small data ([arXiv:2608.14819](https://arxiv.org/pdf/2608.14819)).

### Multi-branch LoRA for small-data audio transformers

The DCASE 2024 anomalous sound detection paper (arXiv:2508.12230) proposes Fully-Connected Multi-Branch LoRA: multiple parallel LoRA branches (best: 8 branches, rank 32) with connections between branches. On BEATs (90M params), this outperformed full fine-tuning (65.15 vs. 63.42 hmean) while keeping original weights frozen. The paper also found that **freezing original weights helps retain pre-trained knowledge and reduce overfitting** — consistent with the PETL paper's findings on MERT ([arXiv:2508.12230](https://arxiv.org/html/2508.12230v1)).

### Hierarchical classification with consistency loss

The cross-cultural era paper (arXiv:2608.10980) jointly predicts 4 hierarchical temporal levels (decade, half-decade, quarter-decade, year) with independent softmax heads and a hierarchical consistency loss penalizing parent-child violations. Your hierarchical experiment (coarse 3-way → fine 2-way) showed better ordinal agreement but lower raw accuracy — the consistency loss may help recover the accuracy gap while preserving the ordinal smoothness ([arXiv:2608.10980](https://arxiv.org/pdf/2608.10980)).

### Stems as complementary inputs

Your Demucs experiment (mixture 47.1% > vocals-only 40.2% > accompaniment-only 37.3%) suggests stems contain complementary information. Consider:
- **Multi-input ensemble**: train separate probes on mixture, vocals, and accompaniment; ensemble their probability outputs.
- **Feature concatenation**: extract MERT embeddings separately for each stem and concatenate.
- **Demucs 4-source separation** (htdemucs_ft): vocals, bass, drums, other — each may carry different market/era cues.

### Repeated StratifiedKFold for model selection

You noted that artist IDs are unavailable within the training split, preventing artist-grouped k-fold. Repeated StratifiedKFold (e.g., 5-fold × 5 repeats) is the best available alternative. It doesn't solve artist leakage within folds (artists may appear in both train and validation folds), but it reduces the variance of model selection compared to a single split. Report this as a known limitation.

### Addressing the 2010s bias

The cross-cultural era paper found that the 2010s class had the lowest accuracy across all architectures (28.9%–50.9%), attributed to "rapid diversification of genres" and "greater diversification of production techniques." Your AF3 result (never predicts "2010s" as top choice) mirrors this. If your Task 1 has similar issues, consider:
- Class-weighted loss to upweight the 2010s.
- Oversampling the 2010s during training.
- Checking whether your 2010s training clips are stylistically diverse (which would make the class harder to learn).

---

## Confidence Summary

| Finding | Confidence | Basis |
|---|---|---|
| Cross-cultural era paper's 67-71% macro accuracy is the most comparable benchmark for Task 1 | High | Directly fetched, verified artist-disjoint split |
| Geographic prediction is structurally harder than temporal (H1) | High | Supported by Técnico Lisboa thesis finding of "poor" geographic performance |
| Probing should outperform full fine-tuning on your dataset size | High | PETL paper directly tested this on MERT with similar task structure |
| MuQ may capture different features than MERT due to different pretraining | Medium | Different pretraining target is confirmed; performance gain is speculative |
| AF3 probability vectors as stacker features will improve Task 2 | Medium-High | Stacking principle well-established; specific gain depends on error correlation |
| Layer-wise LR decay will reduce overfitting | Medium | Standard practice in vision; less directly tested on MERT |
| SpecAugment/mixup will help small-data fine-tuning | Medium-High | Standard in audio classification; specific gain on your task is untested |
| Supervised contrastive learning could help Task 1 | Medium | Proposed in era-recognition paper; not tested on your specific dataset |
| AF-Whisper encoder extraction is feasible | Low-Medium | Model is open-weight; technical feasibility of encoder-only extraction unconfirmed |

---

## Suggested Next Experiment Queue

1. **MuQ frozen-probe sweep** (all 12 layers × 3 classifiers) on both tasks — ~2 hours of compute, directly testable.
2. **AF3 probability vectors as stacker features** on Task 2 — requires running AF3 on training set with constrained scoring, then training logistic/SVM on combined features.
3. **LoRA fine-tuning** (rank 4-8) of MERT-v1-330M with SpecAugment + mixup + LLRD, both tasks.
4. **Validation-tuned late fusion** of AF3 + MERT probe probabilities on Task 2.
5. **Normalized confusion matrix analysis** on Task 2 validation set — identify which market pairs are most confused.
6. **MusicFM-MSD frozen-probe sweep** on both tasks — different architecture, may capture complementary features.
7. **Soft-label distillation** from AF3 to MERT-based classifier on Task 2 (if steps 1-6 don't close the gap).
8. **Multi-stem ensemble** (mixture + vocals + accompaniment probes, probability-averaged).
