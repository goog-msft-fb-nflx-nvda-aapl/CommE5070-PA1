# Round 5 — Lecture-Grounded Independent Research
## CommE5070 PA1: Release-Decade (Task 1) and Release-Market (Task 2) Classification

Sources surveyed: Lecture 02 "Music Classification" (101 slides), Lecture 02b "Music Foundation Models" (29 slides), Lecture 03 "Language & Music" (86 slides), all by Yi-Hsuan Yang, 2026 edition. Every lecture-cited method was cross-checked against the project's "already tried" list (Round-4 prompt), and the not-yet-tried items were verified against their live model cards / repos (Sept 2026). **Every model, dataset and tool mentioned is linked in §15 (Link index)**; key items are also linked inline at first use.

---

## 0. Executive summary

The lectures contain a small number of methods you have not tried that are genuinely new to this project. Ranked by expected value per unit of effort:

| Rank | Idea | Task | Lecture source | Status vs. prior rounds | Effort |
|---|---|---|---|---|---|
| 1 | **MERT-v2-30s frozen probe** (and as a new fine-tuning backbone) | Both | L02b slides 24–26 | New. Not in Round-4 report | Low |
| 2 | **Learned softmax-weighted sum of all layers** as the probe input | Both | L02b slide 19 | New (you swept single layers) | Very low |
| 3 | **LoRA fine-tuning of AF3 itself** (LLM LoRA + projector), label as the target | Task 2 | L03 slides 80, 83 | New (LoRA was only tried on MERT) | Medium–high |
| 4 | **Retrieval-augmented in-context learning with AF3** (exemplars chosen by MuQ kNN) | Task 2 | L03 slides 80–81 | New (only zero-shot / CoT considered) | Medium |
| 5 | **"Whisper-lineage encoder" hypothesis test** via MARBLE's Qwen2.5-Omni encoder | Task 2 | L02b slide 29 | Partly overlaps Round-4 B1/A1; reframed as one diagnostic | Low |
| 6 | **CLaMP 3 audio embedding as a frozen probe** | Task 2 (and 1) | L02b slide 29 | Contrastive family tried (CLAP), but not a metadata-trained one; different use | Low |
| 7 | **PupuJEPA-Large** as a fusion member (2D-patch JEPA, a new pretraining family) | Both | L02b slides 25, 27 | New | Low–medium |
| 8 | **GTZAN-style hand-crafted features in a "wide & deep" probe** | Task 1 | L02 slides 25–37, 47 | Partly overlaps Round-4 A3 (loudness only) | Low |
| 9 | **Caption-as-features** (AF3 / LP-MusicCaps captions → text embedding → probe) | Both | L03 slides 15, 22–25, 33–34 | New | Medium |
| 10 | **Label-aware augmentation ablation** for the from-scratch Short-Chunk CNN | Task 1 | L02 slides 87–91 | New | Low–medium |

The two most important observations from the lectures are these.

First, **MERT-v2-30s** (listed on L02b slide 24–25) is almost certainly the single highest-value untried encoder. It is native 24 kHz mono, trained on 30-second excerpts (exactly your clip length), 632M parameters, and its model card reports frozen-probe GTZAN genre accuracy of 91.72 versus 83.8 for MuQ and 77.6 for MERT-Large, and MTG-Jamendo genre ROC-AUC of 88.01 versus 85.4 for MuQ. Your best frozen encoder so far (MuQ) is now clearly second-tier on the genre-type tasks most correlated with decade and market.

Second, the lecture's list of four ways to use an audio-language model (L03 slide 80: zero-shot, zero-shot CoT, in-context learning, fine-tuning) exposes that the project has only ever used the first. Since AF3 zero-shot is the ceiling on Task 2, the two unused modes that operate *on AF3 itself* (ICL and LoRA fine-tuning) are the most direct routes past 58.8%.

---

## 1. Coverage audit: lecture content vs. what the project has done

### Lecture 02 — Music Classification

| Lecture item | Tried? | Notes |
|---|---|---|
| Single-label softmax + CE, Top-1/Top-3 | Yes | |
| Hand-crafted timbral/rhythm/pitch features (GTZAN; spectral centroid, rolloff, contrast, flux, MFCC, beat histogram, pitch histogram) | **No** | Round-4 proposed loudness/crest factor only. See §8 |
| Wide & deep (hand-crafted + learned features concatenated), slide 47 | **No** | See §8 |
| CQT / chromagram inputs, slides 49–58 | **No** | All CNN work used log-mel. See §8.3 |
| STFT parameter hygiene (window/hop matched to task), slides 60–64 | Unknown | Worth a one-line check of the Short-Chunk CNN front end at 24 kHz (§8.3) |
| 1D / 2D / sample-level CNNs, CRNN | Partly | Short-Chunk CNN only |
| Short-Chunk CNN | Yes | Part of Task 1 ensemble |
| Data augmentation (torchaudio_augmentations, degradation toolbox, pyrubberband) | **No (not reported)** | See §10 |
| PANNs / AST / HTS-AT pretrained | Indirectly | HTS-AT is the CLAP audio tower, tried only in zero-shot mode |
| Artist-split, confusion matrix diagnostics | Yes | |

### Lecture 02b — Music Foundation Models

| Lecture item | Tried? | Notes |
|---|---|---|
| SimCLR / CLMR contrastive SSL, linear probe on *h* not *z* | No | Too little in-domain data for pretraining; see §11 |
| MERT v1 (95M/330M), layer-wise probing | Yes | |
| **MERT "finetuning strategy": learnable softmax-weighted sum of all layers** (slide 19) | **No** | You swept single layers. See §3 |
| MARBLE benchmark / MARBLE toolkit | No | Useful as infrastructure; §5 |
| BEATs | No | General-audio; low priority |
| MuQ | Yes | Current best frozen encoder |
| **MERT-v2-30s / MERT-v2-FullSong** (slides 24–26) | **No** | §2 — top recommendation |
| Dasheng-1.2B | No (proposed in Round 4) | Lecture table shows it below MuQ on genre (81.4 vs 83.8) — deprioritise |
| **PupuJEPA** (slides 25, 27) | **No** | §7 |
| Music-JEPA | Not applicable | Piano world model |
| **CLaMP 3, MuQ-MuLan, Qwen2.5-Omni encoder** (slide 29 table) | **No** | §5, §6 |
| MusicFM | Yes | Neutral |

### Lecture 03 — Language & Music

| Lecture item | Tried? | Notes |
|---|---|---|
| CLAP (LAION) zero-shot | Yes | Clear negative |
| Microsoft CLAP, MuQ-MuLan, MusicCoCa | No | MuQ-MuLan/MusicCoCa probed as embeddings: low priority (§6) |
| LP-MusicCaps / captioning; captions as training data (slides 22–25, 33–34) | **No** | §9 |
| LLark, SALMONN | No | LLark has no public checkpoint; SALMONN is a weaker general ALM |
| Qwen2-Audio, AF3 zero-shot | Yes | AF3 is the Task-2 ceiling |
| **Zero-shot CoT** (slide 80) | No (proposed in Round 4) | |
| **In-context learning with (audio, label) exemplars** (slides 80–81) | **No** | §4 |
| **Fine-tuning the ALM: LoRA on LLM + full projector** (slides 80, 83) | **No** | §3b — the lecture's own example (LLM2Fx-Tools) uses exactly this recipe |

---

## 2. MERT-v2-30s — the top recommendation (both tasks)

**What it is.** [`m-a-p/MERT-v2-30s`](https://huggingface.co/m-a-p/MERT-v2-30s) (HF, `trust_remote_code=True`, CC BY-NC 4.0 — fine for coursework). 24 kHz mono, 632M params, 24 Conformer layers, 1024-dim, 25 Hz frame rate, bidirectional, pretrained on 30-second excerpts. A continuation, [`m-a-p/MERT-v2-FullSong`](https://huggingface.co/m-a-p/MERT-v2-FullSong), exists with the same interface. Technical report is "coming soon"; cite MERT (ICLR 2024) + MARBLE (NeurIPS 2023) per the model card.

**Why it is mechanistically new relative to what you tried.** Per the architecture figure on L02b slide 24, MERT-v2's masked-prediction targets are RVQ codes synthesized offline from a *concatenation of MuQ layer-7 features and Qwen2-Audio layer-32 features*. That is, it is partly a distillation of an audio-language model's deep (semantic) layer into a music encoder. None of your encoders has that property. Qwen2-Audio was weak as a zero-shot classifier for you, but its layer-32 hidden states can carry information its text head does not verbalise well; MERT-v2 packages that into a probe-friendly encoder.

**Why it should suit your data specifically.**
Your clips are 30 s at 24 kHz: MERT-v2-30s is trained at exactly that duration and rate, so no resampling and no 5 s windowing (MERT-v1 used 5 s context — L02b slide 18 — which is a plausible reason multi-crop TTA helped MERT-v1 but hurt the longer-context MuQ).
Reported frozen-probe MARBLE numbers from the model card (same table reproduced on L02b slide 25):

| Model | Params | GTZAN genre acc | MTG genre ROC | MTG top-50 ROC | EMO arousal R² |
|---|---|---|---|---|---|
| MERT-v1-330M ("MERT-Large") | 330M | 77.6 | 86.1 | 82.6 | 76.1 |
| MuQ | 310M | 83.8 | 85.4 | 83.0 | 76.4 |
| MusicFM | 330M | 84.1 | 85.3 | 81.9 | 74.4 |
| PupuJEPA-Large | 307M | 86.9 | 86.1 | 82.8 | 76.8 |
| **MERT-v2-30s** | 632M | **91.72** | **88.01** | **84.18** | **80.01** |

Caveats: (1) the baselines in that table were copied from the PupuJEPA paper, not rerun, and protocols may differ; (2) GTZAN has known artist/recording leakage, so a 8-point GTZAN margin will not translate one-to-one to an artist-disjoint task. The MTG-Jamendo margins (+2–3 ROC points) are the more trustworthy signal. Still, every column favours MERT-v2.

**Protocol.**
1. Extract all 24 `hidden_states` once per clip (mean-pooled with the returned `feature_attention_mask`, as in the model card snippet). Store to disk; ~1,300 clips × 24 × 1024 floats is small.
2. Run your existing layer sweep × {LogReg, SVM, MLP} × PCA ablation, but inside 5-fold artist-grouped CV on the training split (see §12). The model card's own layer guide suggests where to look first: GTZAN genre → L23 (probe LR 5e-3); MTG genre → L19; MTG mood → L16; EmoMusic → all-layer MLP.
3. Also run the learned weighted-layer-sum probe (§3).
4. If the frozen probe beats MuQ in CV log-loss, try it as the backbone for the full fine-tune that currently anchors your Task 1 ensemble (replacing MERT-v1-330M). Memory is ~2× MERT-v1-330M; use gradient checkpointing and bf16.

**Expected outcome.** The most likely single change to move both tasks' frozen-probe numbers. For Task 2, it may also close part of the gap to AF3 because of the Qwen2-Audio-derived targets (Qwen2-Audio uses a Whisper-large-v3 encoder, which relates to the hypothesis in §5).

---

## 3. Two cheap probe-level changes the lectures prescribe

### 3a. Learned softmax-weighted sum of all layers (L02b slide 19)

The lecture's MERT "finetuning strategy" is: freeze the backbone, learn `z = Σ_l softmax(w_l) · h_l` over all layers, then a single linear layer. You have instead chosen one layer by sweep. The weighted sum is a different estimator: it adds only L scalars (24 for MERT, 13 for MuQ) and lets the probe blend, e.g., a mid layer carrying production/timbre with a top layer carrying style. With ~1,000 training clips, it is also less vulnerable to the winner's-curse of picking the best of 24 layers by validation score (which inflates the apparent validation accuracy of a single-layer choice).

Implementation: a 20-line PyTorch module on cached features; L2 on the linear weights, no regularization on `w_l` (or mild entropy regularization toward uniform). Train with CE (or SORD soft targets for Task 1, per Round 4 C1). Compare in the same CV folds as the single best layer.

### 3b. What the lecture says about fine-tuning *vs.* probing

L02b slide 19 frames linear probing as the default adaptation, and L03 slide 80 frames fine-tuning as the fourth and heaviest option. Your LoRA-on-MERT result (below frozen probing) is consistent with the common small-data finding that adapting an SSL encoder with ~800 labels rarely beats a good frozen probe. The important distinction the lecture draws is *which* model you adapt: LoRA on an SSL encoder has to create the task knowledge; LoRA on an ALM that already scores 58.8% zero-shot only has to calibrate existing knowledge. That motivates §3c.

### 3c. LoRA fine-tuning of AF3 (Task 2) — L03 slides 80, 83

**Recipe (mirrors the lecture's LLM2Fx-Tools example on slide 83; Doh et al., ICLR 2026, [arXiv:2512.01559](https://arxiv.org/abs/2512.01559), code [github.com/SonyResearch/LLM2Fx](https://github.com/SonyResearch/LLM2Fx)):** LoRA on the Qwen2.5-7B backbone, full fine-tune of the audio adaptor/projector, frozen AF-Whisper. Training target: the prompt you already use for teacher-forced scoring, with the correct market label as the single answer token. Score at inference exactly as now (label log-probabilities), so the output is directly comparable to zero-shot AF3.

**Why this is not a repeat of the closed LoRA line.** Your five LoRA runs adapted MERT (an encoder with no task prior) and plateaued at 38–43% on Task 2. Here the adapted model starts at 58.8%, so the objective is correcting a strong but miscalibrated prior (e.g., your confusion analysis shows US/UK/DE confusions). With 798 examples, 1–3 epochs of LoRA rank 8–16 is the realistic regime.

**Guardrails (to avoid another 65.7%-style mirage).**
Fit inside 5-fold artist-grouped CV on the training split; choose epochs and LR on OOF log-loss; evaluate on validation once. Use 4-bit base weights (QLoRA) to fit a single 24 GB GPU; 30-s audio at AF-Whisper's frame rate fits comfortably. Freeze AF-Whisper to keep the parameter count small. Also try the minimal variant: train *only* the projector (no LLM LoRA), which is the lowest-capacity adaptation that can still re-map audio evidence.

**Cost/risk.** The highest-effort item here, and the transformers-compatibility work you did for MuFun suggests budgeting debugging time. But it is the only lecture-sanctioned method that acts directly on the model that defines your Task 2 ceiling.

---

## 4. Retrieval-augmented in-context learning with AF3 (Task 2) — L03 slides 80–81

L03 slide 80 defines ICL as prepending (audio, text) exemplars before the query, with no gradient updates; slide 81 cites a 2026 arXiv paper applying textual and multimodal ICL to music valence/arousal with LLMs (I could not locate the arXiv ID for this paper; get it from the lecturer or the slide before citing it).

**Proposed variant, which combines two things you already have in a new way:** for each query clip, retrieve its k nearest training clips in your best frozen-probe embedding space (MuQ now, MERT-v2 if §2 succeeds), and prepend them as labelled exemplars ("This recording was released in Brazil.") before the scored question. Constrain k so that audio length stays within AF3's context: k = 3–6 exemplars trimmed to 10 s each (your own finding that a 10-s crop works as well as 30 s on Task 2 justifies the trim).

**Why this is mechanistically different from your fusion attempts.** Probability-level fusion combined AF3's output with the probe's output after the fact, and the OOF-fitted weight collapsed to AF3. Retrieval-ICL feeds the probe's *neighbourhood structure* into AF3's reasoning, so AF3 can compare the query against acoustically similar, labelled examples — the information enters before the decision rather than being averaged with it.

**Controls needed to interpret the result.** (i) random exemplars (one per class) vs. nearest-neighbour exemplars, to separate "ICL helps" from "retrieval helps"; (ii) nearest-neighbour exemplars with shuffled labels, to check AF3 is using the labels and not just the extra audio; (iii) a pure kNN vote over the same neighbours with no AF3, as the non-ALM baseline. Artist-disjointness still holds because exemplars come only from the training split.

**Feasibility check first.** Confirm that the HF AF3 processor accepts multiple audio items in one conversation turn (AF3-Chat is advertised as multi-turn/multi-audio; verify in the processor code before building on it).

---

## 5. Diagnostic: is AF3 winning Task 2 because of its *Whisper-lineage encoder*?

This is the single most useful hypothesis to test, because it explains your whole Task 2 picture. Every music-SSL encoder you tried (MERT, MuQ, MusicFM, CultureMERT) tops out at 47–52% on Task 2, while AF3 reaches 58.8%, and Brazil (the one market with a distinct sung language) is almost perfectly separated. AF3's encoder (AF-Whisper) and Qwen2-Audio's encoder both descend from Whisper-large-v3, which was trained for multilingual speech recognition and therefore encodes phonetic/language identity; music SSL encoders are not trained to preserve that.

**Test (low cost via the lecture's MARBLE toolkit, [github.com/a43992899/MARBLE](https://github.com/a43992899/MARBLE), L02b slide 29).** MARBLE's encoder list includes a `Qwen2_5OmniEncoder` (Qwen2.5-Omni audio tower, [arXiv:2503.20215](https://arxiv.org/abs/2503.20215); [github.com/QwenLM/Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni); HF [`Qwen/Qwen2.5-Omni-7B`](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)) alongside MERT, MuQ, MusicFM, Dasheng, CLaMP 3 and MuQ-MuLan. Probe the Omni audio tower (and, if you do Round-4 B1, AF-Whisper) with the same probe as MuQ, on both tasks.

**How to read the result.** If the Whisper-lineage encoder is at or above AF3's 58.8% on Task 2 but *not* competitive on Task 1, the Task 2 bottleneck is language/phonetic information, and the right next steps are Round-4 A1 (explicit sung-language ID) and B4 (hierarchical Romance-vs-Anglo classifier). If it is also weak on Task 2, AF3's advantage lies in the LLM's world knowledge, which argues for §3c/§4 instead. Either answer tells you where to spend the remaining budget, which is why this is ranked above more speculative encoders.

Overlap note: Round 4 proposed probing AF-Whisper (B1) and Whisper LID (A1) as improvements; this section re-uses them as a *controlled diagnostic* across encoder families, which is new.

---

## 6. CLaMP 3 and the other text–audio models (L02b slide 29, L03 slides 42–53)

### CLaMP 3 — worth one run on Task 2

[`sander-wood/clamp3`](https://huggingface.co/sander-wood/clamp3) (HF) / [github.com/sanderwood/clamp3](https://github.com/sanderwood/clamp3); [arXiv:2502.10362](https://arxiv.org/abs/2502.10362), ACL 2025. Contrastively aligns sheet music, MIDI, audio and multilingual text. Two facts make it relevant here and distinguish it from the LAION-CLAP negative:

1. Its training text is music metadata: the [M4-RAG](https://huggingface.co/datasets/sander-wood/m4-rag) corpus of 2.31M music-text pairs across 27 languages and 194 countries. Country/region and era are exactly the kind of fields it was trained to align with audio; LAION-CLAP was trained mostly on sound-event captions.
2. The model card explicitly lists zero-shot classification by genre and region.

Its audio tower consumes MERT features (with a 640-second context), so it is effectively a metadata-supervised head on top of MERT-v1.

**Recommended use:** probe CLaMP 3's *audio embedding* with your standard probe (the mechanism that works for you), rather than repeating contrastive zero-shot (the mechanism that failed). Zero-shot with country prompts ("a 1980s Brazilian release") is a cheap secondary check. Overlap note: this is in the contrastive family you tried, but the difference in training text (metadata vs. sound captions) is the specific thing worth testing.

### MuQ-MuLan, MusicCoCa, Microsoft CLAP — low priority

MuQ-MuLan ([`OpenMuQ/MuQ-MuLan-large`](https://huggingface.co/OpenMuQ/MuQ-MuLan-large)) is MuQ further trained contrastively on text; its audio embedding will be strongly correlated with MuQ, which you already have. MusicCoCa (Magenta RealTime, [github.com/magenta/magenta-realtime](https://github.com/magenta/magenta-realtime)) operates on 10-s slices at 16 kHz and is designed for generation conditioning (L03 slide 50). Microsoft CLAP ([github.com/microsoft/CLAP](https://github.com/microsoft/CLAP)) is trained on general-audio captions (L03 slide 45), the same failure mode as LAION-CLAP. L03 slide 52 also notes CLAP-style models capture coarse global semantics but struggle with fine-grained attributes, which is consistent with your CLAP negative.

---

## 7. PupuJEPA — a new pretraining family for ensemble diversity

[arXiv:2606.25713](https://arxiv.org/abs/2606.25713) ("Frequency-Aware Self-Supervised Music Representation Learning", Gu et al. 2026); code at [github.com/sizigi/PupuJEPA](https://github.com/sizigi/PupuJEPA); checkpoints linked from the authors' project page ([yichenggu.com/PupuJEPA](https://www.yichenggu.com/PupuJEPA/)). Note: L02b slide 27 links `github.com/sizigi/PupuM2D`; the repo cited in the paper's BibTeX is `sizigi/PupuJEPA` — use that.

It predicts latent embeddings of masked 2D spectrogram patches (L02b slide 27) with an EMA target encoder — a JEPA, i.e., the fourth pre-training strategy on L02b slide 4, which none of your encoders uses (MERT/MuQ/MusicFM are all masked-token prediction on 1D sequences). PupuJEPA-Large (307M) reports GTZAN 86.9 vs MuQ 83.8.

**Why it matters for you specifically:** your fusions keep collapsing to one member because the members make correlated errors. A model from a different pretraining family and input geometry (2D patches, 4×16 asymmetric patches) is the most plausible source of *decorrelated* errors among open encoders. Evaluate it as a standalone probe, then as an OOF-fitted fusion partner with the best of MuQ / MERT-v2. If it does not improve fusion under OOF fitting, drop it.

---

## 8. Hand-crafted features and input representations (Lecture 02)

### 8.1 A "wide & deep" probe (Task 1 primarily) — L02 slides 25–37, 47

L02 slide 47 describes combining hand-crafted features with learned ones in a wide-and-deep architecture. Concretely: compute clip-level statistics (mean and std over frames) of spectral centroid, bandwidth, rolloff (0.85 and 0.99), spectral contrast (7 bands), spectral flatness, spectral flux, zero-crossing rate, 20 MFCCs, 12 chroma, RMS/crest factor, plus GTZAN-style rhythm features (tempo, beat-histogram peak ratios, beat strength — L02 slide 27). That is roughly 120 dims. Standardize, concatenate to the PCA-reduced SSL embedding, and fit the same probe.

**Why decade might benefit.** Recording and mastering technology changed monotonically across your six decades: tape bandwidth and noise, analogue vs. digital reverb, dynamic range compression (the loudness war, Round-4 A3), and 808/drum-machine rhythm regularity in the 1980s. SSL encoders are trained to be somewhat invariant to exactly these "nuisance" production properties, so explicit descriptors can be complementary. Your finding that errors are cleanly ordinal is consistent with a smooth, technology-driven signal.

**Important caveat specific to your data.** Your audio is 24 kHz, so everything above 12 kHz is gone. High-frequency extension (a classic era cue) is truncated for all decades; use rolloff at 0.85 rather than 0.99 as the primary brightness feature and do not over-interpret absolute spectral bandwidth. Also check whether any loudness normalization happened in preprocessing before trusting RMS/crest factor (as Round 4 warned).

**Overlap note:** Round 4's A3 covered loudness/dynamic-range descriptors only; this extends it to the full GTZAN timbre/rhythm/pitch set and the wide-and-deep fusion design.

### 8.2 Feature-importance as analysis

Even if the wide-and-deep probe gives only a small gain, a permutation-importance or ablation table over feature groups (timbre vs. rhythm vs. pitch vs. dynamics) is a good report section: it answers *what* acoustic properties distinguish decades, which is the analysis the course's slide on confusion matrices (L02 slide 101) points toward.

### 8.3 CQT / chroma branch for the from-scratch CNN — L02 slides 49–64

All your CNN training used log-mel. L02 slide 58 contrasts timbre (mel → MFCC) and harmonic (CQT → chroma) representations; harmonic vocabulary does shift across decades (and between Brazilian and Anglo 1980s pop). A cheap test is a second Short-Chunk CNN on a 84-bin CQT (or chroma) input, fused with the log-mel CNN. Low priority, but it is exactly what the lecture recommends and cheap to run. While editing the CNN front end, verify the STFT window/hop match what the Short-Chunk CNN was designed for (the original uses 16 kHz input; at 24 kHz the same n_fft covers a different time span — L02 slide 64's warning).

---

## 9. Caption-as-features (Both tasks) — L03 slides 15, 22–25, 33–34

L03 describes music captioning (LP-MusicCaps, captions every 10 s and then summarized by an LLM, slide 25) and uses Qwen2-Audio captions to build training data for a text-to-music model (MuseControlLite pipeline, slide 34). Applied here: ask AF3 (your strongest ALM on Task 2) or Music Flamingo (strongest on Task 1) for a *structured* free-text description of each clip — sung language, vocal style, instrumentation, production character (reverb, drum machine, synth type), and apparent era — then embed the text with a sentence-transformer and use it as a probe feature, or feed it to a small text classifier.

**Why this is different from what you did.** Your ALM work asked the model for the label directly (teacher-forced scoring). Captioning asks for intermediate, verbalised evidence and lets a *trained* classifier learn how that evidence maps to your labels. This sidesteps the label-prior miscalibration of zero-shot scoring and the 38% invalid-output problem of Music Flamingo (any caption is usable input). It also yields interpretable failure analysis.

**Cost.** One generation pass per clip (~1,300 clips). Keep generation deterministic and cache it.

---

## 10. Label-aware data augmentation (Task 1, from-scratch CNN) — L02 slides 87–91

The lecture lists `torchaudio_augmentations` (polarity, noise, gain, high/low-pass, delay, pitch shift, reverb), the Audio Degradation Toolbox, and `pyrubberband` time-stretch/pitch-shift. For decade and market, many of these *alter the very production cues that define the label*: filtering changes bandwidth, reverb changes room/era signature, gain and degradation change dynamics and noise floor. The standard CLMR augmentation chain (L02b slide 11) is therefore risky for your tasks.

**Proposed ablation (useful whether it wins or loses):** train the Short-Chunk CNN with (A) no augmentation, (B) "label-preserving" augmentation only (random crop, polarity inversion, time-stretch ±5%, pitch shift ±1 semitone), (C) "production-altering" augmentation (filters, reverb, noise, degradation). The prediction is B ≥ A > C for decade. If C hurts, that is direct evidence that production cues carry decade information — a clean, reportable result. This applies only to the from-scratch CNN; for frozen probes, augmentation changes little.

---

## 11. What the lectures suggest *not* to do

Self-supervised contrastive pre-training on your own data (SimCLR/CLMR, L02b slides 7–16) needs large unlabelled corpora and batch sizes (L03 slide 45 notes batch size matters, up to thousands). With ~1,300 clips, it will not produce an encoder competitive with MERT-v2/MuQ. Skip it.

Training an ALM from scratch or building an instruction dataset (LLark-style, L03 slides 60–70) is out of scope for a coursework assignment.

Dasheng-1.2B (proposed in Round 4 as A4) sits below MuQ on both GTZAN genre (81.4 vs 83.8) and MTG top-50 in the lecture's own comparison table; if compute is limited, MERT-v2 and PupuJEPA should come before it.

---

## 12. Experimental design for this round

This round introduces several new encoders, so the multiple-comparison risk that produced the 65.7% mirage is higher. Suggested protocol:

1. **Selection stage (training split only).** For each candidate (MERT-v2 layers, weighted-sum probe, PupuJEPA, CLaMP 3, Omni encoder, wide-and-deep, captions), run 5-fold *artist-grouped* CV on the training split, repeated with 3 seeds. Primary metric: OOF log-loss; secondary: OOF Top-1 and Top-3. Select at most two candidates per task.
2. **Fusion fitting.** Fit fusion weights on OOF predictions only (as you already do).
3. **One validation evaluation** for the selected configurations, with the paired bootstrap and McNemar you already use, plus the corrected resampled t-test on the CV folds (Round-4 D1).
4. **Test set once**, at the end.

Pre-register the candidate list and the selection rule before looking at validation, and report every candidate's CV result (including negatives) so the final number is defensible.

---

## 13. Suggested order of work

1. Extract MERT-v2-30s hidden states for all clips; run the layer sweep and the weighted-sum probe in CV. (One afternoon.)
2. Run the §5 diagnostic via MARBLE's Omni encoder (and AF-Whisper if the Round-4 B1 code exists).
3. Branch on §5's result: if the Whisper-lineage encoder explains Task 2, prioritize sung-language ID and the hierarchical classifier (Round 4); otherwise prioritize AF3 retrieval-ICL (§4) and AF3 LoRA (§3c).
4. Add PupuJEPA and CLaMP 3 as probes and OOF fusion partners.
5. Task 1 extras: wide-and-deep hand-crafted features, SORD targets (Round 4), and the augmentation ablation.
6. Optional: caption-as-features.

---

## 14. Source list and citation confidence

| Claim / resource | Source | Confidence |
|---|---|---|
| MERT-v2-30s specs, MARBLE table, layer guide, citation instruction | HF model card `m-a-p/MERT-v2-30s` (fetched Sept 2026) | Verified |
| MERT-v2 pretraining targets from MuQ L7 + Qwen2-Audio L32 | L02b slide 24 architecture figure (also on model card) | Verified from figure; technical report not yet released |
| MERT-v2-FullSong continues pretraining from 30s model | HF model card `m-a-p/MERT-v2-FullSong` | Verified |
| PupuJEPA method and MARBLE results; checkpoints link | arXiv:2606.25713; `github.com/sizigi/PupuJEPA` | Verified |
| CLaMP 3: M4-RAG 2.31M pairs, 27 languages, 194 countries; zero-shot by genre/region; audio via MERT features | HF `sander-wood/clamp3`; `github.com/sanderwood/clamp3`; ACL 2025 | Verified |
| MuQ-MuLan HF ID `OpenMuQ/MuQ-MuLan-large`; Qwen2.5-Omni encoder arXiv:2503.20215; MARBLE toolkit `github.com/a43992899/MARBLE` | L02b slide 29 | From lecture slide; links in §15 not independently re-fetched |
| Four ALM usage modes; ICL definition; LoRA-LLM + full-projector recipe | L03 slides 80, 83; LLM2Fx-Tools = arXiv:2512.01559, github.com/SonyResearch/LLM2Fx | Verified |
| 2026 music valence/arousal ICL paper | L03 slide 81 | **arXiv ID not found** — obtain from the slide/lecturer before citing |
| MERT weighted layer-sum probe | L02b slide 19; Li et al., ICLR 2024 | From lecture |
| Wide & deep; GTZAN features; CQT/chroma; STFT parameter warning; augmentation libraries | L02 slides 25–37, 47, 49–64, 87–91 | From lecture |

---

## 15. Link index (every model, dataset and tool mentioned)

Status key: **V** = verified by fetching/searching the live page this round; **L** = URL/ID taken from the lecture slides or the Round-4 prompt (already verified by you in earlier rounds); **K** = canonical repo from general knowledge, not re-fetched — spot-check before citing.

### Encoders / representation models

| Name | Links | Paper | Status |
|---|---|---|---|
| MERT-v2-30s | https://huggingface.co/m-a-p/MERT-v2-30s | tech report pending; cite MERT + MARBLE | V |
| MERT-v2-FullSong | https://huggingface.co/m-a-p/MERT-v2-FullSong | — | V |
| MERT-v1-330M ("MERT-Large") / MERT-v1 | https://huggingface.co/m-a-p/MERT-v1-330M · https://github.com/yizhilll/MERT | arXiv:2306.00107 (ICLR 2024) | V / L |
| MuQ | https://huggingface.co/OpenMuQ/MuQ-large-msd-iter · https://github.com/tencent-ailab/MuQ | arXiv:2501.01108 | L |
| MuQ-MuLan | https://huggingface.co/OpenMuQ/MuQ-MuLan-large | arXiv:2501.01108 | L |
| MusicFM | https://github.com/minzwon/musicfm | arXiv:2311.03318 | L |
| CultureMERT-95M | https://huggingface.co/ntua-slp/CultureMERT-95M | — | L |
| PupuJEPA (Large/Huge) | https://github.com/sizigi/PupuJEPA · https://www.yichenggu.com/PupuJEPA/ (lecture slide links github.com/sizigi/PupuM2D) | arXiv:2606.25713 | V |
| Dasheng / Dasheng-1.2B | https://github.com/richermans/dasheng · https://huggingface.co/mispeech/dasheng-1.2B | arXiv:2406.06992 | L |
| CLaMP 3 | https://huggingface.co/sander-wood/clamp3 · https://github.com/sanderwood/clamp3 | arXiv:2502.10362 (ACL 2025) | V |
| Qwen2.5-Omni audio tower | https://github.com/QwenLM/Qwen2.5-Omni · https://huggingface.co/Qwen/Qwen2.5-Omni-7B | arXiv:2503.20215 | L / K (HF ID) |
| BEATs | https://github.com/microsoft/unilm/tree/master/beats | arXiv:2212.09058 (ICML 2023) | K |
| Whisper / Whisper-large-v3 | https://github.com/openai/whisper · https://huggingface.co/openai/whisper-large-v3 | arXiv:2212.04356 | V (Round 4) |
| AF-Whisper (AF3's encoder) | ships inside AF3: https://github.com/NVIDIA/audio-flamingo | arXiv:2507.08128 | L |
| MARBLE-table baselines (AudioMAE++, MATPAC++, A-JEPA) | numbers reproduced from PupuJEPA Tables III/V; see arXiv:2606.25713 for their original sources | — | V (via PupuJEPA) |

### Audio-language / text–audio models

| Name | Links | Paper | Status |
|---|---|---|---|
| Audio Flamingo 3 (AF3) | https://huggingface.co/nvidia/audio-flamingo-3-hf · https://github.com/NVIDIA/audio-flamingo | arXiv:2507.08128 | L |
| Music Flamingo | https://huggingface.co/nvidia/music-flamingo-2601-hf | — | L |
| Qwen2-Audio | https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct · https://github.com/QwenLM/Qwen2-Audio | arXiv:2407.10759 | L |
| LLark | https://github.com/spotify-research/llark (code only, no checkpoint) | arXiv:2310.07160 (ICML 2024) | L |
| SALMONN | https://github.com/bytedance/SALMONN | arXiv:2310.13289 (ICLR 2024) | L |
| LLM2Fx-Tools | https://github.com/SonyResearch/LLM2Fx | arXiv:2512.01559 (ICLR 2026) | V |
| LP-MusicCaps | https://github.com/seungheondoh/lp-music-caps | arXiv:2307.16372 (ISMIR 2023) | L |
| MuseControlLite data pipeline (Qwen2-Audio captioning) | https://github.com/fundwotsai2001/Text-to-music-dataset-preparation | ICML 2025 | L |
| LAION-CLAP / HTS-AT | https://github.com/LAION-AI/CLAP · https://huggingface.co/laion/clap-htsat-fused · https://github.com/RetroCirce/HTS-Audio-Transformer | arXiv:2211.06687; arXiv:2202.00874 | L / K (HTS-AT repo) |
| Microsoft CLAP | https://github.com/microsoft/CLAP | arXiv:2206.04769 | L |
| MusicCoCa (Magenta RealTime) | https://github.com/magenta/magenta-realtime | "Live music models," arXiv 2025 | L |
| 2026 music valence/arousal ICL paper (L03 slide 81) | **not found** — get from lecturer | — | — |

### Classification architectures, SSL frameworks, tools

| Name | Links | Status |
|---|---|---|
| Short-Chunk CNN | https://github.com/minzwon/sota-music-tagging-models | L |
| PANNs | https://github.com/qiuqiangkong/audioset_tagging_cnn | L |
| AST | https://github.com/YuanGongND/ast | L |
| CLMR | https://github.com/Spijkervet/CLMR | K |
| SimCLR | https://github.com/google-research/simclr | K |
| MARBLE toolkit / benchmark | https://github.com/a43992899/MARBLE · https://marble-bm.shef.ac.uk/ | L |
| torchaudio_augmentations | https://github.com/Spijkervet/torchaudio-augmentations | K |
| Audio Degradation Toolbox | https://github.com/sevagh/audio-degradation-toolbox | L |
| pyrubberband | https://github.com/bmcfee/pyrubberband | L |
| librosa (hand-crafted features) | https://github.com/librosa/librosa | L |
| sentence-transformers (caption embeddings) | https://github.com/UKPLab/sentence-transformers · https://sbert.net | K |

### Datasets

| Name | Links | Status |
|---|---|---|
| GTZAN | https://www.tensorflow.org/datasets/catalog/gtzan | L |
| MTG-Jamendo | https://github.com/MTG/mtg-jamendo-dataset | L |
| MagnaTagATune | https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset | L |
| EmoMusic (1000 Songs) | https://cvml.unige.ch/databases/emoMusic/ | K |
| GiantSteps Key | https://github.com/GiantSteps/giantsteps-key-dataset | K |
| Chords1217 | evaluated via MARBLE: https://github.com/a43992899/MARBLE | L |
| M4-RAG (CLaMP 3 training data) | https://huggingface.co/datasets/sander-wood/m4-rag | V |

