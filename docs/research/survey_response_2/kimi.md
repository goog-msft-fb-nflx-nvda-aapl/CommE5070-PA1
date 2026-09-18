Here's the synthesized research brief, organized by your six questions. Confidence flags are inline.

---

## 1. Published work on release-decade / release-market classification

**Bottom line: I found no published paper that trains/evaluates audio classifiers for release-decade or release-market prediction with accuracy numbers you can compare against.** This appears to be a genuinely novel task formulation — which means your numbers can't be sanity-checked externally, but also that the assignment's benchmark is the reference point. What exists nearby:

- **Geographic/cultural origin classification** (closest analog to Task 2): Gómez et al. achieved **86.7% on *binary* Western vs. non-Western** discrimination using 23 handcrafted timbral/tonal/rhythmic features on ~6,000 recordings . The UCI "Geographical Origin of Music" dataset (traditional/ethnic music, 33 countries, MARSYAS features) is the standard multi-country benchmark . A recent deep-learning thesis (LSTM on MFCCs + Whisper language ID, 22k samples, 56 classes) reports **33.0% accuracy vs. a 1.79% random baseline** — with heavy confusion into US/Laos and explicit caveats that "country borders may not accurately represent cultural or musical boundaries" . Your 47% over 6 balanced classes is broadly consistent with this literature at much smaller class count.
- **Discogs metadata as supervision** (relevant lineage you can cite): Alonso-Jiménez, Serra & Bogdanov, "Music Representation Learning Based on Editorial Metadata from Discogs" (ISMIR 2022) learned track embeddings *from editorial metadata* (decade, label, country, genre fields) rather than audio ; Bogdanov & Serra (ISMIR 2017) quantified music trends from Discogs metadata . Nobody seems to have used these metadata fields as *audio classification targets* — worth stating in your report as a gap you've filled.
- **Artist-disjoint evaluation**: most genre/mood benchmarks are not artist-disjoint, so genre numbers (GTZAN ~85-93%) are not comparable to yours. The ISMIR 2025 paper "Universal Music Representations? Evaluating Foundation Models on World Music Corpora" is one of the few evaluating foundation models with probing + SFT on non-Western corpora, finding Qwen2-Audio the overall best and — notably — that **MERT-330M trained on extra Western music does *worse* than MERT-95M** on non-Western data . That finding is directly relevant to Task 2 (see below).

**Confidence: high** that no directly comparable published benchmark exists (this was a broad search; absence-of-evidence flagged as such).

## 2. Why Task 2 (market) resists what Task 1 (decade) rewards

The literature supports a coherent mechanistic explanation for your six asymmetries:

**(a) Era has strong, monotonic, technology-driven acoustic correlates; market does not.** Decade signals (tape saturation, stereo width, reverb fashions, drum-machine signatures, loudness trends) are distributed across the whole clip — matching your monotonic improvement with segment length for Task 1. Market labels are a *commercial artifact* of which country's label released the record; the genuinely informative cues (vocal language, local production idiom) are temporally localized — consistent with your finding that a single 30s clip beats multi-crop averaging for Task 2: mean-pooling over crops dilutes a sparse cue.

**(b) MERT's pretraining is Western-heavy, so the cultural signal Task 2 needs is exactly what's underrepresented.** The world-music evaluation above found Western-pretrained models transfer poorly to non-Western corpora, and that adding more Western data actively hurt MERT-330M relative to MERT-95M . Task 1's cues (production technology) are universal across Western music → MERT encodes them well in low/intermediate layers (as your layer sweep confirmed: layer 4 best for Task 1).

**(c) Zero-shot ALMs invert the pattern because their advantage is semantic, not acoustic.** Audio Flamingo 3 is a Whisper-large-v3 encoder (AF-Whisper) continued-trained on ~50M audio–text pairs, including a large music-reasoning corpus (AudioSkills-XL), feeding Qwen2.5-7B . Its music knowledge is *text-aligned* — it can name what it hears. A market label is closer to "name the cultural context" (semantic) than "name the decade" (acoustic/technological), so AF3 beating you on Task 2 but not Task 1 is exactly what the architecture predicts. The geographic-origin literature agrees: language + cultural idiom dominate geographic prediction .

**Concrete diagnostics this suggests (cheap, testable):**
1. **Confusion-matrix analysis** on Task 2: I predict US↔UK dominates, and Brazil/Spain/Italy separate mainly via language. Check whether your per-class recall ordering matches language-correlated classes.
2. **Whisper language-ID on vocal segments as an auxiliary feature** concatenated with your MERT embeddings before the SVM/logreg. For 1980s releases, US/UK (English) vs. Brazil (Portuguese) vs. Spain (Spanish) vs. Italy (Italian) vs. Germany (German) is highly language-informative, and this is plausibly *the* feature AF3 exploits. Legal per your definition (market ≠ nationality, but language correlates with market).
3. **Keep single-clip inference for Task 2, multi-crop for Task 1** — your data already validates this asymmetry.

**Confidence: moderate-high.** The mechanism is consistent across three independent literatures (geographic-origin MIR, world-music transfer evaluation, ALM architecture), but no paper states it for this exact task pairing.

## 3. Newer/alternative pretrained encoders worth trying

Concrete model IDs, in priority order for your compute (2 usable H200s, so a 330M-class encoder is comfortable):

| Model | ID / link | Size | Why for your tasks | License |
|---|---|---|---|---|
| **MuQ** | `OpenMuQ/MuQ-large-msd-iter` (HF), code: github.com/tencent-ailab/muq | ~300M, Conformer, Mel-RVQ SSL | Beats MERT/MusicFM on MARBLE suite (GTZAN 85.5 vs MERT 78.6); trained on MSD; lower layers peak on acoustic/timbral tasks (era-relevant), higher on semantics  | MIT code, CC-BY-NC weights |
| **MuQ-MuLan** | `OpenMuQ/MuQ-MuLan-large` | ~700M, music–text contrastive | Lets you **zero-shot prompt-score** "1980s Brazilian pop"-style class descriptions — a free, reproducible analog of your AF3 scoring that might generalize better to Task 2; SOTA zero-shot tagging on MagnaTagATune  | CC-BY-NC |
| **MuFun** | `Yi3852/MuFun-Instruct` (HF) | 9B, Whisper-encoder multi-layer fusion (layers 0/7/15/32) + Qwen3-8B | Music-specific ALM; SOTA on MuCUE by +15 pts over Qwen2.5-Omni; its ablation shows **last-hidden-only loses ~1.25 pts — multi-layer fusion is the key**, mirroring your MERT layer-sweep finding  | open weights (check card) |
| **MusicFM** | github.com/minzwon/musicfm | 330M, BEST-RQ | Same lab as your Short-Chunk CNN baseline; public weights; their own fine-tuning recipe uses lr 1e-5 (encoder) / 1e-4 (probe) with explicit catastrophic-forgetting warnings  | research |
| **CultureMERT** (Kanatas et al., 2025) | arXiv | MERT-derived | Continual pretraining for non-Western music: staged LR reset, selective parameter updates, +4.9% AUC on Turkish/Indian/Greek corpora. You can't re-pretrain on 800 clips, but its **selective-update recipe is directly transferable as a fine-tuning strategy for Task 2**  | — |

**Recommendation: MuQ first** (drop-in replacement for MERT in your existing frozen-probe + SVM pipeline — same effort, new information), then MuQ-MuLan zero-shot scoring for Task 2 (an afternoon of work). The MuQ paper's layer-specialization result suggests sweeping layers for MuQ exactly as you did for MERT.

## 4. Closing the gap to Audio Flamingo 3 on Task 2

Ordered by expected value per GPU-hour:

1. **Teacher-forced KD from the ALM into your classifier.** You already compute ranked label probabilities from AF3 (58.8% direct prompt). Use its soft distributions on the *training* set as KD targets (KL loss) for the frozen-probe/MLP head, or as an auxiliary logit input concatenated with your model's logits in a small stacker. Precedent: multi-distillation from speech+music representation models into a unified student improves few-shot music tasks including singer ID at 20 shots/class . Also classic self-distillation improves small-data audio classification generalization .
2. **AF-Whisper features, not AF3 generation.** AF3's audio tower is a Whisper-large-v3 encoder trained beyond speech (AF-Whisper) . Extract AF-Whisper encoder embeddings from `nvidia/audio-flamingo-3-hf` and feed them (mean-pooled, per-layer) to your SVM — same probe protocol as MERT, no generation. Whisper-based encoders process vocals/language explicitly, which is likely the missing feature. Concatenate MERT-layer-k ⊕ AF-Whisper-layer-j features; PCA to 128; SVM(RBF) — this is the single most promising Task-2 experiment given your infrastructure.
3. **Prompt-embedding zero-shot scoring with MuQ-MuLan or CLAP** as a third ensemble member (probability-weighted, per your existing ensembling code).
4. **Pseudo-label agreement filtering**: train a student on the subset where AF3's top-1 agrees with ground truth (~roughly 59% of data) with higher sample weight; the disagreement set is likely label noise or genuinely ambiguous classes.

**Confidence: moderate.** These are established patterns (ALM→classifier distillation is standard in 2025 practice), but no paper reports them for market classification specifically.

## 5. Overfitting mitigation for fine-tuning MERT on ~800–1000 clips

Ranked by expected benefit for *your exact failure mode* (full FT won Task 1, lost Task 2):

- **LoRA / partial-unfreeze — highest priority.** LoRA often *beats* full fine-tuning specifically in low-data regimes because freezing the base weights prevents catastrophic forgetting . For MERT (BERT-style), apply PEFT `LoraConfig` targeting `q_proj,k_proj,v_proj,out_proj` (+ `intermediate.dense` if underfitting): **r=8–16, α=16–32, dropout=0.1** per the small-dataset (<1k) row of standard tuning tables , adapter lr ~1e-4 vs. head lr ~1e-3. Critically, this directly targets Task 2: your frozen probe (44–47%) beats full FT (44.1%) precisely because full FT destroys mid-layer features you showed are the informative ones (layers 4/7). A partial-unfreeze alternative: **freeze bottom 16 layers, train top 8** (or LoRA on top 8 only).
- **Layer-wise LR decay (discriminative fine-tuning).** Top layer lr 3e-5 with multiplicative decay 0.9 per layer going down (embeddings at ~1e-6); this is the standard Sun et al. 2019 recipe, validated in recent work . Nearly free to add to your existing full-FT run.
- **Augmentation — with task-specific cautions.** Mixup (α=1.5) and SpecAugment-style time/frequency masking are the proven pair for small-data audio tagging ; SpecMix outperforms both alone on acoustic-scene classification . **Caution for Task 1: avoid pitch shifting and tempo shifting** — tape speed and rhythmic idioms are themselves decade cues, and shifting them blurs adjacent decades. Masking + mixup are cue-preserving; pitch/tempo shift are cue-destroying. For Task 2, light augmentation is safer.
- **Cheap stabilizers:** EMA or SWA of weights across the last epochs ; label smoothing 0.1; and **seed your batch order** — your 1–2 pt run-to-run variance likely exceeds several of your Task-2 "losses" (see §6).

## 6. Other 2025-26 practice points for your exact setting

1. **Statistical reality check on Task 2.** n=102 test samples, 6 classes: a single point of Top-1 ≈ 1 sample. A 95% CI on 47% accuracy at n=102 is roughly ±10 pts, and the difference between your frozen probe (47.1%) and full FT (44.1%) is **3 samples — statistically indistinguishable**. Your conclusion "full fine-tuning failed to help Task 2" is directionally supported by val-set behavior, but avoid strong claims from the test set; report McNemar or bootstrap CIs. This reframes the "six asymmetries" too: several are within noise. (High confidence — pure statistics.)
2. **Per-class / stacked ensembling instead of global weights.** Your global weighted ensemble won Task 1; for Task 2, a logistic-regression stacker over member probabilities (or per-class ensemble weights fit on val) handles the case where CNN helps some markets and MERT others.
3. **Secondary metrics.** You already compute quadratic weighted kappa for Task 1's ordinal structure — good; report it alongside accuracy since decades are ordinal (your hierarchical experiment showed the trade-off). For Task 2 add macro-F1 given 102 samples.
4. **Do not chase year-regression further** — your ridge result (27%) matches the broader finding that fine-grained year targets are noisier than decade boundaries; the geographic-origin literature similarly found coarse region > country > coordinates .

---

**If you only do three things:** (1) run the MERT⊕AF-Whisper feature-concatenation probe on Task 2 (§4.2) — it directly tests the "ALM wins via language/semantic features" hypothesis; (2) LoRA-only fine-tune of MERT with r=8–16 on both tasks (§5); (3) MuQ frozen probe as an ensemble member (§3). Each is a small diff from your existing pipeline, and together they map exactly onto the two observed failure modes.