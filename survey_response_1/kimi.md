Here is the sourced survey, organized under your 9 headings. I browsed current sources rather than relying on memory, and I've flagged confidence levels where appropriate.

---

## 1. Encoders for song-level ~30s classification

**MERT-v1-330M** — `m-a-p/MERT-v1-330M` (HF, `trust_remote_code=True`). Confirmed: 330M params, trained on 160K hours, MLM with EnCodec RVQ-VAE acoustic teacher + CQT musical teacher, native **24kHz**, 75 Hz frame rate, 24 transformer layers × 1024-dim (25 hidden states returned).  The HF card's own example does time-mean-pooling per layer and notes you can use a learnable weighted average across layers, explicitly warning layer choice is task-dependent.  **License correction vs. your survey**: MERT-v1-330M is **CC BY-NC 4.0**, not permissive — fine for coursework, but cite it.  MERT-v1-95M and -330M remain the current MERT checkpoints; no v2 exists as of this search.

**MusicFM** — `github.com/minzwon/musicfm`. MSD-pretrained 330M-param Conformer, BEST-RQ objective, **24kHz input** (confirmed by the repo's own 30s/24kHz demo), 25Hz frame rate, d=1024, `get_latent(wav, layer_ix=...)` API.  Pooling is global average over time for sequence-level use. A 2025 layer-wise study (MusicFM + MuQ, 14 tasks) confirms lower layers favor objective tasks (singer/instrument), upper layers semantic ones (genre/structure) — strong empirical support for your layer-sweep plan. 

**Music2Vec** — `m-a-p/music2vec-v1`: 95M, data2vec/BYOL-style, 1000 hrs, **16kHz** (resample from 24kHz needed), 50Hz, 768-dim, 13 layers, MIT-licensed.  Comparable to Jukebox-5B at <2% of its params. 

**CLAP variants** — LAION-CLAP (`LAION-AI/CLAP` repo; `laion/larger_clap_music` on HF): HTSAT audio tower + RoBERTa, **48kHz required** (the repo's quickstart explicitly loads at 48000), 512-d joint space; audio preprocessing in training was mono 48kHz FLAC.  MS-CLAP (`microsoft/CLAP`, `pip install msclap`): CNN14 (2048-d) + BERT, 44.1kHz log-mel, 1024-d joint space; 2022/2023/clapcap weights.  Two newer options worth knowing: **MuQ-MuLan** (music-only text-audio contrastive; zero-shot MagnaTagATune ROC-AUC 79.3 vs LAION-CLAP 73.9) and **SLAP** (Jan 2026, claims SOTA zero-shot classification at 109M audio-text pairs — flag: very new, weights availability unverified).  

**Jukebox-derived**: JukeMIR/Jukebox-5B is the ceiling reference but impractical; the MARBLE benchmark thesis shows MERT-330M matches Jukebox-family averages at 6.6% of the params, and SSL > supervised pretraining for pitch-sensitive tasks. 

**Frozen-probe vs fine-tune at your scale**: MusicFM's own ablation found fine-tuning improved most tasks but *hurt* tagging due to overfitting — direct evidence for LP-first on small data.  The MARBLE evaluation uses frozen probing as the standard protocol. No published head-to-head at ~1K clips exists; treat LP-first as best-supported practice, not proven optimum.

## 2. Prior work on decade / market classification

**Era recognition** — "Music Era Recognition Using Supervised Contrastive Learning and Artist Information" (arXiv 2407.05368, Jul 2024) is confirmed and is the closest prior work: audio-only model gets **54% accuracy with ±3-year tolerance** on MSD; adding artist biographies via multimodal contrastive learning adds **+9%**; for decade-granularity (8 classes) they report 90%+ ACC1.  Crucially, they introduce an **ACC±x tolerance metric** — directly relevant to your §9. Note their splits are *not* clearly artist-disjoint, so your artist-disjoint setup will score lower; their 90% decade number is not a target. 

**YearPredictionMSD (UCI)** — confirmed: 515,345 songs, 1922–2011, 90 features (12 timbre averages + 78 timbre-covariance values), regression task, official split is artist-disjoint explicitly to avoid the "producer effect."   Baselines are kNN and Vowpal Wabbit linear regression; typical MAE lands ~7–9 years (consistent across reproductions). This is your regression sanity floor.

**Geographic origin** — confirmed the two works in your survey: Zhou et al. (1,142 pieces / 73 countries, MARSYAS 68 features + chroma, KNN and random-forest *regression* on great-circle distance, best mean error 3,113 km)  and the UCI Geographical Origin of Music dataset, whose documentation explicitly defines origin as the artist's residence and *excludes all Western music* because "its influence is global" — i.e., these predict **provenance of a musical tradition**, not commercial release market.  I found **no published work on release-market classification of 1980s pop/rock** — your Task 2 appears genuinely unaddressed in the literature; say so in the report.

## 3. Discogs-VI

Confirmed: arXiv 2410.17400 (ISMIR 2024, Araz/Serra/Bogdanov), ~**1.9M versions / 348K cliques**; Discogs-VI-YT 493K versions / 98K cliques mapped to official YouTube uploads.  The paper lists exactly the metadata fields you rely on — "genre, style, record label, release format, release date, master release, and release country" — and its limitations section documents metadata ambiguity (inconsistent credits across releases), which supports your label-noise caveat.  Audio representations (chroma, HPCP, CQT) are available "under request for non-commercial scientific research purposes."  I found **no prior work repurposing Discogs-VI editorial metadata for decade/market classification** — you're on new ground; cite the VI paper for provenance only.

## 4. Classifier heads at ~150-230 samples/class, 768-1024 dims

Nothing here is MIR-specific enough to cite; the honest answer is this is standard small-n-high-p practice: ridge/logistic with C tuned by nested CV, PCA-whitening, and standardization are the defensible defaults, and published music-SSL evaluations (MARBLE, the MusicFM paper) use exactly frozen-embedding + shallow-probe protocols as the standard.  The one MIR-specific data point: MusicFM fine-tuning *degraded* tagging performance via overfitting, reinforcing "linear probe first." Flag in your report that k-fold-CV-then-final-val is your methodological choice, not a literature mandate.

## 5. Augmentation / TTA

SpecAugment (time/freq masking), mixup on log-mels, and multi-crop averaging are standard and uncontroversial — no new literature needed; multi-segment averaging is literally the Short-Chunk CNN's published inference design. On pitch-shifting: I found no paper that directly tests era/market classification under pitch-shift, so your caution argument (pitch drift across pressing/mastering eras is a real cue; the label-meaning constraint) stands as reasoning, not citation. If you want one citation for pitch-aware models being sensitive to exactly this, note MERT uses a CQT musical teacher specifically to capture pitch structure — evidence that SSL music encoders encode absolute-pitch information you'd be perturbing. 

## 6. Source separation

Your Demucs recommendation is confirmed still-current: `pip install -U demucs` (needs Python ≥3.10 per current PyPI), and `demucs --two-stems=vocals file.wav` produces vocals + accompaniment directly; `htdemucs_ft` is ~4× slower but measurably better.   However, the 2026 landscape has moved: **BS-RoFormer** (SDX23 winner) reports ~9.8 dB avg SDR on MUSDB18-HQ vs ~7.7 for htdemucs, with its biggest gain on bass; vocal-focused MDX23C/MDX-Net ensembles top MVSEP's leaderboard (~14 dB vocals).   Practical guidance: keep `htdemucs_ft` as the easy default, but if vocal-stem quality becomes a bottleneck for Task 2, run BS-RoFormer/MDX23C via Ultimate Vocal Remover (`Anjok07/ultimatevocalremovergui`) — same 2-stem output, better vocals, more setup friction. Demucs is MIT-licensed. 

## 7. Audio language models for closed-set classification

**Qwen2-Audio-7B-Instruct**: confirmed 8.2B total (Whisper-large-v3 encoder + Qwen-7B), **input resampled to 16kHz** — so your 24kHz clips *do* get resampled here.  VRAM: ~17GB FP16, ~4.2GB INT4 for the Qwen2-7B backbone; the Whisper encoder adds overhead, putting realistic full-precision at ~18-20GB — trivial for your H200s.  License: **Apache 2.0**.  Bonus: 30s clips are exactly one native Whisper window.

**Audio Flamingo 3**: confirmed — Qwen2.5-7B backbone + AF-Whisper encoder, up to 10 min audio, now integrated into HF Transformers; checkpoints under **NVIDIA OneWay Noncommercial** plus Qwen Research License restrictions (code is MIT).  

**Newer than mid-2025** (worth a mention for "goes beyond"): **Qwen3-Omni** (Sep 2025; open-source SOTA on 32/36 audio benchmarks, 30+ min audio understanding) ; **Kimi-Audio-7B-Instruct** (Apr 2025; open weights, MIT/Apache code, SOTA on MMAU sound/speech and VocalSound) ; and **Bagpiper** (Feb 2026, claims to beat Qwen2.5-Omni on MMAU/AIRBench — flag: brand new, verify before relying). 

**Closed-set prompting practice**: the established evaluation paradigm is **AIR-Bench** ("Benchmarking large audio-language models via *generative comprehension*"), which scores free-form generation with GPT-4 judging and treats unparseable/refusing outputs as failures — this is your canonical citation for reporting invalid-output rates.  For prompt design, published zero-shot audio classification work shows prompt formatting matters substantially — LAION-CLAP's own eval uses a fixed template ("This audio is a <genre> song."), and DCASE 2024 work found properly formatted class-label prompts competitive with elaborate templates.   For ALMs specifically, PALM (few-shot prompt learning for audio LMs) is a useful citation that plain zero-shot generation is a weak baseline worth beating. 

## 8. Loudness war evidence

Well-documented, and you have strong citations. Deruty & Tardieu's analysis of ~4,500 tracks (1969–2010) shows constant RMS-loudness growth from 1982–2005 (~5 dB louder than the '70s), crest factor falling ~3 dB from the early '80s, and brickwall-limiter signatures rising after 1990.  Wikipedia's loudness war entry aggregates the same literature: average rock RMS ~−16.8 dBFS through the 1980s, trend peaking around 2005, with dynamic range *increasing* since — useful for discussing why your 2010s class may be less separable by loudness than the 1990s/2000s.  One nuance worth citing to show sophistication: the same authors argue LRA (loudness *range*) stayed roughly constant across the loudness-war era — so your "rising RMS" feature is real, but "shrinking dynamic range" is contested; measure both before claiming either. 

## 9. Ordinal/ranked metrics for decade classification

There is an established MIR convention: the era-recognition paper's **ACC±x tolerance** metric (accuracy counting predictions within x years as correct) is published practice for exactly this task, and you should adopt/cite it alongside your own mean-absolute-decade-error.  For regression framing, YearPredictionMSD reports year MAE — your "decade MAE" is the ordinal adaptation. Weighted kappa has no era-classification precedent I could find in MIR — fine to use, but present it as a generic ordinal-statistics choice, not an MIR standard. Adjacent-vs-nonadjacent error rate appears to be your own construction; that's fine, just label it as such.

---

**Confidence flags**: (a) MusicFM's *native* sample rate is 24kHz per its demo, but the repo doesn't state it as explicitly as MERT's card — verify with a quick forward pass on a known file; (b) AIR-Bench's exact invalid-answer handling — I cited the benchmark's generative design, but double-check the current scoring script before claiming the specifics; (c) SLAP weights availability is unverified; (d) I found no license info for MusicFM's MSD-pretrained checkpoint — check before the report's model table, since MERT (CC BY-NC) and AF3 (OneWay NC) licenses are confirmed noncommercial.

Want me to fold these findings back into your survey document as an updated version, or draft the citations section in BibTeX for the report?