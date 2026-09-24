# Reproducing the best results

All commands run from the repo root on the GPU server (gsm-gpu2), inside the `pa1_env` conda
environment, with the dataset already placed at the path `src/config.py` expects (`DATASET_DIR`,
not included in this repo — see README). Two environment quirks specific to this box, applied
throughout: (1) GPU index 3 is invisible to the CUDA runtime and breaks any CUDA init in the same
process if left visible — every command that touches `torch.cuda` is run with
`CUDA_VISIBLE_DEVICES=0,1,2` to exclude it; GPU index 0 is the assignment's own documented bad
GPU, so `--device cuda:1` or `cuda:2` is used. (2) `conda run -n pa1_env python ...` silently
buffers stdout on long jobs — invoke the env's python binary directly instead, as shown below.

```bash
conda create -n pa1_env python=3.10 -y && conda activate pa1_env
pip install -r requirements.txt
pip install muq peft   # MuQ encoder (src/muq_features.py) + LoRA (src/finetune_lora.py)
PY=/path/to/miniconda3/envs/pa1_env/bin/python3   # or just `python` once the env is active

# MusicFM (src/musicfm_features.py) -- not pip-installable, clone the reference
# implementation into the project root as `musicfm/` (must be named exactly this --
# its own internal imports are `from musicfm.model...`) and download its MSD checkpoint:
git clone https://github.com/minzwon/musicfm.git
wget -q https://huggingface.co/minzwon/MusicFM/resolve/main/msd_stats.json -O musicfm/data/msd_stats.json
wget -q https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt -O musicfm/data/pretrained_msd.pt

# MuFun (src/mufun_infer.py, negative result, section 6 below) -- also needs ffmpeg,
# not pip-installable:
conda install -n pa1_env -c conda-forge ffmpeg -y

# CLaMP 3 (src/clamp3_features.py, round-5 item 6, negative result -- see WORKLOG.md) --
# their reference implementation pins transformers==4.40.0, which conflicts with pa1_env,
# so it runs in its own dedicated env. Clone into external/ (gitignored, third-party code,
# never committed):
git clone https://github.com/sanderwood/clamp3.git external/clamp3
conda create -n clamp3_env python=3.10.16 -y
/path/to/miniconda3/envs/clamp3_env/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
/path/to/miniconda3/envs/clamp3_env/bin/pip install -r external/clamp3/requirements.txt
# torchaudio is required by their MERT_utils.py but missing from requirements.txt -- the
# line above installs it explicitly, matched to the cu121 torch build. Checkpoint weights
# (MERT-v1-95M + CLaMP3's own) download automatically from Hugging Face on first run.

# PupuM2D-Large (src/pupum2d_features.py, round-5 item 7 -- called "PupuJEPA-Large" in
# TODO.md/the source research doc; verified the real repo is sizigi/PupuM2D, see
# WORKLOG.md). Runs fine inside pa1_env -- just needs timm added:
pip install timm
git clone https://github.com/sizigi/PupuM2D.git external/PupuM2D
# Upstream repo bug: model/__init__.py does `from .pupum2d import *`, a stale module name
# (the actual file is model/pupujepa.py). One-line local patch, required before import works:
echo 'from .pupujepa import *' > external/PupuM2D/model/__init__.py
# Checkpoint (safetensors) + args.json download automatically from HF (spellbrush/PupuM2D)
# on first run of src.pupum2d_features. No further setup needed.
```

## 1. Feature extraction (run once each, cached to `cache/` — gitignored, ~GBs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.mert_features --dataset A --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.mert_features --dataset B --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.muq_features  --dataset A --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.muq_features  --dataset B --device cuda:1

# Post-round-3 encoder checks (negative/neutral results, not part of any best config,
# kept for completeness -- see WORKLOG.md's "CultureMERT" and "MusicFM" sections):
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.culturemert_features --dataset B --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.musicfm_features --dataset A --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.musicfm_features --dataset B --device cuda:1

# Round-5: MERT-v2-30s (m-a-p/MERT-v2-30s) -- new best standalone encoder, both tasks
# (see WORKLOG.md's round-5 section). No known compatibility issues, no special setup.
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.mertv2_features --dataset A --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.mertv2_features --dataset B --device cuda:1

# Round-5 item 5: openai/whisper-large-v3 encoder as a frozen probe (Whisper-lineage
# diagnostic; resolved the linguistic-vs-world-knowledge branch -- see WORKLOG.md).
# Standard HF WhisperModel, no custom code. extract() has no --splits flag, always
# runs all three splits at once (1002/1290 clips depending on dataset).
$PY -m src.whisper_encoder_features --dataset A --device cuda:1
$PY -m src.whisper_encoder_features --dataset B --device cuda:1

# Layer sweep (33 layers = embedding + 32 encoder blocks, + mean_all/concat_last4) +
# classifier/PCA ablation on the winning layer. ablate_mert_layers.py/ablate_classifier_pca.py
# are generic over encoder_name -- reused as-is for the Whisper encoder.
$PY -m src.ablate_mert_layers --dataset A --encoder-name whisper_encoder --n-layers 33
$PY -m src.ablate_mert_layers --dataset B --encoder-name whisper_encoder --n-layers 33
$PY -m src.ablate_classifier_pca --dataset A --layer 8 --encoder-name whisper_encoder   # top1=0.4318
$PY -m src.ablate_classifier_pca --dataset B --layer 20 --encoder-name whisper_encoder  # top1=0.6275

# Round-5 item 6: CLaMP 3 audio embedding -- clean negative result, both tasks (see
# WORKLOG.md). extract() shells out to clamp3_env internally (subprocess), so it can be
# invoked from pa1_env's own python. Single global embedding (no layer structure), so no
# layer sweep -- straight to the classifier/PCA ablation.
$PY -m src.clamp3_features --dataset A --cuda-visible-devices 1
$PY -m src.clamp3_features --dataset B --cuda-visible-devices 1
$PY -m src.ablate_classifier_pca --dataset A --layer 0 --encoder-name clamp3  # top1=0.4924
$PY -m src.ablate_classifier_pca --dataset B --layer 0 --encoder-name clamp3  # top1=0.5098

# Round-5 item 7: PupuM2D-Large frozen probe -- standalone negative both tasks, but see
# the OOF fusion below (WORKLOG.md). Single global embedding, no layer sweep needed.
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.pupum2d_features --dataset A --variant large --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.pupum2d_features --dataset B --variant large --device cuda:1
$PY -m src.ablate_classifier_pca --dataset A --layer 0 --encoder-name pupum2d_large  # top1=0.5152
$PY -m src.ablate_classifier_pca --dataset B --layer 0 --encoder-name pupum2d_large  # top1=0.4804

# OOF fusion of MERT-v2-30s (current best) x PupuM2D-Large -- new Task 1 best point
# estimate (0.5606); Task 2's fusion weight correctly collapses to 100% MERT-v2.
# Requires step 1's mertv2_features extraction to have been run first.
$PY -m src.encoder_fusion_oof --dataset A  # fused top1=0.5606 (alpha=0.6 MERT-v2)
$PY -m src.encoder_fusion_oof --dataset B  # fused top1=0.6471 == MERT-v2-alone (alpha=1.0)
```

## 2. Component models needed for the ensembles below

```bash
# Short-Chunk CNN, from scratch, both tasks (defaults: 40 epochs, batch 32, 3.7s crops)
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.train_scnn --dataset A --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.train_scnn --dataset B --device cuda:1

# Full fine-tune of MERT-v1-330M, Task A only (Task B's fine-tune never beat the frozen probe,
# not part of any best-config here)
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.finetune_mert --dataset A --epochs 15 --batch-size 4 --lr 2e-5 --device cuda:1

# Audio Flamingo 3 zero-shot teacher-forced label scoring, validation split, both prompts,
# both tasks -- needed as an input to the fusion configs below. Output goes to
# results/alm/audioflamingo3_{dataset}/ (unsuffixed -- this is the validation-split path).
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.alm_infer --dataset A --model audioflamingo3 --split validation --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.alm_infer --dataset B --model audioflamingo3 --split validation --device cuda:1

# Same, on the TRAINING split -- needed only for the OOF-refit fusion in step 3b/4b below.
# Output goes to results/alm/audioflamingo3_{dataset}_train/ (split-suffixed, so this can
# never collide with/overwrite the validation-split files above -- alm_infer.py's out_dir
# is parameterized by split for exactly this reason; do not remove that suffixing).
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.alm_infer --dataset A --model audioflamingo3 --split train --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.alm_infer --dataset B --model audioflamingo3 --split train --device cuda:1

# Music Flamingo (--model musicflamingo, same alm_infer.py) -- mixed result, not part of
# any best config: new best zero-shot ALM number for Task A (0.402 vs AF3's 0.379) but
# only ties AF3 on Task B with a 38% invalid-generation rate; not fused with AF3 given the
# shared backbone (correlated errors expected). Also needed for the Task A OOF fusion check
# in WORKLOG.md's "follow-up fusion checks" section -- both splits shown:
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.alm_infer --dataset A --model musicflamingo --split validation --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.alm_infer --dataset B --model musicflamingo --split validation --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.alm_infer --dataset A --model musicflamingo --split train --device cuda:1
```

Frozen-probe classifiers (MERT/MuQ + logreg/SVM/PCA) are refit on the fly by `train_probe.py`
inside `ensemble.py`/`af3_stack.py`/`significance.py` below -- GridSearchCV is deterministic
(`random_state=0` everywhere it appears), so no separate probe-training step is needed.

## 3. Best Task 1 (decade) config

**Round-5 update, read this first**: `m-a-p/MERT-v2-30s` OOF-fused with PupuM2D-Large now
beats every config below (top1=0.5606, α=0.6 MERT-v2 -- see step 1's `encoder_fusion_oof`
command). MERT-v2-30s alone (top1=0.5455, top3=0.8712, higher top3 than the fusion) is the
simpler, more defensible single-model number if fusion complexity isn't wanted. Neither is
independently significance-tested against 3a/3c below (see WORKLOG.md's round-5 section).

```bash
# requires step 1's src.mertv2_features extraction to have been run first
$PY -c "from src.train_probe import run; run('A', layer=10, classifier='logreg', encoder_name='mertv2_30s')"
# for the fusion (top1=0.5606), see step 1's `$PY -m src.encoder_fusion_oof --dataset A`
```

**IMPORTANT, read this before citing a pre-round-5 Task 1 number**: the validation-swept fusion
point estimate (3b below, top1=0.5530) was picked by sweeping the weight directly on the 132-sample
validation set. A leakage-free out-of-fold (OOF) refit — fitting the weight on 1026 training-set
predictions instead — gives a very similar, more defensible number (top1=0.5379, 3a below).
Prefer 3a when a single pre-round-5 number is needed; 3b is kept only as the original point estimate.

**3a. OOF-refit fusion (defensible) — top1=0.5379, top3=0.8636**. Significantly beats AF3-alone
(p=0.0055); not significantly different from the probe alone (p=0.29) or from 3c below.

```bash
# requires step 2's AF3 train-split inference to have been run first
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.af3_stack_oof --dataset A --prompt direct
# prints both NLL-optimal (reported number) and accuracy-optimal alpha variants;
# saved to results/af3_stack_oof/A_muq_large_msd_af3direct.json
```

**3b. Validation-swept point estimate (original, less defensible) — top1=0.5530, top3=0.8485**:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.af3_stack --dataset A --device cuda:1
# best sweep point: MuQ(layer1,svm,noPCA) probe + AF3-direct label probs, w_af3=0.5
# (results/af3_stack/A_muq_large_msd_af3direct.json)
```

**3c. Statistically defensible alternative if a simpler single-model number is preferred —
top1=0.5227, top3=0.8561** (the pre-improvement-round 3-way ensemble):

```bash
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.ensemble --dataset A --mode 3way --device cuda:1
# best sweep point: w_ft=0.20 w_cnn=0.40 w_mert=0.40 (printed + saved to
# results/ensemble_A/3way_sweep.json)
```

## 4. Best Task 2 (market) config

**Round-5 update, read this first**: `m-a-p/MERT-v2-30s` as a plain frozen probe (no AF3, no
calibration) now beats every pre-round-5 config below on its own (top1=0.6471, top3=0.8922),
and OOF-fused with contextually-calibrated AF3 reaches top1=0.6569/top3=0.8922 (best point
estimate found to date; not yet independently significance-tested against 4c). **Coincidence
warning**: this new 0.6569 is numerically identical to the round-3 debunked figure discussed
below but is an entirely different, legitimately-derived config (MERT-v2+calibrated-AF3 via
leakage-free OOF fitting, not a validation-set weight sweep) — do not conflate the two.

```bash
# requires step 1's src.mertv2_features extraction to have been run first
$PY -c "from src.train_probe import run; run('B', layer='mean_all', classifier='logreg', pca_dim=128, encoder_name='mertv2_30s')"
# for the fusion with calibrated AF3, see WORKLOG.md's round-5 section for the exact
# OOF-fitting recipe (mirrors src/af3_stack_oof.py's pattern, applied to this encoder)
```

**IMPORTANT, read this before citing a pre-round-5 Task 2 number**: the validation-swept fusion
(4b below, top1=0.6569 -- the OLD, round-3 figure, see the coincidence warning above) does
**not** reproduce under a leakage-free OOF refit (4a below gives ≈0.57-0.59, essentially
AF3-alone's own score) — that older 0.6569 figure was validation-set overfitting in the
weight choice, confirmed by two independent pieces of evidence (the OOF refit itself, and the
earlier bootstrap/McNemar test already showing p=0.21 vs. AF3-alone). **Do not report the
pre-round-5 0.6569 as a headline number without this caveat; prefer citing ≈0.588 (AF3
zero-shot alone) if citing a pre-round-5 result at all.**

**4a. OOF-refit fusion (the honest number) — top1=0.5686 (NLL-optimal) or 0.5882
(accuracy-optimal), both ≈ AF3-alone's own 0.5882**:

```bash
# requires step 2's AF3 train-split inference to have been run first
env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python $PY -u -m src.af3_stack_oof --dataset B --prompt cot_then_answer
# prints both NLL-optimal and accuracy-optimal alpha variants;
# saved to results/af3_stack_oof/B_muq_large_msd_af3cot_then_answer.json
```

**4c. Round-4 best point estimate — MuQ+Whisper-language-ID concat probe, OOF-fitted fusion
with AF3 — top1=0.5784, top3=0.8529** (not significantly different from 4a/AF3-alone, p=0.73):

```bash
# requires src/stems.py to have separated Task B's Demucs vocal stems first (1002 clips):
$PY -u -m src.stems --dataset B --device cuda:1
# then extract Whisper language-ID features on the vocal stems:
$PY -u -m src.language_id --dataset B --stem vocals --device cuda:1
# then the full probe (standalone langid, MuQ+langid concat, OOF fusion with AF3, all in one):
$PY -u -m src.langid_probe --dataset B --stem vocals
# best point: accuracy-optimal OOF fusion weight; saved to results/langid_probe/B_vocals.json
```

**4b. Validation-swept point estimate (original, NOT reproduced by 4a, kept only for the
record) — top1=0.6569, top3=0.8824**:

```bash
env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python $PY -u -m src.af3_stack --dataset B --device cuda:1
# best sweep point: MuQ(layer2,logreg,noPCA) probe + AF3-cot_then_answer label probs, w_af3=0.4
# (results/af3_stack/B_muq_large_msd_af3cot_then_answer.json)
```

(`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` isn't required by these scripts themselves, but
is kept in the exact invocation used to produce the logged numbers for a byte-for-byte-
reproducible command history; harmless if omitted.)

## 5. Verifying the numbers (significance + error analysis, not needed to reproduce the scores
   themselves, but reproduces every other claim made about them in WORKLOG.md)

```bash
CUDA_VISIBLE_DEVICES=0,1,2 $PY -u -m src.significance --dataset A --n-boot 10000 --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -u -m src.significance --dataset B --n-boot 10000 --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -u -m src.confusion_diagnostic --dataset A --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -u -m src.confusion_diagnostic --dataset B --device cuda:1
```

## 6. Negative/neutral results kept for completeness (none part of any best config)

Every command below reproduces a *documented negative or neutral finding*, not a "best" number
— run to confirm/refute a specific claim in WORKLOG.md, not to chase a higher score.

```bash
# LoRA fine-tune of MERT, attention-only (Task B plateaus at top1=0.382)
CUDA_VISIBLE_DEVICES=0,1,2 $PY -u -m src.finetune_lora --dataset B --epochs 40 \
    --no-time-mask --no-mixup --out-dir results/mert_lora_B_noaug_40ep --device cuda:1

# LoRA retry targeting MLP layers too (Task B improves to 0.431 but still underperforms
# frozen probing -- isolates the "attention-only was the problem" hypothesis)
CUDA_VISIBLE_DEVICES=0,1,2 $PY -u -m src.finetune_lora --dataset B --epochs 20 \
    --no-time-mask --no-mixup --out-dir results/mert_lora_B_allmlp --device cuda:1

# CORAL ordinal head, Task A (top1=0.318, well below the 0.500 nominal-classifier baseline
# on the identical features -- confirmed not a training artifact via a 9-config hyperparam sweep)
$PY -u -m src.coral_ordinal --dataset A

# CultureMERT-95M frozen probe, Task B (top1=0.353, well below MuQ's 0.500)
CUDA_VISIBLE_DEVICES=0,1,2 $PY -u -m src.culturemert_features --dataset B --device cuda:1
$PY -u -c "from src.train_probe import run; run('B', layer=5, classifier='logreg', encoder_name='culturemert_95m')"

# LAION-CLAP zero-shot (A: top1=0.242, B: top1=0.108 -- below random chance on B,
# weakest result this entire project)
CUDA_VISIBLE_DEVICES=0,1,2 $PY -u -m src.clap_zeroshot --dataset A --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -u -m src.clap_zeroshot --dataset B --device cuda:1

# MuFun (9B audio-text LLM) zero-shot (A: top1=0.159, B: top1=0.186 -- both at/near random,
# ~100% invalid free-form generation rate; needed 5 compatibility monkey-patches baked
# into src/mufun_infer.py's load_mufun(), see WORKLOG.md for the full debugging trail --
# also needs PATH to include pa1_env/bin explicitly when invoking python directly, since
# ffmpeg is only discoverable that way, not via `conda run`):
CUDA_VISIBLE_DEVICES=0,1,2 PATH=/path/to/miniconda3/envs/pa1_env/bin:$PATH $PY -u -m src.mufun_infer --dataset A --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 PATH=/path/to/miniconda3/envs/pa1_env/bin:$PATH $PY -u -m src.mufun_infer --dataset B --device cuda:1

# MusicFM frozen probe (A: top1=0.462, B: top1=0.471 -- comparable to MERT-330M, below MuQ;
# OOF-fitted fusion with MuQ adds nothing on either task, see WORKLOG.md)
$PY -u -c "from src.train_probe import run; run('A', layer=3, classifier='logreg', encoder_name='musicfm_msd')"
$PY -u -c "from src.train_probe import run; run('B', layer=6, classifier='logreg', encoder_name='musicfm_msd')"

# Multi-crop TTA with MuQ (A: hurts, best 0.470 vs full-clip 0.508; B: single 10s crop reaches
# 0.520 vs full-clip 0.500 but NOT statistically significant, p=0.82)
CUDA_VISIBLE_DEVICES=0,1,2 $PY -u -m src.sweep_multicrop_muq --dataset A --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -u -m src.sweep_multicrop_muq --dataset B --device cuda:1

# Continuous-score (Brier/Wilcoxon) + one-sided McNemar significance tests -- higher power
# than binarized top1 McNemar; confirms fusion-vs-probe-alone significant on B (p<0.05,
# missed by the binarized test) but fusion-vs-AF3-alone still not significant on B
$PY -u -m src.significance_continuous --dataset A --prompt direct
$PY -u -m src.significance_continuous --dataset B --prompt cot_then_answer

# Round-5 item 3: AF3 LoRA fine-tune, Task 2 (best epoch 2: top1=0.6078, underperforms
# round-4's free contextual calibration 0.6275 and MERT-v2-alone 0.6471)
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.finetune_af3_lora --dataset B --prompt direct \
    --epochs 3 --lr 1e-4 --grad-accum 8 --device cuda:1

# Round-5 item 8: wide-and-deep hand-crafted features, Task 1 (standalone 0.4091, below
# MERT-v2; fused with MERT-v2 ties top1 but improves top3 0.8712 -> 0.8864)
$PY -m src.wide_deep_features --dataset A
$PY -m src.wide_deep_analysis --dataset A

# Round-5 item 10: label-aware augmentation ablation, from-scratch CNN, Task 1 --
# does NOT confirm the predicted B>=A>C ordering (got A=0.3258, B=0.3333, C=0.3409)
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.train_scnn --dataset A --epochs 20 --no-mixup \
    --waveform-augment none --device cuda:1 --out-dir results/scnn_aug_A_none
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.train_scnn --dataset A --epochs 20 --no-mixup \
    --waveform-augment label_preserving --device cuda:1 --out-dir results/scnn_aug_A_labelpreserving
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.train_scnn --dataset A --epochs 20 --no-mixup \
    --waveform-augment production_altering --device cuda:1 --out-dir results/scnn_aug_A_productionaltering

# Round-5 item 9 (last queue item): caption-as-features, both tasks (Task B: real
# moderate signal 0.5392; Task A: weak, 0.3182 -- below the wide-deep hand-crafted features)
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.caption_features --dataset B --model audioflamingo3 --device cuda:1
$PY -m src.ablate_classifier_pca --dataset B --layer 0 --encoder-name caption_embed_audioflamingo3
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.caption_features --dataset A --model musicflamingo --device cuda:1
$PY -m src.ablate_classifier_pca --dataset A --layer 0 --encoder-name caption_embed_musicflamingo
```

## Random seeds / determinism notes

- All sklearn model selection (`train_probe.py`'s `GridSearchCV`/`StratifiedKFold`, PCA) uses
  `random_state=0` — frozen-probe numbers should reproduce exactly given the same cached
  embeddings.
- `src/significance.py`'s bootstrap uses a seeded `np.random.default_rng(0)` — bootstrap CIs
  should reproduce exactly.
- Neural network training (`train_scnn.py`, `finetune_mert.py`, `finetune_lora.py`) is **not**
  seeded end-to-end (data loader shuffling, dropout, CUDA nondeterminism) — expect run-to-run
  variance of a few points on these specific components; this was observed and noted directly in
  `docs/progress/WORKLOG.md` during the original fine-tuning runs. The frozen-probe and fusion
  numbers above, which is where most of this project's precision claims are made, do not depend
  on this randomness.
