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
```

Frozen-probe classifiers (MERT/MuQ + logreg/SVM/PCA) are refit on the fly by `train_probe.py`
inside `ensemble.py`/`af3_stack.py`/`significance.py` below -- GridSearchCV is deterministic
(`random_state=0` everywhere it appears), so no separate probe-training step is needed.

## 3. Best Task 1 (decade) config

**IMPORTANT, read this before citing a Task 1 number**: the validation-swept fusion point
estimate (3b below, top1=0.5530) was picked by sweeping the weight directly on the 132-sample
validation set. A leakage-free out-of-fold (OOF) refit — fitting the weight on 1026 training-set
predictions instead — gives a very similar, more defensible number (top1=0.5379, 3a below).
Prefer 3a when a single number is needed; 3b is kept only as the original point estimate.

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

**IMPORTANT, read this before citing a Task 2 number**: the validation-swept fusion (4b below,
top1=0.6569) does **not** reproduce under a leakage-free OOF refit (4a below gives ≈0.57-0.59,
essentially AF3-alone's own score) — the 0.6569 figure was validation-set overfitting in the
weight choice, confirmed by two independent pieces of evidence (the OOF refit itself, and the
earlier bootstrap/McNemar test already showing p=0.21 vs. AF3-alone). **Do not report 0.6569 as
the headline Task 2 number without this caveat; prefer citing ≈0.588 (AF3 zero-shot alone, or
equivalently the OOF-refit fusion).**

**4a. OOF-refit fusion (the honest number) — top1=0.5686 (NLL-optimal) or 0.5882
(accuracy-optimal), both ≈ AF3-alone's own 0.5882**:

```bash
# requires step 2's AF3 train-split inference to have been run first
env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python $PY -u -m src.af3_stack_oof --dataset B --prompt cot_then_answer
# prints both NLL-optimal and accuracy-optimal alpha variants;
# saved to results/af3_stack_oof/B_muq_large_msd_af3cot_then_answer.json
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

## 6. Negative result kept for completeness (not part of any best config)

LoRA fine-tune of MERT (Task B) plateaus at top1=0.382, well below the frozen-probe baseline —
run to confirm/refute, not to reproduce a "best" number:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 $PY -u -m src.finetune_lora --dataset B --epochs 40 \
    --no-time-mask --no-mixup --out-dir results/mert_lora_B_noaug_40ep --device cuda:1
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
