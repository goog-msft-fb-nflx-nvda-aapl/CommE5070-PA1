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
```

## 1. Feature extraction (run once each, cached to `cache/` — gitignored, ~GBs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.mert_features --dataset A --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.mert_features --dataset B --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.muq_features  --dataset A --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.muq_features  --dataset B --device cuda:1
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
# both tasks -- needed as an input to the fusion configs below
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.alm_infer --dataset A --model audioflamingo3 --split validation --device cuda:1
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.alm_infer --dataset B --model audioflamingo3 --split validation --device cuda:1
```

Frozen-probe classifiers (MERT/MuQ + logreg/SVM/PCA) are refit on the fly by `train_probe.py`
inside `ensemble.py`/`af3_stack.py`/`significance.py` below -- GridSearchCV is deterministic
(`random_state=0` everywhere it appears), so no separate probe-training step is needed.

## 3. Best Task 1 (decade) config

**Statistically defensible number — top1=0.5227, top3=0.8561** (bootstrap/McNemar-confirmed
best; the higher point estimate below is not distinguishable from this at n=132):

```bash
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.ensemble --dataset A --mode 3way --device cuda:1
# best sweep point: w_ft=0.20 w_cnn=0.40 w_mert=0.40 (printed + saved to
# results/ensemble_A/3way_sweep.json)
```

**Best point estimate (not statistically confirmed as better) — top1=0.5530, top3=0.8485**:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 $PY -m src.af3_stack --dataset A --device cuda:1
# best sweep point: MuQ(layer1,svm,noPCA) probe + AF3-direct label probs, w_af3=0.5
# (results/af3_stack/A_muq_large_msd_af3direct.json)
```

## 4. Best Task 2 (market) config

**top1=0.6569, top3=0.8824** — significantly beats every single-model config (p<0.01 vs. MuQ
alone and MERT alone, paired bootstrap + McNemar); not confirmed to beat AF3 alone specifically
(p=0.21) -- see `docs/progress/WORKLOG.md` "Statistical significance" section for the full caveat.

```bash
env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python $PY -u -m src.af3_stack --dataset B --device cuda:1
# best sweep point: MuQ(layer2,logreg,noPCA) probe + AF3-cot_then_answer label probs, w_af3=0.4
# (results/af3_stack/B_muq_large_msd_af3cot_then_answer.json)
```

(`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` isn't required by this script itself, but is
kept in the exact invocation used to produce the logged numbers for a byte-for-byte-reproducible
command history; harmless if omitted.)

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
