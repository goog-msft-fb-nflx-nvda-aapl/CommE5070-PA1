# CommE5070 PA1 — Inference README (Student ID: R13921031)

This README explains how to reproduce the predictions in `R13921031.json` from a test-set
path on your own machine. For the full development history (every experiment tried,
including negative results), see `docs/progress/WORKLOG.md` and `docs/REPRODUCE.md` in
this same folder — this README covers only the two configs actually submitted.

**Google Drive folder (primary — this folder):** https://drive.google.com/drive/folders/13_Ufj30hwtYVkmN2QFgmXyjuOBfn35iZ?usp=sharing

**GitHub repo (full history, all experiments; secondary/mirror):** https://github.com/goog-msft-fb-nflx-nvda-aapl/CommE5070-PA1
(private until the submission deadline per course policy on releasing predictions before
grading; will be made public immediately after 2026-10-05. The URL will not change.)

## 1. What's in this folder

```
README.md                          <- this file
requirements.txt                   <- pip-installable dependencies (inference only)
R13921031.json                     <- the submitted predictions (for reference/cross-check)
checkpoints/
  task_a_mertv2_layer10.joblib     <- {scaler, pca, clf} for Task A's MERT-v2 branch
  task_a_pupum2d.joblib            <- {scaler, pca, clf} for Task A's PupuM2D branch
  task_b_mertv2_meanall.joblib     <- {scaler, pca, clf} for Task B's MERT-v2 branch
src/                                <- full project source (only a few files are used
                                       for inference; the rest is the experiment history)
docs/                               <- development docs (WORKLOG.md, REPRODUCE.md)
```

The `checkpoints/` files are the trained **frozen linear probes** (a scikit-learn
`StandardScaler` + optional `PCA` + `LogisticRegression`, fit on the 1026/798-sample
training sets for Task A/B respectively). They are included for convenience and as a
determinism safety net, but everything is also fully reproducible from source since the
pipeline is deterministic (`random_state=0` everywhere) — `src/final_predictions.py`
refits them from scratch and self-checks against our reported validation numbers before
producing any test prediction (see step 4).

**What is *not* included, per the assignment's instructions**: the released dataset, any
embedding/model cache, and every other checkpoint from this project's ~30 other
experiments (only the two submitted configs' small linear-probe objects are here — the
large pretrained encoder weights below are public and downloaded automatically by the
code, not re-uploaded).

## 2. Models and configs used (with citations)

**Task 1 (release-decade classification)**: `MERT-v2-30s` (layer 10 hidden states, mean-
pooled) linear-probe-fused with `PupuM2D-Large` (global mean-pooled embedding, PCA-128),
fusion weight α=0.6 on MERT-v2 / 0.4 on PupuM2D. Validation top-1 = 0.5606, top-3 = 0.8485
(n=132; not confirmed to beat the simpler MERT-v2-alone number, 0.5455, in significance
testing — see WORKLOG.md — but it is our best point estimate).

**Task 2 (release-market classification)**: `MERT-v2-30s` (mean of all 24 layers'
hidden states, PCA-128) linear-probe-fused with `Audio Flamingo 3` zero-shot teacher-
forced label scoring (`direct` prompt), contextually calibrated (Zhao et al. 2021,
arXiv:2102.09690 — subtract a content-free/near-silent-audio baseline score before the
softmax), fusion weight α=0.8 on MERT-v2 / 0.2 on calibrated AF3. Validation top-1 =
0.6569, top-3 = 0.8922 (n=102; also not confirmed to beat MERT-v2-alone, 0.6471 — see
WORKLOG.md).

Both fusion weights were fit via leakage-free out-of-fold (5-fold) cross-validation on
the training set only — never on validation or test — and are used here as fixed
constants (see `src/encoder_fusion_oof.py` and `src/contextual_calibration.py` for how
they were originally derived).

| Component | Source | License/access |
|---|---|---|
| MERT-v2-30s | https://huggingface.co/m-a-p/MERT-v2-30s | public, `trust_remote_code=True` |
| PupuM2D-Large | code: https://github.com/sizigi/PupuM2D · weights: https://huggingface.co/spellbrush/PupuM2D | public, MIT license |
| Audio Flamingo 3 | https://huggingface.co/nvidia/audio-flamingo-3-hf | public, `trust_remote_code=True` |
| Contextual calibration | Zhao et al. 2021, arXiv:2102.09690 | method, no code dependency |

No API keys or tokens are required — all three models are public and download
automatically via `from_pretrained(...)` on first use (cached under `~/.cache/huggingface`
by default; expect ~2GB for MERT-v2, ~1.2GB for PupuM2D-Large, ~17GB for Audio Flamingo 3).

## 3. Environment setup

```bash
pip install -r requirements.txt

# PupuM2D is not on PyPI -- clone it and fix one upstream bug (their model/__init__.py
# references a stale module name; this one-line patch is required before import works):
git clone https://github.com/sizigi/PupuM2D.git external/PupuM2D
echo 'from .pupujepa import *' > external/PupuM2D/model/__init__.py
```

GPU is strongly recommended (Audio Flamingo 3 is an 8B-parameter model) but not strictly
required — everything here runs in `float32`/CPU-compatible mode, just slower.

## 4. Pointing at the test-set path

Set the dataset root via the `PA1_ROOT` environment variable, or place the dataset so
that this structure exists relative to your working directory:

```
<PA1_ROOT or cwd>/dataset/extracted/dataset_A/manifest.csv
<PA1_ROOT or cwd>/dataset/extracted/dataset_A/audio/*.wav
<PA1_ROOT or cwd>/dataset/extracted/dataset_B/manifest.csv
<PA1_ROOT or cwd>/dataset/extracted/dataset_B/audio/*.wav
```

This is the same layout the original dataset release uses — `manifest.csv` has a `split`
column; rows with `split == "test"` are what gets predicted. If your test set's
`manifest.csv` doesn't set a `label` column for test rows (ours doesn't — it's blind),
that's expected and handled.

## 5. Running inference

```bash
export PA1_ROOT=/path/to/your/dataset/root   # if not running from the project root

# Step 1: extract MERT-v2-30s features (both tasks) and PupuM2D-Large features (Task A)
# for the test split specifically (train/validation are only needed if you want to
# re-verify the reported validation numbers, not for generating test predictions).
python -m src.mertv2_features --dataset A --device cuda:0
python -m src.mertv2_features --dataset B --device cuda:0
python -m src.pupum2d_features --dataset A --variant large --device cuda:0

# Step 2: Audio Flamingo 3 zero-shot scoring on Task B's test split (direct prompt)
python -m src.alm_infer --dataset B --model audioflamingo3 --split test --device cuda:0

# Step 3: generate predictions (also re-verifies against our reported validation
# numbers first, and refuses to write test predictions if that check fails)
python -m src.final_predictions --student-id R13921031
```

This produces `results/R13921031.json` in the required format:
```json
{"dataset_A": {"<sample_id>": ["<top1>", "<top2>", "<top3>"], ...},
 "dataset_B": {"<sample_id>": ["<top1>", "<top2>", "<top3>"], ...}}
```

All 132 Dataset A test samples and 102 Dataset B test samples are included exactly once,
each with exactly 3 labels in descending confidence order.

## 6. Notes on reproducibility

- All scikit-learn model selection (`GridSearchCV`/`StratifiedKFold`, `PCA`) uses
  `random_state=0` — results should reproduce exactly given the same cached embeddings.
- Audio Flamingo 3's teacher-forced scoring is a pure forward pass (no sampling), so it
  is also exactly deterministic.
- If `python -m src.final_predictions` reports a validation-reproduction `MISMATCH`
  rather than `MATCH`, do not trust the resulting test predictions — this indicates an
  environment or data difference from what produced the numbers in this report, and
  should be reported rather than silently used.
