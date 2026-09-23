"""Contextual calibration (Zhao et al. 2021, arXiv:2102.09690) for AF3's zero-shot label
scores -- round-4 queue item 3. Estimates the model's label prior using a content-free
input (near-silent audio, same prompt), then divides it out of every real clip's label
distribution: p_calibrated(label|x) proportional to p(label|x) / p(label|null). In our
teacher-forced average-log-prob scoring, this is just `score(label|x) - score(label|null)`
per candidate before the softmax -- a single correction vector per (dataset, prompt),
computed once from one extra inference call, then applied to every already-computed
AF3 raw.json (no new per-clip inference needed).

Unlike the earlier weight-sweep approach (which overfit the validation set), this is
parameter-free given the null estimate -- it cannot overfit the same way, since nothing
is fit against our labels at all, only against the model's own unconditional bias.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.metrics import summarize
from src.significance import bootstrap_ci_top1, compare


def estimate_null_scores(dataset_key, prompt_name, device="cuda", n_repeats=3, duration_s=30.0, sr=24000):
    """Score the 6 candidate labels on near-silent (very low-amplitude noise, not
    exact zeros -- avoids any special-cased silence handling in the audio front-end)
    audio, averaged over a few noise draws for stability. This is the content-free
    input estimate of the model's own label prior."""
    from src.alm_infer import PROMPTS, load_audio_flamingo3, DECADE_ALIASES, MARKET_ALIASES

    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    axis = "decade this track was released in" if dataset_key == "A" else "release market of this track"
    choices_str = ", ".join(labels)
    prompt_spec = PROMPTS[prompt_name]
    prompt_text = prompt_spec["template"].format(axis=axis, choices=choices_str)
    answer_prefix = prompt_spec["answer_prefix"]

    _generate_fn, score_fn = load_audio_flamingo3(device)
    rng = np.random.default_rng(0)
    all_scores = []
    for _ in range(n_repeats):
        y = (rng.standard_normal(int(duration_s * sr)) * 1e-4).astype(np.float32)
        scores = score_fn(y, sr, prompt_text, labels, answer_prefix=answer_prefix)
        all_scores.append(scores)
    null_scores = np.mean(all_scores, axis=0)
    return dict(zip(labels, null_scores.tolist()))


def apply_calibration(dataset_key, prompt_name, split, null_scores, model_name="audioflamingo3"):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    label_to_int = spec["label_to_int"]
    dirname = f"{model_name}_{dataset_key}" if split == "validation" else f"{model_name}_{dataset_key}_{split}"
    path = os.path.join(RESULTS_DIR, "alm", dirname, f"{prompt_name}_raw.json")
    with open(path) as f:
        raw = json.load(f)
    null_vec = np.array([null_scores[lab] for lab in labels])

    probs_uncal, probs_cal, ids = [], [], []
    for row in raw:
        scores = np.array([row["label_scores"][lab] for lab in labels])
        p_uncal = np.exp(scores - scores.max())
        p_uncal = p_uncal / p_uncal.sum()
        cal_scores = scores - null_vec
        p_cal = np.exp(cal_scores - cal_scores.max())
        p_cal = p_cal / p_cal.sum()
        probs_uncal.append(p_uncal)
        probs_cal.append(p_cal)
        ids.append(row["sample_id"])
    return np.stack(probs_uncal), np.stack(probs_cal), ids


def run(dataset_key, prompt_name, device="cuda", n_boot=10000):
    from src.data import load_manifest
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    n_class = len(labels)
    label_to_int = spec["label_to_int"]

    null_scores = estimate_null_scores(dataset_key, prompt_name, device=device)
    print(f"[{dataset_key}/{prompt_name}] null (content-free) label scores: {null_scores}")

    probs_uncal, probs_cal, ids = apply_calibration(dataset_key, prompt_name, "validation", null_scores)
    rows, _ = load_manifest(dataset_key, "validation")
    row_by_id = {r["sample_id"]: r for r in rows}
    y_val = np.array([label_to_int[row_by_id[sid]["label"]] for sid in ids])

    m_uncal = summarize(probs_uncal, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
    m_cal = summarize(probs_cal, y_val, n_class, ordinal=spec["ordinal"], label_names=labels)
    print(f"[{dataset_key}/{prompt_name}] uncalibrated: top1={m_uncal['top1']:.4f} top3={m_uncal['top3']:.4f}")
    print(f"[{dataset_key}/{prompt_name}] calibrated:   top1={m_cal['top1']:.4f} top3={m_cal['top3']:.4f}")

    cis = {"uncalibrated": bootstrap_ci_top1(probs_uncal, y_val, n_boot=n_boot),
           "calibrated": bootstrap_ci_top1(probs_cal, y_val, n_boot=n_boot)}
    for name, ci in cis.items():
        print(f"  {name}: top1={ci['top1']:.4f} 95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")
    comp = compare("calibrated", probs_cal, "uncalibrated", probs_uncal, y_val, n_boot=n_boot)
    print(f"  calibrated vs uncalibrated: {comp['paired_bootstrap']}")

    out = {"null_scores": null_scores, "uncalibrated_metrics": m_uncal, "calibrated_metrics": m_cal,
           "individual_ci": cis, "calibrated_vs_uncalibrated": comp}
    out_dir = os.path.join(RESULTS_DIR, "contextual_calibration")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}_{prompt_name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{dataset_key}/{prompt_name}] saved -> {out_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(args.dataset, args.prompt, device=args.device)
