"""AF3-probability late fusion (IMPROVEMENT_FINDINGS.md section 4, item 2 -- the
cheapest closing-the-gap-to-AF3 move, reusing alm_infer.py's already-computed
teacher-forced label scores and a weighted-sum sweep in the style of ensemble.py).

Not presumed to help -- swept and measured for BOTH tasks, all backbone/prompt
combinations we have on hand, not just the Task-2 case the queue item named.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.train_probe import load_cached, run as run_probe
from src.ensemble import finetuned_mert_probs, align
from src.metrics import summarize

# best frozen-probe config per (dataset, encoder), taken directly from each
# encoder's own layer-sweep + classifier/PCA ablation results already logged
# in WORKLOG.md -- not re-searched here.
FROZEN_CONFIGS = {
    ("A", "mert_v1_330m"): {"layer": 4, "classifier": "svm", "pca_dim": 64},
    ("B", "mert_v1_330m"): {"layer": 7, "classifier": "logreg", "pca_dim": None},
    ("A", "muq_large_msd"): {"layer": 1, "classifier": "svm", "pca_dim": None},
    ("B", "muq_large_msd"): {"layer": 2, "classifier": "logreg", "pca_dim": None},
}


def frozen_probe_probs(dataset_key, encoder_name):
    cfg = FROZEN_CONFIGS[(dataset_key, encoder_name)]
    _metrics, fitted = run_probe(dataset_key, layer=cfg["layer"], classifier=cfg["classifier"],
                                  pca_dim=cfg["pca_dim"], encoder_name=encoder_name,
                                  out_dir=os.path.join(RESULTS_DIR, "af3_stack_refit"))
    cache_dir = os.path.join("cache", encoder_name, dataset_key)
    data = load_cached(dataset_key, cache_dir, cfg["layer"])
    X_val, y_val, val_ids = data["validation"]
    X_val_s = fitted["scaler"].transform(X_val)
    if fitted["pca"] is not None:
        X_val_s = fitted["pca"].transform(X_val_s)
    probs = fitted["clf"].predict_proba(X_val_s)
    return probs, y_val, list(val_ids)


def af3_probs(dataset_key, prompt_name="direct"):
    """Softmax the teacher-forced label log-prob scores already computed by
    alm_infer.py (stored per-sample in *_raw.json) -- these are genuine ranked
    scores over the closed label set, not a one-hot stand-in from free-form
    generation. Label order must match spec['labels'] (same order alm_infer.py
    iterated when it wrote label_scores)."""
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    path = os.path.join(RESULTS_DIR, "alm", f"audioflamingo3_{dataset_key}", f"{prompt_name}_raw.json")
    with open(path) as f:
        raw = json.load(f)
    probs, y, ids = [], [], []
    label_to_int = spec["label_to_int"]
    for row in raw:
        scores = np.array([row["label_scores"][lab] for lab in labels])
        p = np.exp(scores - scores.max())
        p = p / p.sum()
        probs.append(p)
        ids.append(row["sample_id"])
    # y_true isn't stored in raw.json -- pull it from the manifest via the cached
    # frozen-probe loader's val split, which already carries sample_id->label.
    return np.stack(probs), ids


def sweep(dataset_key, backbone_name, backbone_probs, backbone_labels, backbone_ids,
          af3_p, af3_ids, prompt_name, out_dir):
    spec = DATASETS[dataset_key]
    n_class = len(spec["labels"])
    af3_aligned = align(backbone_probs, backbone_ids, af3_p, af3_ids)

    results = {}
    best = (None, -1.0)
    for w_af3 in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        ens = w_af3 * af3_aligned + (1 - w_af3) * backbone_probs
        metrics = summarize(ens, backbone_labels, n_class, ordinal=spec["ordinal"], label_names=spec["labels"])
        key = f"af3{w_af3:.1f}_{backbone_name}{1 - w_af3:.1f}"
        results[key] = {"top1": metrics["top1"], "top3": metrics["top3"]}
        if metrics["top1"] > best[1]:
            best = (key, metrics["top1"])
        print(f"[{dataset_key}/{backbone_name}/af3-{prompt_name}] w_af3={w_af3:.1f} "
              f"top1={metrics['top1']:.4f} top3={metrics['top3']:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}_{backbone_name}_af3{prompt_name}.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "best": {"config": best[0], "top1": best[1]}}, f, indent=2)
    print(f"[{dataset_key}/{backbone_name}/af3-{prompt_name}] BEST {best[0]} top1={best[1]:.4f} -> {out_path}")
    return results, best


def run_all(dataset_key, device="cuda"):
    out_dir = os.path.join(RESULTS_DIR, "af3_stack")
    backbones = {}
    for (dk, enc), _cfg in FROZEN_CONFIGS.items():
        if dk != dataset_key:
            continue
        probs, y_val, val_ids = frozen_probe_probs(dataset_key, enc)
        backbones[enc] = (probs, y_val, val_ids)
    if dataset_key == "A":
        ft_p, ft_labels, ft_ids = finetuned_mert_probs(dataset_key, device)
        backbones["mert_finetuned"] = (ft_p, ft_labels, ft_ids)

    for prompt_name in ("direct", "cot_then_answer"):
        af3_p, af3_ids = af3_probs(dataset_key, prompt_name)
        for backbone_name, (probs, y_val, val_ids) in backbones.items():
            sweep(dataset_key, backbone_name, probs, y_val, val_ids, af3_p, af3_ids, prompt_name, out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run_all(args.dataset, device=args.device)
