"""Round-3 queue item 3: higher-power statistical tests on the OOF-fusion comparisons
that matter most right now -- does the fusion (validation-swept or OOF-refit) actually
beat AF3 alone? Binarized top1 McNemar gave p=0.21 (B, validation-swept) / already
resolved as "no" via the OOF refit itself for B, and p=0.29 for A's probe comparison.
This re-tests the same comparisons with a continuous scoring rule (Brier score, paired
Wilcoxon) and a one-sided McNemar (matching our directional hypothesis: fusion >= AF3,
not merely "different"), which use more of the available information per sample than a
binarized right/wrong test.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.af3_stack_oof import (PROBE_CONFIGS, fit_probe, fit_temperature, fit_fusion_weight,
                                fit_fusion_weight_accuracy, oof_probe_probs, load_af3_scores,
                                calibrated_softmax)
from src.train_probe import load_cached
from src.significance import paired_wilcoxon_brier, mcnemar_one_sided, mcnemar_exact


def reconstruct(dataset_key, prompt_name, encoder_name="muq_large_msd"):
    """Rebuild the exact probe-alone / af3-alone-calibrated / oof-fused probability
    vectors from af3_stack_oof.run() -- deterministic given the same cached embeddings
    and AF3 raw scores, so this exactly reproduces what run() computed."""
    probe_cfg = PROBE_CONFIGS[dataset_key]
    layer, classifier = probe_cfg["layer"], probe_cfg["classifier"]
    cache_dir = os.path.join("cache", encoder_name, dataset_key)
    data = load_cached(dataset_key, cache_dir, layer)
    X_train, y_train, train_ids = data["train"]
    X_val, y_val, val_ids = data["validation"]

    af3_train_scores, af3_train_ids = load_af3_scores(dataset_key, prompt_name, "train")
    af3_by_id = {sid: af3_train_scores[i] for i, sid in enumerate(af3_train_ids)}
    af3_train_aligned = np.stack([af3_by_id[sid] for sid in train_ids])
    T, _ = fit_temperature(af3_train_aligned, y_train)
    oof_probe = oof_probe_probs(X_train, y_train, classifier=classifier)
    af3_train_cal = calibrated_softmax(af3_train_aligned, T)
    alpha, _ = fit_fusion_weight(oof_probe, af3_train_cal, y_train)
    alpha_acc, _ = fit_fusion_weight_accuracy(oof_probe, af3_train_cal, y_train)

    scaler, clf = fit_probe(X_train, y_train, classifier=classifier)
    probe_val = clf.predict_proba(scaler.transform(X_val))
    af3_val_scores, af3_val_ids = load_af3_scores(dataset_key, prompt_name, "validation")
    af3_by_id_val = {sid: af3_val_scores[i] for i, sid in enumerate(af3_val_ids)}
    af3_val_aligned = np.stack([af3_by_id_val[sid] for sid in val_ids])
    af3_val_cal = calibrated_softmax(af3_val_aligned, T)

    fused_nll = alpha * probe_val + (1 - alpha) * af3_val_cal
    fused_acc = alpha_acc * probe_val + (1 - alpha_acc) * af3_val_cal
    return {"probe_alone": probe_val, "af3_alone": af3_val_cal,
            "oof_fused_nll": fused_nll, "oof_fused_acc": fused_acc}, y_val


def run(dataset_key, prompt_name):
    spec = DATASETS[dataset_key]
    n_class = len(spec["labels"])
    configs, y_val = reconstruct(dataset_key, prompt_name)

    results = {}
    for fused_name in ("oof_fused_nll", "oof_fused_acc"):
        for other_name in ("af3_alone", "probe_alone"):
            wilcoxon = paired_wilcoxon_brier(configs[fused_name], configs[other_name], y_val, n_class)
            mcnemar_1s = mcnemar_one_sided(configs[fused_name], configs[other_name], y_val)
            mcnemar_2s = mcnemar_exact(configs[fused_name], configs[other_name], y_val)
            key = f"{fused_name}_vs_{other_name}"
            results[key] = {"wilcoxon_brier": wilcoxon, "mcnemar_one_sided": mcnemar_1s,
                             "mcnemar_two_sided": mcnemar_2s}
            print(f"[{dataset_key}] {key}:")
            print(f"    Brier: {fused_name}={wilcoxon['mean_brier_a']:.4f} {other_name}={wilcoxon['mean_brier_b']:.4f} "
                  f"Wilcoxon p={wilcoxon['wilcoxon_p']:.4f}")
            print(f"    McNemar one-sided p={mcnemar_1s['p_one_sided']:.4f} "
                  f"(two-sided p={mcnemar_2s['p_value']:.4f}) n01={mcnemar_1s['n01']} n10={mcnemar_1s['n10']}")

    out_dir = os.path.join(RESULTS_DIR, "significance_continuous")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}_af3{prompt_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{dataset_key}] saved -> {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--prompt", required=True)
    args = p.parse_args()
    run(args.dataset, args.prompt)
