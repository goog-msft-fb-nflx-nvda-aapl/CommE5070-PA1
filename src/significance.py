"""Statistical significance for Task 2 (and, since it's now equally relevant, Task 1)
comparisons -- queue item from IMPROVEMENT_FINDINGS.md section 2/6, made urgent by the
2026-09-18 AF3-fusion result (src/af3_stack.py), whose weight was picked by sweeping the
validation set itself and needs an honest robustness check before being trusted.

Two things computed, both on the validation split (n=132 for A, n=102 for B):
1. Bootstrap CI on top1 accuracy for each config individually (resample samples with
   replacement, n_boot times, recompute accuracy each time).
2. Paired bootstrap + exact McNemar test for specific "does X really beat Y" claims --
   paired because the same resampled indices are used for both configs each draw, which
   is the statistically correct way to test a *difference* (unpaired CIs can overlap even
   when a paired test would show significance, and vice versa).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR
from src.af3_stack import frozen_probe_probs, af3_probs
from src.ensemble import finetuned_mert_probs, cnn_probs, align

RNG = np.random.default_rng(0)


def bootstrap_ci_top1(probs, y_true, n_boot=10000, alpha=0.05):
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(np.float64)
    n = len(y_true)
    boot_accs = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        boot_accs[b] = correct[idx].mean()
    lo, hi = np.percentile(boot_accs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"top1": float(correct.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n": n}


def paired_bootstrap_diff(probs_a, probs_b, y_true, n_boot=10000, alpha=0.05):
    """diff = acc(a) - acc(b), same resample indices used for both each draw."""
    preds_a = probs_a.argmax(axis=1)
    preds_b = probs_b.argmax(axis=1)
    correct_a = (preds_a == y_true).astype(np.float64)
    correct_b = (preds_b == y_true).astype(np.float64)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = RNG.integers(0, n, n)
        diffs[k] = correct_a[idx].mean() - correct_b[idx].mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_a_not_better = float((diffs <= 0).mean())  # fraction of resamples where a does NOT beat b
    return {"diff_mean": float(diffs.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "frac_a_not_better": p_a_not_better}


def mcnemar_exact(probs_a, probs_b, y_true):
    """Exact two-sided McNemar test (binomial on the discordant pairs) -- the standard
    paired test for two classifiers' predictions on the same sample set."""
    from scipy.stats import binomtest
    preds_a = probs_a.argmax(axis=1)
    preds_b = probs_b.argmax(axis=1)
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true
    n01 = int(np.sum(correct_a & ~correct_b))  # a right, b wrong
    n10 = int(np.sum(~correct_a & correct_b))  # a wrong, b right
    n_discordant = n01 + n10
    if n_discordant == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    result = binomtest(min(n01, n10), n_discordant, 0.5)
    return {"n01": n01, "n10": n10, "p_value": float(result.pvalue)}


def compare(name_a, probs_a, name_b, probs_b, y_true, n_boot=10000):
    pb = paired_bootstrap_diff(probs_a, probs_b, y_true, n_boot=n_boot)
    mc = mcnemar_exact(probs_a, probs_b, y_true)
    print(f"  {name_a} vs {name_b}: diff={pb['diff_mean']:+.4f} CI=[{pb['ci_lo']:+.4f},{pb['ci_hi']:+.4f}] "
          f"excludes_zero={pb['excludes_zero']}  McNemar p={mc['p_value']:.4f} (n01={mc['n01']},n10={mc['n10']})")
    return {"paired_bootstrap": pb, "mcnemar": mc}


def run_B(device="cuda:1", n_boot=10000):
    spec = DATASETS["B"]
    muq_p, y, ids = frozen_probe_probs("B", "muq_large_msd")
    mert_p, _, mert_ids = frozen_probe_probs("B", "mert_v1_330m")
    mert_p = align(muq_p, ids, mert_p, mert_ids)
    af3cot_p, af3_ids = af3_probs("B", "cot_then_answer")
    af3cot_p = align(muq_p, ids, af3cot_p, af3_ids)
    af3dir_p, af3_ids2 = af3_probs("B", "direct")
    af3dir_p = align(muq_p, ids, af3dir_p, af3_ids2)

    fused_best = 0.4 * af3cot_p + 0.6 * muq_p  # w_af3=0.4, the af3_stack.py-reported best

    configs = {"muq_alone": muq_p, "mert_alone": mert_p, "af3_cot_alone": af3cot_p,
               "af3_direct_alone": af3dir_p, "fused_best_muq_af3cot": fused_best}

    print("[B] individual bootstrap CIs (n_boot=%d):" % n_boot)
    cis = {}
    for name, p in configs.items():
        ci = bootstrap_ci_top1(p, y, n_boot=n_boot)
        cis[name] = ci
        print(f"  {name}: top1={ci['top1']:.4f} 95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

    print("[B] paired comparisons vs the fused-best config:")
    comps = {}
    for name in ("muq_alone", "mert_alone", "af3_cot_alone", "af3_direct_alone"):
        comps[f"fused_best_vs_{name}"] = compare("fused_best_muq_af3cot", fused_best, name, configs[name], y, n_boot=n_boot)

    out = {"individual_ci": cis, "paired_comparisons": comps}
    out_path = os.path.join(RESULTS_DIR, "significance_B.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[B] saved -> {out_path}")
    return out


def run_A(device="cuda:1", n_boot=10000):
    spec = DATASETS["A"]
    muq_p, y, ids = frozen_probe_probs("A", "muq_large_msd")
    mert_p, _, mert_ids = frozen_probe_probs("A", "mert_v1_330m")
    mert_p = align(muq_p, ids, mert_p, mert_ids)
    ft_p, _, ft_ids = finetuned_mert_probs("A", device)
    ft_p = align(muq_p, ids, ft_p, ft_ids)
    cnn_p, _, cnn_ids = cnn_probs("A", device)
    cnn_p = align(muq_p, ids, cnn_p, cnn_ids)
    af3dir_p, af3_ids = af3_probs("A", "direct")
    af3dir_p = align(muq_p, ids, af3dir_p, af3_ids)

    old_best_ensemble = 0.2 * ft_p + 0.4 * cnn_p + 0.4 * mert_p  # prior WORKLOG best (0.5227)
    fused_best = 0.5 * af3dir_p + 0.5 * muq_p  # w_af3=0.5, the af3_stack.py-reported new best

    configs = {"muq_alone": muq_p, "mert_finetuned_alone": ft_p, "old_best_3way_ensemble": old_best_ensemble,
               "af3_direct_alone": af3dir_p, "fused_best_muq_af3direct": fused_best}

    print("[A] individual bootstrap CIs (n_boot=%d):" % n_boot)
    cis = {}
    for name, p in configs.items():
        ci = bootstrap_ci_top1(p, y, n_boot=n_boot)
        cis[name] = ci
        print(f"  {name}: top1={ci['top1']:.4f} 95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

    print("[A] paired comparisons vs the fused-best config:")
    comps = {}
    for name in ("muq_alone", "mert_finetuned_alone", "old_best_3way_ensemble", "af3_direct_alone"):
        comps[f"fused_best_vs_{name}"] = compare("fused_best_muq_af3direct", fused_best, name, configs[name], y, n_boot=n_boot)

    out = {"individual_ci": cis, "paired_comparisons": comps}
    out_path = os.path.join(RESULTS_DIR, "significance_A.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[A] saved -> {out_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()
    if args.dataset == "A":
        run_A(device=args.device, n_boot=args.n_boot)
    else:
        run_B(device=args.device, n_boot=args.n_boot)
