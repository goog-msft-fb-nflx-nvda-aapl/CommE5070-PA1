"""Round-4 queue item 1 (top-ranked): Whisper sung-language-ID as a classifier feature
for Task 2, standalone and concatenated with the MuQ frozen probe. Uses the exact same
StratifiedKFold(5, shuffle=True, random_state=0) convention as train_probe.py throughout
for consistency with every other reported number in this project (an earlier ad-hoc
check used a different implicit CV split for hyperparameter search and got a
noticeably different point estimate for the same nominal config -- caught and fixed
here by using one canonical, reusable implementation instead of one-off scripts).
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, RESULTS_DIR, CACHE_DIR
from src.train_probe import load_cached
from src.metrics import summarize
from src.significance import bootstrap_ci_top1, compare
from src.af3_stack_oof import load_af3_scores, fit_temperature, calibrated_softmax, fit_fusion_weight, fit_fusion_weight_accuracy

CLASSIFIER_CV = {
    "logreg": (LogisticRegression(max_iter=5000), {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}),
    "svm": (SVC(probability=True, random_state=0, max_iter=20000),
            {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "kernel": ["rbf", "linear"]}),
}


def load_langid_features(dataset_key, stem="vocals"):
    cache_dir = os.path.join(CACHE_DIR, f"whisper_lid_{stem}", dataset_key)
    spec = DATASETS[dataset_key]
    label_to_int = spec["label_to_int"]
    data = {"train": ([], [], []), "validation": ([], [], []), "test": ([], [], [])}
    for fname in os.listdir(cache_dir):
        if not fname.endswith(".npz"):
            continue
        npz = np.load(os.path.join(cache_dir, fname), allow_pickle=True)
        split = str(npz["split"])
        label_str = str(npz["label"])
        label_int = label_to_int[label_str] if label_str else -1
        xs, ys, ids = data[split]
        xs.append(npz["feature"])
        ys.append(label_int)
        ids.append(str(npz["sample_id"]))
    return {k: (np.stack(v[0]) if v[0] else np.empty((0,)), np.array(v[1]), v[2]) for k, v in data.items()}


def fit_probe(X_train, y_train, classifier="logreg"):
    scaler = StandardScaler().fit(X_train)
    X_s = scaler.transform(X_train)
    base, grid = CLASSIFIER_CV[classifier]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(base, grid, cv=cv, scoring="accuracy")
    search.fit(X_s, y_train)
    return scaler, search.best_estimator_, search.best_score_


def oof_probe_probs(X, y, classifier="logreg", n_splits=5, seed=0):
    n_class = len(set(y))
    oof = np.zeros((len(y), n_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, held_idx in cv.split(X, y):
        scaler, clf, _ = fit_probe(X[tr_idx], y[tr_idx], classifier=classifier)
        oof[held_idx] = clf.predict_proba(scaler.transform(X[held_idx]))
    return oof


def run(dataset_key="B", encoder_name="muq_large_msd", encoder_layer=2, stem="vocals", n_boot=10000):
    spec = DATASETS[dataset_key]
    labels = spec["labels"]
    n_class = len(labels)

    lid_data = load_langid_features(dataset_key, stem=stem)
    Xtr_lid, ytr, train_ids = lid_data["train"]
    Xval_lid, yval, val_ids = lid_data["validation"]

    enc_data = load_cached(dataset_key, os.path.join(CACHE_DIR, encoder_name, dataset_key), encoder_layer)
    Xtr_enc, ytr2, train_ids2 = enc_data["train"]
    Xval_enc, yval2, val_ids2 = enc_data["validation"]
    assert list(train_ids) == list(train_ids2) and (ytr == ytr2).all(), "train sample-order mismatch"
    assert list(val_ids) == list(val_ids2) and (yval == yval2).all(), "val sample-order mismatch"

    results = {}

    # 1. language-ID alone
    scaler, clf, cv_score = fit_probe(Xtr_lid, ytr, classifier="logreg")
    lid_probs = clf.predict_proba(scaler.transform(Xval_lid))
    m = summarize(lid_probs, yval, n_class, ordinal=spec["ordinal"], label_names=labels)
    print(f"[{dataset_key}] language-ID alone: cv={cv_score:.4f} top1={m['top1']:.4f} top3={m['top3']:.4f}")
    results["langid_alone"] = {"metrics": m, "cv_score": cv_score}

    # 2. encoder alone (recomputed here for a fair same-run comparison)
    scaler_e, clf_e, cv_e = fit_probe(Xtr_enc, ytr, classifier="logreg")
    enc_probs = clf_e.predict_proba(scaler_e.transform(Xval_enc))
    m_e = summarize(enc_probs, yval, n_class, ordinal=spec["ordinal"], label_names=labels)
    print(f"[{dataset_key}] {encoder_name} layer{encoder_layer} alone: cv={cv_e:.4f} top1={m_e['top1']:.4f} top3={m_e['top3']:.4f}")
    results["encoder_alone"] = {"metrics": m_e, "cv_score": cv_e}

    # 3. concatenated features, one probe
    Xtr_cat = np.concatenate([Xtr_enc, Xtr_lid], axis=1)
    Xval_cat = np.concatenate([Xval_enc, Xval_lid], axis=1)
    for clf_name in ("logreg", "svm"):
        scaler_c, clf_c, cv_c = fit_probe(Xtr_cat, ytr, classifier=clf_name)
        cat_probs = clf_c.predict_proba(scaler_c.transform(Xval_cat))
        m_c = summarize(cat_probs, yval, n_class, ordinal=spec["ordinal"], label_names=labels)
        print(f"[{dataset_key}] {encoder_name}+langid concat ({clf_name}): cv={cv_c:.4f} top1={m_c['top1']:.4f} top3={m_c['top3']:.4f}")
        results[f"concat_{clf_name}"] = {"metrics": m_c, "cv_score": cv_c}
        if clf_name == "logreg":
            cat_probs_logreg = cat_probs

    # 4. OOF-fitted fusion of the concat probe with AF3 (leakage-free, same methodology
    #    as every other fusion this project since the validation-overfitting lesson)
    oof_cat = oof_probe_probs(Xtr_cat, ytr, classifier="logreg")
    af3_train_scores, af3_train_ids = load_af3_scores(dataset_key, "cot_then_answer", "train")
    af3_by_id = {sid: af3_train_scores[i] for i, sid in enumerate(af3_train_ids)}
    af3_train_aligned = np.stack([af3_by_id[sid] for sid in train_ids])
    T, _ = fit_temperature(af3_train_aligned, ytr)
    af3_train_cal = calibrated_softmax(af3_train_aligned, T)
    alpha_nll, _ = fit_fusion_weight(oof_cat, af3_train_cal, ytr)
    alpha_acc, _ = fit_fusion_weight_accuracy(oof_cat, af3_train_cal, ytr)

    af3_val_scores, af3_val_ids = load_af3_scores(dataset_key, "cot_then_answer", "validation")
    af3_by_id_val = {sid: af3_val_scores[i] for i, sid in enumerate(af3_val_ids)}
    af3_val_aligned = np.stack([af3_by_id_val[sid] for sid in val_ids])
    af3_val_cal = calibrated_softmax(af3_val_aligned, T)

    fused_nll = alpha_nll * cat_probs_logreg + (1 - alpha_nll) * af3_val_cal
    fused_acc = alpha_acc * cat_probs_logreg + (1 - alpha_acc) * af3_val_cal
    m_fnll = summarize(fused_nll, yval, n_class, ordinal=spec["ordinal"], label_names=labels)
    m_facc = summarize(fused_acc, yval, n_class, ordinal=spec["ordinal"], label_names=labels)
    print(f"[{dataset_key}] OOF-fused concat+AF3 (NLL, alpha={alpha_nll:.3f}): top1={m_fnll['top1']:.4f} top3={m_fnll['top3']:.4f}")
    print(f"[{dataset_key}] OOF-fused concat+AF3 (acc, alpha={alpha_acc:.3f}): top1={m_facc['top1']:.4f} top3={m_facc['top3']:.4f}")
    results["oof_fused_concat_af3_nll"] = {"metrics": m_fnll, "alpha": alpha_nll}
    results["oof_fused_concat_af3_acc"] = {"metrics": m_facc, "alpha": alpha_acc}

    # significance: concat-alone vs encoder-alone, vs AF3-alone; fused vs concat-alone, vs AF3-alone
    configs = {"langid_alone": lid_probs, "encoder_alone": enc_probs, "concat_logreg": cat_probs_logreg,
               "af3_alone": af3_val_cal, "oof_fused_nll": fused_nll, "oof_fused_acc": fused_acc}
    cis = {name: bootstrap_ci_top1(p, yval, n_boot=n_boot) for name, p in configs.items()}
    for name, ci in cis.items():
        print(f"  {name}: top1={ci['top1']:.4f} 95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

    comps = {}
    for a, b in [("concat_logreg", "encoder_alone"), ("concat_logreg", "af3_alone"),
                 ("oof_fused_acc", "af3_alone"), ("oof_fused_acc", "concat_logreg")]:
        comps[f"{a}_vs_{b}"] = compare(a, configs[a], b, configs[b], yval, n_boot=n_boot)
        print(f"  {a} vs {b}: {comps[f'{a}_vs_{b}']['paired_bootstrap']}")

    out = {"results": results, "individual_ci": cis, "comparisons": comps}
    out_dir = os.path.join(RESULTS_DIR, "langid_probe")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}_{stem}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{dataset_key}] saved -> {out_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="B", choices=["A", "B"])
    p.add_argument("--stem", default="vocals", choices=["vocals", "no_vocals"])
    args = p.parse_args()
    run(args.dataset, stem=args.stem)
