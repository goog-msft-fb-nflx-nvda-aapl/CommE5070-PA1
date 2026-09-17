"""Generate the required submission JSON: top-3 labels per test sample, both datasets.
Combines predictions from any {dataset: probs, ids, label_names} source (SCNN multi-crop
eval, or a probe's predict_proba) written by each track's own driver script into cache/.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR, DATASETS


def probs_to_top3_json(probs, ids, label_names):
    out = {}
    for i, sample_id in enumerate(ids):
        top3_idx = np.argsort(-probs[i])[:3]
        out[sample_id] = [label_names[j] for j in top3_idx]
    return out


def export_scnn_test_predictions(dataset_key, checkpoint_path, out_json, device="cuda", crop_seconds=3.7, n_crops=8):
    import torch
    from src.scnn import ShortChunkCNN
    from src.train_scnn import eval_multicrop

    spec = DATASETS[dataset_key]
    model = ShortChunkCNN(n_class=len(spec["labels"])).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    probs, _labels, ids, label_names = eval_multicrop(model, dataset_key, "test", device, crop_seconds, n_crops)
    out = probs_to_top3_json(probs, ids, label_names)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{dataset_key}] wrote {len(out)} test predictions to {out_json}")
    return out_json


def export_probe_test_predictions(dataset_key, fitted_scaler, fitted_pca, fitted_clf, layer, out_json,
                                   cache_dir=None, encoder_name="mert_v1_330m"):
    from src.config import CACHE_DIR
    from src.train_probe import load_cached

    cache_dir = cache_dir or os.path.join(CACHE_DIR, encoder_name, dataset_key)
    spec = DATASETS[dataset_key]
    data = load_cached(dataset_key, cache_dir, layer)
    X_test, _y_test, ids = data["test"]
    X_test_s = fitted_scaler.transform(X_test)
    if fitted_pca is not None:
        X_test_s = fitted_pca.transform(X_test_s)
    probs = fitted_clf.predict_proba(X_test_s)
    out = probs_to_top3_json(probs, ids, spec["labels"])
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{dataset_key}] wrote {len(out)} test predictions to {out_json}")
    return out_json


def build_submission(preds_a_path, preds_b_path, out_path, student_id):
    with open(preds_a_path) as f:
        a = json.load(f)
    with open(preds_b_path) as f:
        b = json.load(f)
    submission = {"dataset_A": a, "dataset_B": b}
    out_file = os.path.join(out_path, f"{student_id}.json")
    with open(out_file, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"wrote {out_file}")
    return out_file


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preds-a", required=True, help="json: {sample_id: [top3 labels]} for dataset A test")
    p.add_argument("--preds-b", required=True, help="json: {sample_id: [top3 labels]} for dataset B test")
    p.add_argument("--out-dir", default=RESULTS_DIR)
    p.add_argument("--student-id", required=True)
    args = p.parse_args()
    build_submission(args.preds_a, args.preds_b, args.out_dir, args.student_id)
