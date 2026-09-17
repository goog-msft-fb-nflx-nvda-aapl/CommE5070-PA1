import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score


def top_k_correct(probs, labels, k):
    topk = np.argsort(-probs, axis=1)[:, :k]
    return np.array([labels[i] in topk[i] for i in range(len(labels))])


def top1_top3(probs, labels):
    top1 = top_k_correct(probs, labels, 1).mean()
    top3 = top_k_correct(probs, labels, 3).mean()
    return float(top1), float(top3)


def confusion(labels, preds, n_class, normalize_rows=True):
    cm = confusion_matrix(labels, preds, labels=list(range(n_class)))
    if not normalize_rows:
        return cm
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    return cm, cm_norm


def ordinal_metrics(labels, preds, n_class):
    """Task 1 only (decades have a natural order)."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    abs_err = np.abs(labels - preds)
    mae = float(abs_err.mean())
    adjacent_or_exact = float((abs_err <= 1).mean())
    nonzero_err = abs_err[abs_err > 0]
    adjacent_frac_of_errors = float((nonzero_err == 1).mean()) if len(nonzero_err) else float("nan")
    qwk = float(cohen_kappa_score(labels, preds, weights="quadratic", labels=list(range(n_class))))
    return {
        "mean_abs_decade_error": mae,
        "acc_within_1_decade": adjacent_or_exact,
        "adjacent_fraction_of_errors": adjacent_frac_of_errors,
        "quadratic_weighted_kappa": qwk,
    }


def summarize(probs, labels, n_class, ordinal=False, label_names=None):
    preds = np.argmax(probs, axis=1)
    top1, top3 = top1_top3(probs, labels)
    cm, cm_norm = confusion(labels, preds, n_class)
    out = {
        "top1": top1,
        "top3": top3,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_normalized": cm_norm.tolist(),
        "labels": label_names,
    }
    if ordinal:
        out.update(ordinal_metrics(labels, preds, n_class))
    return out
