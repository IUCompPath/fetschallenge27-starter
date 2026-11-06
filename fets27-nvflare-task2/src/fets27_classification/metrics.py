"""Binary classification metric calculations without extra dependencies."""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLD = 0.5


def binary_metrics(
    labels: list[int],
    probabilities: list[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float | list[list[int]]]:
    """Compute thresholded and ranking metrics for binary classifier outputs.

    Label ``1`` is the positive class (meningioma) and label ``0`` the negative
    class (glioma). The official key metric is ``balanced_accuracy`` at the fixed
    default threshold; all other values are reported alongside it.

    Args:
        labels: Ground-truth integer labels (0 or 1).
        probabilities: Predicted probability of the positive class.
        threshold: Decision threshold applied to ``probabilities``.

    Returns:
        A dictionary with balanced accuracy, accuracy, AUROC, macro F1,
        sensitivity, specificity, MCC, and the 2x2 confusion matrix.
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1_pos = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    neg_precision = tn / max(tn + fn, 1)
    f1_neg = 2 * neg_precision * specificity / max(neg_precision + specificity, 1e-12)
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "accuracy": float((pred == y).mean()),
        "auroc": _auc(y, p),
        "macro_f1": (f1_pos + f1_neg) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mcc": _mcc(tn, fp, fn, tp),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _mcc(tn: int, fp: int, fn: int, tp: int) -> float:
    """Return the Matthews correlation coefficient for a 2x2 confusion matrix.

    MCC is undefined whenever any confusion-matrix marginal is zero; ``0.0`` is
    returned in that degenerate case so rankings stay finite.
    """
    numerator = float(tp * tn - fp * fn)
    denominator = float(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    ) ** 0.5
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Return the rank-based AUROC, or NaN when only one class is present."""
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    for value in np.unique(p):
        tied = np.flatnonzero(p == value)
        if len(tied) > 1:
            ranks[tied] = ranks[tied].mean()
    return float(
        (ranks[y == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )
