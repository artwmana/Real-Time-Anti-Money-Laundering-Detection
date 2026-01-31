from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def _get_proba(model: Any, X) -> np.ndarray:
    """
    Safely obtain positive-class probabilities/scores for ROC/PR curves.
    Falls back to decision_function if predict_proba is unavailable.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        raise ValueError("predict_proba returned unexpected shape")

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.ravel(scores)
        # Min-max to [0,1] to avoid NaNs in roc_auc for signed margins
        span = scores.max() - scores.min() or 1.0
        return (scores - scores.min()) / span

    raise ValueError("Model must implement predict_proba or decision_function")


def evaluate_binary_classifier(
    model: Any, X, y_true, threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute standard binary metrics using probabilities (for AUC/PR) and a
    configurable threshold (for F1/CM). Prevents the common pitfall of using
    hard labels for AUC.
    """
    proba = _get_proba(model, X)
    y_true = np.asarray(y_true)

    preds = (proba >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    f1 = f1_score(y_true, preds)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    cm = confusion_matrix(y_true, preds)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision_curve": precision,
        "recall_curve": recall,
        "confusion_matrix": cm,
    }


def find_best_threshold(
    y_true, proba: np.ndarray, metric: str = "f1"
) -> Tuple[float, float]:
    """
    Grid-search thresholds on probabilities to maximize F1/precision/recall.
    """
    y_true = np.asarray(y_true)
    best_score, best_th = -1.0, 0.5
    thresholds = np.linspace(0.01, 0.99, 99)

    for th in thresholds:
        preds = (proba >= th).astype(int)
        if metric == "precision":
            score = (preds[y_true == 1].sum()) / max(preds.sum(), 1)
        elif metric == "recall":
            score = (preds[y_true == 1].sum()) / max((y_true == 1).sum(), 1)
        else:
            score = f1_score(y_true, preds)

        if score > best_score:
            best_score, best_th = score, th

    return best_th, best_score
