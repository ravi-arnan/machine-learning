"""Threshold selection, confusion matrix metrics, and decision-curve analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_for_recall(y_true, y_prob, target_recall: float = 0.80) -> float:
    """Find the lowest probability threshold achieving target recall on validation data."""
    thresholds = np.unique(np.sort(y_prob))[::-1]
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        if recall_score(y_true, pred, zero_division=0) >= target_recall:
            return float(t)
    return 0.5


def threshold_for_cost_ratio(
    y_true,
    y_prob,
    fn_cost: float = 20.0,
    fp_cost: float = 1.0,
) -> float:
    """Optimal threshold minimizing weighted misclassification cost."""
    thresholds = np.unique(y_prob)
    best_t, best_cost = 0.5, np.inf
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
    return best_t


def evaluate_predictions(y_true, y_prob, threshold: float) -> dict:
    """Return full metric dict plus confusion-matrix cells."""
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "flagged_pct": float(pred.mean() * 100),
    }


def decision_curve(y_true, y_prob, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Net benefit (Vickers & Elkin, 2006) across risk thresholds."""
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    n = len(y)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 50)

    rows = []
    prevalence = y.mean()
    for pt in thresholds:
        pred = (p >= pt).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        nb_model = tp / n - (fp / n) * (pt / (1 - pt))
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append(
            {
                "threshold": pt,
                "fn_fp_ratio": round((1 - pt) / pt),
                "net_benefit_model": nb_model,
                "net_benefit_treat_all": nb_all,
                "net_benefit_treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def metrics_at_thresholds(
    y_true,
    y_prob,
    thresholds: list[float],
) -> pd.DataFrame:
    """Compare recall/precision at multiple thresholds (for interactive dashboards)."""
    return pd.DataFrame(
        [evaluate_predictions(y_true, y_prob, t) for t in thresholds]
    )
