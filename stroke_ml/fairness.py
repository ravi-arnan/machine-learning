"""Fairness analysis across demographic groups."""

from __future__ import annotations

import pandas as pd

from stroke_ml.threshold import evaluate_predictions


def fairness_report(
    df: pd.DataFrame,
    y_true,
    y_prob,
    threshold: float,
    group_col: str,
) -> pd.DataFrame:
    """Per-group recall, precision, and flagged rate."""
    rows = []
    for group, idx in df.groupby(group_col).groups.items():
        y_g = y_true.loc[idx] if hasattr(y_true, "loc") else y_true[idx]
        p_g = y_prob[idx] if hasattr(y_prob, "__getitem__") else y_prob
        m = evaluate_predictions(y_g, p_g, threshold)
        rows.append(
            {
                "group": group,
                "n": len(idx),
                "stroke_cases": int(y_g.sum()),
                "recall": m["recall"],
                "precision": m["precision"],
                "flagged_pct": m["flagged_pct"],
                "roc_auc": m["roc_auc"],
            }
        )
    return pd.DataFrame(rows).sort_values("group")


def age_group_fairness(
    df: pd.DataFrame,
    y_true,
    y_prob,
    threshold: float,
) -> pd.DataFrame:
    """Bucket ages for fairness comparison."""
    ages = df["age"].copy()
    bins = [0, 40, 55, 65, 80, 120]
    labels = ["<40", "40-54", "55-64", "65-79", "80+"]
    grouped = pd.cut(ages, bins=bins, labels=labels, right=False)
    tmp = df.copy()
    tmp["age_group"] = grouped
    return fairness_report(tmp, y_true, y_prob, threshold, "age_group")


def full_fairness_summary(
    df: pd.DataFrame,
    y_true,
    y_prob,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    """Run fairness across gender, age bands, and work type."""
    out = {}
    if "gender" in df.columns:
        out["gender"] = fairness_report(df, y_true, y_prob, threshold, "gender")
    out["age_group"] = age_group_fairness(df, y_true, y_prob, threshold)
    if "work_type" in df.columns:
        out["work_type"] = fairness_report(df, y_true, y_prob, threshold, "work_type")
    return out
