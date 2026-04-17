from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    y_true_np = np.asarray(y_true)
    y_prob_np = np.asarray(y_prob)
    y_pred = (y_prob_np >= threshold).astype(int)
    has_both_classes = len(np.unique(y_true_np)) > 1

    metrics = {
        "roc_auc": float(roc_auc_score(y_true_np, y_prob_np)) if has_both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true_np, y_prob_np)) if has_both_classes else float("nan"),
        "brier_score": float(brier_score_loss(y_true_np, y_prob_np)),
        "precision_at_0_5": float(precision_score(y_true_np, y_pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true_np, y_pred, zero_division=0)),
    }

    top_decile_cutoff = np.quantile(y_prob_np, 0.9)
    top_decile_pred = (y_prob_np >= top_decile_cutoff).astype(int)
    metrics["recall_top_decile"] = float(recall_score(y_true_np, top_decile_pred, zero_division=0))
    metrics["precision_top_decile"] = float(precision_score(y_true_np, top_decile_pred, zero_division=0))
    return metrics


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true_np, y_pred_np)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_np, y_pred_np))),
        "r2": float(r2_score(y_true_np, y_pred_np)) if len(y_true_np) > 1 else float("nan"),
    }


def grouped_classification_metrics(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "binary_risk_label",
    prob_col: str = "predicted_probability",
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for group_value, group_df in df.groupby(group_col):
        grouped[str(group_value)] = classification_metrics(group_df[target_col], group_df[prob_col])
    return grouped


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path
