from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
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


def ranking_metrics(
    y_true: pd.Series,
    y_prob: pd.Series,
    top_fracs: tuple[float, ...] = (0.01, 0.05, 0.1),
) -> dict[str, float]:
    y_true_np = np.asarray(y_true)
    y_prob_np = np.asarray(y_prob)
    metrics: dict[str, float] = {}
    total = len(y_prob_np)
    if total == 0:
        return metrics

    order = np.argsort(-y_prob_np)
    ranked_truth = y_true_np[order]
    for frac in top_fracs:
        top_n = max(int(np.ceil(total * frac)), 1)
        selected = ranked_truth[:top_n]
        metrics[f"precision_top_{int(frac * 100)}pct"] = float(selected.mean()) if len(selected) else float("nan")
        metrics[f"capture_top_{int(frac * 100)}pct"] = float(selected.sum() / max(y_true_np.sum(), 1))
    return metrics


def calibration_table(
    y_true: pd.Series,
    y_prob: pd.Series,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    y_true_np = np.asarray(y_true)
    y_prob_np = np.asarray(y_prob)
    if len(y_true_np) == 0:
        return []

    prob_true, prob_pred = calibration_curve(
        y_true_np,
        y_prob_np,
        n_bins=n_bins,
        strategy="quantile",
    )
    quantile_edges = np.quantile(y_prob_np, np.linspace(0.0, 1.0, len(prob_true) + 1))
    rows: list[dict[str, float]] = []
    for idx, (observed, predicted) in enumerate(zip(prob_true, prob_pred)):
        rows.append(
            {
                "bin": float(idx + 1),
                "predicted_mean": float(predicted),
                "observed_rate": float(observed),
                "bin_lower": float(quantile_edges[idx]),
                "bin_upper": float(quantile_edges[idx + 1]),
                "abs_gap": float(abs(predicted - observed)),
            }
        )
    return rows


def expected_calibration_error(calibration_rows: list[dict[str, float]]) -> float:
    if not calibration_rows:
        return float("nan")
    return float(np.mean([row["abs_gap"] for row in calibration_rows]))


def build_presentation_guidance(metrics: dict[str, float]) -> dict[str, Any]:
    precision_at_threshold = metrics.get("precision_at_0_5", float("nan"))
    brier_score = metrics.get("brier_score", float("nan"))
    recommended_mode = "ranked_signal"
    score_field = "relative_risk_score"
    positioning_statement = (
        "A working no-leak wildfire risk ranking baseline with promising discrimination, "
        "but not yet calibrated enough for decision-grade probability claims."
    )
    warning = (
        "Use this model as a ranked wildfire signal. Raw probabilities are not calibrated enough "
        "for literal probability claims."
    )

    if np.isfinite(precision_at_threshold) and np.isfinite(brier_score):
        if precision_at_threshold >= 0.25 and brier_score <= 0.15:
            warning = "Probabilities are usable with caution, but ranked presentation is still preferred."
        else:
            recommended_mode = "ranked_signal"
            score_field = "relative_risk_score"

    return {
        "recommended_mode": recommended_mode,
        "recommended_score_field": score_field,
        "positioning_statement": positioning_statement,
        "warning": warning,
    }


def ranking_stability(
    df: pd.DataFrame,
    score_col: str = "ranking_score",
    entity_cols: tuple[str, ...] = ("cell_id", "month_num"),
) -> dict[str, Any]:
    if score_col not in df.columns or "year" not in df.columns:
        return {"pairs": [], "mean_spearman": float("nan"), "mean_top_decile_overlap": float("nan")}

    pairs: list[dict[str, float | int]] = []
    years = sorted(df["year"].dropna().unique())
    for prev_year, curr_year in zip(years, years[1:]):
        prev_df = df[df["year"] == prev_year][list(entity_cols) + [score_col]].rename(
            columns={score_col: "prev_score"}
        )
        curr_df = df[df["year"] == curr_year][list(entity_cols) + [score_col]].rename(
            columns={score_col: "curr_score"}
        )
        merged = prev_df.merge(curr_df, on=list(entity_cols), how="inner")
        if merged.empty:
            continue

        prev_cutoff = merged["prev_score"].quantile(0.9)
        curr_cutoff = merged["curr_score"].quantile(0.9)
        prev_top = set(
            zip(
                merged.loc[merged["prev_score"] >= prev_cutoff, entity_cols[0]],
                merged.loc[merged["prev_score"] >= prev_cutoff, entity_cols[1]],
            )
        )
        curr_top = set(
            zip(
                merged.loc[merged["curr_score"] >= curr_cutoff, entity_cols[0]],
                merged.loc[merged["curr_score"] >= curr_cutoff, entity_cols[1]],
            )
        )
        union = prev_top | curr_top
        overlap = len(prev_top & curr_top) / len(union) if union else float("nan")
        pairs.append(
            {
                "prev_year": int(prev_year),
                "curr_year": int(curr_year),
                "spearman": float(merged["prev_score"].corr(merged["curr_score"], method="spearman")),
                "top_decile_overlap": float(overlap),
            }
        )

    if not pairs:
        return {"pairs": [], "mean_spearman": float("nan"), "mean_top_decile_overlap": float("nan")}
    return {
        "pairs": pairs,
        "mean_spearman": float(np.nanmean([pair["spearman"] for pair in pairs])),
        "mean_top_decile_overlap": float(np.nanmean([pair["top_decile_overlap"] for pair in pairs])),
    }


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
        "positive_rate": float(np.mean(y_true_np)) if len(y_true_np) else float("nan"),
        "mean_predicted_probability": float(np.mean(y_prob_np)) if len(y_prob_np) else float("nan"),
    }

    top_decile_cutoff = np.quantile(y_prob_np, 0.9)
    top_decile_pred = (y_prob_np >= top_decile_cutoff).astype(int)
    metrics["recall_top_decile"] = float(recall_score(y_true_np, top_decile_pred, zero_division=0))
    metrics["precision_top_decile"] = float(precision_score(y_true_np, top_decile_pred, zero_division=0))
    metrics.update(ranking_metrics(y_true, y_prob))
    return metrics


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true_np, y_pred_np)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_np, y_pred_np))),
        "r2": float(r2_score(y_true_np, y_pred_np)) if len(y_true_np) > 1 else float("nan"),
    }


# Boxes are applied in order. Australia before Asia so the mainland is not counted as SE Asia.
CONTINENT_BOXES: tuple[tuple[str, float, float, float, float], ...] = (
    ("australia", -45.0, -10.0, 112.0, 155.0),
    ("africa", -35.0, 37.5, -18.0, 52.0),
    ("south_america", -56.0, 13.0, -82.0, -34.0),
    ("north_america", 7.0, 84.0, -170.0, -50.0),
    ("europe", 36.0, 72.0, -11.0, 40.0),
    ("asia", -11.0, 78.0, 40.0, 180.0),
)
CONTINENT_ORDER = ("global", "africa", "asia", "australia", "europe", "north_america", "south_america", "other")


def assign_continent(lat: pd.Series | np.ndarray, lon: pd.Series | np.ndarray) -> np.ndarray:
    lat_np = np.asarray(lat, dtype=float)
    lon_np = np.asarray(lon, dtype=float)
    labels = np.full(lat_np.shape, "other", dtype=object)
    assigned = np.zeros(lat_np.shape, dtype=bool)
    for name, lat_min, lat_max, lon_min, lon_max in CONTINENT_BOXES:
        mask = (
            (~assigned)
            & (lat_np >= lat_min)
            & (lat_np <= lat_max)
            & (lon_np >= lon_min)
            & (lon_np <= lon_max)
        )
        labels[mask] = name
        assigned |= mask
    return labels


def _choose_score_columns(df: pd.DataFrame) -> tuple[str, str]:
    if "ranking_score" in df.columns:
        rank_col = "ranking_score"
    elif "raw_predicted_probability" in df.columns:
        rank_col = "raw_predicted_probability"
    elif "predicted_probability" in df.columns:
        rank_col = "predicted_probability"
    else:
        raise ValueError("Prediction table needs ranking_score or predicted_probability")

    prob_col = "predicted_probability" if "predicted_probability" in df.columns else rank_col
    return rank_col, prob_col


def score_one_region(df: pd.DataFrame, region: str) -> dict[str, Any]:
    rank_col, prob_col = _choose_score_columns(df)
    raw = classification_metrics(df["binary_risk_label"], df[rank_col])
    calibrated = classification_metrics(df["binary_risk_label"], df[prob_col])
    cal_rows = calibration_table(df["binary_risk_label"], df[prob_col])
    n_rows = int(len(df))
    n_fires = int(df["binary_risk_label"].sum())
    return {
        "region": region,
        "n_rows": n_rows,
        "n_fires": n_fires,
        "fire_rate": float(n_fires / n_rows) if n_rows else float("nan"),
        "roc_auc": raw["roc_auc"],
        "pr_auc": raw["pr_auc"],
        "precision_top_10pct": raw["precision_top_10pct"],
        "capture_top_10pct": raw["capture_top_10pct"],
        "precision_top_1pct": raw["precision_top_1pct"],
        "brier_score": calibrated["brier_score"],
        "mean_predicted_probability": calibrated["mean_predicted_probability"],
        "expected_calibration_error": expected_calibration_error(cal_rows),
        "precision_at_0_5": calibrated["precision_at_0_5"],
        "classification_raw": raw,
        "classification": calibrated,
    }


def europe_subregion(lat: pd.Series | np.ndarray, lon: pd.Series | np.ndarray, continent: np.ndarray) -> np.ndarray:
    """Split the Europe box, and mark the southern strip the Africa box claims first.

    Continent labels stay as they are. These tags are extra rows so Seville and the
    Algarve, which fall in the Africa box, can be counted without moving Africa's score.
    """
    lat_np = np.asarray(lat, dtype=float)
    lon_np = np.asarray(lon, dtype=float)
    labels = np.full(lat_np.shape, "", dtype=object)
    europe = continent == "europe"
    labels[europe & (lat_np < 46.0)] = "europe_south"
    labels[europe & (lat_np >= 46.0)] = "europe_north"
    fringe = (
        (continent == "africa")
        & (lat_np >= 35.0)
        & (lat_np <= 37.5)
        & (lon_np >= -11.0)
        & (lon_np <= 30.0)
    )
    labels[fringe] = "africa_fringe"
    return labels


def score_continents(df: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"binary_risk_label", "lat_center", "lon_center"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {sorted(missing)}")

    scored = df.copy()
    scored["continent"] = assign_continent(scored["lat_center"], scored["lon_center"])
    rows = [score_one_region(scored, "global")]
    present = set(scored["continent"].unique())
    for region in CONTINENT_ORDER:
        if region in {"global"} or region not in present:
            continue
        rows.append(score_one_region(scored[scored["continent"] == region], region))

    scored["europe_part"] = europe_subregion(scored["lat_center"], scored["lon_center"], scored["continent"].to_numpy())
    for region in ("europe_south", "europe_north", "africa_fringe"):
        part = scored[scored["europe_part"] == region]
        if part.empty:
            continue
        rows.append(score_one_region(part, region))
    return rows


def format_continent_table(rows: list[dict[str, Any]]) -> str:
    headers = (
        "region",
        "n_rows",
        "n_fires",
        "fire_rate",
        "roc_auc",
        "pr_auc",
        "precision_top_10pct",
        "capture_top_10pct",
        "brier_score",
        "mean_predicted_probability",
    )
    widths = {header: len(header) for header in headers}
    rendered: list[dict[str, str]] = []
    for row in rows:
        rendered_row = {}
        for header in headers:
            value = row[header]
            if header in {"n_rows", "n_fires"}:
                text = f"{int(value):,}"
            elif isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            rendered_row[header] = text
            widths[header] = max(widths[header], len(text))
        rendered.append(rendered_row)

    def fmt(row: dict[str, str]) -> str:
        return "  ".join(row[header].ljust(widths[header]) for header in headers)

    return fmt({header: header for header in headers}) + "\n" + "\n".join(fmt(row) for row in rendered)


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
