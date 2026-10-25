from __future__ import annotations

import argparse
import gc
import importlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

try:
    from .evaluate import (
        build_presentation_guidance,
        calibration_table,
        classification_metrics,
        expected_calibration_error,
        grouped_classification_metrics,
        ranking_stability,
        regression_metrics,
        save_metrics,
        score_continents,
    )
except ImportError:  # pragma: no cover - allows direct script execution
    from evaluate import (
        build_presentation_guidance,
        calibration_table,
        classification_metrics,
        expected_calibration_error,
        grouped_classification_metrics,
        ranking_stability,
        regression_metrics,
        save_metrics,
        score_continents,
    )
try:
    from .run_artifacts import create_run_layout, write_json, write_stage_manifest
except ImportError:  # pragma: no cover - allows direct script execution
    from run_artifacts import create_run_layout, write_json, write_stage_manifest


NON_FEATURE_COLUMNS = {
    "cell_id",
    "month",
    "year",
    "month_num",
    "lat_center",
    "lon_center",
    "cell_area_km2",
    "binary_risk_label",
    "burned_fraction",
    "burned_area_km2",
    "burned_pixel_count",
    "burned_event_count",
    "product_year",
}

CALENDAR_PREFIXES = ("month_sin", "month_cos", "time_index")
HISTORY_PREFIXES = ("burned_fraction_", "binary_risk_label_")
LANDCOVER_PREFIXES = ("landcover_",)
TERRAIN_PREFIXES = ("terrain_",)
POPULATION_PREFIXES = ("population_",)
ROADS_PREFIXES = ("roads_",)
FIRMS_PREFIXES = ("firms_",)
DEFAULT_DYNAMIC_PREFIXES = ("era5",)
ABLATION_ARMS = ("history", "era5", "both")
ERA5_FAMILY_TOKENS: dict[str, tuple[str, ...]] = {
    "temperature": ("temperature_2m",),
    "precip": ("precipitation",),
    "soil_moisture": ("soil_moisture",),
    "wind": ("wind_u10", "wind_v10", "wind_speed"),
    "hot_dry": ("hot_dry_factor", "potential_evaporation", "climate_water_balance"),
}
ERA5_FAMILY_ORDER = ("temperature", "precip", "soil_moisture", "wind", "hot_dry")
# A family "matters" if dropping it costs at least this much vs the full both model.
ERA5_KEEP_MIN_PR_AUC_DROP = 0.005
ERA5_KEEP_MIN_ROC_AUC_DROP = 0.003
ERA5_KEEP_MIN_CAPTURE_DROP = 0.005
ABLATION_COMPARISON_FIELDS = (
    "roc_auc",
    "pr_auc",
    "brier_score",
    "precision_at_0_5",
    "recall_at_0_5",
    "precision_top_1pct",
    "precision_top_5pct",
    "precision_top_10pct",
    "capture_top_10pct",
)


@dataclass(frozen=True)
class TrainConfig:
    train_end_year: int = 2020
    valid_start_year: int = 2021
    valid_end_year: int = 2024
    random_state: int = 42
    n_estimators: int = 100
    skip_yearly_backtest: bool = False
    backtest_start_year: int | None = None
    backtest_end_year: int | None = None
    skip_regressor: bool = False
    feature_set: str = "both"
    dynamic_prefixes: tuple[str, ...] = DEFAULT_DYNAMIC_PREFIXES
    drop_era5_families: tuple[str, ...] = ()
    keep_era5_families: tuple[str, ...] | None = None
    calibration_method: str = "isotonic"


def _downcast_feature_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    if "month" in chunk.columns:
        chunk["month"] = pd.to_datetime(chunk["month"]).dt.to_period("M").astype(str)
    float_cols = chunk.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        chunk[float_cols] = chunk[float_cols].astype("float32")
    for column in chunk.select_dtypes(include=["int64"]).columns:
        chunk[column] = pd.to_numeric(chunk[column], downcast="integer")
    return chunk


def read_feature_table(path: str | Path, chunksize: int = 250_000) -> pd.DataFrame:
    chunks = []
    for index, chunk in enumerate(pd.read_csv(path, chunksize=chunksize), start=1):
        chunks.append(_downcast_feature_chunk(chunk))
        if index == 1 or index % 8 == 0:
            LOGGER.info("Read %s feature-table chunks", index)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    return df


def feature_family(column: str, dynamic_prefixes: tuple[str, ...] = DEFAULT_DYNAMIC_PREFIXES) -> str:
    if column.startswith(CALENDAR_PREFIXES):
        return "calendar"
    if column.startswith(HISTORY_PREFIXES):
        return "history"
    if column.startswith(LANDCOVER_PREFIXES):
        return "landcover"
    if column.startswith(TERRAIN_PREFIXES):
        return "terrain"
    if column.startswith(POPULATION_PREFIXES):
        return "population"
    if column.startswith(ROADS_PREFIXES):
        return "roads"
    if column.startswith(FIRMS_PREFIXES):
        return "firms"
    for prefix in dynamic_prefixes:
        if column.startswith(f"{prefix}_"):
            return "era5"
    return "other"


def era5_family_for_column(
    column: str,
    dynamic_prefixes: tuple[str, ...] = DEFAULT_DYNAMIC_PREFIXES,
) -> str | None:
    rest = None
    for prefix in dynamic_prefixes:
        head = f"{prefix}_"
        if column.startswith(head):
            rest = column[len(head) :]
            break
    if rest is None:
        return None

    for family in ERA5_FAMILY_ORDER:
        tokens = sorted(ERA5_FAMILY_TOKENS[family], key=len, reverse=True)
        for token in tokens:
            if rest == token or rest.startswith(f"{token}_"):
                return family
    return "other_era5"


def select_feature_columns(
    df: pd.DataFrame,
    feature_set: str = "both",
    dynamic_prefixes: tuple[str, ...] = DEFAULT_DYNAMIC_PREFIXES,
    drop_era5_families: tuple[str, ...] = (),
    keep_era5_families: tuple[str, ...] | None = None,
) -> list[str]:
    numeric_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
        and not column.endswith("product_year")
        and np.issubdtype(df[column].dtype, np.number)
    ]
    if not numeric_columns:
        raise ValueError("No numeric feature columns found in the feature table")
    if feature_set == "both":
        selected = numeric_columns
    elif feature_set in ("history", "era5"):
        allowed = {"calendar", feature_set}
        selected = [column for column in numeric_columns if feature_family(column, dynamic_prefixes) in allowed]
        if not any(feature_family(column, dynamic_prefixes) == feature_set for column in selected):
            raise ValueError(
                f"No {feature_set} features found. History columns start with burned_fraction_; "
                "ERA5 columns start with era5_."
            )
    else:
        raise ValueError(f"Unknown feature_set {feature_set!r}. Use history, era5, or both.")

    if keep_era5_families is not None:
        keep = set(keep_era5_families)
        selected = [
            column
            for column in selected
            if feature_family(column, dynamic_prefixes) != "era5"
            or era5_family_for_column(column, dynamic_prefixes) in keep
        ]
    if drop_era5_families:
        drop = set(drop_era5_families)
        selected = [
            column
            for column in selected
            if era5_family_for_column(column, dynamic_prefixes) not in drop
        ]
    if not selected:
        raise ValueError("No numeric feature columns remain after applying ERA5 family filters")
    return selected


def temporal_split(df: pd.DataFrame, config: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["year"] <= config.train_end_year].copy()
    valid_df = df[(df["year"] >= config.valid_start_year) & (df["year"] <= config.valid_end_year)].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("Temporal split produced an empty train or validation dataset")
    return train_df, valid_df


def make_model_pair(config: TrainConfig) -> tuple[str, Any, Any]:
    if importlib.util.find_spec("lightgbm") is not None:
        from lightgbm import LGBMClassifier, LGBMRegressor

        return (
            "lightgbm",
            LGBMClassifier(
                n_estimators=config.n_estimators,
                learning_rate=0.05,
                max_depth=-1,
                random_state=config.random_state,
                n_jobs=4,
                verbose=20,
            ),
            LGBMRegressor(
                n_estimators=config.n_estimators,
                learning_rate=0.05,
                max_depth=-1,
                random_state=config.random_state,
                n_jobs=4,
                verbose=20,
            ),
        )

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier, XGBRegressor

        return (
            "xgboost",
            XGBClassifier(
                n_estimators=config.n_estimators,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=config.random_state,
                eval_metric="logloss",
                verbosity=1,
            ),
            XGBRegressor(
                n_estimators=config.n_estimators,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=config.random_state,
                verbosity=1,
            ),
        )

    return (
        "sklearn_hist_gradient_boosting",
        HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.05,
            max_iter=config.n_estimators,
            random_state=config.random_state,
            verbose=1,
        ),
        HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=config.n_estimators,
            random_state=config.random_state,
            verbose=1,
        ),
    )


def make_balanced_sample_weight(y: pd.Series) -> np.ndarray:
    y_np = np.asarray(y)
    if len(np.unique(y_np)) < 2:
        return np.ones_like(y_np, dtype=float)

    positive_count = max((y_np == 1).sum(), 1)
    negative_count = max((y_np == 0).sum(), 1)
    positive_weight = len(y_np) / (2.0 * positive_count)
    negative_weight = len(y_np) / (2.0 * negative_count)
    return np.where(y_np == 1, positive_weight, negative_weight).astype(float)


def fit_classifier(model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    LOGGER.info("Fitting classifier on %s rows x %s features", f"{len(X):,}", X.shape[1])
    sample_weight = make_balanced_sample_weight(y)
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)
    LOGGER.info("Finished classifier fit")
    return model


def fit_regressor(model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    LOGGER.info("Fitting regressor on %s rows x %s features", f"{len(X):,}", X.shape[1])
    model.fit(X, y)
    LOGGER.info("Finished regressor fit")
    return model


def split_model_and_calibration_data(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train_df.sort_values(["year", "month_num", "cell_id"]).copy()
    unique_years = sorted(ordered["year"].unique())
    if len(unique_years) >= 2:
        calibration_year = unique_years[-1]
        model_train_df = ordered[ordered["year"] < calibration_year].copy()
        calibration_df = ordered[ordered["year"] == calibration_year].copy()
        if not model_train_df.empty and calibration_df["binary_risk_label"].nunique() > 1:
            return model_train_df, calibration_df

    split_idx = max(int(len(ordered) * 0.8), 1)
    model_train_df = ordered.iloc[:split_idx].copy()
    calibration_df = ordered.iloc[split_idx:].copy()
    if calibration_df.empty or calibration_df["binary_risk_label"].nunique() < 2:
        return ordered, ordered.iloc[0:0].copy()
    return model_train_df, calibration_df


def fit_probability_calibrator(
    raw_probabilities: np.ndarray,
    y_true: pd.Series,
    method: str = "isotonic",
) -> tuple[Any | None, dict[str, Any]]:
    if method in {"none", "identity"} or len(raw_probabilities) == 0 or y_true.nunique() < 2:
        return None, {"method": "identity", "reason": "insufficient calibration holdout or method=none"}

    if method == "platt":
        calibrator = LogisticRegression(random_state=42)
        sample_weight = make_balanced_sample_weight(y_true)
        calibrator.fit(raw_probabilities.reshape(-1, 1), y_true, sample_weight=sample_weight)
        return calibrator, {"method": "platt_scaling", "rows": int(len(y_true))}

    if method != "isotonic":
        raise ValueError(f"Unknown calibration method {method!r}. Use isotonic, platt, or none.")

    # Fit on natural class frequencies so predicted P tracks the observed fire rate.
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    calibrator.fit(raw_probabilities, np.asarray(y_true))
    return calibrator, {"method": "isotonic", "rows": int(len(y_true))}


def apply_probability_calibrator(calibrator: Any | None, raw_probabilities: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return raw_probabilities
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(raw_probabilities.reshape(-1, 1))[:, 1]
    return np.clip(np.asarray(calibrator.predict(raw_probabilities), dtype=float), 0.0, 1.0)


def add_signal_columns(
    df: pd.DataFrame,
    score_col: str = "ranking_score",
) -> pd.DataFrame:
    out = df.copy()
    ranks = out[score_col].rank(method="first", ascending=False)
    total = max(len(out), 1)
    out["risk_rank"] = ranks.astype(int)
    out["relative_risk_score"] = np.clip(100.0 * (1.0 - ((ranks - 1.0) / total)), 0.0, 100.0)
    out["risk_percentile"] = np.clip(1.0 - ((ranks - 1.0) / total), 0.0, 1.0)
    month_key = "month_num" if "month_num" in out.columns else ("month" if "month" in out.columns else None)
    if month_key is not None:
        out["month_risk_percentile"] = out.groupby(month_key, sort=False)[score_col].rank(
            method="average",
            pct=True,
        )
    bins = [-0.01, 0.5, 0.8, 0.95, 1.01]
    labels = ["monitor", "elevated", "high", "extreme"]
    out["risk_band"] = pd.cut(out["risk_percentile"], bins=bins, labels=labels).astype(str)
    return out


def fit_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    config: TrainConfig,
) -> dict[str, Any]:
    model_family, classifier, regressor = make_model_pair(config)
    model_train_df, calibration_df = split_model_and_calibration_data(train_df)
    classifier = fit_classifier(classifier, model_train_df[feature_columns], model_train_df["binary_risk_label"])
    if config.skip_regressor:
        LOGGER.info("Skipping burned-fraction regressor")
        regressor = None
    else:
        regressor = fit_regressor(regressor, train_df[feature_columns], train_df["burned_fraction"])

    calibrator = None
    calibration_details = {"method": "identity", "reason": "no calibration performed"}
    if not calibration_df.empty:
        raw_calibration_probabilities = classifier.predict_proba(calibration_df[feature_columns])[:, 1]
        calibrator, calibration_details = fit_probability_calibrator(
            raw_calibration_probabilities,
            calibration_df["binary_risk_label"],
            method=config.calibration_method,
        )
        LOGGER.info(
            "Fitted %s calibrator on %s holdout rows",
            calibration_details.get("method"),
            f"{calibration_details.get('rows', 0):,}",
        )

    valid_predictions = valid_df.copy()
    valid_predictions["raw_predicted_probability"] = classifier.predict_proba(valid_df[feature_columns])[:, 1]
    valid_predictions["predicted_probability"] = apply_probability_calibrator(
        calibrator,
        valid_predictions["raw_predicted_probability"].to_numpy(),
    )
    valid_predictions["ranking_score"] = valid_predictions["raw_predicted_probability"]
    if regressor is None:
        valid_predictions["predicted_burned_fraction"] = np.nan
    else:
        valid_predictions["predicted_burned_fraction"] = np.clip(
            regressor.predict(valid_df[feature_columns]),
            0.0,
            1.0,
        )
    valid_predictions = add_signal_columns(valid_predictions, score_col="ranking_score")

    return {
        "model_family": model_family,
        "classifier": classifier,
        "calibrator": calibrator,
        "calibration_details": calibration_details,
        "model_train_rows": int(len(model_train_df)),
        "calibration_rows": int(len(calibration_df)),
        "regressor": regressor,
        "feature_columns": feature_columns,
        "valid_predictions": valid_predictions,
    }


YEARLY_REGION_FIELDS = (
    "region",
    "n_rows",
    "n_fires",
    "fire_rate",
    "roc_auc",
    "pr_auc",
    "precision_top_10pct",
    "capture_top_10pct",
)


def _backtest_year_window(config: TrainConfig) -> tuple[int, int]:
    start_year = config.valid_start_year if config.backtest_start_year is None else config.backtest_start_year
    end_year = config.valid_end_year if config.backtest_end_year is None else config.backtest_end_year
    if start_year > end_year:
        raise ValueError(f"Backtest window {start_year}-{end_year} is empty")
    return start_year, end_year


def _slim_region_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    summaries = []
    for row in score_continents(frame):
        summaries.append({field: row[field] for field in YEARLY_REGION_FIELDS})
    return summaries


def format_yearly_backtest_table(yearly_scores: dict[str, dict[str, Any]]) -> str:
    headers = ("year",) + YEARLY_REGION_FIELDS
    widths = {header: len(header) for header in headers}
    rendered: list[dict[str, str]] = []
    for year, payload in yearly_scores.items():
        for region in payload["regions"]:
            rendered_row = {"year": str(year)}
            for header in YEARLY_REGION_FIELDS:
                value = region[header]
                if header in {"n_rows", "n_fires"}:
                    text = f"{int(value):,}"
                elif isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                rendered_row[header] = text
            rendered.append(rendered_row)
            for header, text in rendered_row.items():
                widths[header] = max(widths[header], len(text))

    def fmt(row: dict[str, str]) -> str:
        return "  ".join(row[header].ljust(widths[header]) for header in headers)

    header_row = {header: header for header in headers}
    return fmt(header_row) + "\n" + "\n".join(fmt(row) for row in rendered)


def yearly_backtest(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: TrainConfig,
    on_year: Any = None,
) -> dict[str, dict[str, Any]]:
    """Refit once per test year on every earlier year. The test year is not in that fit."""
    start_year, end_year = _backtest_year_window(config)
    required = {"year", "binary_risk_label", "lat_center", "lon_center"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Yearly backtest is missing required columns: {sorted(missing)}")

    yearly_scores: dict[str, dict[str, Any]] = {}
    years = df["year"].to_numpy(dtype=int)
    labels = df["binary_risk_label"].to_numpy()
    latitudes = df["lat_center"].to_numpy()
    longitudes = df["lon_center"].to_numpy()
    features = df[feature_columns].to_numpy(dtype=np.float32)
    del df
    gc.collect()
    LOGGER.info(
        "Starting yearly backtest %s to %s (%s features). Each year is trained on earlier years only.",
        start_year,
        end_year,
        len(feature_columns),
    )
    for year in range(start_year, end_year + 1):
        train_idx = np.flatnonzero(years < year)
        test_idx = np.flatnonzero(years == year)
        if train_idx.size == 0 or test_idx.size == 0:
            LOGGER.info(
                "Yearly backtest %s skipped: train %s rows, test %s rows",
                year,
                f"{train_idx.size:,}",
                f"{test_idx.size:,}",
            )
            continue

        LOGGER.info(
            "Yearly backtest %s: train %s rows (%s-%s), test %s rows",
            year,
            f"{train_idx.size:,}",
            int(years[train_idx].min()),
            int(years[train_idx].max()),
            f"{test_idx.size:,}",
        )
        _, classifier, _ = make_model_pair(config)
        classifier = fit_classifier(
            classifier,
            features[train_idx],
            pd.Series(labels[train_idx]),
        )
        scores = classifier.predict_proba(features[test_idx])[:, 1]
        test_frame = pd.DataFrame(
            {
                "lat_center": latitudes[test_idx],
                "lon_center": longitudes[test_idx],
                "binary_risk_label": labels[test_idx],
                "ranking_score": scores,
            }
        )
        regions = _slim_region_rows(test_frame)
        global_row = next(region for region in regions if region["region"] == "global")
        yearly_scores[str(year)] = {
            "n_train_rows": int(train_idx.size),
            "n_test_rows": int(test_idx.size),
            "train_year_min": int(years[train_idx].min()),
            "train_year_max": int(years[train_idx].max()),
            "roc_auc": global_row["roc_auc"],
            "pr_auc": global_row["pr_auc"],
            "precision_top_10pct": global_row["precision_top_10pct"],
            "capture_top_10pct": global_row["capture_top_10pct"],
            "regions": regions,
        }
        LOGGER.info(
            "Yearly backtest %s global ROC %.4f PR %.4f capture@10%% %.4f",
            year,
            global_row["roc_auc"],
            global_row["pr_auc"],
            global_row["capture_top_10pct"],
        )
        del classifier, test_frame, scores
        if on_year is not None:
            on_year(yearly_scores)
    return yearly_scores


def _null_nan(value: Any) -> Any:
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _null_nan(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_null_nan(item) for item in value]
    return value


def _yearly_backtest_payload(
    yearly_scores: dict[str, dict[str, Any]],
    feature_table_path: str | Path,
    feature_columns: list[str],
    config: TrainConfig,
) -> dict[str, Any]:
    start_year, end_year = _backtest_year_window(config)
    return _null_nan(
        {
            "feature_table_path": str(feature_table_path),
            "backtest_start_year": start_year,
            "backtest_end_year": end_year,
            "n_features": len(feature_columns),
            "drop_era5_families": list(config.drop_era5_families),
            "score": "raw ranking probability, the same role as ranking_score",
            "note": (
                "Each test year is a new fit on every earlier year in the table. "
                "The v8 holdout model is a separate fit on 2020-2023 scored on 2024-2025."
            ),
            "years": yearly_scores,
        }
    )


def _write_yearly_backtest_files(
    yearly_scores: dict[str, dict[str, Any]],
    feature_table_path: str | Path,
    feature_columns: list[str],
    config: TrainConfig,
    json_path: Path,
    csv_path: Path,
) -> None:
    save_metrics(
        _yearly_backtest_payload(yearly_scores, feature_table_path, feature_columns, config),
        json_path,
    )
    rows: list[dict[str, Any]] = []
    for year, payload in yearly_scores.items():
        for region in payload["regions"]:
            rows.append(
                {
                    "year": int(year),
                    "n_train_rows": payload["n_train_rows"],
                    "train_year_min": payload["train_year_min"],
                    "train_year_max": payload["train_year_max"],
                    **region,
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def run_yearly_backtest(
    feature_table_path: str | Path,
    config: TrainConfig = TrainConfig(),
    run_root: str | Path | None = None,
    run_name: str | None = None,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Score 2022-2025 (or the configured window) without replacing the holdout model."""
    LOGGER.info("Reading feature table %s", feature_table_path)
    loaded = {"table": read_feature_table(feature_table_path)}
    LOGGER.info("Loaded %s rows x %s cols", f"{len(loaded['table']):,}", loaded["table"].shape[1])
    feature_columns = select_feature_columns(
        loaded["table"],
        feature_set=config.feature_set,
        dynamic_prefixes=config.dynamic_prefixes,
        drop_era5_families=config.drop_era5_families,
        keep_era5_families=config.keep_era5_families,
    )
    LOGGER.info(
        "Feature set %s: %s columns (drop_era5=%s)",
        config.feature_set,
        len(feature_columns),
        list(config.drop_era5_families) or "none",
    )

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_dir = layout.reports_dir
    elif model_dir is not None:
        output_dir = Path(model_dir)
    else:
        raise ValueError("Either model_dir or run_root must be provided")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "yearly_backtest.json"
    csv_path = output_dir / "yearly_backtest.csv"

    def persist(yearly_scores: dict[str, dict[str, Any]]) -> None:
        _write_yearly_backtest_files(
            yearly_scores,
            feature_table_path,
            feature_columns,
            config,
            json_path,
            csv_path,
        )

    yearly_scores = yearly_backtest(loaded.pop("table"), feature_columns, config=config, on_year=persist)
    if not yearly_scores:
        raise ValueError("Yearly backtest produced no years. Check the year window against the feature table.")
    persist(yearly_scores)
    table = format_yearly_backtest_table(yearly_scores)
    LOGGER.info("Yearly backtest:\n%s", table)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="yearly_backtest",
            config={
                "feature_table_path": str(feature_table_path),
                "backtest_start_year": config.backtest_start_year,
                "backtest_end_year": config.backtest_end_year,
                "n_estimators": config.n_estimators,
                "drop_era5_families": list(config.drop_era5_families),
            },
            artifacts={"yearly_backtest_json": str(json_path), "yearly_backtest_csv": str(csv_path)},
            summary={"years": list(yearly_scores)},
        )
    return {
        "years": yearly_scores,
        "table": table,
        "json_path": json_path,
        "csv_path": csv_path,
        "feature_columns": feature_columns,
    }


def summarize_feature_importance(
    model: Any,
    feature_columns: list[str],
) -> list[dict[str, float | str]]:
    if not hasattr(model, "feature_importances_"):
        return []
    pairs = sorted(
        zip(feature_columns, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{"feature": name, "importance": float(score)} for name, score in pairs[:25]]


def collect_holdout_metrics(
    fitted: dict[str, Any],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    config: TrainConfig,
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    valid_predictions = fitted["valid_predictions"]
    raw_classification = classification_metrics(
        valid_predictions["binary_risk_label"],
        valid_predictions["raw_predicted_probability"],
    )
    classification = classification_metrics(
        valid_predictions["binary_risk_label"],
        valid_predictions["predicted_probability"],
    )
    raw_calibration = calibration_table(
        valid_predictions["binary_risk_label"],
        valid_predictions["raw_predicted_probability"],
    )
    calibration = calibration_table(
        valid_predictions["binary_risk_label"],
        valid_predictions["predicted_probability"],
    )
    metrics: dict[str, Any] = {
        "model_family": fitted["model_family"],
        "feature_set": config.feature_set,
        "n_features": len(feature_columns),
        "train_years": [int(train_df["year"].min()), int(train_df["year"].max())],
        "valid_years": [int(valid_df["year"].min()), int(valid_df["year"].max())],
        "classification_raw": raw_classification,
        "classification": classification,
        "calibration": {
            "strategy": fitted["calibration_details"],
            "raw_expected_calibration_error": expected_calibration_error(raw_calibration),
            "expected_calibration_error": expected_calibration_error(calibration),
            "raw_bins": raw_calibration,
            "bins": calibration,
        },
        "presentation_guidance": build_presentation_guidance(classification),
        "per_month": grouped_classification_metrics(valid_predictions, group_col="month_num"),
        "top_features": summarize_feature_importance(fitted["classifier"], feature_columns),
        "ranking_summary": {
            "score_field": "ranking_score",
            "primary_metrics": {
                "roc_auc": raw_classification.get("roc_auc"),
                "pr_auc": raw_classification.get("pr_auc"),
                "precision_top_1pct": raw_classification.get("precision_top_1pct"),
                "precision_top_5pct": raw_classification.get("precision_top_5pct"),
                "precision_top_10pct": raw_classification.get("precision_top_10pct"),
                "capture_top_10pct": raw_classification.get("capture_top_10pct"),
            },
        },
        "ranking_stability": ranking_stability(valid_predictions),
    }
    if not config.skip_regressor:
        metrics["regression"] = regression_metrics(
            valid_predictions["burned_fraction"],
            valid_predictions["predicted_burned_fraction"],
        )
    if config.skip_yearly_backtest or df is None:
        metrics["per_year_backtest"] = {}
    else:
        metrics["per_year_backtest"] = yearly_backtest(df, feature_columns, config=config)
    return metrics


def train_baseline_models(
    feature_table_path: str | Path,
    model_dir: str | Path | None = None,
    config: TrainConfig = TrainConfig(),
    run_root: str | Path | None = None,
    run_name: str | None = None,
    df: pd.DataFrame | None = None,
    write_predictions: bool = True,
) -> dict[str, Any]:
    if df is None:
        LOGGER.info("Reading feature table %s", feature_table_path)
        df = read_feature_table(feature_table_path)
        LOGGER.info("Loaded %s rows x %s cols", f"{len(df):,}", df.shape[1])
    feature_columns = select_feature_columns(
        df,
        feature_set=config.feature_set,
        dynamic_prefixes=config.dynamic_prefixes,
        drop_era5_families=config.drop_era5_families,
        keep_era5_families=config.keep_era5_families,
    )
    LOGGER.info(
        "Feature set %s: %s columns (drop_era5=%s keep_era5=%s)",
        config.feature_set,
        len(feature_columns),
        list(config.drop_era5_families) or "none",
        list(config.keep_era5_families) if config.keep_era5_families is not None else "all",
    )
    train_df, valid_df = temporal_split(df, config=config)
    if config.skip_yearly_backtest:
        del df
        df = None
    LOGGER.info(
        "Split train %s rows (%s-%s) / valid %s rows (%s-%s)",
        f"{len(train_df):,}",
        int(train_df["year"].min()),
        int(train_df["year"].max()),
        f"{len(valid_df):,}",
        int(valid_df["year"].min()),
        int(valid_df["year"].max()),
    )
    fitted = fit_models(train_df, valid_df, feature_columns, config=config)
    metrics = collect_holdout_metrics(
        fitted,
        train_df,
        valid_df,
        feature_columns,
        config=config,
        df=df,
    )
    valid_predictions = fitted["valid_predictions"]

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        model_dir = layout.models_dir
    elif model_dir is None:
        raise ValueError("Either model_dir or run_root must be provided")

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "classifier.pkl").open("wb") as handle:
        pickle.dump(fitted["classifier"], handle)
    if fitted["calibrator"] is not None:
        with (model_dir / "calibrator.pkl").open("wb") as handle:
            pickle.dump(fitted["calibrator"], handle)
    if fitted["regressor"] is not None:
        with (model_dir / "regressor.pkl").open("wb") as handle:
            pickle.dump(fitted["regressor"], handle)
    write_json({"feature_columns": feature_columns}, model_dir / "feature_columns.json")
    write_json(
        {
            "recommended_mode": metrics["presentation_guidance"]["recommended_mode"],
            "recommended_score_field": metrics["presentation_guidance"]["recommended_score_field"],
            "warning": metrics["presentation_guidance"]["warning"],
            "ranking_summary": metrics["ranking_summary"],
            "ranking_stability": metrics["ranking_stability"],
            "classification": metrics["classification"],
            "classification_raw": metrics["classification_raw"],
            "calibration": metrics["calibration"],
        },
        model_dir / "signal_profile.json",
    )
    write_json(
        {
            "feature_table_path": str(feature_table_path),
            "train_config": {
                "train_end_year": config.train_end_year,
                "valid_start_year": config.valid_start_year,
                "valid_end_year": config.valid_end_year,
                "random_state": config.random_state,
                "feature_set": config.feature_set,
                "drop_era5_families": list(config.drop_era5_families),
                "keep_era5_families": (
                    list(config.keep_era5_families) if config.keep_era5_families is not None else None
                ),
                "calibration_method": config.calibration_method,
                "calibration_rows": fitted["calibration_rows"],
                "model_train_rows": fitted["model_train_rows"],
            },
        },
        model_dir / "run_config.json",
    )
    metrics_path = save_metrics(metrics, model_dir / "metrics.json")
    validation_predictions_path = model_dir / "validation_predictions.csv"
    if write_predictions:
        LOGGER.info("Writing validation predictions %s", validation_predictions_path)
        valid_predictions.to_csv(validation_predictions_path, index=False)
    LOGGER.info("Training complete for feature set %s", config.feature_set)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="train_baseline",
            config={
                "feature_table_path": str(feature_table_path),
                "train_config": {
                    "train_end_year": config.train_end_year,
                    "valid_start_year": config.valid_start_year,
                    "valid_end_year": config.valid_end_year,
                    "random_state": config.random_state,
                    "feature_set": config.feature_set,
                },
            },
            artifacts={
                "classifier": str(model_dir / "classifier.pkl"),
                "calibrator": str(model_dir / "calibrator.pkl") if (model_dir / "calibrator.pkl").exists() else None,
                "regressor": str(model_dir / "regressor.pkl") if (model_dir / "regressor.pkl").exists() else None,
                "feature_columns": str(model_dir / "feature_columns.json"),
                "signal_profile": str(model_dir / "signal_profile.json"),
                "run_config": str(model_dir / "run_config.json"),
                "validation_predictions": str(validation_predictions_path) if write_predictions else None,
                "metrics": str(metrics_path),
            },
            summary={
                "model_family": metrics["model_family"],
                "feature_set": config.feature_set,
                "n_features": len(feature_columns),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
            },
        )

    return {
        "metrics": metrics,
        "validation_predictions": valid_predictions,
        "feature_columns": feature_columns,
        "model_dir": model_dir,
    }


def _metric_or_nan(metrics: dict[str, Any], field: str) -> float:
    value = metrics.get(field, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def ablation_comparison_row(arm: str, results: dict[str, Any]) -> dict[str, Any]:
    raw = results["metrics"]["classification_raw"]
    calibrated = results["metrics"]["classification"]
    row: dict[str, Any] = {
        "arm": arm,
        "n_features": len(results["feature_columns"]),
        "model_family": results["metrics"]["model_family"],
    }
    for field in ABLATION_COMPARISON_FIELDS:
        row[field] = _metric_or_nan(raw, field)
        row[f"{field}_calibrated"] = _metric_or_nan(calibrated, field)
    return row


def format_ablation_table(
    rows: list[dict[str, Any]],
    headers: tuple[str, ...] | None = None,
) -> str:
    headers = headers or (
        "arm",
        "n_features",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "precision_at_0_5",
        "precision_top_10pct",
        "capture_top_10pct",
    )
    widths = {header: len(header) for header in headers}
    rendered: list[dict[str, str]] = []
    for row in rows:
        rendered_row = {}
        for header in headers:
            value = row[header]
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            rendered_row[header] = text
            widths[header] = max(widths[header], len(text))
        rendered.append(rendered_row)

    def fmt(row: dict[str, str]) -> str:
        return "  ".join(row[header].ljust(widths[header]) for header in headers)

    header_line = fmt({header: header for header in headers})
    body = "\n".join(fmt(row) for row in rendered)
    return f"{header_line}\n{body}"


def add_reference_deltas(
    rows: list[dict[str, Any]],
    reference_arm: str,
) -> list[dict[str, Any]]:
    reference = next((row for row in rows if row["arm"] == reference_arm), None)
    if reference is None:
        return rows
    updated = []
    for row in rows:
        out = dict(row)
        for field in ABLATION_COMPARISON_FIELDS:
            out[f"{field}_delta_vs_{reference_arm}"] = out[field] - reference[field]
        updated.append(out)
    return updated


def add_history_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return add_reference_deltas(rows, "history")


def run_feature_ablation(
    feature_table_path: str | Path,
    model_dir: str | Path | None = None,
    config: TrainConfig = TrainConfig(),
    run_root: str | Path | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train history-only, ERA5-only, and both on one loaded table and one split."""
    LOGGER.info("Reading feature table once for ablation %s", feature_table_path)
    df = read_feature_table(feature_table_path)
    LOGGER.info("Loaded %s rows x %s cols", f"{len(df):,}", df.shape[1])

    layout = None
    reports_dir: Path
    ablation_root: Path
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        ablation_root = layout.models_dir / "ablation"
        reports_dir = layout.reports_dir
    elif model_dir is not None:
        ablation_root = Path(model_dir) / "ablation"
        reports_dir = Path(model_dir)
    else:
        raise ValueError("Either model_dir or run_root must be provided")

    ablation_config = TrainConfig(
        train_end_year=config.train_end_year,
        valid_start_year=config.valid_start_year,
        valid_end_year=config.valid_end_year,
        random_state=config.random_state,
        n_estimators=config.n_estimators,
        skip_yearly_backtest=True,
        skip_regressor=True,
        dynamic_prefixes=config.dynamic_prefixes,
    )
    LOGGER.info(
        "Ablation uses the same split as the baseline: train through %s, test %s-%s, %s trees. "
        "Regressor and yearly backtest are skipped.",
        ablation_config.train_end_year,
        ablation_config.valid_start_year,
        ablation_config.valid_end_year,
        ablation_config.n_estimators,
    )

    arm_results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for arm in ABLATION_ARMS:
        LOGGER.info("=== Ablation arm: %s ===", arm)
        arm_config = TrainConfig(
            train_end_year=ablation_config.train_end_year,
            valid_start_year=ablation_config.valid_start_year,
            valid_end_year=ablation_config.valid_end_year,
            random_state=ablation_config.random_state,
            n_estimators=ablation_config.n_estimators,
            skip_yearly_backtest=True,
            skip_regressor=True,
            feature_set=arm,
            dynamic_prefixes=ablation_config.dynamic_prefixes,
        )
        results = train_baseline_models(
            feature_table_path=feature_table_path,
            model_dir=ablation_root / arm,
            config=arm_config,
            df=df,
            write_predictions=False,
        )
        row = ablation_comparison_row(arm, results)
        comparison_rows.append(row)
        arm_results[arm] = {
            "n_features": row["n_features"],
            "feature_columns": results["feature_columns"],
            "metrics_path": str(Path(results["model_dir"]) / "metrics.json"),
            "classification_raw": results["metrics"]["classification_raw"],
            "top_features": results["metrics"]["top_features"],
        }
        del results["validation_predictions"]

    comparison_rows = add_history_deltas(comparison_rows)
    comparison = {
        "train_end_year": ablation_config.train_end_year,
        "valid_start_year": ablation_config.valid_start_year,
        "valid_end_year": ablation_config.valid_end_year,
        "n_estimators": ablation_config.n_estimators,
        "feature_table_path": str(feature_table_path),
        "arms": comparison_rows,
        "note": (
            "All arms share calendar features (month_sin/month_cos/time_index). "
            "history = calendar + burn-history lags/rolls. "
            "era5 = calendar + lagged ERA5. "
            "both = every numeric column in the feature table. "
            "Ranking metrics use raw (uncalibrated) probabilities."
        ),
    }
    comparison_json = reports_dir / "ablation_comparison.json"
    comparison_csv = reports_dir / "ablation_comparison.csv"
    write_json(comparison, comparison_json)
    pd.DataFrame(comparison_rows).to_csv(comparison_csv, index=False)
    table = format_ablation_table(comparison_rows)
    LOGGER.info("Ablation comparison:\n%s", table)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="ablation",
            config={
                "feature_table_path": str(feature_table_path),
                "arms": list(ABLATION_ARMS),
                "train_end_year": ablation_config.train_end_year,
                "valid_start_year": ablation_config.valid_start_year,
                "valid_end_year": ablation_config.valid_end_year,
                "n_estimators": ablation_config.n_estimators,
            },
            artifacts={
                "comparison_json": str(comparison_json),
                "comparison_csv": str(comparison_csv),
                **{arm: str(ablation_root / arm / "metrics.json") for arm in ABLATION_ARMS},
            },
            summary={"arms": comparison_rows},
        )

    return {
        "comparison": comparison,
        "table": table,
        "comparison_json": comparison_json,
        "comparison_csv": comparison_csv,
        "arm_results": arm_results,
    }


def present_era5_families(
    columns: Iterable[str],
    dynamic_prefixes: tuple[str, ...] = DEFAULT_DYNAMIC_PREFIXES,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {family: [] for family in ERA5_FAMILY_ORDER}
    grouped["other_era5"] = []
    for column in columns:
        family = era5_family_for_column(column, dynamic_prefixes)
        if family is None:
            continue
        grouped.setdefault(family, []).append(column)
    return grouped


def recommend_kept_era5_families(rows: list[dict[str, Any]]) -> dict[str, Any]:
    both = next(row for row in rows if row["arm"] == "both")
    kept: list[str] = []
    dropped: list[str] = []
    details: list[dict[str, Any]] = []
    for family in ERA5_FAMILY_ORDER:
        row = next((item for item in rows if item["arm"] == f"drop_{family}"), None)
        if row is None:
            continue
        pr_drop = both["pr_auc"] - row["pr_auc"]
        roc_drop = both["roc_auc"] - row["roc_auc"]
        capture_drop = both["capture_top_10pct"] - row["capture_top_10pct"]
        matters = (
            pr_drop >= ERA5_KEEP_MIN_PR_AUC_DROP
            or roc_drop >= ERA5_KEEP_MIN_ROC_AUC_DROP
            or capture_drop >= ERA5_KEEP_MIN_CAPTURE_DROP
        )
        record = {
            "family": family,
            "n_features_removed": int(both["n_features"] - row["n_features"]),
            "pr_auc_drop": pr_drop,
            "roc_auc_drop": roc_drop,
            "capture_top_10pct_drop": capture_drop,
            "keep": matters,
        }
        details.append(record)
        if matters:
            kept.append(family)
        else:
            dropped.append(family)
    return {
        "keep": kept,
        "drop": dropped,
        "details": details,
        "thresholds": {
            "pr_auc": ERA5_KEEP_MIN_PR_AUC_DROP,
            "roc_auc": ERA5_KEEP_MIN_ROC_AUC_DROP,
            "capture_top_10pct": ERA5_KEEP_MIN_CAPTURE_DROP,
        },
        "rule": (
            "Keep a family if dropping it from the full both model costs at least "
            f"{ERA5_KEEP_MIN_PR_AUC_DROP} PR-AUC, {ERA5_KEEP_MIN_ROC_AUC_DROP} ROC-AUC, "
            f"or {ERA5_KEEP_MIN_CAPTURE_DROP} top-10% capture."
        ),
    }


def _ablation_train_config(
    config: TrainConfig,
    feature_set: str = "both",
    drop_era5_families: tuple[str, ...] = (),
    keep_era5_families: tuple[str, ...] | None = None,
) -> TrainConfig:
    return TrainConfig(
        train_end_year=config.train_end_year,
        valid_start_year=config.valid_start_year,
        valid_end_year=config.valid_end_year,
        random_state=config.random_state,
        n_estimators=config.n_estimators,
        skip_yearly_backtest=True,
        skip_regressor=True,
        feature_set=feature_set,
        dynamic_prefixes=config.dynamic_prefixes,
        drop_era5_families=drop_era5_families,
        keep_era5_families=keep_era5_families,
    )


def _fit_named_arm(
    arm: str,
    df: pd.DataFrame,
    feature_table_path: str | Path,
    model_dir: Path,
    config: TrainConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    LOGGER.info("=== Ablation arm: %s ===", arm)
    results = train_baseline_models(
        feature_table_path=feature_table_path,
        model_dir=model_dir,
        config=config,
        df=df,
        write_predictions=False,
    )
    row = ablation_comparison_row(arm, results)
    summary = {
        "n_features": row["n_features"],
        "feature_columns": results["feature_columns"],
        "metrics_path": str(Path(results["model_dir"]) / "metrics.json"),
        "classification_raw": results["metrics"]["classification_raw"],
        "top_features": results["metrics"]["top_features"],
    }
    del results["validation_predictions"]
    return row, summary


def run_era5_family_ablation(
    feature_table_path: str | Path,
    model_dir: str | Path | None = None,
    config: TrainConfig = TrainConfig(),
    run_root: str | Path | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Drop one ERA5 family at a time from the full both model, then retrain the keepers."""
    LOGGER.info("Reading feature table once for ERA5 family ablation %s", feature_table_path)
    df = read_feature_table(feature_table_path)
    LOGGER.info("Loaded %s rows x %s cols", f"{len(df):,}", df.shape[1])

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        ablation_root = layout.models_dir / "era5_family_ablation"
        reports_dir = layout.reports_dir
    elif model_dir is not None:
        ablation_root = Path(model_dir) / "era5_family_ablation"
        reports_dir = Path(model_dir)
    else:
        raise ValueError("Either model_dir or run_root must be provided")

    grouped = present_era5_families(df.columns, config.dynamic_prefixes)
    for family in ERA5_FAMILY_ORDER:
        LOGGER.info("ERA5 family %s: %s columns", family, len(grouped[family]))
    if grouped["other_era5"]:
        LOGGER.warning("Unmapped ERA5 columns left in every arm: %s", grouped["other_era5"])

    base_config = _ablation_train_config(config, feature_set="both")
    LOGGER.info(
        "ERA5 family ablation uses train through %s, test %s-%s, %s trees.",
        base_config.train_end_year,
        base_config.valid_start_year,
        base_config.valid_end_year,
        base_config.n_estimators,
    )

    arm_results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []

    both_row, both_summary = _fit_named_arm(
        "both",
        df,
        feature_table_path,
        ablation_root / "both",
        base_config,
    )
    comparison_rows.append(both_row)
    arm_results["both"] = both_summary

    for family in ERA5_FAMILY_ORDER:
        if not grouped[family]:
            LOGGER.info("Skipping drop_%s: no columns for that family in the feature table", family)
            continue
        row, summary = _fit_named_arm(
            f"drop_{family}",
            df,
            feature_table_path,
            ablation_root / f"drop_{family}",
            _ablation_train_config(config, feature_set="both", drop_era5_families=(family,)),
        )
        comparison_rows.append(row)
        arm_results[f"drop_{family}"] = summary

    comparison_rows = add_reference_deltas(comparison_rows, "both")
    recommendation = recommend_kept_era5_families(comparison_rows)
    LOGGER.info("ERA5 keep: %s", recommendation["keep"] or "none")
    LOGGER.info("ERA5 drop: %s", recommendation["drop"] or "none")

    if recommendation["keep"]:
        kept_row, kept_summary = _fit_named_arm(
            "kept",
            df,
            feature_table_path,
            ablation_root / "kept",
            _ablation_train_config(
                config,
                feature_set="both",
                keep_era5_families=tuple(recommendation["keep"]),
            ),
        )
        comparison_rows.append(kept_row)
        arm_results["kept"] = kept_summary
        comparison_rows = add_reference_deltas(comparison_rows, "both")
    else:
        LOGGER.info("No ERA5 family moved the both-model enough; not retraining a reduced weather set.")

    family_headers = (
        "arm",
        "n_features",
        "roc_auc",
        "pr_auc",
        "precision_top_10pct",
        "capture_top_10pct",
        "pr_auc_delta_vs_both",
        "roc_auc_delta_vs_both",
    )
    comparison = {
        "train_end_year": base_config.train_end_year,
        "valid_start_year": base_config.valid_start_year,
        "valid_end_year": base_config.valid_end_year,
        "n_estimators": base_config.n_estimators,
        "feature_table_path": str(feature_table_path),
        "family_columns": {family: grouped[family] for family in (*ERA5_FAMILY_ORDER, "other_era5")},
        "recommendation": recommendation,
        "arms": comparison_rows,
        "note": (
            "Reference arm `both` is history + all ERA5. "
            "Each drop_* arm removes one ERA5 family from that full model. "
            "`kept` retrains history + only the families that moved PR-AUC/ROC/top-10% capture. "
            "hot_dry includes hot_dry_factor, potential_evaporation, and climate_water_balance."
        ),
    }
    comparison_json = reports_dir / "era5_family_ablation.json"
    comparison_csv = reports_dir / "era5_family_ablation.csv"
    write_json(comparison, comparison_json)
    pd.DataFrame(comparison_rows).to_csv(comparison_csv, index=False)
    table = format_ablation_table(comparison_rows, headers=family_headers)
    LOGGER.info("ERA5 family ablation:\n%s", table)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="era5_family_ablation",
            config={
                "feature_table_path": str(feature_table_path),
                "train_end_year": base_config.train_end_year,
                "valid_start_year": base_config.valid_start_year,
                "valid_end_year": base_config.valid_end_year,
                "n_estimators": base_config.n_estimators,
                "keep": recommendation["keep"],
                "drop": recommendation["drop"],
            },
            artifacts={
                "comparison_json": str(comparison_json),
                "comparison_csv": str(comparison_csv),
                **{arm: str(ablation_root / arm / "metrics.json") for arm in arm_results},
            },
            summary={"arms": comparison_rows, "recommendation": recommendation},
        )

    return {
        "comparison": comparison,
        "table": table,
        "comparison_json": comparison_json,
        "comparison_csv": comparison_csv,
        "arm_results": arm_results,
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline monthly wildfire risk model")
    parser.add_argument("--features", required=True, help="Feature table CSV produced by build_features.py")
    parser.add_argument("--model-dir", help="Output directory for models and metrics")
    parser.add_argument("--train-end-year", type=int, default=2020)
    parser.add_argument("--valid-start-year", type=int, default=2021)
    parser.add_argument("--valid-end-year", type=int, default=2024)
    parser.add_argument("--run-root", help="Optional root directory for a Colab-friendly run layout")
    parser.add_argument("--run-name", help="Optional run folder name inside --run-root")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--skip-yearly-backtest",
        action="store_true",
        help="Skip the per-year refit during a holdout training run.",
    )
    parser.add_argument(
        "--yearly-backtest-only",
        action="store_true",
        help=(
            "Refit once per year and write continent scores. Does not train or replace the holdout model."
        ),
    )
    parser.add_argument(
        "--backtest-start-year",
        type=int,
        help="First test year. Training uses every earlier year in the table. Set 4 uses 2022.",
    )
    parser.add_argument(
        "--backtest-end-year",
        type=int,
        help="Last test year. Set 4 uses 2025.",
    )
    parser.add_argument(
        "--feature-set",
        choices=ABLATION_ARMS,
        default="both",
        help="Train one feature family. Ignored when --ablation is set.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Train history-only, ERA5-only, and both on the same split. Writes ablation_comparison.csv.",
    )
    parser.add_argument(
        "--era5-family-ablation",
        action="store_true",
        help=(
            "Drop temperature, precip, soil moisture, wind, and hot-dry one family at a time "
            "from the full both model, then retrain with only the families that move PR-AUC/ROC."
        ),
    )
    parser.add_argument(
        "--drop-era5-family",
        action="append",
        dest="drop_era5_families",
        choices=list(ERA5_FAMILY_ORDER),
        help="ERA5 family to drop from the working model. Repeatable. Use `wind` for the current baseline.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=("isotonic", "platt", "none"),
        default="isotonic",
        help="Isotonic matches observed fire rates. Platt was the previous scaler and does not fix maps.",
    )
    parser.add_argument(
        "--skip-regressor",
        action="store_true",
        help="Skip the burned-fraction regressor. Maps only need the classifier.",
    )
    parser.add_argument(
        "--render-maps",
        action="store_true",
        help="Render within-month percentile maps after training.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = TrainConfig(
        train_end_year=args.train_end_year,
        valid_start_year=args.valid_start_year,
        valid_end_year=args.valid_end_year,
        n_estimators=args.n_estimators,
        skip_yearly_backtest=args.skip_yearly_backtest,
        backtest_start_year=args.backtest_start_year,
        backtest_end_year=args.backtest_end_year,
        skip_regressor=args.skip_regressor,
        feature_set=args.feature_set,
        drop_era5_families=tuple(args.drop_era5_families or ()),
        calibration_method=args.calibration_method,
    )
    if args.yearly_backtest_only:
        if args.ablation or args.era5_family_ablation:
            parser.error("--yearly-backtest-only does not train ablation arms")
        results = run_yearly_backtest(
            feature_table_path=args.features,
            config=config,
            run_root=args.run_root,
            run_name=args.run_name,
            model_dir=args.model_dir,
        )
        print(results["table"])
        print(json.dumps({"json_path": str(results["json_path"]), "csv_path": str(results["csv_path"])}, indent=2))
        return
    if args.era5_family_ablation:
        results = run_era5_family_ablation(
            feature_table_path=args.features,
            model_dir=args.model_dir,
            config=config,
            run_root=args.run_root,
            run_name=args.run_name,
        )
        print(results["table"])
        print(json.dumps(results["comparison"], indent=2))
        return
    if args.ablation:
        results = run_feature_ablation(
            feature_table_path=args.features,
            model_dir=args.model_dir,
            config=config,
            run_root=args.run_root,
            run_name=args.run_name,
        )
        print(results["table"])
        print(json.dumps(results["comparison"], indent=2))
        return

    results = train_baseline_models(
        feature_table_path=args.features,
        model_dir=args.model_dir,
        config=config,
        run_root=args.run_root,
        run_name=args.run_name,
    )
    if args.render_maps:
        try:
            from .render_maps import render_monthly_risk_maps
        except ImportError:  # pragma: no cover - allows direct script execution
            from render_maps import render_monthly_risk_maps

        prediction_path = Path(results["model_dir"]) / "validation_predictions.csv"
        rendered = render_monthly_risk_maps(
            prediction_path,
            run_root=args.run_root,
            run_name=args.run_name,
            output_dir=None if args.run_root else Path(results["model_dir"]) / "maps",
        )
        print(f"Rendered {len(rendered)} monthly percentile maps")
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
