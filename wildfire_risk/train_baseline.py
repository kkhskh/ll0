from __future__ import annotations

import argparse
import importlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

try:
    from .evaluate import classification_metrics, grouped_classification_metrics, regression_metrics, save_metrics
except ImportError:  # pragma: no cover - allows direct script execution
    from evaluate import classification_metrics, grouped_classification_metrics, regression_metrics, save_metrics
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
    "binary_risk_label",
    "burned_fraction",
    "burned_area_km2",
}


@dataclass(frozen=True)
class TrainConfig:
    train_end_year: int = 2020
    valid_start_year: int = 2021
    valid_end_year: int = 2024
    random_state: int = 42


def read_feature_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").astype(str)
    return df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS and np.issubdtype(df[column].dtype, np.number)
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns found in the feature table")
    return feature_columns


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
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                random_state=config.random_state,
                class_weight="balanced",
            ),
            LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                random_state=config.random_state,
            ),
        )

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier, XGBRegressor

        return (
            "xgboost",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=config.random_state,
                eval_metric="logloss",
            ),
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=config.random_state,
            ),
        )

    return (
        "sklearn_hist_gradient_boosting",
        HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.05,
            max_iter=300,
            random_state=config.random_state,
        ),
        HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=300,
            random_state=config.random_state,
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
    sample_weight = make_balanced_sample_weight(y)
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)
    return model


def fit_regressor(model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    model.fit(X, y)
    return model


def fit_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    config: TrainConfig,
) -> dict[str, Any]:
    model_family, classifier, regressor = make_model_pair(config)
    classifier = fit_classifier(classifier, train_df[feature_columns], train_df["binary_risk_label"])
    regressor = fit_regressor(regressor, train_df[feature_columns], train_df["burned_fraction"])

    valid_predictions = valid_df.copy()
    valid_predictions["predicted_probability"] = classifier.predict_proba(valid_df[feature_columns])[:, 1]
    valid_predictions["predicted_burned_fraction"] = np.clip(
        regressor.predict(valid_df[feature_columns]),
        0.0,
        1.0,
    )

    return {
        "model_family": model_family,
        "classifier": classifier,
        "regressor": regressor,
        "feature_columns": feature_columns,
        "valid_predictions": valid_predictions,
    }


def yearly_backtest(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: TrainConfig,
) -> dict[str, dict[str, float]]:
    yearly_scores: dict[str, dict[str, float]] = {}
    for year in range(config.valid_start_year, config.valid_end_year + 1):
        train_df = df[df["year"] < year].copy()
        test_df = df[df["year"] == year].copy()
        if train_df.empty or test_df.empty:
            continue

        _, classifier, _ = make_model_pair(config)
        classifier = fit_classifier(classifier, train_df[feature_columns], train_df["binary_risk_label"])
        y_prob = classifier.predict_proba(test_df[feature_columns])[:, 1]
        yearly_scores[str(year)] = classification_metrics(test_df["binary_risk_label"], y_prob)
    return yearly_scores


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


def train_baseline_models(
    feature_table_path: str | Path,
    model_dir: str | Path | None = None,
    config: TrainConfig = TrainConfig(),
    run_root: str | Path | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    df = read_feature_table(feature_table_path)
    feature_columns = select_feature_columns(df)
    train_df, valid_df = temporal_split(df, config=config)
    fitted = fit_models(train_df, valid_df, feature_columns, config=config)

    valid_predictions = fitted["valid_predictions"]
    metrics = {
        "model_family": fitted["model_family"],
        "train_years": [int(train_df["year"].min()), int(train_df["year"].max())],
        "valid_years": [int(valid_df["year"].min()), int(valid_df["year"].max())],
        "classification": classification_metrics(
            valid_predictions["binary_risk_label"],
            valid_predictions["predicted_probability"],
        ),
        "regression": regression_metrics(
            valid_predictions["burned_fraction"],
            valid_predictions["predicted_burned_fraction"],
        ),
        "per_year_backtest": yearly_backtest(df, feature_columns, config=config),
        "per_month": grouped_classification_metrics(valid_predictions, group_col="month_num"),
        "top_features": summarize_feature_importance(fitted["classifier"], feature_columns),
    }

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
    with (model_dir / "regressor.pkl").open("wb") as handle:
        pickle.dump(fitted["regressor"], handle)
    write_json({"feature_columns": feature_columns}, model_dir / "feature_columns.json")
    write_json(
        {
            "feature_table_path": str(feature_table_path),
            "train_config": {
                "train_end_year": config.train_end_year,
                "valid_start_year": config.valid_start_year,
                "valid_end_year": config.valid_end_year,
                "random_state": config.random_state,
            },
        },
        model_dir / "run_config.json",
    )
    validation_predictions_path = model_dir / "validation_predictions.csv"
    metrics_path = save_metrics(metrics, model_dir / "metrics.json")
    valid_predictions.to_csv(validation_predictions_path, index=False)

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
                },
            },
            artifacts={
                "classifier": str(model_dir / "classifier.pkl"),
                "regressor": str(model_dir / "regressor.pkl"),
                "feature_columns": str(model_dir / "feature_columns.json"),
                "run_config": str(model_dir / "run_config.json"),
                "validation_predictions": str(validation_predictions_path),
                "metrics": str(metrics_path),
            },
            summary={
                "model_family": metrics["model_family"],
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
            },
        )

    return {
        "metrics": metrics,
        "validation_predictions": valid_predictions,
        "feature_columns": feature_columns,
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
    args = parser.parse_args()

    results = train_baseline_models(
        feature_table_path=args.features,
        model_dir=args.model_dir,
        config=TrainConfig(
            train_end_year=args.train_end_year,
            valid_start_year=args.valid_start_year,
            valid_end_year=args.valid_end_year,
        ),
        run_root=args.run_root,
        run_name=args.run_name,
    )
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
