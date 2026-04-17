from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .run_artifacts import create_run_layout, write_stage_manifest
except ImportError:  # pragma: no cover - allows direct script execution
    from run_artifacts import create_run_layout, write_stage_manifest


ID_COLUMNS = {
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
    "burned_event_count",
}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").astype(str)
    return df


def _prefixed_feature_frame(
    df: pd.DataFrame,
    prefix: str,
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    rename_map = {}
    for column in df.columns:
        if column not in key_columns:
            rename_map[column] = f"{prefix}_{column}" if prefix else column
    return df.rename(columns=rename_map)


def merge_dynamic_sources(
    base_df: pd.DataFrame,
    dynamic_sources: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    merged = base_df.copy()
    for path, prefix in dynamic_sources:
        source_df = read_table(path)
        missing = {"cell_id", "month"} - set(source_df.columns)
        if missing:
            raise ValueError(f"Dynamic source {path} is missing required columns: {sorted(missing)}")
        source_df = _prefixed_feature_frame(source_df, prefix=prefix, key_columns=("cell_id", "month"))
        merged = merged.merge(source_df, on=["cell_id", "month"], how="left")
    return merged


def merge_static_sources(
    base_df: pd.DataFrame,
    static_sources: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    merged = base_df.copy()
    for path, prefix in static_sources:
        source_df = read_table(path)
        missing = {"cell_id"} - set(source_df.columns)
        if missing:
            raise ValueError(f"Static source {path} is missing required columns: {sorted(missing)}")
        if "year" in source_df.columns:
            source_df = _prefixed_feature_frame(source_df, prefix=prefix, key_columns=("cell_id", "year"))
            merged = merged.merge(source_df, on=["cell_id", "year"], how="left")
        else:
            source_df = _prefixed_feature_frame(source_df, prefix=prefix, key_columns=("cell_id",))
            merged = merged.merge(source_df, on=["cell_id"], how="left")
    return merged


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    month_angle = 2.0 * np.pi * out["month_num"] / 12.0
    out["month_sin"] = np.sin(month_angle)
    out["month_cos"] = np.cos(month_angle)
    out["time_index"] = (
        (out["year"] - out["year"].min()) * 12 + out["month_num"] - 1
    ).astype(int)
    return out


def add_history_features(
    df: pd.DataFrame,
    target_col: str = "burned_fraction",
    group_col: str = "cell_id",
    lags: tuple[int, ...] = (1, 3, 6, 12),
    rolling_windows: tuple[int, ...] = (3, 6, 12, 24),
) -> pd.DataFrame:
    out = df.sort_values([group_col, "year", "month_num"]).copy()
    grouped = out.groupby(group_col, group_keys=False)

    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = grouped[target_col].shift(lag)

    shifted = grouped[target_col].shift(1)
    for window in rolling_windows:
        out[f"{target_col}_rollmean_{window}"] = (
            shifted.groupby(out[group_col]).rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        out[f"{target_col}_rollmax_{window}"] = (
            shifted.groupby(out[group_col]).rolling(window=window, min_periods=1).max().reset_index(level=0, drop=True)
        )
        out[f"{target_col}_firesum_{window}"] = (
            grouped["binary_risk_label"].shift(1).groupby(out[group_col]).rolling(window=window, min_periods=1).sum().reset_index(level=0, drop=True)
        )

    return out


def add_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_features = [
        column
        for column in out.columns
        if column not in ID_COLUMNS and np.issubdtype(out[column].dtype, np.number)
    ]

    if not numeric_features:
        return out

    month_means = out.groupby("month_num")[numeric_features].transform("mean")
    for column in numeric_features:
        out[f"{column}_month_anom"] = out[column] - month_means[column]
    return out


def finalize_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    out = add_calendar_features(df)
    out = add_history_features(out)
    out = add_anomaly_features(out)

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].fillna(0.0)
    return out.sort_values(["month", "cell_id"]).reset_index(drop=True)


def build_feature_table(
    labels_path: str | Path,
    output_path: str | Path | None = None,
    dynamic_sources: Iterable[tuple[str, str]] = (),
    static_sources: Iterable[tuple[str, str]] = (),
    run_root: str | Path | None = None,
    run_name: str | None = None,
    output_name: str = "feature_table.csv",
) -> pd.DataFrame:
    labels_df = read_table(labels_path)
    required = {"cell_id", "month", "year", "month_num", "binary_risk_label", "burned_fraction"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"Label table is missing required columns: {sorted(missing)}")

    features_df = merge_dynamic_sources(labels_df, dynamic_sources=dynamic_sources)
    features_df = merge_static_sources(features_df, static_sources=static_sources)
    features_df = finalize_feature_table(features_df)

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_path = layout.features_dir / output_name
    elif output_path is None:
        raise ValueError("Either output_path or run_root must be provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="build_features",
            config={
                "labels_path": str(labels_path),
                "dynamic_sources": [[str(path), prefix] for path, prefix in dynamic_sources],
                "static_sources": [[str(path), prefix] for path, prefix in static_sources],
                "output_name": output_name,
            },
            artifacts={"feature_table": str(output_path)},
            summary={"row_count": int(len(features_df)), "column_count": int(features_df.shape[1])},
        )
    return features_df


def parse_source_arguments(raw_values: list[str] | None) -> list[tuple[str, str]]:
    if not raw_values:
        return []
    parsed = []
    for raw_value in raw_values:
        parts = raw_value.split(":", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                "Source arguments must be in the form `path:prefix`, for example "
                "`climate.csv:era5`"
            )
        parsed.append((parts[0], parts[1]))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a monthly wildfire risk feature table")
    parser.add_argument("--labels", required=True, help="CSV created by build_labels.py")
    parser.add_argument("--output", help="Where to write the feature table")
    parser.add_argument(
        "--dynamic-source",
        action="append",
        help="Dynamic monthly source in the form path:prefix",
    )
    parser.add_argument(
        "--static-source",
        action="append",
        help="Static or annual source in the form path:prefix",
    )
    parser.add_argument("--run-root", help="Optional root directory for a Colab-friendly run layout")
    parser.add_argument("--run-name", help="Optional run folder name inside --run-root")
    parser.add_argument(
        "--output-name",
        default="feature_table.csv",
        help="Feature filename when using --run-root",
    )
    args = parser.parse_args()

    features = build_feature_table(
        labels_path=args.labels,
        output_path=args.output,
        dynamic_sources=parse_source_arguments(args.dynamic_source),
        static_sources=parse_source_arguments(args.static_source),
        run_root=args.run_root,
        run_name=args.run_name,
        output_name=args.output_name,
    )
    print(f"Built feature table with {len(features):,} rows and {features.shape[1]} columns")


if __name__ == "__main__":
    main()
