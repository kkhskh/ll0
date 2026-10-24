from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

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
    "burned_pixel_count",
    "burned_event_count",
    "product_year",
}


def downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        out[float_cols] = out[float_cols].astype("float32")
    return out


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    LOGGER.info("Reading %s", path)
    df = pd.read_csv(path)
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").astype(str)
    df = downcast_floats(df)
    LOGGER.info("Loaded %s rows x %s cols from %s", f"{len(df):,}", df.shape[1], path.name)
    return df


def _prefixed_feature_frame(
    df: pd.DataFrame,
    prefix: str,
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    rename_map = {}
    for column in df.columns:
        if column not in key_columns:
            rename_map[column] = column if (prefix and column.startswith(f"{prefix}_")) else (f"{prefix}_{column}" if prefix else column)
    return df.rename(columns=rename_map)


def filter_labels_to_dynamic_cells(
    labels_df: pd.DataFrame,
    dynamic_sources: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    """Keep only grid cells that appear in at least one dynamic source.

    The full 0.5 degree label table includes ocean. ERA5-Land does not. Training on
    those empty water cells triples RAM without adding fire signal.
    """
    dynamic_sources = list(dynamic_sources)
    if not dynamic_sources:
        return labels_df

    land_cells: set[str] = set()
    for path, _prefix in dynamic_sources:
        cells = pd.read_csv(path, usecols=["cell_id"])["cell_id"].astype(str)
        land_cells.update(cells.unique().tolist())
        LOGGER.info("Dynamic source %s covers %s cells", path, f"{cells.nunique():,}")

    before = len(labels_df)
    out = labels_df[labels_df["cell_id"].astype(str).isin(land_cells)].copy()
    LOGGER.info(
        "Restricted labels from %s to %s rows (%s land cells)",
        f"{before:,}",
        f"{len(out):,}",
        f"{out['cell_id'].nunique():,}",
    )
    if out.empty:
        raise ValueError("No label rows remain after restricting to dynamic-source cells")
    return out


def merge_dynamic_sources(
    base_df: pd.DataFrame,
    dynamic_sources: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    merged = base_df
    for path, prefix in dynamic_sources:
        source_df = read_table(path)
        missing = {"cell_id", "month"} - set(source_df.columns)
        if missing:
            raise ValueError(f"Dynamic source {path} is missing required columns: {sorted(missing)}")
        source_df = _prefixed_feature_frame(source_df, prefix=prefix, key_columns=("cell_id", "month"))
        LOGGER.info("Merging %s onto labels", path)
        merged = merged.merge(source_df, on=["cell_id", "month"], how="left")
        LOGGER.info("Merged shape %s x %s", f"{len(merged):,}", merged.shape[1])
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


def dynamic_feature_columns(df: pd.DataFrame, dynamic_prefixes: Iterable[str]) -> list[str]:
    prefixes = [f"{prefix}_" for prefix in dynamic_prefixes if prefix]
    return [
        column
        for column in df.columns
        if any(column.startswith(prefix) for prefix in prefixes)
        and column not in ID_COLUMNS
        and np.issubdtype(df[column].dtype, np.number)
    ]


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


def add_dynamic_source_features(
    df: pd.DataFrame,
    dynamic_prefixes: Iterable[str],
    group_col: str = "cell_id",
    rolling_windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    out = df.sort_values([group_col, "year", "month_num"]).copy()
    grouped = out.groupby(group_col, group_keys=False)
    source_columns = dynamic_feature_columns(out, dynamic_prefixes=dynamic_prefixes)
    if not source_columns:
        return out

    for column in source_columns:
        shifted = grouped[column].shift(1)
        out[f"{column}_lag_1"] = shifted
        for window in rolling_windows:
            out[f"{column}_rollmean_{window}"] = (
                shifted.groupby(out[group_col]).rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
            )
            out[f"{column}_rollmax_{window}"] = (
                shifted.groupby(out[group_col]).rolling(window=window, min_periods=1).max().reset_index(level=0, drop=True)
            )

    # Drop contemporaneous observed values so the feature table stays usable for forward-looking inference.
    return out.drop(columns=source_columns)


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


def finalize_feature_table(
    df: pd.DataFrame,
    dynamic_prefixes: Iterable[str] = (),
) -> pd.DataFrame:
    LOGGER.info("Adding calendar features")
    out = add_calendar_features(df)
    LOGGER.info("Adding lagged dynamic predictors")
    out = add_dynamic_source_features(out, dynamic_prefixes=dynamic_prefixes)
    LOGGER.info("Adding burn-history features")
    out = add_history_features(out)
    LOGGER.info("Adding month-anomaly features")
    out = add_anomaly_features(out)

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].fillna(0.0)
    out = downcast_floats(out)
    LOGGER.info("Finalizing %s rows x %s cols", f"{len(out):,}", out.shape[1])
    return out.sort_values(["month", "cell_id"]).reset_index(drop=True)


def build_feature_table(
    labels_path: str | Path,
    output_path: str | Path | None = None,
    dynamic_sources: Iterable[tuple[str, str]] = (),
    static_sources: Iterable[tuple[str, str]] = (),
    run_root: str | Path | None = None,
    run_name: str | None = None,
    output_name: str = "feature_table.csv",
    restrict_to_dynamic_cells: bool = True,
) -> pd.DataFrame:
    labels_df = read_table(labels_path)
    required = {"cell_id", "month", "year", "month_num", "binary_risk_label", "burned_fraction"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"Label table is missing required columns: {sorted(missing)}")

    dynamic_sources = list(dynamic_sources)
    static_sources = list(static_sources)
    if restrict_to_dynamic_cells and dynamic_sources:
        labels_df = filter_labels_to_dynamic_cells(labels_df, dynamic_sources=dynamic_sources)

    features_df = merge_dynamic_sources(labels_df, dynamic_sources=dynamic_sources)
    features_df = merge_static_sources(features_df, static_sources=static_sources)
    features_df = finalize_feature_table(
        features_df,
        dynamic_prefixes=[prefix for _, prefix in dynamic_sources],
    )

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_path = layout.features_dir / output_name
    elif output_path is None:
        raise ValueError("Either output_path or run_root must be provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing %s", output_path)
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
                "restrict_to_dynamic_cells": restrict_to_dynamic_cells,
            },
            artifacts={"feature_table": str(output_path)},
            summary={"row_count": int(len(features_df)), "column_count": int(features_df.shape[1])},
        )
    return features_df


def months_before(month: str, count: int) -> list[str]:
    year = int(month[:4])
    month_num = int(month[5:7])
    prior: list[str] = []
    for _ in range(count):
        month_num -= 1
        if month_num == 0:
            month_num = 12
            year -= 1
        prior.append(f"{year:04d}-{month_num:02d}")
    prior.reverse()
    return prior


def lagged_dynamic_columns(
    panel: pd.DataFrame,
    source_df: pd.DataFrame,
    prefix: str,
    lookback_months: int = 12,
) -> tuple[pd.DataFrame, list[str]]:
    """Lag a monthly source on the calendar, including months before the feature table starts.

    Same-month values are not returned. January 2020 gets December 2019, and the 12-month
    mean is the previous 12 calendar months with missing months treated as zero.
    """
    source_df = _prefixed_feature_frame(source_df, prefix=prefix, key_columns=("cell_id", "month"))
    value_columns = [
        column
        for column in source_df.columns
        if column not in {"cell_id", "month"} and np.issubdtype(source_df[column].dtype, np.number)
    ]
    if not value_columns:
        raise ValueError(f"Dynamic source with prefix {prefix!r} has no numeric columns")

    first_month = str(panel["month"].min())
    prior_months = months_before(first_month, lookback_months)
    cells = panel[["cell_id"]].drop_duplicates()
    prior = cells.assign(_key=1).merge(pd.DataFrame({"month": prior_months, "_key": 1}), on="_key").drop(columns="_key")
    prior["year"] = prior["month"].str.slice(0, 4).astype(int)
    prior["month_num"] = prior["month"].str.slice(5, 7).astype(int)
    extended = pd.concat(
        [panel[["cell_id", "month", "year", "month_num"]], prior],
        ignore_index=True,
    )
    extended = extended.merge(source_df, on=["cell_id", "month"], how="left")
    for column in value_columns:
        extended[column] = extended[column].fillna(0.0)
    extended = extended.sort_values(["cell_id", "year", "month_num"]).reset_index(drop=True)

    grouped = extended.groupby("cell_id", group_keys=False)
    new_columns: list[str] = []
    for column in value_columns:
        shifted = grouped[column].shift(1)
        lag_name = f"{column}_lag_1"
        roll_name = f"{column}_rollmean_{lookback_months}"
        extended[lag_name] = shifted
        extended[roll_name] = (
            shifted.groupby(extended["cell_id"])
            .rolling(window=lookback_months, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        new_columns.extend([lag_name, roll_name])
        if column.endswith("detection_count"):
            prefix_name = column[: -len("detection_count")]
            log_lag = f"{prefix_name}detection_log_lag_1"
            log_roll = f"{prefix_name}detection_log_rollmean_{lookback_months}"
            extended[log_lag] = np.log1p(extended[lag_name])
            extended[log_roll] = np.log1p(extended[roll_name])
            new_columns.extend([log_lag, log_roll])

    lagged = extended.loc[extended["month"] >= first_month, ["cell_id", "month", *new_columns]].copy()
    lagged[new_columns] = lagged[new_columns].fillna(0.0)
    return lagged, new_columns


def attach_dynamic_to_feature_table(
    base_features_path: str | Path,
    dynamic_sources: Iterable[tuple[str, str]],
    output_path: str | Path | None = None,
    run_root: str | Path | None = None,
    run_name: str | None = None,
    output_name: str = "feature_table.csv",
    lookback_months: int = 12,
) -> pd.DataFrame:
    """Join monthly sources onto an existing feature table as prior-month lags only.

    The same-month source columns are dropped after the lags are built.
    """
    dynamic_sources = list(dynamic_sources)
    if not dynamic_sources:
        raise ValueError("attach_dynamic_to_feature_table needs at least one --dynamic-source")

    features_df = read_table(base_features_path)
    required = {"cell_id", "month", "year", "month_num"}
    missing = required - set(features_df.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing)}")

    panel = features_df[["cell_id", "month", "year", "month_num"]].copy()
    new_cols: list[str] = []
    for path, prefix in dynamic_sources:
        source_df = read_table(path)
        source_missing = {"cell_id", "month"} - set(source_df.columns)
        if source_missing:
            raise ValueError(f"Dynamic source {path} is missing required columns: {sorted(source_missing)}")
        lagged, columns = lagged_dynamic_columns(
            panel,
            source_df,
            prefix=prefix,
            lookback_months=lookback_months,
        )
        LOGGER.info("Lagged %s into %s columns and dropped the same-month values", path, len(columns))
        features_df = features_df.merge(lagged, on=["cell_id", "month"], how="left")
        new_cols.extend(columns)

    if not new_cols:
        raise ValueError("Dynamic sources did not add any lagged columns")

    month_means = features_df.groupby("month_num")[new_cols].transform("mean")
    for column in list(new_cols):
        anomaly = f"{column}_month_anom"
        features_df[anomaly] = features_df[column] - month_means[column]
        new_cols.append(anomaly)
    features_df[new_cols] = features_df[new_cols].fillna(0.0)
    features_df = downcast_floats(features_df)

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_path = layout.features_dir / output_name
    elif output_path is None:
        raise ValueError("Either output_path or run_root must be provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing %s with %s new columns", output_path, len(new_cols))
    features_df.to_csv(output_path, index=False)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="attach_dynamic_features",
            config={
                "base_features_path": str(base_features_path),
                "dynamic_sources": [[str(path), prefix] for path, prefix in dynamic_sources],
                "output_name": output_name,
                "lookback_months": lookback_months,
            },
            artifacts={"feature_table": str(output_path)},
            summary={
                "row_count": int(len(features_df)),
                "column_count": int(features_df.shape[1]),
                "new_columns": new_cols,
            },
        )
    return features_df


def attach_static_to_feature_table(
    base_features_path: str | Path,
    static_sources: Iterable[tuple[str, str]],
    output_path: str | Path | None = None,
    run_root: str | Path | None = None,
    run_name: str | None = None,
    output_name: str = "feature_table.csv",
) -> pd.DataFrame:
    """Join static/annual columns onto an already-built feature table.

    Used when labels are gone but a previous feature table (for example v3) still exists.
    Does not recompute history or ERA5 lags.
    """
    static_sources = list(static_sources)
    if not static_sources:
        raise ValueError("attach_static_to_feature_table needs at least one --static-source")

    features_df = read_table(base_features_path)
    required = {"cell_id", "year"}
    missing = required - set(features_df.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing)}")

    before_cols = set(features_df.columns)
    features_df = merge_static_sources(features_df, static_sources=static_sources)
    new_cols = [column for column in features_df.columns if column not in before_cols]
    if not new_cols:
        raise ValueError("Static sources did not add any new columns")

    numeric_new = [column for column in new_cols if np.issubdtype(features_df[column].dtype, np.number)]
    if numeric_new:
        month_means = features_df.groupby("month_num")[numeric_new].transform("mean")
        for column in numeric_new:
            features_df[f"{column}_month_anom"] = features_df[column] - month_means[column]
            new_cols.append(f"{column}_month_anom")
        features_df[new_cols] = features_df[new_cols].fillna(0.0)
        features_df = downcast_floats(features_df)

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_path = layout.features_dir / output_name
    elif output_path is None:
        raise ValueError("Either output_path or run_root must be provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing %s with %s new columns", output_path, len(new_cols))
    features_df.to_csv(output_path, index=False)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="attach_static_features",
            config={
                "base_features_path": str(base_features_path),
                "static_sources": [[str(path), prefix] for path, prefix in static_sources],
                "output_name": output_name,
            },
            artifacts={"feature_table": str(output_path)},
            summary={
                "row_count": int(len(features_df)),
                "column_count": int(features_df.shape[1]),
                "new_columns": new_cols,
            },
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
    parser.add_argument("--labels", help="CSV created by build_labels.py")
    parser.add_argument(
        "--base-features",
        help=(
            "Existing feature table to attach sources onto. Static sources are copied on. "
            "Dynamic sources are joined as prior-month lags and the same-month values are dropped."
        ),
    )
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
    parser.add_argument(
        "--keep-all-cells",
        action="store_true",
        help="Keep ocean/empty cells that are missing from dynamic sources. Uses much more RAM.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.base_features and args.dynamic_source:
        features = attach_dynamic_to_feature_table(
            base_features_path=args.base_features,
            dynamic_sources=parse_source_arguments(args.dynamic_source),
            output_path=args.output,
            run_root=args.run_root,
            run_name=args.run_name,
            output_name=args.output_name,
        )
    elif args.base_features:
        features = attach_static_to_feature_table(
            base_features_path=args.base_features,
            static_sources=parse_source_arguments(args.static_source),
            output_path=args.output,
            run_root=args.run_root,
            run_name=args.run_name,
            output_name=args.output_name,
        )
    else:
        if not args.labels:
            raise ValueError("Provide --labels, or --base-features to attach static columns onto an existing table")
        features = build_feature_table(
            labels_path=args.labels,
            output_path=args.output,
            dynamic_sources=parse_source_arguments(args.dynamic_source),
            static_sources=parse_source_arguments(args.static_source),
            run_root=args.run_root,
            run_name=args.run_name,
            output_name=args.output_name,
            restrict_to_dynamic_cells=not args.keep_all_cells,
        )
    print(f"Built feature table with {len(features):,} rows and {features.shape[1]} columns")


if __name__ == "__main__":
    main()
