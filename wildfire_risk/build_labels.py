from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

try:
    from .grid import GridSpec, LabelSpec, assign_points_to_grid, create_global_grid, create_month_index
except ImportError:  # pragma: no cover - allows direct script execution
    from grid import GridSpec, LabelSpec, assign_points_to_grid, create_global_grid, create_month_index
try:
    from .run_artifacts import create_run_layout, write_stage_manifest
except ImportError:  # pragma: no cover - allows direct script execution
    from run_artifacts import create_run_layout, write_stage_manifest

try:
    from pyhdf.SD import SD, SDC
except ImportError:  # pragma: no cover - handled with a clear runtime error
    SD = None
    SDC = None


MCD64A1_FILENAME_PATTERN = re.compile(
    r"^MCD64A1\.A(?P<year>\d{4})(?P<doy>\d{3})\.(?P<tile>h\d{2}v\d{2})\.\d{3}\..+\.hdf$"
)
UPPER_LEFT_PATTERN = re.compile(r"UpperLeftPointMtrs=\(([-0-9.]+),([-0-9.]+)\)")
LOWER_RIGHT_PATTERN = re.compile(r"LowerRightMtrs=\(([-0-9.]+),([-0-9.]+)\)")
MCD64A1_PIXEL_SIZE_M = 463.312716528
MCD64A1_PIXEL_AREA_KM2 = (MCD64A1_PIXEL_SIZE_M ** 2) / 1_000_000.0
MODIS_SINUSOIDAL_PROJ = "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"


@dataclass(frozen=True)
class LabelBuilderConfig:
    grid_spec: GridSpec = GridSpec()
    label_spec: LabelSpec = LabelSpec()
    lat_col: str = "lat"
    lon_col: str = "lon"
    date_col: str = "date"
    burned_area_col: str = "burned_area_km2"


def ensure_mcd64a1_dependencies() -> None:
    if SD is None or SDC is None:
        raise ImportError(
            "Reading MCD64A1 HDF files requires `pyhdf`. Install it with `python -m pip install pyhdf`."
        )


def load_burn_records(input_path: str | Path, config: LabelBuilderConfig) -> pd.DataFrame:
    input_path = Path(input_path)
    df = pd.read_csv(input_path)

    required = {config.lat_col, config.lon_col, config.date_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing)}")

    out = df.copy()
    out[config.date_col] = pd.to_datetime(out[config.date_col], utc=False)
    if config.burned_area_col not in out.columns:
        out[config.burned_area_col] = 1.0
    out[config.burned_area_col] = out[config.burned_area_col].fillna(0.0).astype(float)
    out["month"] = out[config.date_col].dt.to_period("M").astype(str)
    return out


def build_full_label_index(config: LabelBuilderConfig, start_month: str, end_month: str) -> pd.DataFrame:
    grid = create_global_grid(config.grid_spec)
    month_index = create_month_index(start_month, end_month)
    return grid[["cell_id", "lat_center", "lon_center", "cell_area_km2"]].merge(
        month_index[["month", "year", "month_num"]],
        how="cross",
    )


def finalize_label_table(
    monthly: pd.DataFrame,
    config: LabelBuilderConfig,
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    full_index = build_full_label_index(config=config, start_month=start_month, end_month=end_month)
    labels = full_index.merge(monthly, on=["cell_id", "month"], how="left")
    labels["burned_area_km2"] = labels["burned_area_km2"].fillna(0.0)
    labels["burned_event_count"] = labels["burned_event_count"].fillna(0).astype(int)
    labels["burned_fraction"] = (
        labels["burned_area_km2"] / labels["cell_area_km2"].replace(0.0, np.nan)
    ).fillna(0.0)
    labels["burned_fraction"] = labels["burned_fraction"].clip(0.0, 1.0)
    labels["binary_risk_label"] = (
        labels["burned_fraction"] >= config.label_spec.burned_fraction_threshold
    ).astype(int)
    return labels.sort_values(["month", "cell_id"]).reset_index(drop=True)


def aggregate_burn_records(
    burn_records: pd.DataFrame,
    config: LabelBuilderConfig,
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    assigned = assign_points_to_grid(
        burn_records,
        spec=config.grid_spec,
        lat_col=config.lat_col,
        lon_col=config.lon_col,
    )
    assigned = assigned[assigned["month"].between(start_month, end_month)]

    monthly = (
        assigned.groupby(["cell_id", "month"], as_index=False)
        .agg(
            burned_area_km2=(config.burned_area_col, "sum"),
            burned_event_count=(config.burned_area_col, "size"),
        )
    )
    return finalize_label_table(
        monthly=monthly,
        config=config,
        start_month=start_month,
        end_month=end_month,
    )


def parse_mcd64a1_filename(file_path: str | Path) -> tuple[str, int, int, str]:
    file_name = Path(file_path).name
    match = MCD64A1_FILENAME_PATTERN.match(file_name)
    if not match:
        raise ValueError(f"File name does not match expected MCD64A1 pattern: {file_name}")

    year = int(match.group("year"))
    doy = int(match.group("doy"))
    tile_id = match.group("tile")
    date_value = datetime(year, 1, 1) + timedelta(days=doy - 1)
    month = pd.Timestamp(date_value).to_period("M").strftime("%Y-%m")
    return month, year, doy, tile_id


def parse_mcd64a1_extent(struct_metadata: str) -> tuple[float, float, float, float]:
    upper_left = UPPER_LEFT_PATTERN.search(struct_metadata)
    lower_right = LOWER_RIGHT_PATTERN.search(struct_metadata)
    if upper_left is None or lower_right is None:
        raise ValueError("Could not parse UpperLeftPointMtrs / LowerRightMtrs from StructMetadata.0")
    return (
        float(upper_left.group(1)),
        float(upper_left.group(2)),
        float(lower_right.group(1)),
        float(lower_right.group(2)),
    )


def iter_mcd64a1_files(
    input_path: str | Path,
    start_month: str,
    end_month: str,
) -> dict[str, list[Path]]:
    folder = Path(input_path)
    if not folder.is_dir():
        raise ValueError(f"MCD64A1 input path must be a directory, got {folder}")

    grouped: dict[str, list[Path]] = defaultdict(list)
    for file_path in sorted(folder.glob("*.hdf")):
        try:
            month, _, _, _ = parse_mcd64a1_filename(file_path)
        except ValueError:
            continue
        if start_month <= month <= end_month:
            grouped[month].append(file_path)

    if not grouped:
        raise ValueError(
            f"No MCD64A1 `.hdf` files between {start_month} and {end_month} were found in {folder}"
        )
    return dict(sorted(grouped.items()))


def aggregate_mcd64a1_tile(
    file_path: str | Path,
    grid_spec: GridSpec,
    transformer: Transformer,
) -> dict[str, int]:
    ensure_mcd64a1_dependencies()
    hdf = SD(str(file_path), SDC.READ)
    try:
        burn_date = hdf.select("Burn Date").get()
        burned_rows, burned_cols = np.where(burn_date > 0)
        if burned_rows.size == 0:
            return {}

        struct_metadata = hdf.attributes().get("StructMetadata.0")
        if struct_metadata is None:
            raise ValueError(f"Missing StructMetadata.0 in {file_path}")
        ulx, uly, lrx, lry = parse_mcd64a1_extent(struct_metadata)
        n_rows, n_cols = burn_date.shape
        pixel_width = (lrx - ulx) / n_cols
        pixel_height = (uly - lry) / n_rows

        xs = ulx + (burned_cols + 0.5) * pixel_width
        ys = uly - (burned_rows + 0.5) * pixel_height
        lons, lats = transformer.transform(xs, ys)

        valid = (
            (lats >= grid_spec.lat_min)
            & (lats < grid_spec.lat_max)
            & (lons >= grid_spec.lon_min)
            & (lons < grid_spec.lon_max)
        )
        if not np.any(valid):
            return {}

        lat_idx = np.floor((lats[valid] - grid_spec.lat_min) / grid_spec.resolution_deg).astype(int)
        lon_idx = np.floor((lons[valid] - grid_spec.lon_min) / grid_spec.resolution_deg).astype(int)
        cell_ids = np.char.add(
            np.char.add(np.char.zfill(lat_idx.astype(str), 3), "_"),
            np.char.zfill(lon_idx.astype(str), 3),
        )
        unique_cells, counts = np.unique(cell_ids, return_counts=True)
        return {str(cell_id): int(count) for cell_id, count in zip(unique_cells.tolist(), counts.tolist())}
    finally:
        hdf.end()


def aggregate_mcd64a1_folder(
    input_path: str | Path,
    config: LabelBuilderConfig,
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    transformer = Transformer.from_crs(MODIS_SINUSOIDAL_PROJ, "EPSG:4326", always_xy=True)
    grouped_files = iter_mcd64a1_files(input_path=input_path, start_month=start_month, end_month=end_month)

    monthly_cell_counts: dict[tuple[str, str], int] = defaultdict(int)
    for month, month_files in grouped_files.items():
        for file_path in month_files:
            tile_counts = aggregate_mcd64a1_tile(
                file_path=file_path,
                grid_spec=config.grid_spec,
                transformer=transformer,
            )
            for cell_id, burned_pixel_count in tile_counts.items():
                monthly_cell_counts[(month, cell_id)] += burned_pixel_count

    if not monthly_cell_counts:
        raise ValueError("No burned pixels were found in the requested MCD64A1 range")

    monthly = pd.DataFrame(
        [
            {
                "month": month,
                "cell_id": cell_id,
                "burned_pixel_count": burned_pixel_count,
                "burned_area_km2": burned_pixel_count * MCD64A1_PIXEL_AREA_KM2,
                "burned_event_count": burned_pixel_count,
            }
            for (month, cell_id), burned_pixel_count in monthly_cell_counts.items()
        ]
    )
    return finalize_label_table(
        monthly=monthly,
        config=config,
        start_month=start_month,
        end_month=end_month,
    )


def build_label_table(
    input_path: str | Path,
    output_path: str | Path | None,
    start_month: str = "2004-01",
    end_month: str = "2024-12",
    resolution_deg: float = 0.5,
    burned_fraction_threshold: float = 0.01,
    input_format: str = "auto",
    run_root: str | Path | None = None,
    run_name: str | None = None,
    output_name: str = "label_table.csv",
) -> pd.DataFrame:
    config = LabelBuilderConfig(
        grid_spec=GridSpec(resolution_deg=resolution_deg),
        label_spec=LabelSpec(burned_fraction_threshold=burned_fraction_threshold),
    )
    input_path = Path(input_path)

    if input_format not in {"auto", "csv", "mcd64a1-hdf"}:
        raise ValueError("input_format must be one of `auto`, `csv`, or `mcd64a1-hdf`")

    inferred_format = input_format
    if inferred_format == "auto":
        inferred_format = "mcd64a1-hdf" if input_path.is_dir() else "csv"

    if inferred_format == "csv":
        burn_records = load_burn_records(input_path, config=config)
        label_table = aggregate_burn_records(
            burn_records,
            config=config,
            start_month=start_month,
            end_month=end_month,
        )
    else:
        label_table = aggregate_mcd64a1_folder(
            input_path=input_path,
            config=config,
            start_month=start_month,
            end_month=end_month,
        )

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_path = layout.labels_dir / output_name
    elif output_path is None:
        raise ValueError("Either output_path or run_root must be provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_table.to_csv(output_path, index=False)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="build_labels",
            config={
                "input_path": str(input_path),
                "input_format": inferred_format,
                "start_month": start_month,
                "end_month": end_month,
                "resolution_deg": resolution_deg,
                "burned_fraction_threshold": burned_fraction_threshold,
                "output_name": output_name,
            },
            artifacts={"label_table": str(output_path)},
            summary={
                "row_count": int(len(label_table)),
                "positive_cells": int(label_table["binary_risk_label"].sum()),
                "burned_area_km2_total": float(label_table["burned_area_km2"].sum()),
            },
        )
    return label_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate burned area records onto a monthly global grid")
    parser.add_argument(
        "--input",
        required=True,
        help="Either a CSV with lat/lon/date columns or a directory of MCD64A1 HDF tiles",
    )
    parser.add_argument("--output", help="Where to write the monthly label table")
    parser.add_argument("--start-month", default="2004-01")
    parser.add_argument("--end-month", default="2024-12")
    parser.add_argument("--resolution-deg", type=float, default=0.5)
    parser.add_argument("--burned-fraction-threshold", type=float, default=0.01)
    parser.add_argument(
        "--input-format",
        choices=("auto", "csv", "mcd64a1-hdf"),
        default="auto",
        help="How to interpret --input. `auto` treats directories as MCD64A1 folders and files as CSVs.",
    )
    parser.add_argument("--run-root", help="Optional root directory for a Colab-friendly run layout")
    parser.add_argument("--run-name", help="Optional run folder name inside --run-root")
    parser.add_argument(
        "--output-name",
        default="label_table.csv",
        help="Label filename when using --run-root",
    )
    args = parser.parse_args()

    label_table = build_label_table(
        input_path=args.input,
        output_path=args.output,
        start_month=args.start_month,
        end_month=args.end_month,
        resolution_deg=args.resolution_deg,
        burned_fraction_threshold=args.burned_fraction_threshold,
        input_format=args.input_format,
        run_root=args.run_root,
        run_name=args.run_name,
        output_name=args.output_name,
    )
    print(
        "Built label table with "
        f"{len(label_table):,} rows spanning {label_table['month'].min()} to {label_table['month'].max()}"
    )


if __name__ == "__main__":
    main()
