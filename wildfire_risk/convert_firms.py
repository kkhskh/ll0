from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .grid import GridSpec, assign_points_to_grid
    from .run_artifacts import create_run_layout, write_stage_manifest
except ImportError:  # pragma: no cover - allows direct script execution
    from grid import GridSpec, assign_points_to_grid
    from run_artifacts import create_run_layout, write_stage_manifest


LOGGER = logging.getLogger(__name__)

# MODIS confidence below 30 is the FIRMS "low" bin. VIIRS uses nominal/high text.
LOW_CONFIDENCE_TEXT = {"low", "l"}
KEEP_CONFIDENCE_TEXT = {"nominal", "high", "n", "h"}


def detection_mask(confidence: pd.Series) -> pd.Series:
    text = confidence.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(confidence, errors="coerce")
    return numeric.ge(30).fillna(False) | text.isin(KEEP_CONFIDENCE_TEXT)


def aggregate_detections(frame: pd.DataFrame, grid_spec: GridSpec) -> pd.DataFrame:
    """Count kept active-fire detections and sum FRP into 0.5 degree cell-months."""
    required = {"latitude", "longitude", "acq_date", "confidence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"FIRMS table is missing columns: {sorted(missing)}")

    work = frame.loc[detection_mask(frame["confidence"])].copy()
    if "type" in work.columns:
        kind = pd.to_numeric(work["type"], errors="coerce")
        work = work.loc[kind.isna() | kind.eq(0)]
    if work.empty:
        return pd.DataFrame(columns=["cell_id", "month", "detection_count", "frp_sum"])

    work["lat"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["lon"] = pd.to_numeric(work["longitude"], errors="coerce")
    if "frp" in work.columns:
        work["frp"] = pd.to_numeric(work["frp"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        work["frp"] = 0.0
    work["month"] = pd.to_datetime(work["acq_date"], errors="coerce").dt.to_period("M").astype(str)
    work = work.dropna(subset=["lat", "lon", "month"])
    work = work.loc[work["month"].ne("NaT")]
    if work.empty:
        return pd.DataFrame(columns=["cell_id", "month", "detection_count", "frp_sum"])

    work = assign_points_to_grid(work, grid_spec, lat_col="lat", lon_col="lon")
    grouped = work.groupby(["cell_id", "month"], as_index=False).agg(
        detection_count=("frp", "size"),
        frp_sum=("frp", "sum"),
    )
    return grouped


def find_firms_csvs(input_path: str | Path) -> list[Path]:
    folder = Path(input_path)
    if not folder.exists():
        raise FileNotFoundError(
            f"FIRMS folder does not exist: {folder}. "
            "Download it with `python -m wildfire_risk.download_external firms --start-month 2019-01 --end-month 2025-12`."
        )
    if folder.is_file():
        return [folder]
    files = sorted(folder.rglob("*.csv"))
    if not files:
        raise ValueError(f"No FIRMS csv files in {folder}")
    return files


def build_firms_table(
    input_path: str | Path,
    output_path: str | Path | None = None,
    resolution_deg: float = 0.5,
    run_root: str | Path | None = None,
    run_name: str | None = None,
    output_name: str = "firms_monthly.csv",
) -> pd.DataFrame:
    files = find_firms_csvs(input_path)
    grid_spec = GridSpec(resolution_deg=resolution_deg)
    LOGGER.info("Reading %s FIRMS window files", len(files))
    parts: list[pd.DataFrame] = []
    for index, path in enumerate(files, start=1):
        frame = pd.read_csv(path)
        if frame.empty or "latitude" not in frame.columns:
            continue
        part = aggregate_detections(frame, grid_spec)
        if not part.empty:
            parts.append(part)
        if index == 1 or index % 50 == 0 or index == len(files):
            LOGGER.info("Aggregated %s/%s window files", index, len(files))
    if not parts:
        raise ValueError(f"No kept FIRMS detections in {input_path}")

    monthly = (
        pd.concat(parts, ignore_index=True)
        .groupby(["cell_id", "month"], as_index=False)[["detection_count", "frp_sum"]]
        .sum()
    )
    monthly["detection_count"] = monthly["detection_count"].astype(int)
    monthly = monthly.sort_values(["month", "cell_id"]).reset_index(drop=True)

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_path = layout.features_dir / output_name
    elif output_path is None:
        raise ValueError("Either output_path or run_root must be provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s cell-months to %s", f"{len(monthly):,}", output_path)
    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="convert_firms",
            config={"input_path": str(input_path), "resolution_deg": resolution_deg, "n_files": len(files)},
            artifacts={"firms_monthly": str(output_path)},
            summary={
                "row_count": int(len(monthly)),
                "n_cells": int(monthly["cell_id"].nunique()),
                "months": [str(monthly["month"].min()), str(monthly["month"].max())],
            },
        )
    return monthly


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate FIRMS active-fire detections onto the wildfire 0.5 degree grid"
    )
    parser.add_argument("--input", required=True, help="FIRMS csv or a directory of window csvs")
    parser.add_argument("--output", help="Where to write firms_monthly.csv")
    parser.add_argument("--resolution-deg", type=float, default=0.5)
    parser.add_argument("--run-root")
    parser.add_argument("--run-name")
    parser.add_argument("--output-name", default="firms_monthly.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    table = build_firms_table(
        input_path=args.input,
        output_path=args.output,
        resolution_deg=args.resolution_deg,
        run_root=args.run_root,
        run_name=args.run_name,
        output_name=args.output_name,
    )
    print(
        f"Built FIRMS table with {len(table):,} cell-months, "
        f"{table['cell_id'].nunique():,} cells, {table['month'].min()} to {table['month'].max()}"
    )


if __name__ == "__main__":
    main()
