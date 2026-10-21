from __future__ import annotations

import argparse
import logging
import re
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    from .grid import GridSpec
    from .run_artifacts import create_run_layout, write_stage_manifest
except ImportError:  # pragma: no cover - allows direct script execution
    from grid import GridSpec
    from run_artifacts import create_run_layout, write_stage_manifest


NASADEM_TILE_PATTERN = re.compile(
    r"(?P<ns>[ns])(?P<lat>\d{2})(?P<ew>[ew])(?P<lon>\d{3})",
    re.IGNORECASE,
)
NASADEM_VOID = -32768
METERS_PER_DEGREE_LAT = 111_320.0
LOGGER = logging.getLogger(__name__)


def parse_nasadem_tile_origin(file_path: str | Path) -> tuple[float, float]:
    match = NASADEM_TILE_PATTERN.search(Path(file_path).name)
    if match is None:
        raise ValueError(f"Could not parse NASADEM tile origin from {file_path}")
    lat = float(match.group("lat"))
    lon = float(match.group("lon"))
    if match.group("ns").lower() == "s":
        lat = -lat
    if match.group("ew").lower() == "w":
        lon = -lon
    return lat, lon


def read_hgt_array(file_path: str | Path) -> np.ndarray:
    path = Path(file_path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            hgt_name = next((name for name in archive.namelist() if name.lower().endswith(".hgt")), None)
            if hgt_name is None:
                raise ValueError(f"No .hgt member in {path}")
            raw = np.frombuffer(archive.read(hgt_name), dtype=">i2")
    else:
        raw = np.fromfile(path, dtype=">i2")

    side = int(round(np.sqrt(raw.size)))
    if side * side != raw.size:
        raise ValueError(f"NASADEM file {path} is not a square HGT array ({raw.size} values)")
    elev = raw.reshape(side, side).astype(np.float32)
    elev[elev <= NASADEM_VOID + 1] = np.nan
    return elev


def elevation_to_slope_deg(elev: np.ndarray, lat_south: float, stride: int = 1) -> np.ndarray:
    side = elev.shape[0]
    sample_spacing_deg = 1.0 / max(side - 1, 1)
    dy_m = METERS_PER_DEGREE_LAT * sample_spacing_deg * stride
    rows = np.arange(0, side, stride)
    lats = lat_south + 1.0 - (rows + 0.5) * sample_spacing_deg
    dx_m = METERS_PER_DEGREE_LAT * np.cos(np.deg2rad(lats)) * sample_spacing_deg * stride
    dx_m = np.clip(dx_m, 1.0, None)

    sampled = elev[::stride, ::stride]
    fill = np.nanmedian(sampled)
    if not np.isfinite(fill):
        fill = 0.0
    d_row, d_col = np.gradient(np.nan_to_num(sampled, nan=fill))
    d_elev_dy = d_row / dy_m
    d_elev_dx = d_col / dx_m[:, None]
    return np.degrees(np.arctan(np.hypot(d_elev_dx, d_elev_dy)))


def pixel_lat_lon(lat_south: float, lon_west: float, side: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    sample_spacing_deg = 1.0 / max(side - 1, 1)
    rows = np.arange(0, side, stride)
    cols = np.arange(0, side, stride)
    lats = lat_south + 1.0 - (rows + 0.5) * sample_spacing_deg
    lons = lon_west + (cols + 0.5) * sample_spacing_deg
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    return lat_grid.ravel(), lon_grid.ravel()


def accumulate_terrain(
    stats: dict[str, np.ndarray],
    elev: np.ndarray,
    slope: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    grid_spec: GridSpec,
) -> int:
    valid = (
        np.isfinite(elev)
        & np.isfinite(slope)
        & (lats >= grid_spec.lat_min)
        & (lats < grid_spec.lat_max)
        & (lons >= grid_spec.lon_min)
        & (lons < grid_spec.lon_max)
    )
    if not np.any(valid):
        return 0

    lat_idx = np.floor((lats[valid] - grid_spec.lat_min) / grid_spec.resolution_deg).astype(int)
    lon_idx = np.floor((lons[valid] - grid_spec.lon_min) / grid_spec.resolution_deg).astype(int)
    packed = lat_idx.astype(np.int64) * grid_spec.n_lon + lon_idx.astype(np.int64)
    elev_v = elev[valid].astype(np.float64)
    slope_v = slope[valid].astype(np.float64)
    unique_cells, inverse = np.unique(packed, return_inverse=True)
    counts = np.bincount(inverse)
    elev_sum = np.bincount(inverse, weights=elev_v)
    elev_sumsq = np.bincount(inverse, weights=elev_v * elev_v)
    slope_sum = np.bincount(inverse, weights=slope_v)
    n_lon = grid_spec.n_lon
    for packed_cell, count, e_sum, e_sumsq, s_sum in zip(unique_cells, counts, elev_sum, elev_sumsq, slope_sum):
        cell_id = f"{int(packed_cell) // n_lon:03d}_{int(packed_cell) % n_lon:03d}"
        bucket = stats.get(cell_id)
        if bucket is None:
            bucket = np.zeros(4, dtype=np.float64)
            stats[cell_id] = bucket
        bucket[0] += count
        bucket[1] += e_sum
        bucket[2] += e_sumsq
        bucket[3] += s_sum
    return int(valid.sum())


def stats_to_rows(stats: dict[str, np.ndarray]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for cell_id, bucket in stats.items():
        count = float(bucket[0])
        if count <= 0:
            continue
        mean_elev = bucket[1] / count
        var = max(bucket[2] / count - mean_elev * mean_elev, 0.0)
        rows.append(
            {
                "cell_id": cell_id,
                "has_terrain": 1,
                "elevation_mean_m": float(mean_elev),
                "elevation_std_m": float(np.sqrt(var)),
                "slope_mean_deg": float(bucket[3] / count),
                "terrain_valid_pixel_count": int(count),
            }
        )
    return rows


def iter_nasadem_files(input_path: str | Path) -> list[Path]:
    folder = Path(input_path)
    if not folder.exists():
        raise FileNotFoundError(
            f"NASADEM folder does not exist: {folder}. "
            "Download tiles first with `python -m wildfire_risk.download_external nasadem --bbox ...`."
        )
    if not folder.is_dir():
        raise ValueError(f"NASADEM input path must be a directory, got {folder}")

    files = sorted(folder.glob("*.zip")) + sorted(folder.glob("*.hgt"))
    if not files:
        raise ValueError(f"No NASADEM `.zip` or `.hgt` files were found in {folder}")
    return files


def aggregate_nasadem_folder(
    input_path: str | Path,
    grid_spec: GridSpec | None = None,
    stride: int = 10,
    log_every: int = 25,
) -> pd.DataFrame:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    grid_spec = grid_spec or GridSpec()
    files = iter_nasadem_files(input_path)
    LOGGER.info("Starting NASADEM aggregation for %s tiles from %s (stride=%s)", len(files), input_path, stride)

    stats: dict[str, np.ndarray] = {}
    started = perf_counter()
    processed = 0
    for file_path in files:
        lat_south, lon_west = parse_nasadem_tile_origin(file_path)
        elev = read_hgt_array(file_path)
        slope = elevation_to_slope_deg(elev, lat_south=lat_south, stride=stride)
        lats, lons = pixel_lat_lon(lat_south, lon_west, elev.shape[0], stride)
        sampled_elev = elev[::stride, ::stride].ravel()
        accumulate_terrain(stats, sampled_elev, slope.ravel(), lats, lons, grid_spec)
        processed += 1
        if processed % log_every == 0 or processed == len(files):
            elapsed = perf_counter() - started
            rate = processed / elapsed if elapsed else 0.0
            LOGGER.info("Processed %s/%s NASADEM tiles (%.1f tiles/min)", processed, len(files), rate * 60.0)

    rows = stats_to_rows(stats)
    if not rows:
        raise ValueError(f"NASADEM aggregation produced no land cells from {input_path}")
    return pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)


def build_nasadem_table(
    input_path: str | Path,
    output_path: str | Path | None = None,
    resolution_deg: float = 0.5,
    stride: int = 10,
    run_root: str | Path | None = None,
    run_name: str | None = None,
    output_name: str = "nasadem_terrain.csv",
) -> pd.DataFrame:
    terrain = aggregate_nasadem_folder(
        input_path=input_path,
        grid_spec=GridSpec(resolution_deg=resolution_deg),
        stride=stride,
    )

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_path = layout.features_dir / output_name
    elif output_path is None:
        raise ValueError("Either output_path or run_root must be provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    terrain.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s rows to %s", f"{len(terrain):,}", output_path)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="convert_nasadem",
            config={"input_path": str(input_path), "resolution_deg": resolution_deg, "stride": stride},
            artifacts={"nasadem_terrain": str(output_path)},
            summary={"row_count": int(len(terrain)), "n_cells": int(terrain["cell_id"].nunique())},
        )
    return terrain


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate NASADEM HGT tiles onto the wildfire 0.5 degree grid")
    parser.add_argument("--input", required=True, help="Directory of NASADEM .zip or .hgt tiles")
    parser.add_argument("--output", help="Where to write nasadem_terrain.csv")
    parser.add_argument("--resolution-deg", type=float, default=0.5)
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Sample every Nth 1-arc-second pixel. 10 is enough for 0.5 degree means.",
    )
    parser.add_argument("--run-root", help="Optional Colab run root")
    parser.add_argument("--run-name", help="Optional run folder name inside --run-root")
    parser.add_argument("--output-name", default="nasadem_terrain.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    table = build_nasadem_table(
        input_path=args.input,
        output_path=args.output,
        resolution_deg=args.resolution_deg,
        stride=args.stride,
        run_root=args.run_root,
        run_name=args.run_name,
        output_name=args.output_name,
    )
    print(f"Built NASADEM table with {len(table):,} cells")


if __name__ == "__main__":
    main()
