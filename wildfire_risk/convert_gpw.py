from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .grid import GridSpec
    from .run_artifacts import create_run_layout, write_stage_manifest
except ImportError:  # pragma: no cover - allows direct script execution
    from grid import GridSpec
    from run_artifacts import create_run_layout, write_stage_manifest


LOGGER = logging.getLogger(__name__)


def density_member_name(names: list[str]) -> str:
    tifs = [name for name in names if name.lower().endswith((".tif", ".tiff"))]
    density = [
        name
        for name in tifs
        if "density" in name.lower() and "qual" not in name.lower() and "context" not in name.lower()
    ]
    if len(density) == 1:
        return density[0]
    if len(tifs) == 1:
        return tifs[0]
    if density:
        return sorted(density, key=len)[0]
    raise ValueError(f"No population-density GeoTIFF in {names}")


def aggregate_density_array(
    values: np.ndarray,
    lat_north: float,
    lon_west: float,
    pixel_height_deg: float,
    pixel_width_deg: float,
    grid_spec: GridSpec,
    nodata: float | None = None,
) -> pd.DataFrame:
    """Mean population density for each 0.5 degree cell. Row 0 is the northern edge."""
    grid = np.asarray(values, dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError(f"Density raster must be 2D, got shape {grid.shape}")
    n_rows, n_cols = grid.shape
    valid = np.isfinite(grid)
    if nodata is not None and np.isfinite(nodata):
        valid &= grid != nodata
    valid &= grid >= 0.0
    if not np.any(valid):
        raise ValueError("Population raster has no valid pixels")

    row_idx, col_idx = np.where(valid)
    lats = lat_north - (row_idx + 0.5) * pixel_height_deg
    lons = lon_west + (col_idx + 0.5) * pixel_width_deg
    lons = ((lons + 180.0) % 360.0) - 180.0
    inside = (
        (lats >= grid_spec.lat_min)
        & (lats < grid_spec.lat_max)
        & (lons >= grid_spec.lon_min)
        & (lons < grid_spec.lon_max)
    )
    if not np.any(inside):
        raise ValueError("No valid population pixels fall inside the wildfire grid")

    lat_bin = np.floor((lats[inside] - grid_spec.lat_min) / grid_spec.resolution_deg).astype(int)
    lon_bin = np.floor((lons[inside] - grid_spec.lon_min) / grid_spec.resolution_deg).astype(int)
    packed = lat_bin.astype(np.int64) * grid_spec.n_lon + lon_bin.astype(np.int64)
    density = grid[row_idx[inside], col_idx[inside]]
    unique_cells, inverse = np.unique(packed, return_inverse=True)
    counts = np.bincount(inverse)
    totals = np.bincount(inverse, weights=density)
    n_lon = grid_spec.n_lon
    rows = []
    for packed_cell, count, total in zip(unique_cells, counts, totals):
        mean_density = float(total / count)
        rows.append(
            {
                "cell_id": f"{int(packed_cell) // n_lon:03d}_{int(packed_cell) % n_lon:03d}",
                "has_population": 1,
                "population_density": mean_density,
                "population_log_density": float(np.log1p(mean_density)),
                "population_valid_pixel_count": int(count),
            }
        )
    return pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)


def read_density_raster(path: Path) -> tuple[np.ndarray, float, float, float, float, float | None]:
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("Reading GPWv4 GeoTIFFs requires rasterio. Install it with `pip install rasterio`.") from exc

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            member = density_member_name(archive.namelist())
        source = f"zip://{path}!{member}"
    else:
        source = str(path)

    with rasterio.open(source) as dataset:
        values = dataset.read(1)
        transform = dataset.transform
        nodata = dataset.nodata
    pixel_width = float(transform.a)
    pixel_height = abs(float(transform.e))
    lon_west = float(transform.c)
    lat_north = float(transform.f)
    if pixel_width <= 0 or pixel_height <= 0:
        raise ValueError(f"Unexpected GPW transform {transform}")
    return values, lat_north, lon_west, pixel_height, pixel_width, None if nodata is None else float(nodata)


def find_gpw_raster(input_path: str | Path) -> Path:
    folder = Path(input_path)
    if not folder.exists():
        raise FileNotFoundError(
            f"GPW folder does not exist: {folder}. "
            "Download it with `python -m wildfire_risk.download_external gpw --years 2020 --resolution 2pt5_min`."
        )
    if folder.is_file():
        return folder
    candidates = sorted(folder.glob("*.zip")) + sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    if not candidates:
        raise ValueError(f"No GPW .zip or .tif files in {folder}")
    density = [path for path in candidates if "density" in path.name.lower() or path.suffix.lower() == ".zip"]
    return density[0] if density else candidates[0]


def build_gpw_table(
    input_path: str | Path,
    output_path: str | Path | None = None,
    resolution_deg: float = 0.5,
    run_root: str | Path | None = None,
    run_name: str | None = None,
    output_name: str = "gpw_population.csv",
) -> pd.DataFrame:
    raster_path = find_gpw_raster(input_path)
    LOGGER.info("Reading GPW raster %s", raster_path)
    values, lat_north, lon_west, pixel_height, pixel_width, nodata = read_density_raster(raster_path)
    LOGGER.info(
        "GPW grid %s x %s, pixel %.4f x %.4f deg, north %.3f west %.3f",
        values.shape[0],
        values.shape[1],
        pixel_height,
        pixel_width,
        lat_north,
        lon_west,
    )
    population = aggregate_density_array(
        values,
        lat_north=lat_north,
        lon_west=lon_west,
        pixel_height_deg=pixel_height,
        pixel_width_deg=pixel_width,
        grid_spec=GridSpec(resolution_deg=resolution_deg),
        nodata=nodata,
    )

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_path = layout.features_dir / output_name
    elif output_path is None:
        raise ValueError("Either output_path or run_root must be provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    population.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s rows to %s", f"{len(population):,}", output_path)
    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="convert_gpw",
            config={"input_path": str(raster_path), "resolution_deg": resolution_deg},
            artifacts={"gpw_population": str(output_path)},
            summary={"row_count": int(len(population)), "n_cells": int(population["cell_id"].nunique())},
        )
    return population


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate GPWv4 population density onto the wildfire 0.5 degree grid")
    parser.add_argument("--input", required=True, help="GPW zip, tif, or directory containing one")
    parser.add_argument("--output", help="Where to write gpw_population.csv")
    parser.add_argument("--resolution-deg", type=float, default=0.5)
    parser.add_argument("--run-root")
    parser.add_argument("--run-name")
    parser.add_argument("--output-name", default="gpw_population.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    table = build_gpw_table(
        input_path=args.input,
        output_path=args.output,
        resolution_deg=args.resolution_deg,
        run_root=args.run_root,
        run_name=args.run_name,
        output_name=args.output_name,
    )
    print(f"Built GPW table with {len(table):,} cells")


if __name__ == "__main__":
    main()
