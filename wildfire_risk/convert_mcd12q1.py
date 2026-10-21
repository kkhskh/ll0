from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    from .build_labels import parse_mcd64a1_extent
    from .grid import GridSpec
    from .run_artifacts import create_run_layout, write_stage_manifest
except ImportError:  # pragma: no cover - allows direct script execution
    from build_labels import parse_mcd64a1_extent
    from grid import GridSpec
    from run_artifacts import create_run_layout, write_stage_manifest

try:
    from pyhdf.SD import SD, SDC
except ImportError:  # pragma: no cover - handled with a clear runtime error
    SD = None
    SDC = None


MCD12Q1_FILENAME_PATTERN = re.compile(
    r"^MCD12Q1\.A(?P<year>\d{4})(?P<doy>\d{3})\.(?P<tile>h\d{2}v\d{2})\.\d{3}\..+\.hdf$"
)
MODIS_SINUSOIDAL_RADIUS_M = 6371007.181
IGBP_FILL = 255
IGBP_N_CLASSES = 17
LC_TYPE1_LAYER = "LC_Type1"
IGBP_GROUPS: dict[str, tuple[int, ...]] = {
    "water": (0,),
    "forest": (1, 2, 3, 4, 5),
    "shrub": (6, 7),
    "savanna": (8, 9),
    "grass": (10,),
    "wetland": (11,),
    "crop": (12, 14),
    "urban": (13,),
    "snow": (15,),
    "barren": (16,),
}
LANDCOVER_COLUMNS = (
    "land_cover_majority",
    "land_cover_valid_pixel_count",
    *(f"frac_{name}" for name in IGBP_GROUPS),
)
LOGGER = logging.getLogger(__name__)


def ensure_mcd12q1_dependencies() -> None:
    if SD is None or SDC is None:
        raise ImportError(
            "Reading MCD12Q1 HDF files requires `pyhdf`. Install it with `python -m pip install pyhdf`."
        )


def parse_mcd12q1_filename(file_path: str | Path) -> tuple[int, str]:
    file_name = Path(file_path).name
    match = MCD12Q1_FILENAME_PATTERN.match(file_name)
    if not match:
        raise ValueError(f"File name does not match expected MCD12Q1 pattern: {file_name}")
    return int(match.group("year")), match.group("tile")


def sinusoidal_xy_to_lonlat(
    xs: np.ndarray,
    ys: np.ndarray,
    radius_m: float = MODIS_SINUSOIDAL_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.asarray(ys, dtype=float) / radius_m
    cos_lat = np.cos(lat_rad)
    lon_rad = np.divide(
        np.asarray(xs, dtype=float),
        radius_m * cos_lat,
        out=np.zeros_like(lat_rad),
        where=np.abs(cos_lat) > 1e-12,
    )
    lons = np.degrees(lon_rad)
    lats = np.degrees(lat_rad)
    lons = ((lons + 180.0) % 360.0) - 180.0
    return lons, lats


def igbp_group_name(class_code: int) -> str | None:
    for name, codes in IGBP_GROUPS.items():
        if class_code in codes:
            return name
    return None


def iter_mcd12q1_files(
    input_path: str | Path,
    start_year: int,
    end_year: int,
) -> dict[int, list[Path]]:
    folder = Path(input_path)
    if not folder.exists():
        raise FileNotFoundError(
            f"MCD12Q1 folder does not exist: {folder}. "
            "Download tiles first with `python -m wildfire_risk.download_external mcd12q1`."
        )
    if not folder.is_dir():
        raise ValueError(f"MCD12Q1 input path must be a directory, got {folder}")

    grouped: dict[int, list[Path]] = defaultdict(list)
    for file_path in sorted(folder.glob("*.hdf")):
        try:
            year, _tile = parse_mcd12q1_filename(file_path)
        except ValueError:
            continue
        if start_year <= year <= end_year:
            grouped[year].append(file_path)

    if not grouped:
        raise ValueError(
            f"No MCD12Q1 `.hdf` files between {start_year} and {end_year} were found in {folder}"
        )
    return dict(sorted(grouped.items()))


def pixel_centers(
    n_rows: int,
    n_cols: int,
    ulx: float,
    uly: float,
    lrx: float,
    lry: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(0, n_rows, stride)
    cols = np.arange(0, n_cols, stride)
    pixel_width = (lrx - ulx) / n_cols
    pixel_height = (uly - lry) / n_rows
    xs = ulx + (cols + 0.5) * pixel_width
    ys = uly - (rows + 0.5) * pixel_height
    xx, yy = np.meshgrid(xs, ys)
    return xx.ravel(), yy.ravel(), rows, cols


def accumulate_igbp_counts(
    class_counts: dict[str, np.ndarray],
    classes: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    grid_spec: GridSpec,
) -> int:
    valid = (
        (classes >= 0)
        & (classes < IGBP_N_CLASSES)
        & (lats >= grid_spec.lat_min)
        & (lats < grid_spec.lat_max)
        & (lons >= grid_spec.lon_min)
        & (lons < grid_spec.lon_max)
    )
    if not np.any(valid):
        return 0

    lat_idx = np.floor((lats[valid] - grid_spec.lat_min) / grid_spec.resolution_deg).astype(int)
    lon_idx = np.floor((lons[valid] - grid_spec.lon_min) / grid_spec.resolution_deg).astype(int)
    class_values = classes[valid].astype(int)
    n_lon = grid_spec.n_lon
    packed = lat_idx.astype(np.int64) * n_lon + lon_idx.astype(np.int64)
    pair = packed * IGBP_N_CLASSES + class_values
    unique_pairs, counts = np.unique(pair, return_counts=True)
    unique_cells = unique_pairs // IGBP_N_CLASSES
    unique_classes = unique_pairs % IGBP_N_CLASSES
    for packed_cell, class_code, count in zip(unique_cells, unique_classes, counts):
        cell_id = f"{int(packed_cell) // n_lon:03d}_{int(packed_cell) % n_lon:03d}"
        bucket = class_counts.get(cell_id)
        if bucket is None:
            bucket = np.zeros(IGBP_N_CLASSES, dtype=np.int64)
            class_counts[cell_id] = bucket
        bucket[int(class_code)] += int(count)
    return int(valid.sum())


def counts_to_rows(class_counts: dict[str, np.ndarray], product_year: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for cell_id, counts in class_counts.items():
        valid_pixels = int(counts.sum())
        if valid_pixels == 0:
            continue
        fractions = counts / valid_pixels
        row: dict[str, float | int | str] = {
            "cell_id": cell_id,
            "product_year": int(product_year),
            "land_cover_majority": int(np.argmax(counts)),
            "land_cover_valid_pixel_count": valid_pixels,
        }
        for name, codes in IGBP_GROUPS.items():
            row[f"frac_{name}"] = float(fractions[list(codes)].sum())
        rows.append(row)
    return rows


def assign_feature_years(
    product_df: pd.DataFrame,
    year_offset: int = 1,
    fill_first_year: bool = True,
) -> pd.DataFrame:
    if product_df.empty:
        return product_df.copy()
    if "product_year" not in product_df.columns:
        raise ValueError("Land-cover table needs product_year before the no-leak year shift")

    shifted = product_df.copy()
    shifted["year"] = shifted["product_year"].astype(int) + int(year_offset)
    if year_offset == 0 or not fill_first_year:
        return shifted.reset_index(drop=True)

    first_product_year = int(product_df["product_year"].min())
    if first_product_year in set(shifted["year"].astype(int)):
        return shifted.reset_index(drop=True)

    fallback = product_df[product_df["product_year"].astype(int) == first_product_year].copy()
    fallback["year"] = first_product_year
    LOGGER.warning(
        "No land-cover product for %s; using %s MCD12Q1 for that calendar year only",
        first_product_year,
        first_product_year,
    )
    return pd.concat([fallback, shifted], ignore_index=True)


def aggregate_mcd12q1_tile(
    file_path: str | Path,
    class_counts: dict[str, np.ndarray],
    grid_spec: GridSpec,
    stride: int,
) -> int:
    ensure_mcd12q1_dependencies()
    hdf = SD(str(file_path), SDC.READ)
    try:
        land_cover = np.asarray(hdf.select(LC_TYPE1_LAYER).get())
        struct_metadata = hdf.attributes().get("StructMetadata.0")
        if struct_metadata is None:
            raise ValueError(f"Missing StructMetadata.0 in {file_path}")
        ulx, uly, lrx, lry = parse_mcd64a1_extent(struct_metadata)
        n_rows, n_cols = land_cover.shape
        xs, ys, rows, cols = pixel_centers(n_rows, n_cols, ulx, uly, lrx, lry, stride)
        classes = land_cover[np.ix_(rows, cols)].ravel()
        lons, lats = sinusoidal_xy_to_lonlat(xs, ys)
        return accumulate_igbp_counts(class_counts, classes, lats, lons, grid_spec)
    finally:
        hdf.end()


def aggregate_mcd12q1_folder(
    input_path: str | Path,
    start_year: int,
    end_year: int,
    grid_spec: GridSpec | None = None,
    stride: int = 1,
    year_offset: int = 1,
    fill_first_year: bool = True,
    log_every: int = 25,
) -> pd.DataFrame:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    grid_spec = grid_spec or GridSpec()
    grouped_files = iter_mcd12q1_files(input_path, start_year=start_year, end_year=end_year)
    total_files = sum(len(files) for files in grouped_files.values())
    LOGGER.info(
        "Starting MCD12Q1 aggregation for %s years and %s files from %s (stride=%s, year_offset=%s)",
        len(grouped_files),
        total_files,
        input_path,
        stride,
        year_offset,
    )

    product_frames: list[pd.DataFrame] = []
    processed = 0
    started = perf_counter()
    for year, files in grouped_files.items():
        class_counts: dict[str, np.ndarray] = {}
        year_start = perf_counter()
        for file_path in files:
            aggregate_mcd12q1_tile(file_path, class_counts, grid_spec=grid_spec, stride=stride)
            processed += 1
            if processed % log_every == 0 or processed == total_files:
                elapsed = perf_counter() - started
                rate = processed / elapsed if elapsed else 0.0
                LOGGER.info(
                    "Processed %s/%s MCD12Q1 tiles (%.1f tiles/min)",
                    processed,
                    total_files,
                    rate * 60.0,
                )
        rows = counts_to_rows(class_counts, product_year=year)
        LOGGER.info(
            "Year %s: %s cells from %s tiles in %.1f min",
            year,
            f"{len(rows):,}",
            len(files),
            (perf_counter() - year_start) / 60.0,
        )
        if rows:
            product_frames.append(pd.DataFrame(rows))

    if not product_frames:
        raise ValueError(f"MCD12Q1 aggregation produced no land cells from {input_path}")

    product_df = pd.concat(product_frames, ignore_index=True)
    return assign_feature_years(product_df, year_offset=year_offset, fill_first_year=fill_first_year)


def build_mcd12q1_table(
    input_path: str | Path,
    output_path: str | Path | None = None,
    start_year: int = 2019,
    end_year: int = 2024,
    resolution_deg: float = 0.5,
    stride: int = 1,
    year_offset: int = 1,
    fill_first_year: bool = True,
    run_root: str | Path | None = None,
    run_name: str | None = None,
    output_name: str = "mcd12q1_landcover.csv",
) -> pd.DataFrame:
    landcover = aggregate_mcd12q1_folder(
        input_path=input_path,
        start_year=start_year,
        end_year=end_year,
        grid_spec=GridSpec(resolution_deg=resolution_deg),
        stride=stride,
        year_offset=year_offset,
        fill_first_year=fill_first_year,
    )
    landcover = landcover.sort_values(["year", "cell_id"]).reset_index(drop=True)

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_path = layout.features_dir / output_name
    elif output_path is None:
        raise ValueError("Either output_path or run_root must be provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    landcover.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s rows to %s", f"{len(landcover):,}", output_path)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="convert_mcd12q1",
            config={
                "input_path": str(input_path),
                "start_year": start_year,
                "end_year": end_year,
                "resolution_deg": resolution_deg,
                "stride": stride,
                "year_offset": year_offset,
                "fill_first_year": fill_first_year,
            },
            artifacts={"mcd12q1_landcover": str(output_path)},
            summary={
                "row_count": int(len(landcover)),
                "n_cells": int(landcover["cell_id"].nunique()),
                "feature_years": sorted(int(year) for year in landcover["year"].unique()),
                "product_years": sorted(int(year) for year in landcover["product_year"].unique()),
            },
        )
    return landcover


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate MCD12Q1 IGBP land cover onto the wildfire 0.5 degree grid"
    )
    parser.add_argument("--input", required=True, help="Directory of MCD12Q1 HDF tiles")
    parser.add_argument("--output", help="Where to write mcd12q1_landcover.csv")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--resolution-deg", type=float, default=0.5)
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Sample every Nth 500 m pixel. 4 is enough for 0.5 degree fractions; 1 is a full census.",
    )
    parser.add_argument(
        "--year-offset",
        type=int,
        default=1,
        help="Join land cover from product year Y onto calendar year Y+offset. Default 1 is no-leak.",
    )
    parser.add_argument(
        "--no-fill-first-year",
        action="store_true",
        help="Do not copy the earliest product year onto itself when the previous year is missing",
    )
    parser.add_argument("--run-root", help="Optional Colab run root")
    parser.add_argument("--run-name", help="Optional run folder name inside --run-root")
    parser.add_argument("--output-name", default="mcd12q1_landcover.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    table = build_mcd12q1_table(
        input_path=args.input,
        output_path=args.output,
        start_year=args.start_year,
        end_year=args.end_year,
        resolution_deg=args.resolution_deg,
        stride=args.stride,
        year_offset=args.year_offset,
        fill_first_year=not args.no_fill_first_year,
        run_root=args.run_root,
        run_name=args.run_name,
        output_name=args.output_name,
    )
    print(
        f"Built MCD12Q1 table with {len(table):,} rows, "
        f"{table['cell_id'].nunique():,} cells, years {sorted(table['year'].unique())}"
    )


if __name__ == "__main__":
    main()
