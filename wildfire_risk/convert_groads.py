from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .grid import EARTH_RADIUS_KM, GridSpec, approximate_cell_area_km2
    from .run_artifacts import create_run_layout, write_stage_manifest
except ImportError:  # pragma: no cover - allows direct script execution
    from grid import EARTH_RADIUS_KM, GridSpec, approximate_cell_area_km2
    from run_artifacts import create_run_layout, write_stage_manifest


LOGGER = logging.getLogger(__name__)


def _wrap_dlon(lon0: np.ndarray, lon1: np.ndarray) -> np.ndarray:
    return (lon1 - lon0 + 180.0) % 360.0 - 180.0


def _segment_km(lon0: np.ndarray, lat0: np.ndarray, lon1: np.ndarray, lat1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dlon = _wrap_dlon(lon0, lon1)
    mean_lat = np.deg2rad((lat0 + lat1) * 0.5)
    dx = np.deg2rad(dlon) * np.cos(mean_lat) * EARTH_RADIUS_KM
    dy = np.deg2rad(lat1 - lat0) * EARTH_RADIUS_KM
    return np.hypot(dx, dy), dlon


def _pack_cells(lon: np.ndarray, lat: np.ndarray, grid_spec: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    lon = ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lat = np.asarray(lat, dtype=np.float64)
    inside = (
        (lat >= grid_spec.lat_min)
        & (lat < grid_spec.lat_max)
        & (lon >= grid_spec.lon_min)
        & (lon < grid_spec.lon_max)
    )
    packed = np.zeros(lat.shape, dtype=np.int64)
    if np.any(inside):
        lat_bin = np.floor((lat[inside] - grid_spec.lat_min) / grid_spec.resolution_deg).astype(np.int64)
        lon_bin = np.floor((lon[inside] - grid_spec.lon_min) / grid_spec.resolution_deg).astype(np.int64)
        packed[inside] = lat_bin * grid_spec.n_lon + lon_bin
    return packed, inside


def _add_segment_batch(
    lengths: np.ndarray,
    lon0: np.ndarray,
    lat0: np.ndarray,
    lon1: np.ndarray,
    lat1: np.ndarray,
    grid_spec: GridSpec,
    step_deg: float,
) -> None:
    finite = np.isfinite(lon0) & np.isfinite(lat0) & np.isfinite(lon1) & np.isfinite(lat1)
    if not np.any(finite):
        return
    lon0, lat0, lon1, lat1 = lon0[finite], lat0[finite], lon1[finite], lat1[finite]
    km, dlon = _segment_km(lon0, lat0, lon1, lat1)
    chord = np.hypot(dlon, lat1 - lat0)
    keep = (km > 0.0) & np.isfinite(km)
    if not np.any(keep):
        return
    lon0, lat0, lon1, lat1 = lon0[keep], lat0[keep], lon1[keep], lat1[keep]
    km, dlon, chord = km[keep], dlon[keep], chord[keep]

    short = chord <= step_deg
    if np.any(short):
        mid_lon = lon0[short] + dlon[short] / 2.0
        mid_lat = (lat0[short] + lat1[short]) / 2.0
        packed, inside = _pack_cells(mid_lon, mid_lat, grid_spec)
        if np.any(inside):
            np.add.at(lengths, packed[inside], km[short][inside])

    if not np.any(~short):
        return
    for lo0, la0, lo1, la1, length, delta_lon, span in zip(
        lon0[~short], lat0[~short], lon1[~short], lat1[~short], km[~short], dlon[~short], chord[~short]
    ):
        n_steps = int(np.ceil(span / step_deg))
        t = (np.arange(n_steps) + 0.5) / n_steps
        packed, inside = _pack_cells(lo0 + delta_lon * t, la0 + (la1 - la0) * t, grid_spec)
        if np.any(inside):
            np.add.at(lengths, packed[inside], length / n_steps)


def accumulate_road_lines(
    lines: list[tuple[np.ndarray, np.ndarray]],
    grid_spec: GridSpec,
    step_deg: float | None = None,
) -> pd.DataFrame:
    """Sum road length, in km, into each 0.5 degree cell."""
    if step_deg is None:
        step_deg = grid_spec.resolution_deg / 5.0
    if step_deg <= 0:
        raise ValueError("step_deg must be positive")
    lengths = np.zeros(grid_spec.n_cells, dtype=np.float64)
    for lon, lat in lines:
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        if lon.size < 2 or lat.size < 2 or lon.size != lat.size:
            continue
        _add_segment_batch(lengths, lon[:-1], lat[:-1], lon[1:], lat[1:], grid_spec, step_deg)
    return lengths_to_rows(lengths, grid_spec)


def lengths_to_rows(lengths: np.ndarray, grid_spec: GridSpec) -> pd.DataFrame:
    occupied = np.flatnonzero(lengths > 0.0)
    if occupied.size == 0:
        raise ValueError("Road lines did not land in any grid cell")
    lat_bin = occupied // grid_spec.n_lon
    lon_bin = occupied % grid_spec.n_lon
    lat_center = grid_spec.lat_min + (lat_bin + 0.5) * grid_spec.resolution_deg
    area = approximate_cell_area_km2(lat_center, grid_spec.resolution_deg)
    density = lengths[occupied] / area
    return (
        pd.DataFrame(
            {
                "cell_id": [f"{int(lat):03d}_{int(lon):03d}" for lat, lon in zip(lat_bin, lon_bin)],
                "roads_present": 1,
                "roads_length_km": lengths[occupied],
                "roads_density_km_per_km2": density,
                "roads_log_density": np.log1p(density),
            }
        )
        .sort_values("cell_id")
        .reset_index(drop=True)
    )


def _line_parts(geometry: dict | None):
    if not geometry:
        return
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geom_type == "LineString" and coordinates:
        yield coordinates
    elif geom_type == "MultiLineString":
        for part in coordinates or []:
            if part:
                yield part
    elif geom_type == "GeometryCollection":
        for part in geometry.get("geometries") or []:
            yield from _line_parts(part)


def _xy_from_coordinates(coordinates) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return None
    return points[:, 0], points[:, 1]


def read_road_lines(gdb_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    try:
        import fiona
    except ImportError as exc:
        raise ImportError("Reading gROADS requires fiona. Install it with `pip install fiona`.") from exc

    layers = list(fiona.listlayers(str(gdb_path)))
    if not layers:
        raise ValueError(f"No layers in {gdb_path}")
    LOGGER.info("gROADS layers: %s", ", ".join(layers))
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for layer in layers:
        with fiona.open(str(gdb_path), layer=layer) as source:
            geometry_name = str(source.schema.get("geometry") or "")
            if geometry_name and "line" not in geometry_name.lower() and "unknown" not in geometry_name.lower():
                LOGGER.info("Skipping layer %s (%s)", layer, geometry_name)
                continue
            LOGGER.info("Reading %s features from %s", f"{len(source):,}", layer)
            for index, feature in enumerate(source, start=1):
                for part in _line_parts(feature.get("geometry")):
                    xy = _xy_from_coordinates(part)
                    if xy is not None:
                        lines.append(xy)
                if index % 200_000 == 0:
                    LOGGER.info("Read %s features from %s", f"{index:,}", layer)
    if not lines:
        raise ValueError(f"No road lines in {gdb_path}")
    LOGGER.info("Loaded %s road lines", f"{len(lines):,}")
    return lines


def find_groads_input(input_path: str | Path) -> Path:
    folder = Path(input_path)
    if not folder.exists():
        raise FileNotFoundError(
            f"gROADS folder does not exist: {folder}. "
            "Download it with `python -m wildfire_risk.download_external groads --region global`."
        )
    if folder.is_file():
        return folder
    gdbs = sorted(folder.rglob("*.gdb"))
    zips = sorted(folder.glob("*.zip"))
    candidates = gdbs + zips
    if not candidates:
        raise ValueError(f"No gROADS .gdb or .zip files in {folder}")
    global_gdb = [path for path in candidates if "global" in path.name.lower() and "gdb" in path.name.lower()]
    return global_gdb[0] if global_gdb else candidates[0]


def extract_gdb(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        roots = []
        for name in archive.namelist():
            root = name.split("/")[0]
            if root.endswith(".gdb") and root not in roots:
                roots.append(root)
        if not roots:
            raise ValueError(f"No .gdb directory inside {zip_path}")
        gdb_name = next((name for name in roots if "global" in name.lower()), roots[0])
        dest_root = Path("/content") if Path("/content").is_dir() else zip_path.parent
        dest = dest_root / gdb_name
        if dest.is_dir() and any(dest.glob("*.gdbtable")):
            LOGGER.info("Using extracted geodatabase %s", dest)
            return dest
        LOGGER.info("Extracting %s to %s", zip_path.name, dest)
        archive.extractall(dest_root)
    if not dest.is_dir():
        raise FileNotFoundError(f"Expected geodatabase at {dest} after extracting {zip_path}")
    return dest


def resolve_gdb(input_path: str | Path) -> Path:
    found = find_groads_input(input_path)
    if found.is_dir() and found.suffix.lower() == ".gdb":
        return found
    if found.suffix.lower() == ".zip":
        return extract_gdb(found)
    if found.is_dir():
        gdbs = sorted(found.rglob("*.gdb"))
        if gdbs:
            return gdbs[0]
    raise ValueError(f"Could not find a gROADS geodatabase in {found}")


def build_groads_table(
    input_path: str | Path,
    output_path: str | Path | None = None,
    resolution_deg: float = 0.5,
    run_root: str | Path | None = None,
    run_name: str | None = None,
    output_name: str = "groads_roads.csv",
    step_deg: float | None = None,
) -> pd.DataFrame:
    gdb_path = resolve_gdb(input_path)
    LOGGER.info("Reading gROADS geodatabase %s", gdb_path)
    lines = read_road_lines(gdb_path)
    grid_spec = GridSpec(resolution_deg=resolution_deg)
    roads = accumulate_road_lines(lines, grid_spec, step_deg=step_deg)

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_path = layout.features_dir / output_name
    elif output_path is None:
        raise ValueError("Either output_path or run_root must be provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    roads.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s rows to %s", f"{len(roads):,}", output_path)
    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="convert_groads",
            config={"input_path": str(gdb_path), "resolution_deg": resolution_deg, "step_deg": step_deg},
            artifacts={"groads_roads": str(output_path)},
            summary={"row_count": int(len(roads)), "n_cells": int(roads["cell_id"].nunique())},
        )
    return roads


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate gROADS road length onto the wildfire 0.5 degree grid")
    parser.add_argument("--input", required=True, help="gROADS zip, .gdb directory, or a folder containing one")
    parser.add_argument("--output", help="Where to write groads_roads.csv")
    parser.add_argument("--resolution-deg", type=float, default=0.5)
    parser.add_argument("--step-deg", type=float, help="Sampling step for lines that cross a grid cell")
    parser.add_argument("--run-root")
    parser.add_argument("--run-name")
    parser.add_argument("--output-name", default="groads_roads.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    table = build_groads_table(
        input_path=args.input,
        output_path=args.output,
        resolution_deg=args.resolution_deg,
        run_root=args.run_root,
        run_name=args.run_name,
        output_name=args.output_name,
        step_deg=args.step_deg,
    )
    print(f"Built gROADS table with {len(table):,} cells")


if __name__ == "__main__":
    main()
