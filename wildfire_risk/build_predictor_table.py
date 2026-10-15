from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .grid import GridSpec, assign_points_to_grid
    from .run_artifacts import create_run_layout, write_stage_manifest
except ImportError:  # pragma: no cover - allows direct script execution
    from grid import GridSpec, assign_points_to_grid
    from run_artifacts import create_run_layout, write_stage_manifest


LOGGER = logging.getLogger(__name__)

# ERA5-Land monthly means ship short variable names, accumulations in metres per day, and
# a downward-flux sign convention where evaporation is negative. Each entry maps a source
# variable onto the predictor name plus the multiplier that puts it in wildfire-friendly units.
ERA5_LAND_VARIABLES: dict[str, tuple[str, float]] = {
    "t2m": ("temperature_2m", 1.0),  # kelvin
    "tp": ("precipitation", 1000.0),  # m/day -> mm/day
    "swvl1": ("soil_moisture", 1.0),  # m3/m3
    "pev": ("potential_evaporation", -1000.0),  # m/day downward -> mm/day of evaporative demand
    "u10": ("wind_u10", 1.0),  # m/s
    "v10": ("wind_v10", 1.0),  # m/s
}


def infer_column(candidates: Iterable[str], available: Iterable[str], label: str) -> str:
    available_set = set(available)
    for candidate in candidates:
        if candidate in available_set:
            return candidate
    raise ValueError(f"Could not infer `{label}`. Tried: {list(candidates)}")


def parse_rename_arguments(raw_values: list[str] | None) -> dict[str, str]:
    if not raw_values:
        return {}
    mapping: dict[str, str] = {}
    for raw_value in raw_values:
        source, target = raw_value.split(":", maxsplit=1)
        mapping[source] = target
    return mapping


def load_source_frame(
    input_path: str | Path,
    variables: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    input_path = Path(input_path)
    rename_map = rename_map or {}

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        try:
            import xarray as xr
        except ImportError as exc:  # pragma: no cover - import checked at runtime in Colab
            raise RuntimeError(
                "xarray is required to read NetCDF predictor files. Install `xarray h5netcdf netcdf4` first."
            ) from exc

        ds = xr.open_dataset(input_path)
        time_name = infer_column(("time", "valid_time", "month"), ds.coords, "time coordinate")
        lat_name = infer_column(("lat", "latitude", "y"), ds.coords, "latitude coordinate")
        lon_name = infer_column(("lon", "longitude", "x"), ds.coords, "longitude coordinate")
        selected_variables = variables or list(ds.data_vars)
        missing = sorted(set(selected_variables) - set(ds.data_vars))
        if missing:
            raise ValueError(f"Variables not found in {input_path}: {missing}")
        df = ds[selected_variables].to_dataframe().reset_index()
        df = df.rename(columns={time_name: "month", lat_name: "lat", lon_name: "lon"})

    df = df.rename(columns=rename_map)

    month_column = infer_column(("month", "time", "date"), df.columns, "month column")
    lat_column = infer_column(("lat", "latitude"), df.columns, "latitude column")
    lon_column = infer_column(("lon", "longitude"), df.columns, "longitude column")

    variable_columns = variables or [
        column
        for column in df.columns
        if column not in {month_column, lat_column, lon_column}
    ]
    missing_variables = sorted(set(variable_columns) - set(df.columns))
    if missing_variables:
        raise ValueError(f"Variables not found in source frame: {missing_variables}")

    out = df[[month_column, lat_column, lon_column, *variable_columns]].copy()
    out = out.rename(columns={month_column: "month", lat_column: "lat", lon_column: "lon"})
    out["month"] = pd.to_datetime(out["month"]).dt.to_period("M").astype(str)
    return out.dropna(subset=["month", "lat", "lon"])


def aggregate_predictor_frame(
    source_df: pd.DataFrame,
    variables: list[str],
    grid_spec: GridSpec,
) -> pd.DataFrame:
    assigned = assign_points_to_grid(source_df, spec=grid_spec, lat_col="lat", lon_col="lon")
    if assigned.empty:
        raise ValueError("No source records fell inside the target grid")

    aggregated = (
        assigned.groupby(["cell_id", "month"], as_index=False)[variables]
        .mean()
        .sort_values(["month", "cell_id"])
        .reset_index(drop=True)
    )
    return aggregated


def _axis_grid_index(
    values: np.ndarray,
    axis_min: float,
    axis_max: float,
    resolution_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    inside = (values >= axis_min) & (values < axis_max)
    index = np.floor((values - axis_min) / resolution_deg).astype(np.int64)
    return index, inside


def aggregate_era5_land_netcdf(
    input_path: str | Path,
    grid_spec: GridSpec,
    variables: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
    start_month: str | None = None,
    end_month: str | None = None,
    log_every: int = 12,
) -> pd.DataFrame:
    """Aggregate a global ERA5-Land monthly-means NetCDF onto the wildfire grid.

    A global 0.1 degree file holds hundreds of millions of cell-months, so this reads one
    timestep at a time and reduces it with bincount instead of materialising a dataframe.
    """
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - import checked at runtime
        raise RuntimeError(
            "xarray is required to read ERA5-Land NetCDF files. Install `xarray h5netcdf netcdf4` first."
        ) from exc

    dataset = xr.open_dataset(input_path)
    try:
        time_name = infer_column(("valid_time", "time", "month"), dataset.coords, "time coordinate")
        lat_name = infer_column(("latitude", "lat", "y"), dataset.coords, "latitude coordinate")
        lon_name = infer_column(("longitude", "lon", "x"), dataset.coords, "longitude coordinate")

        gridded_variables = [
            name
            for name, array in dataset.data_vars.items()
            if {lat_name, lon_name}.issubset(set(array.dims))
        ]
        selected = list(variables) if variables else gridded_variables
        unknown = sorted(set(selected) - set(gridded_variables))
        if unknown:
            raise ValueError(
                f"Variables are missing or not gridded in {input_path}: {unknown}. "
                f"Available gridded variables: {sorted(gridded_variables)}"
            )

        name_map = {source: target for source, (target, _) in ERA5_LAND_VARIABLES.items()}
        scale_map = {source: scale for source, (_, scale) in ERA5_LAND_VARIABLES.items()}
        name_map.update(rename_map or {})

        latitudes = np.asarray(dataset[lat_name].values, dtype=float)
        longitudes = np.asarray(dataset[lon_name].values, dtype=float)
        # CDS delivers ERA5-Land on a 0 to 360 longitude axis; the wildfire grid spans -180 to 180.
        longitudes = np.where(longitudes > 180.0, longitudes - 360.0, longitudes)

        lat_index, lat_inside = _axis_grid_index(
            latitudes, grid_spec.lat_min, grid_spec.lat_max, grid_spec.resolution_deg
        )
        lon_index, lon_inside = _axis_grid_index(
            longitudes, grid_spec.lon_min, grid_spec.lon_max, grid_spec.resolution_deg
        )
        flat_cell_index = lat_index[:, None] * grid_spec.n_lon + lon_index[None, :]
        inside_grid = lat_inside[:, None] & lon_inside[None, :]
        n_cells = grid_spec.n_cells

        timestamps = pd.to_datetime(dataset[time_name].values)
        months = [stamp.to_period("M").strftime("%Y-%m") for stamp in timestamps]
        wanted = [
            (position, month)
            for position, month in enumerate(months)
            if (start_month is None or month >= start_month)
            and (end_month is None or month <= end_month)
        ]
        if not wanted:
            raise ValueError(
                f"No ERA5-Land timesteps between {start_month} and {end_month} in {input_path}"
            )

        LOGGER.info(
            "Aggregating %s ERA5-Land months and %s variables onto a %.2f degree grid",
            len(wanted),
            len(selected),
            grid_spec.resolution_deg,
        )

        started_at = perf_counter()
        monthly_frames: list[pd.DataFrame] = []
        for step, (position, month) in enumerate(wanted, start=1):
            planes: dict[str, np.ndarray] = {}
            for source_name in selected:
                plane = np.asarray(
                    dataset[source_name].isel({time_name: position}).values, dtype=np.float64
                )
                planes[source_name] = plane * scale_map.get(source_name, 1.0)

            if {"u10", "v10"}.issubset(planes):
                planes["wind_speed"] = np.hypot(planes["u10"], planes["v10"])

            cell_totals: dict[str, np.ndarray] = {}
            cell_counts: dict[str, np.ndarray] = {}
            for source_name, plane in planes.items():
                # ERA5-Land masks ocean as NaN, so each variable needs its own valid-pixel count.
                valid = inside_grid & np.isfinite(plane)
                flat = flat_cell_index[valid]
                cell_totals[source_name] = np.bincount(
                    flat, weights=plane[valid], minlength=n_cells
                )
                cell_counts[source_name] = np.bincount(flat, minlength=n_cells)

            observed_cells = np.flatnonzero(
                np.add.reduce([counts for counts in cell_counts.values()]) > 0
            )
            if observed_cells.size == 0:
                continue

            frame = pd.DataFrame(
                {
                    "cell_id": [
                        f"{cell // grid_spec.n_lon:03d}_{cell % grid_spec.n_lon:03d}"
                        for cell in observed_cells.tolist()
                    ],
                    "month": month,
                }
            )
            for source_name in planes:
                counts = cell_counts[source_name][observed_cells]
                totals = cell_totals[source_name][observed_cells]
                means = np.divide(
                    totals, counts, out=np.full(totals.shape, np.nan), where=counts > 0
                )
                frame[name_map.get(source_name, source_name)] = means
            monthly_frames.append(frame)

            if step == 1 or step == len(wanted) or step % max(log_every, 1) == 0:
                elapsed_minutes = (perf_counter() - started_at) / 60.0
                LOGGER.info(
                    "Aggregated %s (%s/%s) with %s active cells | elapsed %.1fm",
                    month,
                    step,
                    len(wanted),
                    observed_cells.size,
                    elapsed_minutes,
                )

        if not monthly_frames:
            raise ValueError("No ERA5-Land pixels fell inside the target grid")

        aggregated = (
            pd.concat(monthly_frames, ignore_index=True)
            .sort_values(["month", "cell_id"])
            .reset_index(drop=True)
        )
        LOGGER.info(
            "Finished ERA5-Land aggregation in %.1fm with %s cell-months",
            (perf_counter() - started_at) / 60.0,
            len(aggregated),
        )
        return aggregated
    finally:
        dataset.close()


def add_common_wildfire_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"temperature_2m", "soil_moisture"}.issubset(out.columns):
        out["hot_dry_factor"] = out["temperature_2m"] / out["soil_moisture"].clip(lower=1e-6)
    if {"precipitation", "potential_evaporation"}.issubset(out.columns):
        out["climate_water_balance"] = out["precipitation"] - out["potential_evaporation"]
    if {"ndvi", "evi"}.issubset(out.columns):
        out["vegetation_stress_proxy"] = out["ndvi"] - out["evi"]
    return out


def build_predictor_table(
    input_path: str | Path,
    output_path: str | Path | None = None,
    variables: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
    resolution_deg: float = 0.5,
    derive_wildfire_features: bool = True,
    run_root: str | Path | None = None,
    run_name: str | None = None,
    output_name: str = "predictor_table.csv",
    input_format: str = "auto",
    start_month: str | None = None,
    end_month: str | None = None,
) -> pd.DataFrame:
    if input_format not in {"auto", "tabular", "era5-land-netcdf"}:
        raise ValueError("input_format must be one of `auto`, `tabular`, or `era5-land-netcdf`")

    grid_spec = GridSpec(resolution_deg=resolution_deg)

    if input_format == "era5-land-netcdf":
        predictors = aggregate_era5_land_netcdf(
            input_path=input_path,
            grid_spec=grid_spec,
            variables=variables,
            rename_map=rename_map,
            start_month=start_month,
            end_month=end_month,
        )
    else:
        source_df = load_source_frame(
            input_path=input_path, variables=variables, rename_map=rename_map
        )
        resolved_variables = variables or [
            column for column in source_df.columns if column not in {"month", "lat", "lon"}
        ]
        predictors = aggregate_predictor_frame(
            source_df=source_df,
            variables=resolved_variables,
            grid_spec=grid_spec,
        )
        if start_month is not None:
            predictors = predictors[predictors["month"] >= start_month]
        if end_month is not None:
            predictors = predictors[predictors["month"] <= end_month]
        predictors = predictors.reset_index(drop=True)

    variables = [column for column in predictors.columns if column not in {"cell_id", "month"}]
    if derive_wildfire_features:
        predictors = add_common_wildfire_derivatives(predictors)

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_path = layout.features_dir / output_name
    elif output_path is None:
        raise ValueError("Either output_path or run_root must be provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictors.to_csv(output_path, index=False)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="build_predictor_table",
            config={
                "input_path": str(input_path),
                "variables": variables,
                "rename_map": rename_map or {},
                "resolution_deg": resolution_deg,
                "derive_wildfire_features": derive_wildfire_features,
                "output_name": output_name,
                "input_format": input_format,
                "start_month": start_month,
                "end_month": end_month,
            },
            artifacts={"predictor_table": str(output_path)},
            summary={"row_count": int(len(predictors)), "column_count": int(predictors.shape[1])},
        )
    return predictors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a monthly wildfire predictor table from gridded CSV or NetCDF data")
    parser.add_argument("--input", required=True, help="Path to a CSV or NetCDF source file")
    parser.add_argument("--output", help="Where to write the predictor table")
    parser.add_argument(
        "--variable",
        action="append",
        help="Variable to extract. Repeat for multiple variables. Defaults to all data variables.",
    )
    parser.add_argument(
        "--rename",
        action="append",
        help="Optional source-to-output rename in the form original:new_name",
    )
    parser.add_argument("--resolution-deg", type=float, default=0.5, help="Target wildfire grid resolution in degrees")
    parser.add_argument(
        "--disable-derivatives",
        action="store_true",
        help="Do not compute common wildfire derivative predictors like hot_dry_factor",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "tabular", "era5-land-netcdf"),
        default="auto",
        help=(
            "How to read --input. `tabular` loads the whole file into memory and suits small CSV or "
            "NetCDF extracts. `era5-land-netcdf` streams a global ERA5-Land monthly-means file one "
            "timestep at a time and applies ERA5 naming, units, and longitude conventions."
        ),
    )
    parser.add_argument("--start-month", help="Optional inclusive first month, as YYYY-MM")
    parser.add_argument("--end-month", help="Optional inclusive last month, as YYYY-MM")
    parser.add_argument("--run-root", help="Optional root directory for a Colab-friendly run layout")
    parser.add_argument("--run-name", help="Optional run folder name inside --run-root")
    parser.add_argument("--output-name", default="predictor_table.csv", help="Feature filename when using --run-root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    predictors = build_predictor_table(
        input_path=args.input,
        output_path=args.output,
        variables=args.variable,
        rename_map=parse_rename_arguments(args.rename),
        resolution_deg=args.resolution_deg,
        derive_wildfire_features=not args.disable_derivatives,
        run_root=args.run_root,
        run_name=args.run_name,
        output_name=args.output_name,
        input_format=args.input_format,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    print(f"Built predictor table with {len(predictors):,} rows and {predictors.shape[1]} columns")


if __name__ == "__main__":
    main()
