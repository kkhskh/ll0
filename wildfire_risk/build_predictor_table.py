from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .grid import GridSpec, assign_points_to_grid
    from .run_artifacts import create_run_layout, write_stage_manifest
except ImportError:  # pragma: no cover - allows direct script execution
    from grid import GridSpec, assign_points_to_grid
    from run_artifacts import create_run_layout, write_stage_manifest


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
) -> pd.DataFrame:
    source_df = load_source_frame(input_path=input_path, variables=variables, rename_map=rename_map)
    variables = variables or [column for column in source_df.columns if column not in {"month", "lat", "lon"}]
    predictors = aggregate_predictor_frame(
        source_df=source_df,
        variables=variables,
        grid_spec=GridSpec(resolution_deg=resolution_deg),
    )
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
    parser.add_argument("--run-root", help="Optional root directory for a Colab-friendly run layout")
    parser.add_argument("--run-name", help="Optional run folder name inside --run-root")
    parser.add_argument("--output-name", default="predictor_table.csv", help="Feature filename when using --run-root")
    args = parser.parse_args()

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
    )
    print(f"Built predictor table with {len(predictors):,} rows and {predictors.shape[1]} columns")


if __name__ == "__main__":
    main()
