"""Build a monthly fire-weather table from daily ERA5-Land.

Each month is downloaded, reduced onto the 0.5 degree grid, and deleted. The
Canadian FWI state is kept between months, so a stopped run continues at the
next month. 2019 is included so January 2020 has a real prior year.

The Copernicus daily-statistics dataset does not serve precipitation, so rain
is the hourly ERA5-Land total summed to a day. Temperature, dewpoint, and wind
come from the daily statistics. Accept both dataset licences, and keep the API
key out of the chat:

https://cds.climate.copernicus.eu/datasets/derived-era5-land-daily-statistics
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
"""

from __future__ import annotations

import argparse
import calendar
import logging
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .build_predictor_table import _axis_grid_index, infer_column
    from .fire_weather import FireWeatherState, relative_humidity, summarize_month, summary_frame
    from .grid import GridSpec
except ImportError:  # pragma: no cover - allows direct script execution
    from build_predictor_table import _axis_grid_index, infer_column
    from fire_weather import FireWeatherState, relative_humidity, summarize_month, summary_frame
    from grid import GridSpec


LOGGER = logging.getLogger(__name__)

DAILY_DATASET = "derived-era5-land-daily-statistics"
HOURLY_DATASET = "reanalysis-era5-land"
OUTPUT_COLUMNS = ("cell_id", "month", "hot_dry_days", "fwi_mean", "fwi_max", "worst_7day")


def emit(message: str) -> None:
    print(message, flush=True)
    LOGGER.info(message)


def reduce_plane_to_cells(
    plane: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    grid_spec: GridSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of a lat/lon plane onto the coarse grid. NaN is ignored."""
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    longitudes = np.where(longitudes > 180.0, longitudes - 360.0, longitudes)
    plane = np.asarray(plane, dtype=float)
    lat_index, lat_inside = _axis_grid_index(
        latitudes, grid_spec.lat_min, grid_spec.lat_max, grid_spec.resolution_deg
    )
    lon_index, lon_inside = _axis_grid_index(
        longitudes, grid_spec.lon_min, grid_spec.lon_max, grid_spec.resolution_deg
    )
    flat = lat_index[:, None] * grid_spec.n_lon + lon_index[None, :]
    inside = lat_inside[:, None] & lon_inside[None, :] & np.isfinite(plane)
    totals = np.bincount(flat[inside], weights=plane[inside], minlength=grid_spec.n_cells).astype(float)
    counts = np.bincount(flat[inside], minlength=grid_spec.n_cells).astype(float)
    return totals, counts


def _cell_latitudes(grid_spec: GridSpec) -> np.ndarray:
    lat_idx = np.arange(grid_spec.n_cells) // grid_spec.n_lon
    return grid_spec.lat_min + (lat_idx + 0.5) * grid_spec.resolution_deg


def _cell_ids(indexes: np.ndarray, n_lon: int) -> np.ndarray:
    return np.array([f"{int(index) // n_lon:03d}_{int(index) % n_lon:03d}" for index in indexes], dtype=object)


def _open_dataset(path: Path):
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - Colab installs xarray before the run
        raise RuntimeError("xarray is required. Install `xarray h5netcdf netcdf4` first.") from exc
    return xr.open_dataset(path)


def _coord_names(dataset) -> tuple[str, str, str]:
    time_name = infer_column(("valid_time", "time"), dataset.coords, "time coordinate")
    lat_name = infer_column(("latitude", "lat"), dataset.coords, "latitude coordinate")
    lon_name = infer_column(("longitude", "lon"), dataset.coords, "longitude coordinate")
    return time_name, lat_name, lon_name


def _variable(dataset, candidates: tuple[str, ...]) -> str:
    return infer_column(candidates, dataset.data_vars, "data variable")


def _dates_in_dataset(dataset) -> list[pd.Timestamp]:
    time_name, _, _ = _coord_names(dataset)
    return list(pd.to_datetime(dataset[time_name].values))


def _fill_daily_mean(
    dataset,
    variable: str,
    grid_spec: GridSpec,
    date_index: dict[pd.Timestamp, int],
    means: np.ndarray,
    counts: np.ndarray,
) -> None:
    time_name, lat_name, lon_name = _coord_names(dataset)
    latitudes = np.asarray(dataset[lat_name].values, dtype=float)
    longitudes = np.asarray(dataset[lon_name].values, dtype=float)
    stamps = pd.to_datetime(dataset[time_name].values)
    for position, stamp in enumerate(stamps):
        day = pd.Timestamp(stamp).normalize()
        if day not in date_index:
            continue
        plane = np.asarray(dataset[variable].isel({time_name: position}).values, dtype=float)
        total, count = reduce_plane_to_cells(plane, latitudes, longitudes, grid_spec)
        slot = date_index[day]
        means[slot] = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
        counts[slot] = count


def _fill_daily_precip_mm(
    dataset,
    grid_spec: GridSpec,
    date_index: dict[pd.Timestamp, int],
    precip_mm: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Sum hourly precipitation in metres onto each coarse cell, then convert to mm."""
    time_name, lat_name, lon_name = _coord_names(dataset)
    variable = _variable(dataset, ("tp", "total_precipitation"))
    latitudes = np.asarray(dataset[lat_name].values, dtype=float)
    longitudes = np.asarray(dataset[lon_name].values, dtype=float)
    stamps = pd.to_datetime(dataset[time_name].values)
    for position, stamp in enumerate(stamps):
        day = pd.Timestamp(stamp).normalize()
        if day not in date_index:
            continue
        plane = np.asarray(dataset[variable].isel({time_name: position}).values, dtype=float)
        total, count = reduce_plane_to_cells(plane, latitudes, longitudes, grid_spec)
        slot = date_index[day]
        hourly_mm = np.divide(total * 1000.0, count, out=np.zeros(total.shape, dtype=float), where=count > 0)
        precip_mm[slot] = precip_mm[slot] + hourly_mm
        counts[slot] = np.maximum(counts[slot], count)


def month_arrays_from_files(
    tmax_path: Path,
    tmean_path: Path,
    wind_path: Path,
    precip_path: Path,
    grid_spec: GridSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Return tmax C, mean temp C, RH, wind km/h, precip mm, valid, and the dates."""
    tmax_ds = _open_dataset(tmax_path)
    tmean_ds = _open_dataset(tmean_path)
    wind_ds = _open_dataset(wind_path)
    precip_ds = _open_dataset(precip_path)
    try:
        stamps = [pd.Timestamp(stamp).normalize() for stamp in _dates_in_dataset(tmax_ds)]
        dates = sorted(set(stamps))
        date_index = {day: index for index, day in enumerate(dates)}
        n_days = len(dates)
        shape = (n_days, grid_spec.n_cells)
        tmax = np.full(shape, np.nan)
        tmean = np.full(shape, np.nan)
        dewpoint = np.full(shape, np.nan)
        wind_u = np.full(shape, np.nan)
        wind_v = np.full(shape, np.nan)
        precip = np.zeros(shape)
        temp_count = np.zeros(shape)
        precip_count = np.zeros(shape)
        _fill_daily_mean(tmax_ds, _variable(tmax_ds, ("t2m", "2m_temperature")), grid_spec, date_index, tmax, temp_count)
        _fill_daily_mean(
            tmean_ds,
            _variable(tmean_ds, ("t2m", "2m_temperature")),
            grid_spec,
            date_index,
            tmean,
            np.zeros(shape),
        )
        _fill_daily_mean(
            tmean_ds,
            _variable(tmean_ds, ("d2m", "2m_dewpoint_temperature")),
            grid_spec,
            date_index,
            dewpoint,
            np.zeros(shape),
        )
        _fill_daily_mean(wind_ds, _variable(wind_ds, ("u10", "10m_u_component_of_wind")), grid_spec, date_index, wind_u, np.zeros(shape))
        _fill_daily_mean(wind_ds, _variable(wind_ds, ("v10", "10m_v_component_of_wind")), grid_spec, date_index, wind_v, np.zeros(shape))
        _fill_daily_precip_mm(precip_ds, grid_spec, date_index, precip, precip_count)
    finally:
        tmax_ds.close()
        tmean_ds.close()
        wind_ds.close()
        precip_ds.close()

    valid = temp_count > 0
    with np.errstate(invalid="ignore"):
        tmean_peak = np.nanmax(tmean)
        tmax_peak = np.nanmax(tmax)
    if np.isfinite(tmean_peak) and tmean_peak > 150.0:
        tmean = tmean - 273.15
        dewpoint = dewpoint - 273.15
    if np.isfinite(tmax_peak) and tmax_peak > 150.0:
        tmax = tmax - 273.15
    rh = relative_humidity(tmean, dewpoint)
    wind_kmh = np.hypot(wind_u, wind_v) * 3.6
    return tmax, tmean, rh, wind_kmh, precip, valid, dates


def _load_state(path: Path, n_cells: int) -> tuple[FireWeatherState, str | None]:
    if not path.exists():
        return FireWeatherState.start(n_cells), None
    saved = np.load(path, allow_pickle=False)
    state = FireWeatherState(
        ffmc=saved["ffmc"],
        dmc=saved["dmc"],
        dc=saved["dc"],
        fwi_hist=saved["fwi_hist"],
    )
    last_month = str(saved["last_month"].item())
    if state.ffmc.shape != (n_cells,):
        raise ValueError(f"Saved fire-weather state in {path} has {state.ffmc.size} cells, expected {n_cells}")
    return state, last_month


def _save_state(path: Path, state: FireWeatherState, last_month: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / "fireweather_state_tmp.npz"
    np.savez(
        temporary,
        ffmc=state.ffmc,
        dmc=state.dmc,
        dc=state.dc,
        fwi_hist=state.fwi_hist,
        last_month=np.array(last_month),
    )
    temporary.replace(path)


def _month_span(start_month: str, end_month: str) -> list[str]:
    return [str(period) for period in pd.period_range(start_month, end_month, freq="M")]


def _days(year: int, month: int) -> list[str]:
    count = calendar.monthrange(year, month)[1]
    return [f"{day:02d}" for day in range(1, count + 1)]


def _cds_client():
    try:
        import cdsapi
    except ImportError as exc:  # pragma: no cover - installed in the Colab cell
        raise RuntimeError("cdsapi is required. Install it with `pip install cdsapi`.") from exc
    return cdsapi.Client(timeout=600, quiet=False, retry_max=8)


def _retrieve(client, dataset: str, request: dict, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        emit(f"Already downloaded {target.name}")
        return target
    emit(f"Requesting {target.name}")
    client.retrieve(dataset, request, str(target))
    if zipfile.is_zipfile(target):
        with zipfile.ZipFile(target) as bundle:
            members = [name for name in bundle.namelist() if name.endswith(".nc")]
            if len(members) != 1:
                raise ValueError(f"{target} contained {members}, expected one NetCDF")
            extracted = target.with_suffix(".extracted.nc")
            with bundle.open(members[0]) as source, extracted.open("wb") as sink:
                sink.write(source.read())
        target.unlink()
        extracted.replace(target)
    emit(f"Downloaded {target.name} ({target.stat().st_size / 1e6:.0f} MB)")
    return target


def download_month(client, year: int, month: int, raw_dir: Path) -> dict[str, Path]:
    days = _days(year, month)
    label = f"{year:04d}-{month:02d}"
    tmax = _retrieve(
        client,
        DAILY_DATASET,
        {
            "variable": ["2m_temperature"],
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "day": days,
            "daily_statistic": "daily_maximum",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "data_format": "netcdf",
        },
        raw_dir / f"tmax_{label}.nc",
    )
    tmean = _retrieve(
        client,
        DAILY_DATASET,
        {
            "variable": ["2m_temperature", "2m_dewpoint_temperature"],
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "day": days,
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "data_format": "netcdf",
        },
        raw_dir / f"tmean_{label}.nc",
    )
    wind = _retrieve(
        client,
        DAILY_DATASET,
        {
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "day": days,
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "data_format": "netcdf",
        },
        raw_dir / f"wind_{label}.nc",
    )
    precip = _retrieve(
        client,
        HOURLY_DATASET,
        {
            "variable": ["total_precipitation"],
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "day": days,
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        raw_dir / f"precip_{label}.nc",
    )
    return {"tmax": tmax, "tmean": tmean, "wind": wind, "precip": precip}


def build_fire_weather_table(
    output_path: str | Path,
    work_dir: str | Path,
    start_month: str = "2019-01",
    end_month: str = "2025-12",
    resolution_deg: float = 0.5,
    download: bool = True,
    raw_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download each month, append its fire-weather rows, and keep the FWI state."""
    output_path = Path(output_path)
    work_dir = Path(work_dir)
    shard_dir = work_dir / "months"
    raw_dir = Path(raw_dir) if raw_dir is not None else work_dir / "raw"
    shard_dir.mkdir(parents=True, exist_ok=True)
    grid_spec = GridSpec(resolution_deg=resolution_deg)
    state_path = work_dir / "fireweather_state.npz"
    state, last_month = _load_state(state_path, grid_spec.n_cells)
    latitudes = _cell_latitudes(grid_spec)
    cell_ids = _cell_ids(np.arange(grid_spec.n_cells), grid_spec.n_lon)
    months = _month_span(start_month, end_month)
    client = _cds_client() if download else None

    emit(f"Fire weather {start_month} to {end_month}: {len(months)} months")
    for index, month in enumerate(months, start=1):
        shard_path = shard_dir / f"fireweather_{month}.csv"
        if last_month is not None and month <= last_month and shard_path.exists():
            emit(f"Fire weather {index}/{len(months)} {month} already built")
            continue
        if last_month is not None and month <= last_month and not shard_path.exists():
            raise ValueError(
                f"State is at {last_month} but {shard_path.name} is missing. "
                "Keep the month files next to the state file."
            )
        emit(f"Fire weather {index}/{len(months)} {month}")
        if not download:
            raise ValueError(f"No shard for {month} and downloading is off")
        year, month_num = int(month[:4]), int(month[5:7])
        paths = download_month(client, year, month_num, raw_dir)
        tmax, tmean, rh, wind, precip, valid, _dates = month_arrays_from_files(
            paths["tmax"], paths["tmean"], paths["wind"], paths["precip"], grid_spec
        )
        summary, state = summarize_month(
            tmean, rh, wind, precip, valid, latitudes, month_num, state, tmax_c=tmax
        )
        frame = summary_frame(summary, cell_ids, month)
        frame.to_csv(shard_path, index=False)
        _save_state(state_path, state, month)
        for path in paths.values():
            path.unlink(missing_ok=True)
        emit(
            f"Built {month}: {len(frame):,} cells, "
            f"mean hot-dry days {frame['hot_dry_days'].mean():.2f}, "
            f"mean FWI {frame['fwi_mean'].mean():.2f}"
        )

    frames = [pd.read_csv(shard_dir / f"fireweather_{month}.csv") for month in months]
    table = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    emit(f"Wrote {output_path} with {len(table):,} cell-months")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lagged-ready monthly fire-weather from daily ERA5-Land")
    parser.add_argument("--output", required=True, help="Monthly cell table: hot_dry_days, fwi_mean, fwi_max, worst_7day")
    parser.add_argument("--work-dir", required=True, help="Resume files and raw downloads. Raw files are deleted after each month.")
    parser.add_argument("--start-month", default="2019-01")
    parser.add_argument("--end-month", default="2025-12")
    parser.add_argument("--resolution-deg", type=float, default=0.5)
    parser.add_argument("--raw-dir", help="Large daily files. Deleted after each month. Defaults to WORK_DIR/raw.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if not os.environ.get("CDSAPI_URL") and not Path.home().joinpath(".cdsapirc").exists():
        raise SystemExit("CDS credentials are missing. Write ~/.cdsapirc before this command. Do not paste the key into chat.")
    build_fire_weather_table(
        output_path=args.output,
        work_dir=args.work_dir,
        start_month=args.start_month,
        end_month=args.end_month,
        resolution_deg=args.resolution_deg,
        raw_dir=args.raw_dir,
    )


if __name__ == "__main__":
    main()
