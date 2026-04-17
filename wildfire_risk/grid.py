from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class GridSpec:
    resolution_deg: float = 0.5
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0

    def __post_init__(self) -> None:
        if self.resolution_deg <= 0:
            raise ValueError("resolution_deg must be positive")
        if self.lat_min >= self.lat_max:
            raise ValueError("lat_min must be smaller than lat_max")
        if self.lon_min >= self.lon_max:
            raise ValueError("lon_min must be smaller than lon_max")

    @property
    def n_lat(self) -> int:
        return int(round((self.lat_max - self.lat_min) / self.resolution_deg))

    @property
    def n_lon(self) -> int:
        return int(round((self.lon_max - self.lon_min) / self.resolution_deg))

    @property
    def n_cells(self) -> int:
        return self.n_lat * self.n_lon


@dataclass(frozen=True)
class LabelSpec:
    burned_fraction_threshold: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 <= self.burned_fraction_threshold <= 1.0:
            raise ValueError("burned_fraction_threshold must be between 0 and 1")


def create_month_index(start_month: str = "2004-01", end_month: str = "2024-12") -> pd.DataFrame:
    periods = pd.period_range(start=start_month, end=end_month, freq="M")
    month_starts = periods.to_timestamp("M").to_period("M").to_timestamp()
    return pd.DataFrame(
        {
            "month": periods.astype(str),
            "month_start": month_starts,
            "year": periods.year.astype(int),
            "month_num": periods.month.astype(int),
        }
    )


def create_global_grid(spec: GridSpec) -> pd.DataFrame:
    lat_edges = np.arange(spec.lat_min, spec.lat_max, spec.resolution_deg)
    lon_edges = np.arange(spec.lon_min, spec.lon_max, spec.resolution_deg)
    lat_idx, lon_idx = np.meshgrid(np.arange(len(lat_edges)), np.arange(len(lon_edges)), indexing="ij")

    lat_min = lat_edges[lat_idx.ravel()]
    lon_min = lon_edges[lon_idx.ravel()]
    lat_max = lat_min + spec.resolution_deg
    lon_max = lon_min + spec.resolution_deg
    lat_center = lat_min + spec.resolution_deg / 2.0
    lon_center = lon_min + spec.resolution_deg / 2.0

    grid = pd.DataFrame(
        {
            "lat_idx": lat_idx.ravel().astype(int),
            "lon_idx": lon_idx.ravel().astype(int),
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_center": lat_center,
            "lon_center": lon_center,
        }
    )
    grid["cell_id"] = (
        grid["lat_idx"].astype(int).map(lambda value: f"{value:03d}")
        + "_"
        + grid["lon_idx"].astype(int).map(lambda value: f"{value:03d}")
    )
    grid["cell_area_km2"] = approximate_cell_area_km2(grid["lat_center"], spec.resolution_deg)
    return grid[
        [
            "cell_id",
            "lat_idx",
            "lon_idx",
            "lat_min",
            "lat_max",
            "lon_min",
            "lon_max",
            "lat_center",
            "lon_center",
            "cell_area_km2",
        ]
    ]


def approximate_cell_area_km2(lat_center: Iterable[float] | pd.Series, resolution_deg: float) -> np.ndarray:
    lat_center = np.asarray(list(lat_center) if not isinstance(lat_center, np.ndarray) else lat_center, dtype=float)
    dlat = np.deg2rad(resolution_deg)
    dlon = np.deg2rad(resolution_deg)
    lat1 = np.deg2rad(lat_center - resolution_deg / 2.0)
    lat2 = np.deg2rad(lat_center + resolution_deg / 2.0)
    area = (EARTH_RADIUS_KM ** 2) * dlon * (np.sin(lat2) - np.sin(lat1))
    return np.abs(area)


def normalize_month_column(df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    out = df.copy()
    out[month_col] = pd.to_datetime(out[month_col]).dt.to_period("M").astype(str)
    return out


def assign_points_to_grid(
    df: pd.DataFrame,
    spec: GridSpec,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Input data must include `{lat_col}` and `{lon_col}` columns")

    out = df.copy()
    out = out[out[lat_col].between(spec.lat_min, spec.lat_max, inclusive="left")]
    out = out[out[lon_col].between(spec.lon_min, spec.lon_max, inclusive="left")]

    lat_idx = np.floor((out[lat_col] - spec.lat_min) / spec.resolution_deg).astype(int)
    lon_idx = np.floor((out[lon_col] - spec.lon_min) / spec.resolution_deg).astype(int)
    out["lat_idx"] = lat_idx
    out["lon_idx"] = lon_idx
    out["cell_id"] = [
        f"{lat_value:03d}_{lon_value:03d}"
        for lat_value, lon_value in zip(lat_idx, lon_idx)
    ]
    return out
