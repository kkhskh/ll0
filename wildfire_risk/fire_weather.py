"""Daily fire-weather rolled up to one row per cell-month.

The index is the Canadian Fire Weather Index, driven by daily temperature, humidity,
wind, and precipitation. Day length follows the northern-hemisphere monthly factors,
shifted six months south of the equator, and blended toward the annual mean inside
the tropics. Hot-dry days are the count of days at or above 30 C, at or below 40%
relative humidity, and under 1 mm of rain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Van Wagner 1987 monthly day-length factors for the Duff Moisture Code and Drought Code.
_DMC_DAY_LENGTH = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0])
_DC_DAY_LENGTH = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])

HOT_DRY_TEMP_C = 30.0
HOT_DRY_RH = 40.0
HOT_DRY_PRECIP_MM = 1.0
FFMC_START = 85.0
DMC_START = 6.0
DC_START = 15.0


def relative_humidity(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    """Relative humidity in percent from the Magnus formula."""
    temp_c = np.asarray(temp_c, dtype=float)
    dewpoint_c = np.asarray(dewpoint_c, dtype=float)
    vapour = np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    saturation = np.exp((17.625 * temp_c) / (243.04 + temp_c))
    return np.clip(100.0 * vapour / saturation, 0.0, 100.0)


def seasonal_day_length(lat: np.ndarray, month: int, northern: np.ndarray) -> np.ndarray:
    """Monthly day-length factor. Southern summer uses the northern July curve."""
    lat = np.asarray(lat, dtype=float)
    month_index = np.where(lat >= 0.0, month - 1, (month - 1 + 6) % 12)
    seasonal = northern[month_index.astype(int)]
    weight = np.clip((np.abs(lat) - 15.0) / 15.0, 0.0, 1.0)
    return (1.0 - weight) * float(northern.mean()) + weight * seasonal


def _ffmc_next(
    ffmc_prev: np.ndarray,
    temp_c: np.ndarray,
    rh: np.ndarray,
    wind_kmh: np.ndarray,
    precip_mm: np.ndarray,
) -> np.ndarray:
    temp_c = np.asarray(temp_c, dtype=float)
    rh = np.clip(np.asarray(rh, dtype=float), 0.0, 100.0)
    wind_kmh = np.clip(np.asarray(wind_kmh, dtype=float), 0.0, None)
    precip_mm = np.clip(np.asarray(precip_mm, dtype=float), 0.0, None)
    ffmc_prev = np.clip(np.asarray(ffmc_prev, dtype=float), 0.0, 101.0)

    moisture = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)
    rain = np.maximum(precip_mm - 0.5, 0.0)
    wet = precip_mm > 0.5
    rain_term = np.exp(-100.0 / (251.0 - moisture)) * (1.0 - np.exp(-6.93 / np.maximum(rain, 1e-6)))
    wetted = moisture + 42.5 * rain * rain_term
    heavy = wet & (moisture > 150.0)
    wetted = np.where(
        heavy,
        wetted + 0.0015 * (moisture - 150.0) ** 2 * np.sqrt(rain),
        wetted,
    )
    wetted = np.minimum(wetted, 250.0)
    moisture = np.where(wet, wetted, moisture)

    ed = (
        0.942 * rh**0.679
        + 11.0 * np.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - temp_c) * (1.0 - np.exp(-0.115 * rh))
    )
    ew = (
        0.618 * rh**0.753
        + 10.0 * np.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - temp_c) * (1.0 - np.exp(-0.115 * rh))
    )
    drying = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (1.0 - (rh / 100.0) ** 8)
    drying = drying * 0.581 * np.exp(0.0365 * temp_c)
    wetting = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7)
    wetting = wetting + 0.0694 * np.sqrt(wind_kmh) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
    wetting = wetting * 0.581 * np.exp(0.0365 * temp_c)
    dried = ed + (moisture - ed) * np.power(10.0, -np.maximum(drying, 0.0))
    moistened = ew - (ew - moisture) * np.power(10.0, -np.maximum(wetting, 0.0))
    moisture = np.where(moisture > ed, dried, np.where(moisture < ew, moistened, moisture))
    ffmc = 59.5 * (250.0 - moisture) / (147.2 + moisture)
    return np.clip(ffmc, 0.0, 101.0)


def _dmc_next(
    dmc_prev: np.ndarray,
    temp_c: np.ndarray,
    rh: np.ndarray,
    precip_mm: np.ndarray,
    day_length: np.ndarray,
) -> np.ndarray:
    temp_c = np.maximum(np.asarray(temp_c, dtype=float), -1.1)
    rh = np.clip(np.asarray(rh, dtype=float), 0.0, 100.0)
    precip_mm = np.clip(np.asarray(precip_mm, dtype=float), 0.0, None)
    dmc_prev = np.maximum(np.asarray(dmc_prev, dtype=float), 0.0)
    rk = 1.894 * (temp_c + 1.1) * (100.0 - rh) * day_length * 1e-6

    rain = np.maximum(0.92 * precip_mm - 1.27, 0.0)
    moisture = 20.0 + np.exp(5.6348 - dmc_prev / 43.43)
    log_dmc = np.log(np.maximum(dmc_prev, 1e-6))
    slope = np.where(
        dmc_prev <= 33.0,
        100.0 / (0.5 + 0.3 * np.maximum(dmc_prev, 0.0)),
        np.where(dmc_prev <= 65.0, 14.0 - 1.3 * log_dmc, 6.2 * log_dmc - 17.2),
    )
    wetted = moisture + 1000.0 * rain / (48.77 + slope * rain)
    after_rain = 244.72 - 43.43 * np.log(np.maximum(wetted - 20.0, 1e-6))
    after_rain = np.maximum(after_rain, 0.0)
    carried = np.where(precip_mm > 1.5, after_rain, dmc_prev)
    return np.maximum(carried + 100.0 * rk, 0.0)


def _dc_next(
    dc_prev: np.ndarray,
    temp_c: np.ndarray,
    precip_mm: np.ndarray,
    day_length: np.ndarray,
) -> np.ndarray:
    temp_c = np.maximum(np.asarray(temp_c, dtype=float), -2.8)
    precip_mm = np.clip(np.asarray(precip_mm, dtype=float), 0.0, None)
    dc_prev = np.maximum(np.asarray(dc_prev, dtype=float), 0.0)
    evap = np.maximum((0.36 * (temp_c + 2.8) + day_length) / 2.0, 0.0)
    rain = np.maximum(0.83 * precip_mm - 1.27, 0.0)
    stored = 800.0 * np.exp(-dc_prev / 400.0)
    wetted = stored + 3.937 * rain
    after_rain = 400.0 * np.log(800.0 / np.maximum(wetted, 1e-6))
    after_rain = np.maximum(after_rain, 0.0)
    carried = np.where(precip_mm > 2.8, after_rain, dc_prev)
    return np.maximum(carried + evap, 0.0)


def _isi(ffmc: np.ndarray, wind_kmh: np.ndarray) -> np.ndarray:
    moisture = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    spread = 19.115 * np.exp(-0.1386 * moisture) * (1.0 + moisture**5.31 / 4.93e7)
    return spread * np.exp(0.05039 * np.clip(wind_kmh, 0.0, None))


def _bui(dmc: np.ndarray, dc: np.ndarray) -> np.ndarray:
    dmc = np.maximum(dmc, 0.0)
    dc = np.maximum(dc, 0.0)
    light = 0.8 * dmc * dc / np.maximum(dmc + 0.4 * dc, 1e-6)
    heavy = dmc - (1.0 - 0.8 * dc / np.maximum(dmc + 0.4 * dc, 1e-6)) * (0.92 + (0.0114 * dmc) ** 1.7)
    bui = np.where((dmc == 0.0) & (dc == 0.0), 0.0, np.where(dmc <= 0.4 * dc, light, heavy))
    return np.maximum(bui, 0.0)


def _fwi_from_isi_bui(isi: np.ndarray, bui: np.ndarray) -> np.ndarray:
    mid = 0.1 * isi * (0.626 * np.power(np.maximum(bui, 0.0), 0.809) + 2.0)
    high = 0.1 * isi * (1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)))
    build = np.where(bui <= 80.0, mid, high)
    scaled = np.array(build, dtype=float, copy=True)
    high = build > 1.0
    if np.any(high):
        scaled[high] = np.exp(2.72 * np.power(0.434 * np.log(build[high]), 0.647))
    return np.where(build <= 1.0, build, scaled)


def fwi_step(
    ffmc_prev: np.ndarray,
    dmc_prev: np.ndarray,
    dc_prev: np.ndarray,
    temp_c: np.ndarray,
    rh: np.ndarray,
    wind_kmh: np.ndarray,
    precip_mm: np.ndarray,
    lat: np.ndarray,
    month: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance one day. Returns the new FFMC, DMC, DC, and FWI."""
    ffmc = _ffmc_next(ffmc_prev, temp_c, rh, wind_kmh, precip_mm)
    dmc = _dmc_next(dmc_prev, temp_c, rh, precip_mm, seasonal_day_length(lat, month, _DMC_DAY_LENGTH))
    dc = _dc_next(dc_prev, temp_c, precip_mm, seasonal_day_length(lat, month, _DC_DAY_LENGTH))
    fwi = _fwi_from_isi_bui(_isi(ffmc, wind_kmh), _bui(dmc, dc))
    return ffmc, dmc, dc, fwi


@dataclass
class FireWeatherState:
    ffmc: np.ndarray
    dmc: np.ndarray
    dc: np.ndarray
    fwi_hist: np.ndarray

    @classmethod
    def start(cls, n_cells: int) -> "FireWeatherState":
        return cls(
            ffmc=np.full(n_cells, FFMC_START, dtype=np.float64),
            dmc=np.full(n_cells, DMC_START, dtype=np.float64),
            dc=np.full(n_cells, DC_START, dtype=np.float64),
            fwi_hist=np.full((6, n_cells), np.nan, dtype=np.float64),
        )


def summarize_month(
    temp_c: np.ndarray,
    rh: np.ndarray,
    wind_kmh: np.ndarray,
    precip_mm: np.ndarray,
    valid: np.ndarray,
    lat: np.ndarray,
    month: int,
    state: FireWeatherState,
    tmax_c: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], FireWeatherState]:
    """Roll a month of daily cell weather into hot-dry days, FWI, and the worst 7-day mean.

    Arrays are shaped (days, cells). ``temp_c`` drives the fire-weather index.
    ``tmax_c`` decides hot-dry days, and falls back to ``temp_c`` when omitted.
    Invalid days leave that cell's codes unchanged. The worst 7-day value uses the
    previous month's last days, carried in the state.
    """
    temp_c = np.asarray(temp_c, dtype=float)
    tmax_c = temp_c if tmax_c is None else np.asarray(tmax_c, dtype=float)
    n_days, n_cells = temp_c.shape
    if state.ffmc.shape != (n_cells,):
        raise ValueError("Fire-weather state does not match the number of cells")

    hot_dry = np.zeros(n_cells, dtype=float)
    valid_days = np.zeros(n_cells, dtype=float)
    fwi_sum = np.zeros(n_cells, dtype=float)
    fwi_max = np.full(n_cells, -np.inf, dtype=float)
    worst_7day = np.full(n_cells, -np.inf, dtype=float)
    ffmc = state.ffmc.copy()
    dmc = state.dmc.copy()
    dc = state.dc.copy()
    history = state.fwi_hist.copy()

    for day in range(n_days):
        day_valid = np.asarray(valid[day], dtype=bool)
        next_ffmc, next_dmc, next_dc, fwi = fwi_step(
            ffmc,
            dmc,
            dc,
            temp_c[day],
            rh[day],
            wind_kmh[day],
            precip_mm[day],
            lat,
            month,
        )
        ffmc = np.where(day_valid, next_ffmc, ffmc)
        dmc = np.where(day_valid, next_dmc, dmc)
        dc = np.where(day_valid, next_dc, dc)
        fwi = np.where(day_valid, fwi, np.nan)
        history = np.roll(history, -1, axis=0)
        history[-1] = fwi
        with np.errstate(invalid="ignore"):
            window = np.nanmean(history, axis=0)
        hot = (
            day_valid
            & (tmax_c[day] >= HOT_DRY_TEMP_C)
            & (rh[day] <= HOT_DRY_RH)
            & (precip_mm[day] < HOT_DRY_PRECIP_MM)
        )
        hot_dry += hot.astype(float)
        valid_days += day_valid.astype(float)
        fwi_sum = np.where(day_valid, fwi_sum + np.nan_to_num(fwi), fwi_sum)
        fwi_max = np.where(day_valid, np.maximum(fwi_max, np.nan_to_num(fwi, nan=-np.inf)), fwi_max)
        worst_7day = np.where(day_valid, np.maximum(worst_7day, np.nan_to_num(window, nan=-np.inf)), worst_7day)

    observed = valid_days > 0
    fwi_mean = np.divide(fwi_sum, valid_days, out=np.full(n_cells, np.nan), where=observed)
    summary = {
        "hot_dry_days": np.where(observed, hot_dry, np.nan),
        "fwi_mean": fwi_mean,
        "fwi_max": np.where(observed, fwi_max, np.nan),
        "worst_7day": np.where(observed, worst_7day, np.nan),
        "observed": observed,
    }
    return summary, FireWeatherState(ffmc=ffmc, dmc=dmc, dc=dc, fwi_hist=history)


def summary_frame(
    summary: dict[str, np.ndarray],
    cell_ids: np.ndarray,
    month: str,
) -> pd.DataFrame:
    observed = np.asarray(summary["observed"], dtype=bool)
    frame = pd.DataFrame(
        {
            "cell_id": np.asarray(cell_ids)[observed],
            "month": month,
            "hot_dry_days": summary["hot_dry_days"][observed],
            "fwi_mean": summary["fwi_mean"][observed],
            "fwi_max": summary["fwi_max"][observed],
            "worst_7day": summary["worst_7day"][observed],
        }
    )
    return frame.sort_values(["month", "cell_id"]).reset_index(drop=True)
