"""Continent-tiled overlay of public power assets on monthly burned-area scores.

v8 scores are a global 0.5° monthly burned-area model. This module streams the
prediction CSV once, splits it by `evaluate.assign_continent` (Africa first in
the processing order; Australia-before-Asia in the box list), and joins public
power assets onto each region's own peak-fire month.

The join uses `grid.assign_points_to_grid`. Lines are sampled every 0.1° and the
cell with the highest `ranking_score` is kept. EPSG:3310 is not used.

Horizon is the calendar month: this month's burned-area probability. This is not
a spread, smoke, or loss model. Missing source files are recorded by URL and the
region continues without invented assets or scores.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse

import numpy as np
import pandas as pd

try:
    from .evaluate import CONTINENT_BOXES, CONTINENT_ORDER, assign_continent
    from .grid import GridSpec, assign_points_to_grid
except ImportError:  # pragma: no cover - allows direct script execution
    from evaluate import CONTINENT_BOXES, CONTINENT_ORDER, assign_continent
    from grid import GridSpec, assign_points_to_grid

LOGGER = logging.getLogger(__name__)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "month",
    "year",
    "month_num",
    "lat_center",
    "lon_center",
    "binary_risk_label",
    "predicted_probability",
    "ranking_score",
    "risk_percentile",
    "month_risk_percentile",
    "risk_band",
)
RANKED_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "asset_name",
    "asset_type",
    "region",
    "cell_id",
    "month",
    "predicted_probability",
    "risk_band",
    "month_risk_percentile",
    "n_high_cells",
    "n_extreme_cells",
    "reason",
    "asset_source",
)
LINE_SAMPLE_STEP_DEG = 0.1
PREDICTION_CHUNKSIZE = 250_000
PROCESS_ORDER: tuple[str, ...] = tuple(name for name in CONTINENT_ORDER if name != "global")

OSM_POWER_TMM_URL = (
    "https://energydata.info/dataset/e0e878a7-7712-465d-a4c2-69b2b3518513/"
    "resource/e0f33c3e-6dab-41d1-919d-d266f300d0bc/download/osm_power_tmm.zip"
)
OSM_POWER_TMM_LICENSE = "CC-BY-4.0 (World Bank Global Transmission Network, OSM 2016 extract)"
OSM_POWER_PARQUET_URL = "https://parquetry.geomermaids.com/gmwid/latest/power_line.parquet"
OSM_POWER_PARQUET_LICENSE = "ODbL 1.0, © OpenStreetMap contributors (Geofabrik extracts via parquetry/gmwid)"
GPPD_URL = (
    "https://raw.githubusercontent.com/wri/global-power-plant-database/master/"
    "output_database/global_power_plant_database.csv"
)
GPPD_LICENSE = "CC-BY-4.0, World Resources Institute Global Power Plant Database"
HIFLD_TRANSMISSION_URL = (
    "https://services2.arcgis.com/LYMgRMwHfrWWEg3s/arcgis/rest/services/"
    "HIFLD_US_Electric_Power_Transmission_Lines/FeatureServer/0"
)
HIFLD_SUBSTATIONS_URL = (
    "https://services5.arcgis.com/HDRa0B57OVrv2E1q/ArcGIS/rest/services/"
    "Electric_Substations/FeatureServer/0"
)
HIFLD_LICENSE = "U.S. public domain / HIFLD Open (Homeland Infrastructure Foundation-Level Data)"
CALIFORNIA_BBOX: tuple[float, float, float, float] = (32.5, 42.1, -124.5, -114.1)
USER_AGENT = "HephaelionWildfireAssetExposure/1.0 (research overlay; DESIGN_HEPHAELION/wildfire_risk)"

_WKT_PAIR = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


@dataclass
class SourceStatus:
    source: str
    url: str
    status: str
    n_assets: int = 0
    license: str = ""
    note: str = ""
    region: str = "global"
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailureCache:
    """Remember HTTP 403 (and similar hard failures) so a missing zip is not retried per region."""

    path: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "FailureCache":
        path = Path(path)
        entries: dict[str, dict[str, Any]] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                entries = loaded
        return cls(path=path, entries=entries)

    def blocked(self, url: str) -> dict[str, Any] | None:
        entry = self.entries.get(url)
        if not entry:
            return None
        if int(entry.get("status", 0)) in {401, 403, 404}:
            return entry
        return None

    def remember(self, url: str, status: int, note: str = "") -> None:
        self.entries[url] = {
            "status": int(status),
            "note": note,
            "cached_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")


def empty_ranked_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in RANKED_COLUMNS})


def require_predictions_file(predictions_path: str | Path) -> Path:
    path = Path(predictions_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def continent_bbox(region: str) -> tuple[float, float, float, float] | None:
    for name, lat_min, lat_max, lon_min, lon_max in CONTINENT_BOXES:
        if name == region:
            return float(lat_min), float(lat_max), float(lon_min), float(lon_max)
    return None


def cell_ids_for_points(
    lats: Iterable[float],
    lons: Iterable[float],
    spec: GridSpec | None = None,
) -> list[str]:
    spec = spec or GridSpec()
    points = pd.DataFrame({"lat": np.asarray(list(lats), dtype=float), "lon": np.asarray(list(lons), dtype=float)})
    if points.empty:
        return []
    assigned = assign_points_to_grid(points, spec, lat_col="lat", lon_col="lon")
    return assigned["cell_id"].astype(str).tolist()


def unique_cell_ids_along_line(
    lons: Iterable[float],
    lats: Iterable[float],
    spec: GridSpec | None = None,
    step_deg: float = LINE_SAMPLE_STEP_DEG,
) -> list[str]:
    """Sample a polyline every `step_deg` degrees and return unique `grid.py` cell ids."""
    spec = spec or GridSpec()
    lon_np = np.asarray(list(lons), dtype=float)
    lat_np = np.asarray(list(lats), dtype=float)
    if lon_np.size == 0 or lat_np.size == 0 or lon_np.size != lat_np.size:
        return []
    sample_lons = [float(lon_np[0])]
    sample_lats = [float(lat_np[0])]
    for lon0, lat0, lon1, lat1 in zip(lon_np[:-1], lat_np[:-1], lon_np[1:], lat_np[1:]):
        dlon = float(lon1 - lon0)
        dlat = float(lat1 - lat0)
        span = float(np.hypot(dlon, dlat))
        n_steps = max(int(np.ceil(span / step_deg)), 1)
        for step in range(1, n_steps + 1):
            t = step / n_steps
            sample_lons.append(lon0 + dlon * t)
            sample_lats.append(lat0 + dlat * t)
    return list(dict.fromkeys(cell_ids_for_points(sample_lats, sample_lons, spec=spec)))


def parse_wkt_line_parts(wkt: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parse LINESTRING / MULTILINESTRING WKT into (lons, lats) parts. No shapely required."""
    if not wkt or not isinstance(wkt, str):
        return []
    text = wkt.strip()
    if text.upper().startswith("MULTILINESTRING"):
        inner = text[text.find("(") :]
        part_texts = re.findall(r"\(([^()]+)\)", inner)
    elif text.upper().startswith("LINESTRING"):
        inner = text[text.find("(") + 1 : text.rfind(")")]
        part_texts = [inner]
    else:
        part_texts = [text]
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    for part_text in part_texts:
        pairs = [(float(lon), float(lat)) for lon, lat in _WKT_PAIR.findall(part_text)]
        if len(pairs) < 2:
            continue
        lons = np.asarray([pair[0] for pair in pairs], dtype=float)
        lats = np.asarray([pair[1] for pair in pairs], dtype=float)
        parts.append((lons, lats))
    return parts


def peak_month_for_region(pred_df: pd.DataFrame, month: str | None = None) -> str:
    """Peak-fire month inside one region. Africa's peak must not overwrite Australia's."""
    if month:
        return str(month)
    if pred_df.empty or "month" not in pred_df.columns:
        raise ValueError("Prediction table needs a month column to choose a peak-fire month")
    working = pred_df.copy()
    working["month"] = pd.to_datetime(working["month"]).dt.to_period("M").astype(str)
    grouped = working.groupby("month", sort=True)
    summary = pd.DataFrame(
        {
            "n_fires": grouped["binary_risk_label"].sum() if "binary_risk_label" in working.columns else 0,
            "mean_score": grouped["ranking_score"].mean() if "ranking_score" in working.columns else 0.0,
            "n_rows": grouped.size(),
        }
    )
    summary = summary.sort_values(["n_fires", "mean_score", "n_rows"], ascending=[False, False, False])
    return str(summary.index[0])


def reason_for_asset_type(asset_type: str) -> str:
    if asset_type == "transmission_line":
        return (
            "Highest ranking_score cell among 0.1-degree samples along this line. "
            "This is this month's burned-area probability, not spread, smoke, or loss."
        )
    return (
        "Asset sits in this cell. This is this month's burned-area probability, "
        "not spread, smoke, or loss."
    )


def overlay_assets_on_predictions(
    assets: pd.DataFrame,
    predictions: pd.DataFrame,
    spec: GridSpec | None = None,
) -> pd.DataFrame:
    """Pandas grid join: keep the highest ranking_score cell per asset.

    Works without geopandas. `assets` must include `cell_ids` (list of grid.py ids)
    or a single `cell_id`. Assets that miss the prediction land grid are dropped,
    not scored by invention.
    """
    spec = spec or GridSpec()
    if assets is None or assets.empty or predictions is None or predictions.empty:
        return empty_ranked_frame()

    working = assets.copy()
    if "cell_ids" not in working.columns:
        if "cell_id" in working.columns:
            working["cell_ids"] = working["cell_id"].map(lambda value: [value] if pd.notna(value) else [])
        elif {"lat", "lon"}.issubset(working.columns):
            working["cell_ids"] = [
                cell_ids_for_points([lat], [lon], spec=spec)
                for lat, lon in zip(working["lat"], working["lon"])
            ]
        else:
            raise ValueError("Assets need cell_ids, cell_id, or lat/lon to join the 0.5° grid")

    pred = predictions.copy()
    if "month" in pred.columns:
        pred["month"] = pd.to_datetime(pred["month"]).dt.to_period("M").astype(str)
    pred = pred.drop_duplicates(subset=["cell_id"], keep="first")
    score_cols = [
        column
        for column in (
            "cell_id",
            "month",
            "ranking_score",
            "predicted_probability",
            "risk_band",
            "month_risk_percentile",
        )
        if column in pred.columns
    ]
    pred_lookup = pred[score_cols].copy()
    pred_lookup["cell_id"] = pred_lookup["cell_id"].astype(str)

    exploded = working.explode("cell_ids", ignore_index=True).rename(columns={"cell_ids": "cell_id"})
    exploded = exploded[exploded["cell_id"].notna() & (exploded["cell_id"].astype(str) != "")]
    if exploded.empty:
        return empty_ranked_frame()
    exploded["cell_id"] = exploded["cell_id"].astype(str)
    joined = exploded.merge(pred_lookup, on="cell_id", how="inner")
    if joined.empty:
        return empty_ranked_frame()
    if "ranking_score" not in joined.columns:
        raise ValueError("Prediction table needs ranking_score")

    joined["ranking_score"] = pd.to_numeric(joined["ranking_score"], errors="coerce")
    joined = joined.dropna(subset=["ranking_score"])
    if joined.empty:
        return empty_ranked_frame()

    counts = joined.groupby("asset_id", sort=False).agg(
        n_high_cells=("risk_band", lambda series: int((series.astype(str) == "high").sum())),
        n_extreme_cells=("risk_band", lambda series: int((series.astype(str) == "extreme").sum())),
    )
    best_idx = joined.groupby("asset_id", sort=False)["ranking_score"].idxmax()
    best = joined.loc[best_idx].copy()
    best = best.merge(counts, on="asset_id", how="left")
    if "region" not in best.columns:
        best["region"] = "global"
    if "asset_name" not in best.columns:
        best["asset_name"] = best["asset_id"]
    if "asset_type" not in best.columns:
        best["asset_type"] = "unknown"
    if "asset_source" not in best.columns:
        best["asset_source"] = "unknown"
    best["reason"] = best["asset_type"].map(reason_for_asset_type)
    best["n_high_cells"] = best["n_high_cells"].fillna(0).astype(int)
    best["n_extreme_cells"] = best["n_extreme_cells"].fillna(0).astype(int)
    best["predicted_probability"] = pd.to_numeric(best["predicted_probability"], errors="coerce")
    best["month_risk_percentile"] = pd.to_numeric(best["month_risk_percentile"], errors="coerce")
    ranked = best.loc[:, [column for column in RANKED_COLUMNS if column in best.columns]].copy()
    for column in RANKED_COLUMNS:
        if column not in ranked.columns:
            ranked[column] = pd.NA
    ranked = ranked.loc[:, list(RANKED_COLUMNS)]
    return ranked.sort_values(
        ["predicted_probability", "n_extreme_cells", "n_high_cells"],
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


def stream_predictions_by_continent(
    predictions_path: str | Path,
    dest_dir: str | Path,
    region: str | None = None,
    month: str | None = None,
    chunksize: int = PREDICTION_CHUNKSIZE,
    bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, Path]:
    """One-pass stream of validation_predictions.csv, split by continent.

    Only `PREDICTION_COLUMNS` are read. Africa is first in PROCESS_ORDER; box
    assignment still uses evaluate.CONTINENT_BOXES (Australia before Asia).
    """
    predictions_path = require_predictions_file(predictions_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(predictions_path, nrows=0)
    usecols = [column for column in PREDICTION_COLUMNS if column in header.columns]
    missing = [column for column in ("cell_id", "month", "lat_center", "lon_center", "ranking_score") if column not in usecols]
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {missing}")

    writers: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    try:
        for chunk in pd.read_csv(predictions_path, usecols=usecols, chunksize=chunksize):
            if "month" in chunk.columns:
                chunk["month"] = pd.to_datetime(chunk["month"]).dt.to_period("M").astype(str)
            if month:
                chunk = chunk[chunk["month"] == str(month)]
            if chunk.empty:
                continue
            if bbox is not None:
                lat_min, lat_max, lon_min, lon_max = bbox
                chunk = chunk[
                    chunk["lat_center"].between(lat_min, lat_max)
                    & chunk["lon_center"].between(lon_min, lon_max)
                ]
                if chunk.empty:
                    continue
            if "month_num" not in chunk.columns:
                chunk["month_num"] = pd.to_datetime(chunk["month"]).dt.month
            chunk = chunk.copy()
            chunk["continent"] = assign_continent(chunk["lat_center"], chunk["lon_center"])
            if region:
                chunk = chunk[chunk["continent"] == region]
            for continent, part in chunk.groupby("continent", sort=False):
                name = str(continent)
                if name not in writers:
                    path = dest_dir / f"{name}.csv"
                    writers[name] = path.open("a", encoding="utf-8", newline="")
                    paths[name] = path
                    header_needed = True
                else:
                    header_needed = False
                part.to_csv(writers[name], index=False, header=header_needed)
    finally:
        for handle in writers.values():
            handle.close()
    return paths


def _http_session():
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def download_url(
    url: str,
    dest: Path,
    failure_cache: FailureCache,
    timeout: int = 120,
) -> SourceStatus:
    blocked = failure_cache.blocked(url)
    if blocked:
        return SourceStatus(
            source=Path(urlparse(url).path).name or url,
            url=url,
            status="cached_failure",
            note=f"Cached HTTP {blocked.get('status')} from {blocked.get('cached_at_utc')}: {blocked.get('note', '')}".strip(),
            path=str(dest),
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return SourceStatus(
            source=dest.name,
            url=url,
            status="ok",
            note="Using cached download",
            path=str(dest),
        )
    session = _http_session()
    try:
        response = session.get(url, stream=True, timeout=timeout, allow_redirects=True)
    except Exception as exc:  # noqa: BLE001 - record the exact URL and continue
        failure_cache.remember(url, status=0, note=f"{type(exc).__name__}: {exc}")
        return SourceStatus(source=dest.name, url=url, status="missing", note=str(exc), path=str(dest))
    if response.status_code in {401, 403, 404}:
        failure_cache.remember(url, status=response.status_code, note=response.reason or "")
        response.close()
        return SourceStatus(
            source=dest.name,
            url=url,
            status=f"http_{response.status_code}",
            note=f"HTTP {response.status_code} for {url}",
            path=str(dest),
        )
    try:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    handle.write(block)
    except Exception as exc:  # noqa: BLE001
        if dest.exists():
            dest.unlink()
        return SourceStatus(source=dest.name, url=url, status="missing", note=str(exc), path=str(dest))
    finally:
        response.close()
    return SourceStatus(source=dest.name, url=url, status="ok", note="Downloaded", path=str(dest))


def _shapefile_field_map(reader: Any) -> dict[str, int]:
    return {name[0]: index for index, name in enumerate(reader.fields[1:])}


def _shapefile_lines(shp_path: Path, spec: GridSpec | None = None) -> pd.DataFrame:
    import shapefile

    spec = spec or GridSpec()
    reader = shapefile.Reader(str(shp_path))
    fields = _shapefile_field_map(reader)
    rows: list[dict[str, Any]] = []
    for index, shape_record in enumerate(reader.iterShapeRecords()):
        shape = shape_record.shape
        record = shape_record.record
        parts = list(shape.parts) + [len(shape.points)]
        cell_ids: list[str] = []
        first_lon: float | None = None
        first_lat: float | None = None
        for start, end in zip(parts, parts[1:]):
            points = shape.points[start:end]
            if len(points) < 2:
                continue
            lons = [xy[0] for xy in points]
            lats = [xy[1] for xy in points]
            if first_lon is None:
                first_lon, first_lat = float(lons[0]), float(lats[0])
            cell_ids.extend(unique_cell_ids_along_line(lons, lats, spec=spec))
        cell_ids = list(dict.fromkeys(cell_ids))
        if not cell_ids or first_lat is None or first_lon is None:
            continue
        name = ""
        for key in ("NAME", "name", "OPERATOR", "OWNER", "VOLTAGE", "TYPE"):
            if key in fields:
                value = record[fields[key]]
                if value not in (None, ""):
                    name = str(value)
                    break
        asset_id = f"osm_tmm_{index}"
        if "FID" in fields:
            asset_id = f"osm_tmm_{record[fields['FID']]}"
        region = str(assign_continent([first_lat], [first_lon])[0])
        rows.append(
            {
                "asset_id": str(asset_id),
                "asset_name": name or asset_id,
                "asset_type": "transmission_line",
                "cell_ids": cell_ids,
                "asset_source": "world_bank_osm_power_tmm",
                "region": region,
                "lat": first_lat,
                "lon": first_lon,
            }
        )
    return pd.DataFrame(rows)


def load_world_bank_osm_lines(
    cache_dir: Path,
    failure_cache: FailureCache,
    spec: GridSpec | None = None,
) -> tuple[pd.DataFrame, SourceStatus]:
    dest = cache_dir / "osm_power_tmm.zip"
    status = download_url(OSM_POWER_TMM_URL, dest, failure_cache)
    status.source = "world_bank_osm_power_tmm"
    status.license = OSM_POWER_TMM_LICENSE
    if status.status != "ok":
        status.n_assets = 0
        return pd.DataFrame(), status
    extract_dir = cache_dir / "osm_power_tmm"
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(dest) as archive:
            archive.extractall(extract_dir)
    except zipfile.BadZipFile as exc:
        status.status = "missing"
        status.note = f"Zip downloaded but could not be read: {exc}"
        return pd.DataFrame(), status
    shp_paths = list(extract_dir.rglob("*.shp"))
    if not shp_paths:
        status.status = "missing"
        status.note = "Zip contained no shapefile"
        return pd.DataFrame(), status
    try:
        frame = _shapefile_lines(shp_paths[0], spec=spec)
    except ImportError:
        status.status = "missing"
        status.note = "pyshp is required to read osm_power_tmm.zip without geopandas"
        return pd.DataFrame(), status
    status.n_assets = int(len(frame))
    status.note = f"Read {len(frame)} transmission lines from {shp_paths[0].name}"
    return frame, status


def load_osm_parquet_lines_for_bbox(
    bbox: tuple[float, float, float, float],
    spec: GridSpec | None = None,
    url: str = OSM_POWER_PARQUET_URL,
) -> tuple[pd.DataFrame, SourceStatus]:
    """ODbL OSM transmission lines (power=line) for one lat/lon box. No World Bank retry."""
    spec = spec or GridSpec()
    lat_min, lat_max, lon_min, lon_max = bbox
    status = SourceStatus(
        source="osm_geofabrik_power_line_parquet",
        url=url,
        status="ok",
        license=OSM_POWER_PARQUET_LICENSE,
    )
    try:
        import duckdb
    except ImportError:
        status.status = "missing"
        status.note = "duckdb is required to read the GeoParquet OSM line extract without geopandas"
        return pd.DataFrame(), status
    sql = """
        SELECT osm_id, name, operator, type, lifecycle, voltage_kv,
               ST_AsText(geometry) AS wkt
        FROM read_parquet(?)
        WHERE bbox.xmin <= ? AND bbox.xmax >= ?
          AND bbox.ymin <= ? AND bbox.ymax >= ?
          AND type = 'line'
          AND coalesce(lifecycle, 'active') IN ('active', 'construction')
    """
    try:
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")
        frame = con.execute(sql, [url, lon_max, lon_min, lat_max, lat_min]).fetchdf()
    except Exception as exc:  # noqa: BLE001
        status.status = "missing"
        status.note = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(), status
    rows: list[dict[str, Any]] = []
    for record in frame.itertuples(index=False):
        parts = parse_wkt_line_parts(getattr(record, "wkt", "") or "")
        cell_ids: list[str] = []
        for lons, lats in parts:
            cell_ids.extend(unique_cell_ids_along_line(lons, lats, spec=spec))
        cell_ids = list(dict.fromkeys(cell_ids))
        if not cell_ids:
            continue
        osm_id = getattr(record, "osm_id")
        name = getattr(record, "name", None) or getattr(record, "operator", None) or f"osm_way_{osm_id}"
        rows.append(
            {
                "asset_id": f"osm_line_{osm_id}",
                "asset_name": str(name),
                "asset_type": "transmission_line",
                "cell_ids": cell_ids,
                "asset_source": "osm_geofabrik_power_line_parquet",
            }
        )
    assets = pd.DataFrame(rows)
    status.n_assets = int(len(assets))
    status.note = (
        f"ODbL OSM power=line features in bbox lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}]: {len(assets)}"
    )
    return assets, status


def load_gppd_plants(
    cache_dir: Path,
    failure_cache: FailureCache,
    spec: GridSpec | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> tuple[pd.DataFrame, SourceStatus]:
    spec = spec or GridSpec()
    dest = cache_dir / "global_power_plant_database.csv"
    status = download_url(GPPD_URL, dest, failure_cache)
    status.source = "wri_gppd"
    status.license = GPPD_LICENSE
    if status.status != "ok":
        return pd.DataFrame(), status
    plants = pd.read_csv(dest, usecols=lambda name: name in {
        "gppd_idnr",
        "name",
        "latitude",
        "longitude",
        "capacity_mw",
        "primary_fuel",
        "country_long",
        "owner",
    })
    plants = plants.dropna(subset=["latitude", "longitude", "gppd_idnr"])
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        plants = plants[
            plants["latitude"].between(lat_min, lat_max) & plants["longitude"].between(lon_min, lon_max)
        ]
    if plants.empty:
        status.n_assets = 0
        status.note = "GPPD CSV had no plants in the requested bbox"
        return pd.DataFrame(), status
    assigned = assign_points_to_grid(
        plants.rename(columns={"latitude": "lat", "longitude": "lon"}),
        spec,
        lat_col="lat",
        lon_col="lon",
    )
    assigned = assigned.copy()
    assigned["asset_id"] = assigned["gppd_idnr"].astype(str)
    assigned["asset_name"] = assigned["name"].fillna(assigned["asset_id"]).astype(str)
    assigned["asset_type"] = "plant"
    assigned["asset_source"] = "wri_gppd"
    assigned["cell_ids"] = assigned["cell_id"].map(lambda value: [str(value)])
    assigned["region"] = assign_continent(assigned["lat"], assigned["lon"])
    status.n_assets = int(len(assigned))
    status.note = f"Loaded {len(assigned)} WRI GPPD generation sites (asset_type=plant, not substations)"
    return assigned, status


def _arcgis_query_geojson(
    layer_url: str,
    bbox: tuple[float, float, float, float],
    failure_cache: FailureCache,
    page_size: int = 2000,
) -> tuple[list[dict[str, Any]], SourceStatus]:
    lat_min, lat_max, lon_min, lon_max = bbox
    status = SourceStatus(source=layer_url.rsplit("/", 2)[-2], url=layer_url, status="ok", license=HIFLD_LICENSE)
    blocked = failure_cache.blocked(layer_url)
    if blocked:
        status.status = "cached_failure"
        status.note = f"Cached HTTP {blocked.get('status')} for {layer_url}"
        return [], status
    session = _http_session()
    features: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = {
            "where": "1=1",
            "geometry": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": 4326,
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        try:
            response = session.get(f"{layer_url}/query", params=params, timeout=120)
        except Exception as exc:  # noqa: BLE001
            status.status = "missing"
            status.note = str(exc)
            return features, status
        if response.status_code in {401, 403, 404}:
            failure_cache.remember(layer_url, status=response.status_code, note=response.reason or "")
            status.status = f"http_{response.status_code}"
            status.note = f"HTTP {response.status_code} for {layer_url}"
            return features, status
        try:
            payload = response.json()
        except ValueError:
            status.status = "missing"
            status.note = f"Non-JSON HIFLD response ({response.status_code})"
            return features, status
        if payload.get("error"):
            status.status = "missing"
            status.note = str(payload["error"])
            return features, status
        batch = payload.get("features") or []
        features.extend(batch)
        exceeded = bool(payload.get("exceededTransferLimit") or payload.get("properties", {}).get("exceededTransferLimit"))
        if not batch or not exceeded:
            if len(batch) < page_size:
                break
            offset += len(batch)
            continue
        offset += len(batch)
    status.n_assets = int(len(features))
    status.note = f"Fetched {len(features)} GeoJSON features"
    return features, status


def _geojson_feature_to_asset(
    feature: dict[str, Any],
    asset_type: str,
    asset_source: str,
    spec: GridSpec,
    id_prefix: str,
    index: int,
) -> dict[str, Any] | None:
    geometry = feature.get("geometry") or {}
    props = feature.get("properties") or {}
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    cell_ids: list[str] = []
    if geom_type == "Point" and coordinates:
        cell_ids = cell_ids_for_points([coordinates[1]], [coordinates[0]], spec=spec)
    elif geom_type == "LineString" and coordinates:
        lons = [xy[0] for xy in coordinates]
        lats = [xy[1] for xy in coordinates]
        cell_ids = unique_cell_ids_along_line(lons, lats, spec=spec)
    elif geom_type == "MultiLineString" and coordinates:
        collected: list[str] = []
        for part in coordinates:
            lons = [xy[0] for xy in part]
            lats = [xy[1] for xy in part]
            collected.extend(unique_cell_ids_along_line(lons, lats, spec=spec))
        cell_ids = list(dict.fromkeys(collected))
    elif geom_type == "MultiPoint" and coordinates:
        lats = [xy[1] for xy in coordinates]
        lons = [xy[0] for xy in coordinates]
        cell_ids = list(dict.fromkeys(cell_ids_for_points(lats, lons, spec=spec)))
    if not cell_ids:
        return None
    raw_id = props.get("ID") or props.get("OBJECTID") or props.get("FID") or index
    name = (
        props.get("NAME")
        or props.get("SUB_1")
        or props.get("OWNER")
        or props.get("NAICS_DESC")
        or f"{id_prefix}_{raw_id}"
    )
    return {
        "asset_id": f"{id_prefix}_{raw_id}",
        "asset_name": str(name),
        "asset_type": asset_type,
        "cell_ids": cell_ids,
        "asset_source": asset_source,
    }


def load_hifld_assets(
    bbox: tuple[float, float, float, float],
    failure_cache: FailureCache,
    spec: GridSpec | None = None,
    include_lines: bool = True,
    include_substations: bool = True,
    region: str = "north_america",
) -> tuple[pd.DataFrame, list[SourceStatus]]:
    spec = spec or GridSpec()
    frames: list[pd.DataFrame] = []
    reports: list[SourceStatus] = []
    if include_lines:
        features, status = _arcgis_query_geojson(HIFLD_TRANSMISSION_URL, bbox, failure_cache)
        status.source = "hifld_transmission_lines"
        status.region = region
        status.license = HIFLD_LICENSE
        assets = []
        for index, feature in enumerate(features):
            row = _geojson_feature_to_asset(
                feature,
                asset_type="transmission_line",
                asset_source="hifld_transmission_lines",
                spec=spec,
                id_prefix="hifld_line",
                index=index,
            )
            if row:
                row["region"] = region
                assets.append(row)
        status.n_assets = int(len(assets))
        reports.append(status)
        if assets:
            frames.append(pd.DataFrame(assets))
    if include_substations:
        features, status = _arcgis_query_geojson(HIFLD_SUBSTATIONS_URL, bbox, failure_cache)
        status.source = "hifld_substations"
        status.region = region
        status.license = HIFLD_LICENSE
        assets = []
        for index, feature in enumerate(features):
            row = _geojson_feature_to_asset(
                feature,
                asset_type="substation",
                asset_source="hifld_substations",
                spec=spec,
                id_prefix="hifld_sub",
                index=index,
            )
            if row:
                row["region"] = region
                assets.append(row)
        status.n_assets = int(len(assets))
        reports.append(status)
        if assets:
            frames.append(pd.DataFrame(assets))
    if not frames:
        return pd.DataFrame(), reports
    return pd.concat(frames, ignore_index=True), reports


def collect_region_assets(
    region: str,
    cache_dir: Path,
    failure_cache: FailureCache,
    global_lines: pd.DataFrame | None,
    global_line_status: SourceStatus | None,
    plants: pd.DataFrame,
    spec: GridSpec | None = None,
) -> tuple[pd.DataFrame, list[SourceStatus]]:
    """Assemble real assets for one continent. Never invent lines or scores."""
    spec = spec or GridSpec()
    reports: list[SourceStatus] = []
    frames: list[pd.DataFrame] = []
    bbox = continent_bbox(region)
    if global_line_status is not None:
        region_status = SourceStatus(**{**global_line_status.to_dict(), "region": region})
        reports.append(region_status)
    if global_lines is not None and not global_lines.empty:
        lines = global_lines.copy()
        if "region" in lines.columns:
            lines = lines[lines["region"] == region]
        if not lines.empty:
            lines["region"] = region
            frames.append(lines)
    elif bbox is not None:
        parquet_lines, parquet_status = load_osm_parquet_lines_for_bbox(bbox, spec=spec)
        parquet_status.region = region
        reports.append(parquet_status)
        if not parquet_lines.empty:
            parquet_lines = parquet_lines.copy()
            parquet_lines["region"] = region
            frames.append(parquet_lines)
        elif region == "north_america":
            LOGGER.info("OSM lines missing for north_america; falling back to HIFLD FeatureServers")
            hifld, hifld_reports = load_hifld_assets(bbox, failure_cache, spec=spec, region=region)
            reports.extend(hifld_reports)
            if not hifld.empty:
                frames.append(hifld)

    if not plants.empty:
        region_plants = plants[plants["region"] == region] if "region" in plants.columns else plants
        frames.append(region_plants)
        reports.append(
            SourceStatus(
                source="wri_gppd",
                url=GPPD_URL,
                status="ok" if not region_plants.empty else "missing",
                n_assets=int(len(region_plants)),
                license=GPPD_LICENSE,
                region=region,
                note="Generation sites only (asset_type=plant)",
            )
        )

    if not frames:
        return pd.DataFrame(), reports
    combined = pd.concat(frames, ignore_index=True)
    if "region" in combined.columns:
        combined = combined[combined["region"].isna() | (combined["region"] == region)].copy()
        combined["region"] = region
    else:
        combined["region"] = region
    return combined, reports


def build_demo_portfolio(ranked: pd.DataFrame, limit: int = 40) -> pd.DataFrame:
    """Public-data demo book, not a customer book. Named plants first, then other assets."""
    if ranked.empty:
        return empty_ranked_frame()
    named = ranked[ranked["asset_name"].astype(str).str.len() > 0].copy()
    plants = named[named["asset_type"] == "plant"]
    hot = plants[plants["risk_band"].isin(["high", "extreme"])]
    ordered = pd.concat(
        [hot, plants[~plants["asset_id"].isin(hot["asset_id"])], named[~named["asset_id"].isin(plants["asset_id"])]],
        ignore_index=True,
    ).drop_duplicates(subset=["asset_id"])
    return ordered.head(limit).reset_index(drop=True)


def overlay_region_predictions(
    pred_df: pd.DataFrame,
    assets: pd.DataFrame,
    region: str,
    month: str | None = None,
    spec: GridSpec | None = None,
) -> tuple[pd.DataFrame, str]:
    chosen_month = peak_month_for_region(pred_df, month=month)
    month_pred = pred_df.copy()
    month_pred["month"] = pd.to_datetime(month_pred["month"]).dt.to_period("M").astype(str)
    month_pred = month_pred[month_pred["month"] == chosen_month]
    LOGGER.info(
        "Region %s peak-fire month %s (%s cells, %s assets)",
        region,
        chosen_month,
        f"{len(month_pred):,}",
        f"{len(assets):,}",
    )
    ranked = overlay_assets_on_predictions(assets, month_pred, spec=spec)
    if not ranked.empty:
        ranked["region"] = region
        ranked["month"] = chosen_month
    return ranked, chosen_month


def iter_process_regions(available: Iterable[str], region: str | None = None) -> Iterator[str]:
    wanted = set(available)
    order = [region] if region else list(PROCESS_ORDER)
    for name in order:
        if name in wanted:
            yield name
    if region is None:
        for name in sorted(wanted):
            if name not in PROCESS_ORDER:
                yield name
