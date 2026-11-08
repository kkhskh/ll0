# Worldwide wildfire asset exposure

Overlay public power assets on Hephaelion v8 monthly burned-area scores. Horizon
is the calendar month. Say **this month's burned-area probability**. This overlay
does not simulate spread, smoke, or dollar losses, and it does not invent assets
or scores.

`--scope california` is the original HIFLD California demo. `--scope global`
tiles the prediction CSV by continent so one Africa peak-fire month cannot
overwrite Australia.

v8 and v9 are not retrained. `wildfire_risk/build_fire_weather.py` and
`wildfire_risk/fire_weather.py` are not edited.

## Why not California-only

HIFLD is US-only. The v8 prediction CSV is about 4 GB. A single global peak-fire
month is always Africa. Continents use `evaluate.py` `CONTINENT_BOXES` (Australia
before Asia so the mainland is not counted as SE Asia). Processing order is
Africa-first from `CONTINENT_ORDER`.

| Region | lat_min | lat_max | lon_min | lon_max |
|---|---:|---:|---:|---:|
| australia | -45 | -10 | 112 | 155 |
| africa | -35 | 37.5 | -18 | 52 |
| south_america | -56 | 13 | -82 | -34 |
| north_america | 7 | 84 | -170 | -50 |
| europe | 36 | 72 | -11 | 40 |
| asia | -11 | 78 | 40 | 180 |

## Architecture

1. Stream `validation_predictions.csv` with `usecols` limited to `cell_id`,
   `month`, `year`, `month_num`, `lat_center`, `lon_center`, `binary_risk_label`,
   `predicted_probability`, `ranking_score`, `risk_percentile`,
   `month_risk_percentile`, `risk_band`. Split by continent in one pass.
2. Assign points with `wildfire_risk/grid.py` 0.5° (`assign_points_to_grid`).
   Sample lines every 0.1° and unique `cell_id`s. Keep the highest
   `ranking_score`. Count high and extreme cells. Do not use EPSG:3310.
3. If geopandas is missing, the pandas grid join still runs. Matplotlib uses the
   Agg backend. Outputs go through `create_run_layout` / `write_stage_manifest`
   into a **new** run folder. v8 model artifacts are not overwritten.

## CLI

```bash
python -m wildfire_risk.asset_exposure \
  --predictions PATH \
  --run-root artifacts/wildfire_risk_runs \
  --scope global \
  [--month 2024-07] \
  [--region africa]
```

`--scope california` keeps the HIFLD California path.

## Asset sources

Only real public files. If a source file is missing, record the exact URL and
skip that region's lines or points. Do not invent assets.

| Source | URL | Role | License | Status from this project |
|---|---|---|---|---|
| World Bank OSM Global Transmission Network (`osm_power_tmm.zip`) | https://energydata.info/dataset/e0e878a7-7712-465d-a4c2-69b2b3518513/resource/e0f33c3e-6dab-41d1-919d-d266f300d0bc/download/osm_power_tmm.zip | Global transmission lines | CC-BY-4.0 | HTTP 403. Cached in `source_fetch_failures.json`. Not retried per region. |
| OSM GeoParquet `power_line.parquet` | https://parquetry.geomermaids.com/gmwid/latest/power_line.parquet | Global `power=line` fallback (Geofabrik extracts, `type=line` only) | ODbL 1.0, © OpenStreetMap contributors | HTTP 200. HEAD 508,983,537 bytes. Used when the World Bank zip is blocked. |
| WRI Global Power Plant Database | https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv | Generation sites, `asset_type=plant` (not substations) | CC-BY-4.0 | HTTP 200. |
| HIFLD US Electric Power Transmission Lines | https://services2.arcgis.com/LYMgRMwHfrWWEg3s/arcgis/rest/services/HIFLD_US_Electric_Power_Transmission_Lines/FeatureServer/0 | California demo; North America fallback lines | U.S. public domain / HIFLD Open | HTTP 200. |
| HIFLD Electric Substations | https://services5.arcgis.com/HDRa0B57OVrv2E1q/ArcGIS/rest/services/Electric_Substations/FeatureServer/0 | California demo; North America fallback substations | U.S. public domain / HIFLD Open | HTTP 200. |

The GeoParquet fallback is a real OSM-derived global line extract, not modeled
gridfinder lines. Credit © OpenStreetMap contributors and keep derivatives under
ODbL 1.0.

## Outputs

Written under `artifacts/wildfire_risk_runs/wildfire-asset-exposure-global/` when
`validation_predictions.csv` is present:

- `reports/ranked_asset_exposure_{region}.csv`
- `reports/ranked_asset_exposure_global.csv`
- `reports/demo_utility_portfolio_global.csv` (demo book, not a customer book)
- `reports/coverage_report.csv`
- `maps/asset_exposure_{region}_{month}.png`
- `README.md`

Ranked columns: `asset_id`, `asset_name`, `asset_type`, `region`, `cell_id`,
`month`, `predicted_probability`, `risk_band`, `month_risk_percentile`,
`n_high_cells`, `n_extreme_cells`, `reason`, `asset_source`.

If the prediction CSV is not on disk, this repository ships code, offline tests,
and this coverage document only. Do not fabricate a ranked book.
