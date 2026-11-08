# Wildfire asset exposure

This run overlays public power assets on Hephaelion v8 monthly burned-area scores.

**This is this month's burned-area probability.** The horizon is the calendar
month. The overlay does **not** simulate fire spread, smoke transport, or dollar
losses. A high score means the 0.5° cell that holds the asset ranked high for
burned area this month, not that a fire will reach the asset.

The `demo_utility_portfolio_*.csv` file is a **demo book built from public
plant and line records, not a customer book.** Do not treat it as an insured
portfolio, a utility's GIS, or a loss run.

## Model

- Scores come from the existing v8 global 0.5° monthly burned-area model
  (train 2020–2023, holdout 2024–2025). This overlay does not retrain v8 or v9.
- Join uses `wildfire_risk/grid.py` (same formula as `assign_points_to_grid`).
  Lines are sampled every 0.1°. ranking_score is S_max; length_weighted_score is
  S_len. score_range is not a confidence interval. See ASSET_EXPOSURE_MATH.md.
- Each continent keeps its own peak-fire month so Africa does not overwrite
  Australia. Pass `--month YYYY-MM` to force one month.

## Asset sources

Only real public files are used. If a download fails, the exact URL is recorded
in `coverage_report.csv` and that region continues without invented assets.

| Source | Role | License |
|---|---|---|
| World Bank OSM extract `osm_power_tmm.zip` | Global transmission lines | CC-BY-4.0 |
| OSM GeoParquet `power_line.parquet` | Global `power=line` fallback when the World Bank zip is blocked | ODbL 1.0, © OpenStreetMap contributors |
| WRI Global Power Plant Database | Generation sites (`asset_type=plant`) | CC-BY-4.0 |
| HIFLD transmission lines + substations | California demo, and North America fallback | U.S. public domain / HIFLD Open |

World Bank URL (previously HTTP 403 from this project):
https://energydata.info/dataset/e0e878a7-7712-465d-a4c2-69b2b3518513/resource/e0f33c3e-6dab-41d1-919d-d266f300d0bc/download/osm_power_tmm.zip

OSM GeoParquet fallback:
https://parquetry.geomermaids.com/gmwid/latest/power_line.parquet

GPPD:
https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv

HIFLD lines:
https://services2.arcgis.com/LYMgRMwHfrWWEg3s/arcgis/rest/services/HIFLD_US_Electric_Power_Transmission_Lines/FeatureServer/0

HIFLD substations:
https://services5.arcgis.com/HDRa0B57OVrv2E1q/ArcGIS/rest/services/Electric_Substations/FeatureServer/0

## Ranked columns

See `wildfire_risk/ASSET_EXPOSURE_MATH.md` for numbered equations.

asset_id, asset_name, asset_type, region, cell_id, month, ranking_score (S_max),
length_weighted_score (S_len), predicted_probability, calibrated_probability,
risk_band, month_risk_percentile, n_cells, n_high_cells, n_extreme_cells,
extreme_cell_fraction, score_range (U, not a CI), score_std, inspect_rank,
reason, asset_source.

Demo book extra columns: high_or_extreme_share, herfindahl_top_cells, n_demo_assets.
Those are concentration of this public demo book, not customer TVaR.

## This run

Scope: `global`.

**SYNTHETIC DEMO.** v8 `validation_predictions.csv` was absent. Do not treat these scores as this month's burned-area probability from v8. The columns implement ASSET_EXPOSURE_MATH.md eqs. (6)–(13) on labeled toy assets in Africa, Australia, and North America.

Per-region months used by the toy overlay:
- africa: 2024-07
- australia: 2024-01
- north_america: 2024-09
