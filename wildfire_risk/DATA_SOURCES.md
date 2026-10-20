# Wildfire Risk Data Inventory

This package treats the existing `EO4WildFires` `.nc` files as a separate event-level task.
The monthly wildfire risk pipeline uses these first-wave global sources instead:

| Name | Role | Coverage | Resolution | Expected columns after preprocessing |
|---|---|---|---|---|
| `MCD64A1` | Primary wildfire label source using monthly burned area | `2000-present` | `Monthly`, `500 m global` | `month`, `lat`, `lon`, `burned_area_km2` |
| `NASA FIRMS` | Supplementary active-fire counts and ignition density proxy | `2000-present` | `Daily/point detections`, aggregated monthly | `date`, `lat`, `lon`, `confidence`, `frp` |
| `ERA5-Land` | Monthly climate and drought predictors | `1950-present` | `Monthly`, `0.1 degree global` | `month`, `cell_id`, `temperature_2m`, `precipitation`, `wind_speed`, `soil_moisture` |
| `MCD12Q1` | Annual land-cover and vegetation context | `2001-present` | `Annual`, `500 m global` | `year`, `cell_id`, `land_cover_class` |
| `NASADEM` | Static terrain features | `Static` | `30 m where available` | `cell_id`, `elevation_m`, `slope_deg`, `aspect_deg` |
| `GPWv4` | Population density / human pressure | `2000, 2005, 2010, 2015, 2020` | `Five-year snapshots`, `30 arc-second global` | `year`, `cell_id`, `population_density` |
| `gROADS` or `GRIP` | Road access / infrastructure pressure | `Static or slowly changing` | `Vector or raster` | `cell_id`, `road_density_km_per_km2` |

Reference URLs:

- `MCD64A1`: <https://lpdaac.usgs.gov/products/mcd64a1v061/>
- `NASA FIRMS`: <https://firms.modaps.eosdis.nasa.gov/download/>
- `ERA5-Land`: <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means>
- `MCD12Q1`: <https://lpdaac.usgs.gov/products/mcd12q1v061/>
- `NASADEM`: <https://www.earthdata.nasa.gov/data/catalog/lpcloud-nasadem-sc-001>
- `GPWv4`: <https://www.earthdata.nasa.gov/data/projects/gpw>
- `gROADS`: <https://www.earthdata.nasa.gov/data/catalog/sedac-ciesin-sedac-groads-v1-1.0>

## Local acquisition status

| Source | Raw data on disk | Built predictor/label table |
|---|---|---|
| `MCD64A1` | Yes, 19832 HDF tiles for `2020-01` to `2026-02` in `datasets/wildfires/mcd64a1` | Not built. See `MCD64A1_INGESTION.md` |
| `ERA5-Land` | Yes, `datasets/wildfires/era5_land/b09a78a9a0989d1946e3597dbfbc11ca.nc`, global `0.1` degree monthly means for `2020-01` to `2025-12` | Yes, `datasets/wildfires/era5_land/era5_land_predictors_2020_2025.csv` |
| `NASA FIRMS` | No | No |
| `MCD12Q1` | Downloaded on Colab (~630 tiles, 2020-2024). Add 2019 for a no-leak 2020 join. | Converter added; table not built yet |
| `NASADEM` | No | No |
| `GPWv4` | No | No |
| `gROADS` | No | No |

Use `download_external.py` to fetch the five outstanding sources. It resolves granules through
NASA's CMR API, so the collection concept IDs stay correct as new revisions are published:

| Subcommand | Collection | Concept ID |
|---|---|---|
| `mcd12q1` | `MCD12Q1` v061 | `C2484079608-LPCLOUD` |
| `nasadem` | `NASADEM_HGT` v001 | `C2763264762-LPCLOUD` |
| `nasadem --product slope` | `NASADEM_SC` v001 | `C2763264770-LPCLOUD` |
| `gpw` | `CIESIN_SEDAC_GPWv4_POPDENS_R11` v4.11 | `C3540910651-ESDIS` |
| `groads` | `CIESIN_SEDAC_GROADS_V1` v1.0 | `C3550192909-ESDIS` |
| `firms` | FIRMS area API, not CMR | n/a |

## Credentials

All five downloads need an Earthdata Login account, and FIRMS needs its own map key.

1. Register at <https://urs.earthdata.nasa.gov/users/new>.
2. Either generate a token at <https://urs.earthdata.nasa.gov/profile> and `export EARTHDATA_TOKEN=...`,
   or add `machine urs.earthdata.nasa.gov login USER password PASS` to `~/.netrc` and `chmod 600 ~/.netrc`.
3. Request a FIRMS map key at <https://firms.modaps.eosdis.nasa.gov/api/map_key/> and `export FIRMS_MAP_KEY=...`.
4. Confirm the setup with `python wildfire_risk/download_external.py check-credentials`.

LP DAAC and SEDAC collections also require a one-time EULA acceptance per collection. If a download
returns `HTTP 401` or `403` with valid credentials, open the collection's landing page while signed in
and accept the agreement, then re-run the subcommand.

## Download volume

`NASADEM` is about `14000` tiles and roughly `180 GB` globally, which is far more terrain detail than a
`0.5` degree grid needs. The `nasadem` subcommand therefore refuses to run globally unless you pass
`--allow-global`; prefer `--bbox west,south,east,north` for the regions you actually model.
