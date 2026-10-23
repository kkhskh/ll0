# External Predictor Tables

This pipeline can now build real monthly predictor tables for wildfire modeling from gridded CSV or NetCDF sources.

The target format consumed by `build_features.py` is:

- `cell_id`
- `month`
- one or more predictor columns such as:
  - `temperature_2m`
  - `precipitation`
  - `soil_moisture`
  - `potential_evaporation`
  - `ndvi`
  - `evi`
  - `spi_3`

## Builder Script

Use `wildfire_risk/build_predictor_table.py` to aggregate source grids onto the wildfire grid.

It accepts:

- monthly gridded `CSV`
- monthly gridded `NetCDF`

It outputs:

- one row per `cell_id/month`
- mean-aggregated predictor values
- optional derived wildfire features such as `hot_dry_factor`

## Input Formats

`--input-format` selects how the source file is read:

- `tabular` loads the whole file into memory. Use it for small CSV or NetCDF extracts.
- `era5-land-netcdf` streams a global ERA5-Land monthly-means file one timestep at a time.
- `auto` (the default) behaves like `tabular`.

A global ERA5-Land file holds `72 x 1801 x 3600`, or roughly `467` million, cell-months per variable.
That does not fit in memory as a dataframe, so `era5-land-netcdf` reduces each timestep with
`bincount` and never materialises the full grid.

## ERA5-Land Example

This is the exact command used to build the predictor table that is currently on disk:

```bash
python wildfire_risk/build_predictor_table.py \
  --input datasets/wildfires/era5_land/b09a78a9a0989d1946e3597dbfbc11ca.nc \
  --input-format era5-land-netcdf \
  --variable t2m --variable tp --variable swvl1 \
  --variable pev --variable u10 --variable v10 \
  --resolution-deg 0.5 \
  --output datasets/wildfires/era5_land/era5_land_predictors_2020_2025.csv
```

It produced `6,793,560` rows covering `94,355` land cells across `2020-01` to `2025-12`, with columns
`temperature_2m`, `precipitation`, `soil_moisture`, `potential_evaporation`, `wind_u10`, `wind_v10`,
`wind_speed`, `hot_dry_factor`, and `climate_water_balance`.

### Conventions the ERA5-Land path applies for you

ERA5-Land does not arrive in modelling-ready form, so this path normalises three things:

- **Longitude.** CDS delivers a `0` to `360` axis; the wildfire grid spans `-180` to `180`.
- **Accumulation units.** `tp` and `pev` arrive in metres per day and become millimetres per day.
- **Evaporation sign.** `pev` uses a downward-flux convention where evaporation is negative. It is
  flipped so that a larger `potential_evaporation` means more atmospheric drying, which makes
  `climate_water_balance = precipitation - potential_evaporation` read correctly as a deficit.

Variables are also renamed from the short ERA5 names to the predictor names in `data_sources.py`.
Pass `--rename` to override any of them.

### Caveats worth knowing

- `temperature_2m` stays in kelvin. `hot_dry_factor` divides it by soil moisture, so a conversion to
  celsius would make that ratio unstable near `0 C`.
- `wind_speed` is the magnitude of the monthly *mean* wind vector, not the mean of hourly wind
  speeds. It is biased low wherever wind direction varies through the month. Treat it as a prevailing
  wind indicator rather than a gust or fire-weather wind.
- ERA5-Land `pev` assumes no moisture stress, so it runs high in arid regions. That is a known
  product characteristic, not an ingestion bug; the land-mean works out to about `1590 mm/year`.

Then feed the result into feature building:

```bash
python wildfire_risk/build_features.py \
  --labels "artifacts/wildfire_risk_runs/labels_full_2020_2025.csv" \
  --dynamic-source "datasets/wildfires/era5_land/era5_land_predictors_2020_2025.csv:era5" \
  --run-root "artifacts/wildfire_risk_runs" \
  --run-name "wildfire-risk-mvp-2020-2025-noleak-v3"
```

## Drought Example

For monthly drought indicators such as `spi_3`, `spei_6`, or soil moisture anomalies:

```bash
python wildfire_risk/build_predictor_table.py \
  --input "/content/drive/MyDrive/Hephaelion/drought_monthly.nc" \
  --variable spi_3 \
  --variable spei_6 \
  --run-root "/content/drive/MyDrive/Hephaelion/runs" \
  --run-name "wildfire-risk-mvp-2020-2025-noleak-v3" \
  --output-name "drought_predictors.csv"
```

Then:

```bash
python wildfire_risk/build_features.py \
  --labels "/content/drive/MyDrive/Hephaelion/labels_full_2020_2025.csv" \
  --dynamic-source "/content/drive/MyDrive/Hephaelion/runs/wildfire-risk-mvp-2020-2025-noleak-v3/features/era5_predictors.csv:era5" \
  --dynamic-source "/content/drive/MyDrive/Hephaelion/runs/wildfire-risk-mvp-2020-2025-noleak-v3/features/drought_predictors.csv:drought" \
  --run-root "/content/drive/MyDrive/Hephaelion/runs" \
  --run-name "wildfire-risk-mvp-2020-2025-noleak-v3"
```

## Vegetation Example

If you have monthly NDVI/EVI rasters or NetCDF exported to a grid:

```bash
python wildfire_risk/build_predictor_table.py \
  --input "/content/drive/MyDrive/Hephaelion/vegetation_monthly.nc" \
  --variable ndvi \
  --variable evi \
  --run-root "/content/drive/MyDrive/Hephaelion/runs" \
  --run-name "wildfire-risk-mvp-2020-2025-noleak-v3" \
  --output-name "vegetation_predictors.csv"
```

Then:

```bash
python wildfire_risk/build_features.py \
  --labels "/content/drive/MyDrive/Hephaelion/labels_full_2020_2025.csv" \
  --dynamic-source "/content/drive/MyDrive/Hephaelion/runs/wildfire-risk-mvp-2020-2025-noleak-v3/features/era5_predictors.csv:era5" \
  --dynamic-source "/content/drive/MyDrive/Hephaelion/runs/wildfire-risk-mvp-2020-2025-noleak-v3/features/drought_predictors.csv:drought" \
  --dynamic-source "/content/drive/MyDrive/Hephaelion/runs/wildfire-risk-mvp-2020-2025-noleak-v3/features/vegetation_predictors.csv:veg" \
  --run-root "/content/drive/MyDrive/Hephaelion/runs" \
  --run-name "wildfire-risk-mvp-2020-2025-noleak-v3"
```

## MCD12Q1 Land Cover

Annual IGBP land cover is a static/slow feature. `convert_mcd12q1.py` reads the HDF tiles,
aggregates 500 m `LC_Type1` pixels onto the 0.5 degree grid, and writes class fractions plus the
majority class.

The join is no-leak by default: the map for product year `Y` is used as the `year = Y+1` feature.
If 2019 tiles are missing, 2020 months get the 2020 map as a one-year fallback.

```bash
python -m wildfire_risk.convert_mcd12q1 \
  --input datasets/wildfires/mcd12q1 \
  --start-year 2019 \
  --end-year 2024 \
  --stride 4 \
  --run-root artifacts/wildfire_risk_runs \
  --run-name wildfire-risk-mvp-2020-2025-noleak-v4
```

Then rebuild features without overwriting the v3 table:

```bash
python wildfire_risk/build_features.py \
  --labels artifacts/wildfire_risk_runs/labels_full_2020_2025.csv \
  --dynamic-source datasets/wildfires/era5_land/era5_land_predictors_2020_2025.csv:era5 \
  --static-source artifacts/wildfire_risk_runs/wildfire-risk-mvp-2020-2025-noleak-v4/features/mcd12q1_landcover.csv:landcover \
  --run-root artifacts/wildfire_risk_runs \
  --run-name wildfire-risk-mvp-2020-2025-noleak-v4
```

## NASADEM Terrain

NASADEM is static elevation and slope. Do not download the global archive. Western North America
(`-130,25,-100,52`) is the only region where this run can change the continent score.

`convert_nasadem.py` reads `NASADEM_HGT` `.zip` / `.hgt` tiles and writes mean elevation, elevation
std, mean slope, and `has_terrain` on the 0.5 degree grid. Cells outside the downloaded box stay
unjoined and fill as zero, so Africa is not given fake sea-level terrain unless `has_terrain=1`.

```bash
python -m wildfire_risk.convert_nasadem \
  --input datasets/wildfires/nasadem \
  --stride 10 \
  --run-root artifacts/wildfire_risk_runs \
  --run-name wildfire-risk-mvp-2020-2025-noleak-v5
```

Then attach onto the v3 feature table (the working model; do not rebuild from labels):

```bash
python wildfire_risk/build_features.py \
  --base-features artifacts/wildfire_risk_runs/wildfire-risk-mvp-2020-2025-noleak-v3/features/feature_table.csv \
  --static-source artifacts/wildfire_risk_runs/wildfire-risk-mvp-2020-2025-noleak-v5/features/nasadem_terrain.csv:terrain \
  --run-root artifacts/wildfire_risk_runs \
  --run-name wildfire-risk-mvp-2020-2025-noleak-v5
```

## GPWv4 Population

Use the 2020 population-density snapshot at 2.5 arc minutes. It is one zip, about 50 MB.
The 2020 map is joined as a static column on every month from 2020 through 2025.

```bash
python -m wildfire_risk.download_external gpw --years 2020 --resolution 2pt5_min
python -m wildfire_risk.convert_gpw \
  --input datasets/wildfires/gpw \
  --run-root artifacts/wildfire_risk_runs \
  --run-name wildfire-risk-mvp-2020-2025-noleak-v6
```

Attach onto the v3 feature table and write the big CSV on local disk before copying it to Drive.

## Getting The Raw Sources

`download_external.py` fetches the sources that are not on disk yet. It needs Earthdata Login
credentials, and FIRMS additionally needs a map key. See `DATA_SOURCES.md` for setup, then:

```bash
python wildfire_risk/download_external.py check-credentials
python wildfire_risk/download_external.py list

python wildfire_risk/download_external.py gpw --years 2020 --resolution 2pt5_min
python wildfire_risk/download_external.py groads --region global
python wildfire_risk/download_external.py mcd12q1 --start-year 2019 --end-year 2024
python wildfire_risk/download_external.py nasadem --bbox -130,25,-100,52
python wildfire_risk/download_external.py firms --start-month 2020-01 --end-month 2025-12
```

Downloads resume and skip files that are already complete, so re-running after an interruption only
fetches what is missing.

## Important Modeling Note

`build_features.py` now treats dynamic external predictors as forecast-safe features:

- contemporaneous source values are not kept as direct same-month training inputs
- lagged and rolling variants are created per grid cell
- this reduces the risk of leaking same-month observed weather or vegetation conditions into the target month
