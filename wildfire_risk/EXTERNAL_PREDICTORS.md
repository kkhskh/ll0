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

## ERA5 Example

If your NetCDF contains coordinates like `time`, `latitude`, `longitude` and variables like `t2m`, `tp`, `swvl1`, `pev`, run:

```bash
python wildfire_risk/build_predictor_table.py \
  --input "/content/drive/MyDrive/Hephaelion/era5_monthly.nc" \
  --variable t2m \
  --variable tp \
  --variable swvl1 \
  --variable pev \
  --rename t2m:temperature_2m \
  --rename tp:precipitation \
  --rename swvl1:soil_moisture \
  --rename pev:potential_evaporation \
  --run-root "/content/drive/MyDrive/Hephaelion/runs" \
  --run-name "wildfire-risk-mvp-2020-2025-noleak-v3" \
  --output-name "era5_predictors.csv"
```

Then feed the result into feature building:

```bash
python wildfire_risk/build_features.py \
  --labels "/content/drive/MyDrive/Hephaelion/labels_full_2020_2025.csv" \
  --dynamic-source "/content/drive/MyDrive/Hephaelion/runs/wildfire-risk-mvp-2020-2025-noleak-v3/features/era5_predictors.csv:era5" \
  --run-root "/content/drive/MyDrive/Hephaelion/runs" \
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

## Important Modeling Note

`build_features.py` now treats dynamic external predictors as forecast-safe features:

- contemporaneous source values are not kept as direct same-month training inputs
- lagged and rolling variants are created per grid cell
- this reduces the risk of leaking same-month observed weather or vegetation conditions into the target month
