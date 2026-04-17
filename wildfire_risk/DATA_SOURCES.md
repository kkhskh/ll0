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
