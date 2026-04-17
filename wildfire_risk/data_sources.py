from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DataSourceSpec:
    name: str
    role: str
    temporal_coverage: str
    temporal_resolution: str
    spatial_resolution: str
    format_hint: str
    access_url: str
    expected_columns: tuple[str, ...]
    notes: str


FIRST_WAVE_SOURCES: tuple[DataSourceSpec, ...] = (
    DataSourceSpec(
        name="MCD64A1",
        role="Primary wildfire label source using monthly burned area",
        temporal_coverage="2000-present",
        temporal_resolution="Monthly",
        spatial_resolution="500 m global",
        format_hint="Raster or exported grid-level CSV",
        access_url="https://lpdaac.usgs.gov/products/mcd64a1v061/",
        expected_columns=("month", "lat", "lon", "burned_area_km2"),
        notes=(
            "Use this as the core supervision target. Aggregate burned pixels into the coarse grid and "
            "derive burned_fraction and binary_risk_label."
        ),
    ),
    DataSourceSpec(
        name="NASA FIRMS",
        role="Supplementary active-fire counts and ignition density proxy",
        temporal_coverage="2000-present",
        temporal_resolution="Daily to monthly aggregate",
        spatial_resolution="Point detections",
        format_hint="CSV, JSON, or shapefile exported to CSV",
        access_url="https://firms.modaps.eosdis.nasa.gov/download/",
        expected_columns=("date", "lat", "lon", "confidence", "frp"),
        notes=(
            "Use as a support feature or validation signal, not the primary burned-area label. Monthly counts, "
            "FRP totals, and detection density are useful coarse-grid predictors."
        ),
    ),
    DataSourceSpec(
        name="ERA5-Land",
        role="Monthly climate and drought predictors",
        temporal_coverage="1950-present",
        temporal_resolution="Monthly",
        spatial_resolution="0.1 degree global",
        format_hint="NetCDF exported to monthly grid-level CSV",
        access_url="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means",
        expected_columns=("month", "cell_id", "temperature_2m", "precipitation", "wind_speed", "soil_moisture"),
        notes=(
            "Use only variables available before the month being predicted. Add anomaly and trailing-window "
            "features after aggregation to the wildfire risk grid."
        ),
    ),
    DataSourceSpec(
        name="MCD12Q1",
        role="Annual land-cover and vegetation context",
        temporal_coverage="2001-present",
        temporal_resolution="Annual",
        spatial_resolution="500 m global",
        format_hint="Raster or exported grid-level CSV",
        access_url="https://lpdaac.usgs.gov/products/mcd12q1v061/",
        expected_columns=("year", "cell_id", "land_cover_class"),
        notes=(
            "Join as a slow-changing annual feature. For a monthly table, forward-fill within the same year."
        ),
    ),
    DataSourceSpec(
        name="NASADEM",
        role="Static terrain features including elevation and slope",
        temporal_coverage="Static",
        temporal_resolution="Static",
        spatial_resolution="30 m where available",
        format_hint="Raster or exported grid-level CSV",
        access_url="https://www.earthdata.nasa.gov/data/catalog/lpcloud-nasadem-sc-001",
        expected_columns=("cell_id", "elevation_m", "slope_deg", "aspect_deg"),
        notes=(
            "Aggregate high-resolution terrain into mean elevation, mean slope, and circular aspect summaries per cell."
        ),
    ),
    DataSourceSpec(
        name="GPWv4",
        role="Static or slowly changing population density",
        temporal_coverage="2000, 2005, 2010, 2015, 2020",
        temporal_resolution="Five-year snapshots",
        spatial_resolution="30 arc-second global",
        format_hint="Raster or exported grid-level CSV",
        access_url="https://www.earthdata.nasa.gov/data/projects/gpw",
        expected_columns=("year", "cell_id", "population_density"),
        notes=(
            "Interpolate to yearly values or step-forward the nearest snapshot. Use as a human-pressure feature."
        ),
    ),
    DataSourceSpec(
        name="gROADS_or_GRIP",
        role="Road access and infrastructure pressure",
        temporal_coverage="Static or slowly changing",
        temporal_resolution="Static",
        spatial_resolution="Vector or raster",
        format_hint="Vector/raster exported to grid-level CSV",
        access_url="https://www.earthdata.nasa.gov/data/catalog/sedac-ciesin-sedac-groads-v1-1.0",
        expected_columns=("cell_id", "road_density_km_per_km2"),
        notes=(
            "Road density is optional for the first baseline, but it helps explain human ignition pressure."
        ),
    ),
)


def source_inventory_rows(sources: Iterable[DataSourceSpec] = FIRST_WAVE_SOURCES) -> list[dict]:
    return [asdict(source) for source in sources]


def export_source_inventory(output_path: str | Path, fmt: str = "json") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        output_path.write_text(json.dumps(source_inventory_rows(), indent=2), encoding="utf-8")
        return output_path

    if fmt == "md":
        lines = [
            "# Wildfire Risk Data Inventory",
            "",
            "| Name | Role | Coverage | Resolution | Expected Columns | URL |",
            "|---|---|---|---|---|---|",
        ]
        for source in FIRST_WAVE_SOURCES:
            lines.append(
                "| {name} | {role} | {coverage} | {resolution} | `{columns}` | {url} |".format(
                    name=source.name,
                    role=source.role,
                    coverage=source.temporal_coverage,
                    resolution=f"{source.temporal_resolution}, {source.spatial_resolution}",
                    columns=", ".join(source.expected_columns),
                    url=source.access_url,
                )
            )
            lines.append(f"|  | Notes |  |  | {source.notes} |  |")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    raise ValueError("fmt must be either `json` or `md`")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the first-wave wildfire risk data inventory")
    parser.add_argument("--output", required=True, help="Path to the output json or md file")
    parser.add_argument("--format", choices=("json", "md"), default="json")
    args = parser.parse_args()

    written_path = export_source_inventory(args.output, fmt=args.format)
    print(f"Wrote source inventory to {written_path}")


if __name__ == "__main__":
    main()
