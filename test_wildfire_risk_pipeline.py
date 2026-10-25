import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wildfire_risk.build_features import build_feature_table, lagged_dynamic_columns
from wildfire_risk.build_labels import build_label_table
from wildfire_risk.build_predictor_table import build_predictor_table
from wildfire_risk.download_external import firms_date_windows, glue_negative_option_values, modis_tiles_for_bbox
from wildfire_risk.grid import GridSpec, create_global_grid, create_month_index
from wildfire_risk.convert_gpw import aggregate_density_array, density_member_name
from wildfire_risk.convert_firms import aggregate_detections
from wildfire_risk.convert_groads import accumulate_road_lines, extract_gdb, gdb_directories
from wildfire_risk.convert_nasadem import (
    accumulate_terrain,
    elevation_to_slope_deg,
    parse_nasadem_tile_origin,
    stats_to_rows,
)
from wildfire_risk.convert_mcd12q1 import (
    accumulate_igbp_counts,
    assign_feature_years,
    counts_to_rows,
    igbp_group_name,
    parse_mcd12q1_filename,
    sinusoidal_xy_to_lonlat,
)
from wildfire_risk.evaluate import assign_continent, europe_subregion, score_continents
from wildfire_risk.render_maps import render_monthly_risk_maps
from wildfire_risk.train_baseline import (
    TrainConfig,
    apply_probability_calibrator,
    era5_family_for_column,
    feature_family,
    fit_probability_calibrator,
    run_era5_family_ablation,
    run_feature_ablation,
    run_yearly_backtest,
    select_feature_columns,
    train_baseline_models,
    yearly_backtest,
)


class WildfireRiskPipelineTest(unittest.TestCase):
    def test_feature_builder_and_trainer_exclude_same_month_label_leakage(self) -> None:
        labels = pd.DataFrame(
            {
                "cell_id": ["000_000", "000_000", "000_000"],
                "month": ["2024-01", "2024-02", "2024-03"],
                "year": [2024, 2024, 2024],
                "month_num": [1, 2, 3],
                "lat_center": [0.25, 0.25, 0.25],
                "lon_center": [0.25, 0.25, 0.25],
                "cell_area_km2": [1000.0, 1000.0, 1000.0],
                "burned_area_km2": [0.0, 10.0, 20.0],
                "burned_fraction": [0.0, 0.01, 0.02],
                "binary_risk_label": [0, 1, 1],
                "burned_pixel_count": [0, 5, 10],
                "burned_event_count": [0, 1, 2],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            labels_path = Path(tmp_dir) / "labels.csv"
            labels.to_csv(labels_path, index=False)

            features = build_feature_table(labels_path=labels_path, output_path=Path(tmp_dir) / "features.csv")
            self.assertNotIn("burned_pixel_count_month_anom", features.columns)
            self.assertNotIn("burned_event_count_month_anom", features.columns)

            feature_columns = select_feature_columns(features)
            self.assertNotIn("burned_pixel_count", feature_columns)
            self.assertNotIn("burned_event_count", feature_columns)
            self.assertNotIn("cell_area_km2", feature_columns)
            self.assertIn("burned_fraction_lag_1", feature_columns)

    def test_dynamic_external_predictors_are_lagged_before_training(self) -> None:
        labels = pd.DataFrame(
            {
                "cell_id": ["000_000", "000_000", "000_000", "000_000"],
                "month": ["2020-01", "2020-02", "2020-03", "2020-04"],
                "year": [2020, 2020, 2020, 2020],
                "month_num": [1, 2, 3, 4],
                "lat_center": [0.25, 0.25, 0.25, 0.25],
                "lon_center": [0.25, 0.25, 0.25, 0.25],
                "cell_area_km2": [1000.0, 1000.0, 1000.0, 1000.0],
                "burned_area_km2": [0.0, 5.0, 15.0, 0.0],
                "burned_fraction": [0.0, 0.005, 0.015, 0.0],
                "binary_risk_label": [0, 1, 1, 0],
                "burned_pixel_count": [0, 2, 6, 0],
                "burned_event_count": [0, 1, 1, 0],
            }
        )
        climate = pd.DataFrame(
            {
                "cell_id": ["000_000", "000_000", "000_000", "000_000"],
                "month": ["2020-01", "2020-02", "2020-03", "2020-04"],
                "era5_temperature_2m": [20.0, 21.0, 23.0, 22.0],
                "era5_precipitation": [100.0, 80.0, 60.0, 70.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            labels_path = tmp_path / "labels.csv"
            climate_path = tmp_path / "climate.csv"
            labels.to_csv(labels_path, index=False)
            climate.to_csv(climate_path, index=False)

            features = build_feature_table(
                labels_path=labels_path,
                output_path=tmp_path / "features.csv",
                dynamic_sources=[(str(climate_path), "era5")],
            )
            self.assertNotIn("era5_temperature_2m", features.columns)
            self.assertIn("era5_temperature_2m_lag_1", features.columns)
            self.assertIn("era5_precipitation_rollmean_3", features.columns)

            history_cols = select_feature_columns(features, feature_set="history")
            era5_cols = select_feature_columns(features, feature_set="era5")
            both_cols = select_feature_columns(features, feature_set="both")
            self.assertIn("month_sin", history_cols)
            self.assertIn("month_sin", era5_cols)
            self.assertTrue(any(column.startswith("burned_fraction_") for column in history_cols))
            self.assertFalse(any(column.startswith("era5_") for column in history_cols))
            self.assertTrue(any(column.startswith("era5_") for column in era5_cols))
            self.assertFalse(any(column.startswith("burned_fraction_") for column in era5_cols))
            self.assertEqual(set(history_cols) | set(era5_cols), set(both_cols))

    def test_build_predictor_table_from_monthly_csv(self) -> None:
        source = pd.DataFrame(
            {
                "month": ["2020-01-15", "2020-01-15", "2020-02-15"],
                "lat": [0.1, 0.3, 0.1],
                "lon": [0.1, 0.3, 0.1],
                "temperature_2m": [20.0, 24.0, 22.0],
                "soil_moisture": [100.0, 80.0, 90.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "source.csv"
            source.to_csv(source_path, index=False)

            predictors = build_predictor_table(
                input_path=source_path,
                output_path=tmp_path / "predictors.csv",
                variables=["temperature_2m", "soil_moisture"],
                resolution_deg=0.5,
            )
            self.assertIn("hot_dry_factor", predictors.columns)
            january = predictors[predictors["month"] == "2020-01"].iloc[0]
            self.assertAlmostEqual(january["temperature_2m"], 22.0)
            self.assertAlmostEqual(january["soil_moisture"], 90.0)

    def test_era5_land_netcdf_applies_product_conventions(self) -> None:
        xr = pytest.importorskip("xarray")

        # ERA5-Land ships a 0 to 360 longitude axis, so 300 degrees east is 60 degrees west.
        latitudes = np.array([10.0, 10.1, 10.2])
        longitudes = np.array([300.0, 300.1, 300.2])
        times = pd.to_datetime(["2020-01-01", "2020-02-01"])
        shape = (len(times), len(latitudes), len(longitudes))

        def plane(value: float) -> np.ndarray:
            return np.full(shape, value, dtype="float32")

        dataset = xr.Dataset(
            {
                "t2m": (("valid_time", "latitude", "longitude"), plane(300.0)),
                "tp": (("valid_time", "latitude", "longitude"), plane(0.004)),
                "swvl1": (("valid_time", "latitude", "longitude"), plane(0.2)),
                "pev": (("valid_time", "latitude", "longitude"), plane(-0.006)),
                "u10": (("valid_time", "latitude", "longitude"), plane(3.0)),
                "v10": (("valid_time", "latitude", "longitude"), plane(4.0)),
            },
            coords={"valid_time": times, "latitude": latitudes, "longitude": longitudes},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "era5_land.nc"
            dataset.to_netcdf(source_path)

            predictors = build_predictor_table(
                input_path=source_path,
                output_path=tmp_path / "era5_predictors.csv",
                input_format="era5-land-netcdf",
                variables=["t2m", "tp", "swvl1", "pev", "u10", "v10"],
                resolution_deg=0.5,
            )

            self.assertEqual(sorted(predictors["month"].unique()), ["2020-01", "2020-02"])
            # lat 10.0 -> index 200, lon -60.0 -> index 240 on the half-degree global grid.
            self.assertEqual(predictors["cell_id"].unique().tolist(), ["200_240"])

            january = predictors[predictors["month"] == "2020-01"].iloc[0]
            self.assertAlmostEqual(january["temperature_2m"], 300.0, places=3)
            self.assertAlmostEqual(january["precipitation"], 4.0, places=3)
            # Downward-flux evaporation is flipped so larger values mean more drying.
            self.assertAlmostEqual(january["potential_evaporation"], 6.0, places=3)
            self.assertAlmostEqual(january["wind_speed"], 5.0, places=3)
            self.assertAlmostEqual(january["climate_water_balance"], -2.0, places=3)

    def test_era5_land_month_filter_rejects_empty_ranges(self) -> None:
        xr = pytest.importorskip("xarray")

        dataset = xr.Dataset(
            {"t2m": (("valid_time", "latitude", "longitude"), np.full((1, 2, 2), 290.0, dtype="float32"))},
            coords={
                "valid_time": pd.to_datetime(["2020-01-01"]),
                "latitude": np.array([10.0, 10.1]),
                "longitude": np.array([20.0, 20.1]),
            },
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "era5_land.nc"
            dataset.to_netcdf(source_path)

            with self.assertRaises(ValueError):
                build_predictor_table(
                    input_path=source_path,
                    output_path=Path(tmp_dir) / "out.csv",
                    input_format="era5-land-netcdf",
                    start_month="2021-01",
                    end_month="2021-12",
                )

    def test_modis_tiles_for_bbox_matches_sinusoidal_layout(self) -> None:
        # California sits on the h08v04/h08v05 tile boundary in the MODIS sine grid.
        california = modis_tiles_for_bbox("-125,32,-114,42")
        self.assertIn("h08v04", california)
        self.assertIn("h08v05", california)
        self.assertLessEqual(len(california), 8)

        # A single equatorial degree must not fan out across many tiles. Longitude 0 is the
        # exact seam between h17 and h18, so landing on both of those is correct.
        equator = modis_tiles_for_bbox("0,0,1,1")
        self.assertIn("h18v08", equator)
        self.assertLessEqual(len(equator), 2)

        with self.assertRaises(ValueError):
            modis_tiles_for_bbox("10,0,-10,5")

        glued = glue_negative_option_values(
            ["nasadem", "--bbox", "-130,25,-100,52", "--dest", "/tmp/nasadem"]
        )
        self.assertEqual(glued[1], "--bbox=-130,25,-100,52")

    def test_firms_windows_tile_the_requested_period_without_gaps(self) -> None:
        windows = list(firms_date_windows("2020-01", "2020-02"))
        self.assertEqual(windows[0][0].isoformat(), "2020-01-01")
        self.assertEqual(sum(day_count for _, day_count in windows), 60)
        for (start, day_count), (next_start, _) in zip(windows, windows[1:]):
            self.assertEqual((next_start - start).days, day_count)
        self.assertTrue(all(1 <= day_count <= 5 for _, day_count in windows))

    def test_end_to_end_pipeline_on_synthetic_data(self) -> None:
        rng = np.random.default_rng(7)
        grid_spec = GridSpec(resolution_deg=30.0)
        grid = create_global_grid(grid_spec)
        month_index = create_month_index("2004-01", "2024-12")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            burns_path = tmp_path / "burns.csv"
            labels_path = tmp_path / "labels.csv"
            climate_path = tmp_path / "climate.csv"
            vegetation_path = tmp_path / "vegetation.csv"
            terrain_path = tmp_path / "terrain.csv"
            population_path = tmp_path / "population.csv"
            features_path = tmp_path / "features.csv"
            model_dir = tmp_path / "models"
            map_dir = tmp_path / "maps"
            run_root = tmp_path / "runs"
            run_name = "synthetic-track-b"

            hotspot_specs = [
                (-30.0, 120.0, {12, 1, 2}),
                (30.0, -120.0, {7, 8, 9}),
                (45.0, 30.0, {6, 7, 8}),
            ]
            burn_records = []
            for month in month_index["month"]:
                year = int(month[:4])
                month_num = int(month[5:7])
                trend = 1.0 + 0.015 * (year - 2004)
                for base_lat, base_lon, hot_months in hotspot_specs:
                    if month_num not in hot_months:
                        continue
                    for _ in range(3):
                        burn_records.append(
                            {
                                "date": f"{month}-15",
                                "lat": base_lat + rng.normal(0.0, 2.0),
                                "lon": base_lon + rng.normal(0.0, 2.0),
                                "burned_area_km2": max(10.0, 1200.0 * trend + rng.normal(0.0, 50.0)),
                            }
                        )

            pd.DataFrame(burn_records).to_csv(burns_path, index=False)

            labels = build_label_table(
                input_path=burns_path,
                output_path=None,
                start_month="2004-01",
                end_month="2024-12",
                resolution_deg=30.0,
                burned_fraction_threshold=1e-4,
                run_root=run_root,
                run_name=run_name,
                output_name="labels.csv",
            )
            self.assertIn("binary_risk_label", labels.columns)
            labels_path = run_root / run_name / "labels" / "labels.csv"
            self.assertTrue(labels_path.exists())
            self.assertTrue((run_root / run_name / "reports" / "build-labels_manifest.json").exists())

            merged_index = grid[["cell_id", "lat_center", "lon_center"]].merge(
                month_index[["month", "year", "month_num"]],
                how="cross",
            )

            climate = merged_index.copy()
            climate["temperature"] = (
                28.0
                - np.abs(climate["lat_center"]) * 0.25
                + 8.0 * np.cos((climate["month_num"] - 1) / 12.0 * 2.0 * np.pi)
                + 0.1 * (climate["year"] - 2004)
            )
            climate["precipitation"] = (
                120.0
                - 35.0 * np.cos((climate["month_num"] - 1) / 12.0 * 2.0 * np.pi)
                + np.abs(climate["lat_center"]) * 0.2
            )
            climate["dryness_index"] = climate["temperature"] / np.clip(climate["precipitation"], 1.0, None)
            climate.to_csv(climate_path, index=False)

            vegetation = merged_index[["cell_id", "month"]].copy()
            vegetation["ndvi_anomaly"] = (
                0.6
                - np.abs(merged_index["lat_center"]) / 120.0
                - 0.3 * np.cos((merged_index["month_num"] - 1) / 12.0 * 2.0 * np.pi)
            )
            vegetation.to_csv(vegetation_path, index=False)

            terrain = grid[["cell_id"]].copy()
            terrain["elevation_m"] = 800.0 + np.abs(grid["lat_center"]) * 10.0
            terrain["slope_deg"] = 3.0 + np.abs(grid["lon_center"]) / 60.0
            terrain.to_csv(terrain_path, index=False)

            population = grid[["cell_id"]].merge(
                pd.DataFrame({"year": sorted(month_index["year"].unique())}),
                how="cross",
            )
            population["population_density"] = 15.0 + np.abs(
                np.sin(np.deg2rad(population["year"] * 3.0))
            ) * 10.0
            population.to_csv(population_path, index=False)

            features = build_feature_table(
                labels_path=labels_path,
                output_path=None,
                dynamic_sources=[
                    (str(climate_path), "era5"),
                    (str(vegetation_path), "veg"),
                ],
                static_sources=[
                    (str(terrain_path), "terrain"),
                    (str(population_path), "population"),
                ],
                run_root=run_root,
                run_name=run_name,
                output_name="features.csv",
            )
            self.assertGreater(features.shape[1], labels.shape[1])
            self.assertTrue((run_root / run_name / "features" / "features.csv").exists())
            self.assertTrue((run_root / run_name / "reports" / "build-features_manifest.json").exists())

            results = train_baseline_models(
                feature_table_path=run_root / run_name / "features" / "features.csv",
                model_dir=None,
                config=TrainConfig(train_end_year=2020, valid_start_year=2021, valid_end_year=2024),
                run_root=run_root,
                run_name=run_name,
            )
            self.assertIn("classification", results["metrics"])
            self.assertIn("classification_raw", results["metrics"])
            self.assertIn("presentation_guidance", results["metrics"])
            self.assertIn("calibration", results["metrics"])
            self.assertIn("ranking_summary", results["metrics"])
            self.assertIn("ranking_stability", results["metrics"])
            self.assertGreaterEqual(results["metrics"]["classification"]["roc_auc"], 0.6)
            self.assertTrue((run_root / run_name / "models" / "metrics.json").exists())
            self.assertTrue((run_root / run_name / "models" / "run_config.json").exists())
            self.assertTrue((run_root / run_name / "models" / "signal_profile.json").exists())
            self.assertIn("raw_predicted_probability", results["validation_predictions"].columns)
            self.assertIn("ranking_score", results["validation_predictions"].columns)
            self.assertIn("month_risk_percentile", results["validation_predictions"].columns)
            self.assertEqual(results["metrics"]["calibration"]["strategy"]["method"], "isotonic")

            rendered = render_monthly_risk_maps(
                run_root / run_name / "models" / "validation_predictions.csv",
                output_dir=None,
                run_root=run_root,
                run_name=run_name,
            )
            self.assertTrue(rendered)
            self.assertTrue((run_root / run_name / "maps" / "top_risk_cells.csv").exists())
            self.assertTrue((run_root / run_name / "reports" / "render-maps_manifest.json").exists())
            top_risk = pd.read_csv(run_root / run_name / "maps" / "top_risk_cells.csv")
            self.assertIn("relative_risk_score", top_risk.columns)
            self.assertIn("risk_band", top_risk.columns)

    def test_feature_ablation_trains_three_arms_on_the_same_split(self) -> None:
        months = pd.period_range("2020-01", "2021-12", freq="M")
        label_rows = []
        climate_rows = []
        for cell_id, lat in (("000_000", 10.0), ("000_001", -20.0)):
            for month in months:
                month_num = int(month.month)
                year = int(month.year)
                is_hotspot = cell_id == "000_000" and month_num in {7, 8, 9}
                burned = 1 if is_hotspot and year == 2021 else 0
                label_rows.append(
                    {
                        "cell_id": cell_id,
                        "month": str(month),
                        "year": year,
                        "month_num": month_num,
                        "lat_center": lat,
                        "lon_center": 20.0,
                        "cell_area_km2": 1000.0,
                        "burned_area_km2": 50.0 * burned,
                        "burned_fraction": 0.05 * burned,
                        "binary_risk_label": burned,
                        "burned_pixel_count": 4 * burned,
                        "burned_event_count": burned,
                    }
                )
                climate_rows.append(
                    {
                        "cell_id": cell_id,
                        "month": str(month),
                        "temperature_2m": 25.0 + (8.0 if is_hotspot else 0.0),
                        "precipitation": 20.0 if is_hotspot else 80.0,
                    }
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            labels_path = tmp_path / "labels.csv"
            climate_path = tmp_path / "climate.csv"
            pd.DataFrame(label_rows).to_csv(labels_path, index=False)
            pd.DataFrame(climate_rows).to_csv(climate_path, index=False)
            features_path = tmp_path / "features.csv"
            build_feature_table(
                labels_path=labels_path,
                output_path=features_path,
                dynamic_sources=[(str(climate_path), "era5")],
            )

            ablation = run_feature_ablation(
                feature_table_path=features_path,
                model_dir=tmp_path / "models",
                config=TrainConfig(
                    train_end_year=2020,
                    valid_start_year=2021,
                    valid_end_year=2021,
                    n_estimators=20,
                    skip_yearly_backtest=True,
                    skip_regressor=True,
                ),
            )
            arms = {row["arm"]: row for row in ablation["comparison"]["arms"]}
            self.assertEqual(set(arms), {"history", "era5", "both"})
            self.assertGreater(arms["era5"]["n_features"], 0)
            self.assertGreater(arms["history"]["n_features"], 0)
            self.assertGreaterEqual(arms["both"]["n_features"], arms["history"]["n_features"])
            self.assertGreaterEqual(arms["both"]["n_features"], arms["era5"]["n_features"])
            self.assertTrue((tmp_path / "models" / "ablation_comparison.csv").exists())
            self.assertTrue((tmp_path / "models" / "ablation" / "history" / "metrics.json").exists())
            self.assertFalse((tmp_path / "models" / "ablation" / "history" / "validation_predictions.csv").exists())
            self.assertFalse((tmp_path / "models" / "classifier.pkl").exists())

    def test_gpw_density_means_land_on_half_degree_cells(self) -> None:
        names = [
            "gpw_v4_data_quality_rev11_2020_2pt5_min.tif",
            "gpw_v4_population_density_rev11_2020_2pt5_min.tif",
        ]
        self.assertTrue(density_member_name(names).startswith("gpw_v4_population_density"))

        values = np.array(
            [
                [10.0, 30.0, -9999.0, 0.0],
                [10.0, 30.0, 5.0, 5.0],
                [100.0, 300.0, 1.0, 3.0],
                [100.0, 300.0, 1.0, 3.0],
            ]
        )
        table = aggregate_density_array(
            values,
            lat_north=1.0,
            lon_west=0.0,
            pixel_height_deg=0.25,
            pixel_width_deg=0.25,
            grid_spec=GridSpec(resolution_deg=0.5),
            nodata=-9999.0,
        )
        by_cell = {row.cell_id: row for row in table.itertuples(index=False)}
        self.assertAlmostEqual(by_cell["181_360"].population_density, 20.0)
        self.assertEqual(by_cell["181_361"].population_valid_pixel_count, 3)
        self.assertAlmostEqual(by_cell["180_360"].population_density, 200.0)
        self.assertEqual(feature_family("population_density"), "population")

    def test_groads_length_lands_on_half_degree_cells(self) -> None:
        from wildfire_risk.grid import EARTH_RADIUS_KM, approximate_cell_area_km2

        grid = GridSpec(resolution_deg=0.5)
        inside = accumulate_road_lines(
            [(np.array([0.25, 0.25]), np.array([0.1, 0.4]))],
            grid,
            step_deg=0.1,
        )
        by_cell = {row.cell_id: row for row in inside.itertuples(index=False)}
        expected_km = EARTH_RADIUS_KM * np.deg2rad(0.3)
        self.assertEqual(list(by_cell), ["180_360"])
        self.assertAlmostEqual(by_cell["180_360"].roads_length_km, expected_km, places=6)
        area = float(approximate_cell_area_km2(np.array([0.25]), 0.5)[0])
        self.assertAlmostEqual(by_cell["180_360"].roads_density_km_per_km2, expected_km / area, places=6)

        crossing = accumulate_road_lines(
            [(np.array([0.4, 0.6]), np.array([0.25, 0.25]))],
            grid,
            step_deg=0.05,
        )
        crossed = {row.cell_id: row.roads_length_km for row in crossing.itertuples(index=False)}
        half = EARTH_RADIUS_KM * np.deg2rad(0.2) * np.cos(np.deg2rad(0.25)) / 2.0
        self.assertAlmostEqual(crossed["180_360"], half, places=5)
        self.assertAlmostEqual(crossed["180_361"], half, places=5)
        self.assertEqual(feature_family("roads_density_km_per_km2"), "roads")

    def test_groads_zip_finds_nested_geodatabase(self) -> None:
        names = [
            "groads-v1-global-gdb/readme.txt",
            "groads-v1-global-gdb/gROADS-v1-global.gdb/a00000001.gdbtable",
        ]
        self.assertEqual(gdb_directories(names), ["groads-v1-global-gdb/gROADS-v1-global.gdb"])
        windows = [r"gROADS-v1-global.gdb\a00000001.gdbtable"]
        self.assertEqual(gdb_directories(windows), ["gROADS-v1-global.gdb"])

        import zipfile

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "groads-v1-global-gdb.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("groads-v1-global-gdb/gROADS-v1-global.gdb/a00000001.gdbtable", b"x")
            gdb = extract_gdb(zip_path)
            self.assertTrue((gdb / "a00000001.gdbtable").is_file())
            self.assertTrue(gdb.name.endswith(".gdb"))

    def test_firms_lags_use_prior_months_only(self) -> None:
        detections = pd.DataFrame(
            {
                "latitude": [0.25, 0.30, 0.25, 0.25, 0.25],
                "longitude": [0.25, 0.30, 0.25, 0.25, 0.80],
                "acq_date": ["2020-01-15", "2020-01-16", "2020-01-17", "2020-01-18", "2020-02-01"],
                "confidence": [80, 90, 10, 80, 80],
                "type": [0, 0, 0, 2, 0],
                "frp": [10.0, 5.0, 100.0, 50.0, 7.0],
            }
        )
        monthly = aggregate_detections(detections, GridSpec(resolution_deg=0.5))
        by_key = {(row.cell_id, row.month): row for row in monthly.itertuples(index=False)}
        self.assertEqual(by_key[("180_360", "2020-01")].detection_count, 2)
        self.assertAlmostEqual(by_key[("180_360", "2020-01")].frp_sum, 15.0)
        self.assertEqual(by_key[("180_361", "2020-02")].detection_count, 1)

        panel = pd.DataFrame(
            {
                "cell_id": ["180_360", "180_360"],
                "month": ["2020-01", "2020-02"],
                "year": [2020, 2020],
                "month_num": [1, 2],
            }
        )
        history = pd.DataFrame(
            {
                "cell_id": ["180_360", "180_360", "180_360"],
                "month": ["2019-12", "2020-01", "2020-02"],
                "detection_count": [6, 4, 10],
                "frp_sum": [1.0, 2.0, 3.0],
            }
        )
        lagged, columns = lagged_dynamic_columns(panel, history, prefix="firms", lookback_months=12)
        by_month = {row.month: row for row in lagged.itertuples(index=False)}
        self.assertNotIn("firms_detection_count", columns)
        self.assertEqual(by_month["2020-01"].firms_detection_count_lag_1, 6.0)
        self.assertAlmostEqual(by_month["2020-01"].firms_detection_count_rollmean_12, 0.5)
        self.assertEqual(by_month["2020-02"].firms_detection_count_lag_1, 4.0)
        self.assertAlmostEqual(by_month["2020-02"].firms_detection_count_rollmean_12, 10.0 / 12.0)
        self.assertAlmostEqual(by_month["2020-02"].firms_detection_log_lag_1, np.log1p(4.0))
        self.assertEqual(feature_family("firms_detection_count_lag_1"), "firms")

    def test_nasadem_tile_name_and_slope_aggregate_to_half_degree_cells(self) -> None:
        self.assertEqual(parse_nasadem_tile_origin("NASADEM_HGT_n38w119.zip"), (38.0, -119.0))
        self.assertEqual(parse_nasadem_tile_origin("s16e012.hgt"), (-16.0, 12.0))

        flat = np.full((9, 9), 1000.0)
        ramp = np.tile(np.linspace(0.0, 80_000.0, 9), (9, 1))
        flat_slope = elevation_to_slope_deg(flat, lat_south=38.0, stride=1)
        ramp_slope = elevation_to_slope_deg(ramp, lat_south=38.0, stride=1)
        self.assertLess(float(np.nanmean(flat_slope)), 1.0)
        self.assertGreater(float(np.nanmean(ramp_slope)), float(np.nanmean(flat_slope)))

        spec = GridSpec(resolution_deg=0.5)
        stats: dict[str, np.ndarray] = {}
        lats = np.full(4, 38.25)
        lons = np.full(4, -118.75)
        accumulate_terrain(
            stats,
            elev=np.array([1000.0, 1100.0, 1200.0, 1300.0]),
            slope=np.array([2.0, 4.0, 6.0, 8.0]),
            lats=lats,
            lons=lons,
            grid_spec=spec,
        )
        rows = stats_to_rows(stats)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["has_terrain"], 1)
        self.assertAlmostEqual(rows[0]["elevation_mean_m"], 1150.0)
        self.assertAlmostEqual(rows[0]["slope_mean_deg"], 5.0)

    def test_mcd12q1_aggregates_igbp_fractions_and_shifts_to_next_year(self) -> None:
        self.assertEqual(parse_mcd12q1_filename("MCD12Q1.A2023001.h27v12.061.2024233123456.hdf"), (2023, "h27v12"))
        lons, lats = sinusoidal_xy_to_lonlat(np.array([0.0]), np.array([0.0]))
        self.assertAlmostEqual(float(lons[0]), 0.0, places=6)
        self.assertAlmostEqual(float(lats[0]), 0.0, places=6)
        self.assertEqual(igbp_group_name(2), "forest")
        self.assertEqual(igbp_group_name(10), "grass")
        self.assertEqual(igbp_group_name(12), "crop")

        spec = GridSpec(resolution_deg=0.5)
        class_counts: dict[str, np.ndarray] = {}
        accumulate_igbp_counts(
            class_counts,
            classes=np.array([10, 10, 10, 2, 255, 0], dtype=int),
            lats=np.array([-25.1, -25.1, -25.1, -25.1, -25.1, -25.1]),
            lons=np.array([135.1, 135.1, 135.1, 135.1, 135.1, 135.1]),
            grid_spec=spec,
        )
        rows = counts_to_rows(class_counts, product_year=2023)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["product_year"], 2023)
        self.assertEqual(row["land_cover_majority"], 10)
        self.assertEqual(row["land_cover_valid_pixel_count"], 5)
        self.assertAlmostEqual(row["frac_grass"], 0.6)
        self.assertAlmostEqual(row["frac_forest"], 0.2)
        self.assertAlmostEqual(row["frac_water"], 0.2)

        shifted = assign_feature_years(pd.DataFrame(rows), year_offset=1, fill_first_year=True)
        self.assertEqual(sorted(shifted["year"].unique().tolist()), [2023, 2024])
        self.assertTrue(((shifted["year"] == 2024) & (shifted["product_year"] == 2023)).any())
        self.assertTrue(((shifted["year"] == 2023) & (shifted["product_year"] == 2023)).any())

        with tempfile.TemporaryDirectory() as tmp_dir:
            labels_path = Path(tmp_dir) / "labels.csv"
            landcover_path = Path(tmp_dir) / "landcover.csv"
            features_path = Path(tmp_dir) / "features.csv"
            pd.DataFrame(
                {
                    "cell_id": ["129_630", "129_630"],
                    "month": ["2023-12", "2024-01"],
                    "year": [2023, 2024],
                    "month_num": [12, 1],
                    "lat_center": [-25.25, -25.25],
                    "lon_center": [135.25, 135.25],
                    "cell_area_km2": [1000.0, 1000.0],
                    "burned_area_km2": [0.0, 10.0],
                    "burned_fraction": [0.0, 0.01],
                    "binary_risk_label": [0, 1],
                    "burned_pixel_count": [0, 4],
                    "burned_event_count": [0, 1],
                }
            ).to_csv(labels_path, index=False)
            shifted[["cell_id", "year", "land_cover_majority", "frac_grass", "frac_forest"]].to_csv(
                landcover_path, index=False
            )
            features = build_feature_table(
                labels_path=labels_path,
                output_path=features_path,
                static_sources=[(str(landcover_path), "landcover")],
                restrict_to_dynamic_cells=False,
            )
            jan = features[features["month"] == "2024-01"].iloc[0]
            dec = features[features["month"] == "2023-12"].iloc[0]
            self.assertAlmostEqual(float(jan["landcover_frac_grass"]), 0.6, places=5)
            self.assertAlmostEqual(float(dec["landcover_frac_grass"]), 0.6, places=5)
            self.assertIn("landcover", {feature_family(col) for col in features.columns})

    def test_continent_boxes_label_known_fire_regions(self) -> None:
        points = pd.DataFrame(
            {
                "name": [
                    "lagos",
                    "chicago",
                    "sao_paulo",
                    "sydney",
                    "paris",
                    "beijing",
                    "jakarta",
                ],
                "lat_center": [6.5, 41.9, -23.5, -33.9, 48.9, 39.9, -6.2],
                "lon_center": [3.4, -87.6, -46.6, 151.2, 2.3, 116.4, 106.8],
            }
        )
        labels = assign_continent(points["lat_center"], points["lon_center"])
        self.assertEqual(
            list(labels),
            ["africa", "north_america", "south_america", "australia", "europe", "asia", "asia"],
        )

        scored = pd.DataFrame(
            {
                "lat_center": [6.5, 6.5, 41.9, 41.9],
                "lon_center": [3.4, 3.4, -87.6, -87.6],
                "binary_risk_label": [1, 0, 0, 1],
                "ranking_score": [0.9, 0.1, 0.2, 0.8],
                "predicted_probability": [0.4, 0.02, 0.03, 0.3],
            }
        )
        rows = {row["region"]: row for row in score_continents(scored)}
        self.assertIn("global", rows)
        self.assertIn("africa", rows)
        self.assertIn("north_america", rows)
        self.assertEqual(rows["africa"]["n_fires"], 1)
        self.assertEqual(rows["north_america"]["n_fires"], 1)

    def test_europe_box_splits_south_from_north_and_keeps_the_africa_fringe(self) -> None:
        points = pd.DataFrame(
            {
                "lat_center": [37.39, 44.84, 48.9, 55.76, 36.75],
                "lon_center": [-5.98, -0.58, 2.3, 37.62, 3.06],
            }
        )
        continent = assign_continent(points["lat_center"], points["lon_center"])
        self.assertEqual(list(continent), ["africa", "europe", "europe", "europe", "africa"])
        parts = europe_subregion(points["lat_center"], points["lon_center"], continent)
        self.assertEqual(list(parts), ["africa_fringe", "europe_south", "europe_north", "europe_north", "africa_fringe"])

        scored = pd.DataFrame(
            {
                "lat_center": [44.84, 44.84, 52.5, 52.5, 37.39],
                "lon_center": [-0.58, -0.58, 13.4, 13.4, -5.98],
                "binary_risk_label": [1, 0, 0, 1, 1],
                "ranking_score": [0.9, 0.2, 0.1, 0.8, 0.7],
                "predicted_probability": [0.4, 0.05, 0.02, 0.3, 0.2],
            }
        )
        rows = {row["region"]: row for row in score_continents(scored)}
        self.assertEqual(rows["europe"]["n_fires"], 2)
        self.assertEqual(rows["europe_south"]["n_fires"], 1)
        self.assertEqual(rows["europe_north"]["n_fires"], 1)
        self.assertEqual(rows["africa_fringe"]["n_fires"], 1)
        self.assertEqual(rows["africa"]["n_fires"], 1)

    def test_isotonic_calibrator_is_monotonic_and_bounded(self) -> None:
        raw = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
        y_true = pd.Series([0, 0, 0, 1, 1, 1])
        calibrator, details = fit_probability_calibrator(raw, y_true, method="isotonic")
        self.assertEqual(details["method"], "isotonic")
        calibrated = apply_probability_calibrator(calibrator, raw)
        self.assertTrue(np.all(calibrated >= 0.0))
        self.assertTrue(np.all(calibrated <= 1.0))
        self.assertGreaterEqual(calibrated[-1], calibrated[0])

    def test_era5_family_filters_drop_one_variable_group(self) -> None:
        self.assertEqual(era5_family_for_column("era5_temperature_2m_lag_1"), "temperature")
        self.assertEqual(era5_family_for_column("era5_precipitation_rollmean_3_month_anom"), "precip")
        self.assertEqual(era5_family_for_column("era5_soil_moisture_lag_1"), "soil_moisture")
        self.assertEqual(era5_family_for_column("era5_wind_speed_rollmax_12"), "wind")
        self.assertEqual(era5_family_for_column("era5_hot_dry_factor_lag_1"), "hot_dry")
        self.assertEqual(era5_family_for_column("era5_climate_water_balance_rollmean_3"), "hot_dry")
        self.assertIsNone(era5_family_for_column("burned_fraction_lag_1"))

        features = pd.DataFrame(
            {
                "cell_id": ["000_000"],
                "month": ["2020-01"],
                "year": [2020],
                "binary_risk_label": [0],
                "burned_fraction": [0.0],
                "month_sin": [0.0],
                "burned_fraction_lag_1": [0.1],
                "era5_temperature_2m_lag_1": [300.0],
                "era5_precipitation_lag_1": [2.0],
                "era5_soil_moisture_lag_1": [0.2],
                "era5_wind_speed_lag_1": [4.0],
                "era5_hot_dry_factor_lag_1": [12.0],
            }
        )
        dropped_temp = select_feature_columns(features, feature_set="both", drop_era5_families=("temperature",))
        self.assertNotIn("era5_temperature_2m_lag_1", dropped_temp)
        self.assertIn("era5_precipitation_lag_1", dropped_temp)
        self.assertIn("burned_fraction_lag_1", dropped_temp)

        kept_hot_dry = select_feature_columns(
            features,
            feature_set="both",
            keep_era5_families=("hot_dry",),
        )
        self.assertIn("era5_hot_dry_factor_lag_1", kept_hot_dry)
        self.assertNotIn("era5_temperature_2m_lag_1", kept_hot_dry)
        self.assertIn("burned_fraction_lag_1", kept_hot_dry)

    def test_era5_family_ablation_skips_missing_families_and_writes_keep_list(self) -> None:
        months = pd.period_range("2020-01", "2021-12", freq="M")
        label_rows = []
        climate_rows = []
        for cell_id, lat in (("000_000", 10.0), ("000_001", -20.0)):
            for month in months:
                month_num = int(month.month)
                year = int(month.year)
                is_hotspot = cell_id == "000_000" and month_num in {7, 8, 9}
                burned = 1 if is_hotspot and year == 2021 else 0
                label_rows.append(
                    {
                        "cell_id": cell_id,
                        "month": str(month),
                        "year": year,
                        "month_num": month_num,
                        "lat_center": lat,
                        "lon_center": 20.0,
                        "cell_area_km2": 1000.0,
                        "burned_area_km2": 50.0 * burned,
                        "burned_fraction": 0.05 * burned,
                        "binary_risk_label": burned,
                        "burned_pixel_count": 4 * burned,
                        "burned_event_count": burned,
                    }
                )
                climate_rows.append(
                    {
                        "cell_id": cell_id,
                        "month": str(month),
                        "temperature_2m": 25.0 + (8.0 if is_hotspot else 0.0),
                        "precipitation": 20.0 if is_hotspot else 80.0,
                    }
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            labels_path = tmp_path / "labels.csv"
            climate_path = tmp_path / "climate.csv"
            pd.DataFrame(label_rows).to_csv(labels_path, index=False)
            pd.DataFrame(climate_rows).to_csv(climate_path, index=False)
            features_path = tmp_path / "features.csv"
            build_feature_table(
                labels_path=labels_path,
                output_path=features_path,
                dynamic_sources=[(str(climate_path), "era5")],
            )

            ablation = run_era5_family_ablation(
                feature_table_path=features_path,
                model_dir=tmp_path / "models",
                config=TrainConfig(
                    train_end_year=2020,
                    valid_start_year=2021,
                    valid_end_year=2021,
                    n_estimators=10,
                    skip_yearly_backtest=True,
                    skip_regressor=True,
                ),
            )
            arms = {row["arm"] for row in ablation["comparison"]["arms"]}
            self.assertIn("both", arms)
            self.assertIn("drop_temperature", arms)
            self.assertIn("drop_precip", arms)
            self.assertNotIn("drop_wind", arms)
            self.assertTrue((tmp_path / "models" / "era5_family_ablation.csv").exists())
            self.assertIn("keep", ablation["recommendation"])
            self.assertIn("drop", ablation["recommendation"])

    def test_yearly_backtest_refits_years_before_the_holdout_window(self) -> None:
        rows = []
        for year in (2020, 2021, 2022, 2023):
            for month in range(1, 5):
                rows.append(
                    {
                        "year": year,
                        "month_num": month,
                        "lat_center": -25.0,
                        "lon_center": 133.0,
                        "binary_risk_label": 1 if month == 1 else 0,
                        "burned_fraction_lag_1": 1.0 if month == 1 else 0.0,
                    }
                )
                rows.append(
                    {
                        "year": year,
                        "month_num": month,
                        "lat_center": 0.0,
                        "lon_center": 20.0,
                        "binary_risk_label": 0,
                        "burned_fraction_lag_1": 0.0,
                    }
                )
        frame = pd.DataFrame(rows)
        config = TrainConfig(
            train_end_year=2023,
            valid_start_year=2024,
            valid_end_year=2025,
            backtest_start_year=2022,
            backtest_end_year=2023,
            n_estimators=5,
        )
        scores = yearly_backtest(frame, ["burned_fraction_lag_1"], config)
        self.assertEqual(list(scores), ["2022", "2023"])
        self.assertEqual(scores["2022"]["n_train_rows"], 16)
        self.assertEqual(scores["2022"]["train_year_max"], 2021)
        self.assertEqual(scores["2022"]["n_test_rows"], 8)
        self.assertEqual(scores["2023"]["n_train_rows"], 24)
        self.assertEqual(scores["2023"]["train_year_max"], 2022)
        regions = {row["region"] for row in scores["2022"]["regions"]}
        self.assertIn("global", regions)
        self.assertIn("australia", regions)
        self.assertIn("africa", regions)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            features_path = tmp_path / "features.csv"
            frame.to_csv(features_path, index=False)
            results = run_yearly_backtest(
                feature_table_path=features_path,
                config=config,
                model_dir=tmp_path / "reports",
            )
            self.assertTrue(results["json_path"].exists())
            self.assertTrue(results["csv_path"].exists())
            self.assertFalse((tmp_path / "reports" / "classifier.pkl").exists())
            written = pd.read_csv(results["csv_path"])
            self.assertEqual(set(written["year"]), {2022, 2023})
            self.assertIn("australia", set(written["region"]))


if __name__ == "__main__":
    unittest.main()
