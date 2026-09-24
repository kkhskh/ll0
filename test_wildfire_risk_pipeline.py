import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_risk.build_features import build_feature_table
from wildfire_risk.build_labels import build_label_table
from wildfire_risk.build_predictor_table import build_predictor_table
from wildfire_risk.grid import GridSpec, create_global_grid, create_month_index
from wildfire_risk.render_maps import render_monthly_risk_maps
from wildfire_risk.train_baseline import TrainConfig, select_feature_columns, train_baseline_models


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


if __name__ == "__main__":
    unittest.main()
