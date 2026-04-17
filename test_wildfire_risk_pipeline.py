import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_risk.build_features import build_feature_table
from wildfire_risk.build_labels import build_label_table
from wildfire_risk.grid import GridSpec, create_global_grid, create_month_index
from wildfire_risk.render_maps import render_monthly_risk_maps
from wildfire_risk.train_baseline import TrainConfig, train_baseline_models


class WildfireRiskPipelineTest(unittest.TestCase):
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
            self.assertGreaterEqual(results["metrics"]["classification"]["roc_auc"], 0.6)
            self.assertTrue((run_root / run_name / "models" / "metrics.json").exists())
            self.assertTrue((run_root / run_name / "models" / "run_config.json").exists())

            rendered = render_monthly_risk_maps(
                run_root / run_name / "models" / "validation_predictions.csv",
                output_dir=None,
                run_root=run_root,
                run_name=run_name,
            )
            self.assertTrue(rendered)
            self.assertTrue((run_root / run_name / "maps" / "top_risk_cells.csv").exists())
            self.assertTrue((run_root / run_name / "reports" / "render-maps_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
