"""Offline tests for worldwide wildfire asset exposure.

These tests never read the 4 GB validation_predictions.csv and do not require
geopandas. The pandas 0.5° grid join is the production path.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from wildfire_risk.asset_exposure import run_asset_exposure
from wildfire_risk.asset_scale import (
    FailureCache,
    PROCESS_ORDER,
    RANKED_COLUMNS,
    cell_ids_for_points,
    overlay_assets_on_predictions,
    overlay_region_predictions,
    peak_month_for_region,
    require_predictions_file,
    stream_predictions_by_continent,
    unique_cell_ids_along_line,
)
from wildfire_risk.evaluate import assign_continent
from wildfire_risk.grid import GridSpec, assign_points_to_grid


def _prediction_row(
    lat: float,
    lon: float,
    month: str,
    ranking_score: float,
    binary_risk_label: int = 0,
    risk_band: str = "monitor",
    predicted_probability: float | None = None,
    month_risk_percentile: float | None = None,
) -> dict:
    spec = GridSpec()
    assigned = assign_points_to_grid(
        pd.DataFrame({"lat": [lat], "lon": [lon]}),
        spec,
        lat_col="lat",
        lon_col="lon",
    ).iloc[0]
    year, month_num = month.split("-")
    score = float(ranking_score)
    return {
        "cell_id": assigned["cell_id"],
        "month": month,
        "year": int(year),
        "month_num": int(month_num),
        "lat_center": float(lat),
        "lon_center": float(lon),
        "binary_risk_label": int(binary_risk_label),
        "predicted_probability": float(predicted_probability if predicted_probability is not None else score),
        "ranking_score": score,
        "risk_percentile": score,
        "month_risk_percentile": float(month_risk_percentile if month_risk_percentile is not None else score),
        "risk_band": risk_band,
    }


class WildfireAssetExposureTest(unittest.TestCase):
    def test_cell_id_matches_grid_py_in_australia_and_africa(self) -> None:
        spec = GridSpec()
        australia = pd.DataFrame({"lat": [-33.9], "lon": [151.2]})
        africa = pd.DataFrame({"lat": [6.5], "lon": [3.4]})
        expected_au = assign_points_to_grid(australia, spec, lat_col="lat", lon_col="lon")["cell_id"].iloc[0]
        expected_af = assign_points_to_grid(africa, spec, lat_col="lat", lon_col="lon")["cell_id"].iloc[0]
        self.assertEqual(cell_ids_for_points([-33.9], [151.2], spec=spec), [expected_au])
        self.assertEqual(cell_ids_for_points([6.5], [3.4], spec=spec), [expected_af])
        self.assertEqual(assign_continent([-33.9], [151.2])[0], "australia")
        self.assertEqual(assign_continent([6.5], [3.4])[0], "africa")
        self.assertEqual(PROCESS_ORDER[0], "africa")

    def test_line_crossing_two_cells_keeps_higher_ranking_score(self) -> None:
        spec = GridSpec()
        # Adjacent 0.5° cells in the Africa box: lon 3.25 and 3.75 at lat 6.75.
        low = _prediction_row(6.75, 3.25, "2024-07", 0.20, risk_band="elevated")
        high = _prediction_row(6.75, 3.75, "2024-07", 0.91, risk_band="extreme")
        self.assertNotEqual(low["cell_id"], high["cell_id"])
        predictions = pd.DataFrame([low, high])
        line_cells = unique_cell_ids_along_line(
            lons=[3.10, 3.90],
            lats=[6.75, 6.75],
            spec=spec,
            step_deg=0.1,
        )
        self.assertIn(low["cell_id"], line_cells)
        self.assertIn(high["cell_id"], line_cells)
        assets = pd.DataFrame(
            [
                {
                    "asset_id": "line_cross",
                    "asset_name": "cross-cell test line",
                    "asset_type": "transmission_line",
                    "region": "africa",
                    "cell_ids": line_cells,
                    "asset_source": "test",
                }
            ]
        )
        ranked = overlay_assets_on_predictions(assets, predictions, spec=spec)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked.iloc[0]["cell_id"], high["cell_id"])
        self.assertAlmostEqual(float(ranked.iloc[0]["predicted_probability"]), 0.91)
        self.assertEqual(int(ranked.iloc[0]["n_extreme_cells"]), 1)
        self.assertGreaterEqual(int(ranked.iloc[0]["n_high_cells"]), 0)
        self.assertEqual(list(ranked.columns), list(RANKED_COLUMNS))
        self.assertIn("this month's burned-area probability", ranked.iloc[0]["reason"])

    def test_per_region_peak_month_does_not_let_africa_overwrite_australia(self) -> None:
        africa_july = _prediction_row(6.5, 3.4, "2024-07", 0.95, binary_risk_label=1, risk_band="extreme")
        africa_jan = _prediction_row(6.5, 3.4, "2024-01", 0.10, binary_risk_label=0, risk_band="monitor")
        aus_jan = _prediction_row(-33.9, 151.2, "2024-01", 0.80, binary_risk_label=1, risk_band="high")
        aus_july = _prediction_row(-33.9, 151.2, "2024-07", 0.05, binary_risk_label=0, risk_band="monitor")
        africa_pred = pd.DataFrame([africa_july, africa_jan])
        aus_pred = pd.DataFrame([aus_jan, aus_july])
        combined = pd.concat([africa_pred, aus_pred], ignore_index=True)
        self.assertEqual(peak_month_for_region(combined), "2024-07")
        self.assertEqual(peak_month_for_region(africa_pred), "2024-07")
        self.assertEqual(peak_month_for_region(aus_pred), "2024-01")

        africa_assets = pd.DataFrame(
            [
                {
                    "asset_id": "plant_lagos",
                    "asset_name": "Lagos test plant",
                    "asset_type": "plant",
                    "region": "africa",
                    "cell_ids": [africa_july["cell_id"]],
                    "asset_source": "test",
                }
            ]
        )
        aus_assets = pd.DataFrame(
            [
                {
                    "asset_id": "plant_sydney",
                    "asset_name": "Sydney test plant",
                    "asset_type": "plant",
                    "region": "australia",
                    "cell_ids": [aus_jan["cell_id"]],
                    "asset_source": "test",
                }
            ]
        )
        africa_ranked, africa_month = overlay_region_predictions(africa_pred, africa_assets, "africa")
        aus_ranked, aus_month = overlay_region_predictions(aus_pred, aus_assets, "australia")
        self.assertEqual(africa_month, "2024-07")
        self.assertEqual(aus_month, "2024-01")
        self.assertEqual(africa_ranked.iloc[0]["month"], "2024-07")
        self.assertEqual(aus_ranked.iloc[0]["month"], "2024-01")
        self.assertAlmostEqual(float(africa_ranked.iloc[0]["predicted_probability"]), 0.95)
        self.assertAlmostEqual(float(aus_ranked.iloc[0]["predicted_probability"]), 0.80)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pred_path = Path(tmp_dir) / "validation_predictions.csv"
            combined.to_csv(pred_path, index=False)
            tiles = stream_predictions_by_continent(pred_path, Path(tmp_dir) / "tiles")
            self.assertIn("africa", tiles)
            self.assertIn("australia", tiles)
            africa_tile = pd.read_csv(tiles["africa"])
            aus_tile = pd.read_csv(tiles["australia"])
            self.assertEqual(peak_month_for_region(africa_tile), "2024-07")
            self.assertEqual(peak_month_for_region(aus_tile), "2024-01")

    def test_missing_predictions_file_raises_file_not_found(self) -> None:
        missing = Path("/tmp/does-not-exist-wildfire-asset-exposure.csv")
        with self.assertRaises(FileNotFoundError):
            require_predictions_file(missing)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                run_asset_exposure(
                    predictions_path=missing,
                    run_root=tmp_dir,
                    scope="global",
                    cache_dir=Path(tmp_dir) / "cache",
                )

    def test_global_run_writes_layout_without_network(self) -> None:
        africa = _prediction_row(6.5, 3.4, "2024-07", 0.9, binary_risk_label=1, risk_band="extreme")
        aus = _prediction_row(-33.9, 151.2, "2024-01", 0.7, binary_risk_label=1, risk_band="high")
        plants = pd.DataFrame(
            [
                {
                    "asset_id": "WRI1000001",
                    "asset_name": "Lagos test plant",
                    "asset_type": "plant",
                    "region": "africa",
                    "cell_ids": [africa["cell_id"]],
                    "asset_source": "wri_gppd",
                },
                {
                    "asset_id": "WRI1000002",
                    "asset_name": "Sydney test plant",
                    "asset_type": "plant",
                    "region": "australia",
                    "cell_ids": [aus["cell_id"]],
                    "asset_source": "wri_gppd",
                },
            ]
        )
        from wildfire_risk.asset_scale import SourceStatus

        empty_lines = pd.DataFrame()
        wb_status = SourceStatus(
            source="world_bank_osm_power_tmm",
            url="https://energydata.info/dataset/e0e878a7-7712-465d-a4c2-69b2b3518513/resource/e0f33c3e-6dab-41d1-919d-d266f300d0bc/download/osm_power_tmm.zip",
            status="http_403",
            note="HTTP 403",
            license="CC-BY-4.0",
        )
        plant_status = SourceStatus(
            source="wri_gppd",
            url="https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv",
            status="ok",
            n_assets=2,
            license="CC-BY-4.0",
        )
        parquet_status = SourceStatus(
            source="osm_geofabrik_power_line_parquet",
            url="https://parquetry.geomermaids.com/gmwid/latest/power_line.parquet",
            status="missing",
            note="skipped in offline test",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            pred_path = tmp / "validation_predictions.csv"
            pd.DataFrame([africa, aus]).to_csv(pred_path, index=False)
            with patch("wildfire_risk.asset_exposure.load_world_bank_osm_lines", return_value=(empty_lines, wb_status)), patch(
                "wildfire_risk.asset_exposure.load_gppd_plants", return_value=(plants, plant_status)
            ), patch(
                "wildfire_risk.asset_scale.load_osm_parquet_lines_for_bbox",
                return_value=(empty_lines, parquet_status),
            ), patch(
                "wildfire_risk.asset_scale.load_hifld_assets",
                return_value=(empty_lines, []),
            ):
                result = run_asset_exposure(
                    predictions_path=pred_path,
                    run_root=tmp / "runs",
                    scope="global",
                    cache_dir=tmp / "cache",
                    run_name="wildfire-asset-exposure-global",
                )
            run_root = Path(result["run_root"])
            self.assertTrue((run_root / "README.md").is_file())
            self.assertTrue((run_root / "reports" / "coverage_report.csv").is_file())
            self.assertTrue((run_root / "reports" / "ranked_asset_exposure_global.csv").is_file())
            self.assertTrue((run_root / "reports" / "demo_utility_portfolio_global.csv").is_file())
            self.assertTrue((run_root / "reports" / "asset-exposure_manifest.json").is_file())
            global_ranked = pd.read_csv(run_root / "reports" / "ranked_asset_exposure_global.csv")
            months = dict(zip(global_ranked["asset_id"], global_ranked["month"]))
            self.assertEqual(months["WRI1000001"], "2024-07")
            self.assertEqual(months["WRI1000002"], "2024-01")
            readme = (run_root / "README.md").read_text(encoding="utf-8")
            self.assertIn("this month's burned-area probability", readme)
            self.assertIn("not a customer book", readme)

    def test_failure_cache_does_not_retry_a_403(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = FailureCache.load(Path(tmp_dir) / "source_fetch_failures.json")
            url = "https://energydata.info/dataset/e0e878a7-7712-465d-a4c2-69b2b3518513/resource/e0f33c3e-6dab-41d1-919d-d266f300d0bc/download/osm_power_tmm.zip"
            cache.remember(url, status=403, note="HTTP 403")
            blocked = cache.blocked(url)
            self.assertIsNotNone(blocked)
            self.assertEqual(int(blocked["status"]), 403)
            reloaded = FailureCache.load(Path(tmp_dir) / "source_fetch_failures.json")
            self.assertIsNotNone(reloaded.blocked(url))


if __name__ == "__main__":
    unittest.main()
