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

from wildfire_risk.asset_exposure import run_asset_exposure, write_synthetic_demo_run
from wildfire_risk.asset_scale import (
    DEMO_COLUMNS,
    FailureCache,
    PROCESS_ORDER,
    RANKED_COLUMNS,
    cell_ids_for_points,
    cell_weights_along_line,
    download_url,
    load_osm_parquet_lines_for_bbox,
    overlay_assets_on_predictions,
    overlay_region_predictions,
    peak_month_for_region,
    require_predictions_file,
    stream_predictions_by_continent,
    unique_cell_ids_along_line,
)
from wildfire_risk.asset_scores import portfolio_concentration
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
        self.assertIn("length_weighted_score", ranked.columns)
        self.assertIn("score_range", ranked.columns)
        self.assertIn("inspect_rank", ranked.columns)
        self.assertGreater(float(ranked.iloc[0]["score_range"]), 0.0)

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
            for column in RANKED_COLUMNS:
                self.assertIn(column, global_ranked.columns)
            demo = pd.read_csv(run_root / "reports" / "demo_utility_portfolio_global.csv")
            for column in DEMO_COLUMNS:
                self.assertIn(column, demo.columns)
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

    def test_download_url_does_not_retry_cached_403(self) -> None:
        url = "https://energydata.info/dataset/e0e878a7-7712-465d-a4c2-69b2b3518513/resource/e0f33c3e-6dab-41d1-919d-d266f300d0bc/download/osm_power_tmm.zip"
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = FailureCache.load(Path(tmp_dir) / "source_fetch_failures.json")
            cache.remember(url, status=403, note="HTTP 403")
            with patch("wildfire_risk.asset_scale._http_session") as session_factory:
                status = download_url(url, Path(tmp_dir) / "osm_power_tmm.zip", cache)
            session_factory.assert_not_called()
            self.assertEqual(status.status, "cached_failure")
            self.assertIn("403", status.note)

    def test_parquet_loader_reads_region_cache_without_network(self) -> None:
        spec = GridSpec()
        low = _prediction_row(6.75, 3.25, "2024-07", 0.2)
        cached = pd.DataFrame(
            [
                {
                    "asset_id": "osm_line_1",
                    "asset_name": "cached line",
                    "asset_type": "transmission_line",
                    "cell_ids": [low["cell_id"]],
                    "cell_weights": {low["cell_id"]: 4},
                    "asset_source": "osm_geofabrik_power_line_parquet",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            cached.to_pickle(cache_dir / "osm_power_line_africa.pkl")
            with patch("wildfire_risk.asset_scale.duckdb", create=True):
                assets, status = load_osm_parquet_lines_for_bbox(
                    (-35.0, 37.5, -18.0, 52.0),
                    spec=spec,
                    cache_dir=cache_dir,
                    region="africa",
                )
            self.assertEqual(status.status, "ok")
            self.assertEqual(len(assets), 1)
            self.assertEqual(assets.iloc[0]["asset_id"], "osm_line_1")
            self.assertIn("ODbL", status.note)


class AssetScoreFormulasTest(unittest.TestCase):
    def test_length_weighted_score_below_max_on_90_10_line(self) -> None:
        spec = GridSpec()
        low = _prediction_row(6.75, 3.25, "2024-07", 0.20, risk_band="elevated")
        high = _prediction_row(6.75, 3.75, "2024-07", 0.91, risk_band="extreme")
        self.assertNotEqual(low["cell_id"], high["cell_id"])
        assets = pd.DataFrame(
            [
                {
                    "asset_id": "line_90_10",
                    "asset_name": "mostly quiet line",
                    "asset_type": "transmission_line",
                    "region": "africa",
                    "cell_ids": [low["cell_id"], high["cell_id"]],
                    "cell_weights": {low["cell_id"]: 90, high["cell_id"]: 10},
                    "asset_source": "test",
                }
            ]
        )
        ranked = overlay_assets_on_predictions(assets, pd.DataFrame([low, high]), spec=spec)
        self.assertEqual(len(ranked), 1)
        row = ranked.iloc[0]
        self.assertEqual(row["cell_id"], high["cell_id"])
        self.assertAlmostEqual(float(row["ranking_score"]), 0.91)
        expected_len = 0.20 * 0.9 + 0.91 * 0.1
        self.assertAlmostEqual(float(row["length_weighted_score"]), expected_len)
        self.assertLess(float(row["length_weighted_score"]), float(row["ranking_score"]))
        self.assertEqual(int(row["n_cells"]), 2)
        self.assertEqual(int(row["n_extreme_cells"]), 1)
        self.assertEqual(int(row["n_high_cells"]), 0)
        self.assertAlmostEqual(float(row["extreme_cell_fraction"]), 0.5)
        self.assertAlmostEqual(float(row["score_range"]), 0.71)
        self.assertGreater(float(row["score_std"]), 0.0)
        self.assertEqual(int(row["inspect_rank"]), 1)
        self.assertIn("not a confidence interval", row["reason"])

    def test_sampled_line_geometry_also_separates_s_max_and_s_len(self) -> None:
        spec = GridSpec()
        low = _prediction_row(6.75, 3.25, "2024-07", 0.20, risk_band="elevated")
        high = _prediction_row(6.75, 3.75, "2024-07", 0.91, risk_band="extreme")
        cell_ids, weights = cell_weights_along_line([3.05, 3.51], [6.75, 6.75], spec=spec, step_deg=0.1)
        self.assertIn(low["cell_id"], weights)
        self.assertIn(high["cell_id"], weights)
        self.assertGreater(weights[low["cell_id"]], weights[high["cell_id"]])
        assets = pd.DataFrame(
            [
                {
                    "asset_id": "line_geom",
                    "asset_name": "sampled 90-ish line",
                    "asset_type": "transmission_line",
                    "region": "africa",
                    "cell_ids": cell_ids,
                    "cell_weights": weights,
                    "asset_source": "test",
                }
            ]
        )
        ranked = overlay_assets_on_predictions(assets, pd.DataFrame([low, high]), spec=spec).iloc[0]
        self.assertAlmostEqual(float(ranked["ranking_score"]), 0.91)
        self.assertLess(float(ranked["length_weighted_score"]), 0.91)
        self.assertGreater(float(ranked["score_range"]), 0.0)

    def test_point_asset_has_zero_score_range(self) -> None:
        plant = _prediction_row(-33.9, 151.2, "2024-01", 0.80, risk_band="high")
        assets = pd.DataFrame(
            [
                {
                    "asset_id": "plant_sydney",
                    "asset_name": "Sydney test plant",
                    "asset_type": "plant",
                    "region": "australia",
                    "cell_ids": [plant["cell_id"]],
                    "cell_weights": {plant["cell_id"]: 1},
                    "asset_source": "test",
                }
            ]
        )
        ranked = overlay_assets_on_predictions(assets, pd.DataFrame([plant])).iloc[0]
        self.assertAlmostEqual(float(ranked["ranking_score"]), float(ranked["length_weighted_score"]))
        self.assertAlmostEqual(float(ranked["score_range"]), 0.0)
        self.assertAlmostEqual(float(ranked["score_std"]), 0.0)
        self.assertEqual(int(ranked["n_cells"]), 1)
        self.assertAlmostEqual(float(ranked["calibrated_probability"]), 0.80)

    def test_demo_herfindahl_and_high_share(self) -> None:
        africa = _prediction_row(6.5, 3.4, "2024-07", 0.9, risk_band="extreme")
        aus = _prediction_row(-33.9, 151.2, "2024-01", 0.7, risk_band="high")
        na = _prediction_row(37.4, -122.1, "2024-09", 0.1, risk_band="monitor")
        ranked = pd.DataFrame(
            [
                {"asset_id": "a1", "cell_id": africa["cell_id"], "risk_band": "extreme", "ranking_score": 0.9},
                {"asset_id": "a2", "cell_id": africa["cell_id"], "risk_band": "extreme", "ranking_score": 0.9},
                {"asset_id": "a3", "cell_id": aus["cell_id"], "risk_band": "high", "ranking_score": 0.7},
                {"asset_id": "a4", "cell_id": na["cell_id"], "risk_band": "monitor", "ranking_score": 0.1},
            ]
        )
        stats = portfolio_concentration(ranked)
        self.assertEqual(stats["n_demo_assets"], 4.0)
        self.assertAlmostEqual(stats["high_or_extreme_share"], 0.75)
        expected_h = (2 / 4) ** 2 + (1 / 4) ** 2 + (1 / 4) ** 2
        self.assertAlmostEqual(stats["herfindahl_top_cells"], expected_h)
        monopoly = portfolio_concentration(ranked.iloc[[0, 1]])
        self.assertAlmostEqual(monopoly["herfindahl_top_cells"], 1.0)

    def test_synthetic_demo_run_writes_math_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = write_synthetic_demo_run(tmp_dir, run_name="wildfire-asset-exposure-global")
            run_root = Path(result["run_root"])
            global_ranked = pd.read_csv(run_root / "reports" / "ranked_asset_exposure_global.csv")
            demo = pd.read_csv(run_root / "reports" / "demo_utility_portfolio_global.csv")
            self.assertGreaterEqual(len(global_ranked), 4)
            self.assertEqual(list(global_ranked.columns), list(RANKED_COLUMNS))
            self.assertEqual(list(demo.columns), list(DEMO_COLUMNS))
            by_id = global_ranked.set_index("asset_id")
            line = by_id.loc["synth_line_af_90_10"]
            self.assertAlmostEqual(float(line["ranking_score"]), 0.91)
            self.assertLess(float(line["length_weighted_score"]), 0.91)
            self.assertAlmostEqual(float(line["score_range"]), 0.71)
            self.assertAlmostEqual(float(by_id.loc["synth_plant_lagos"]["ranking_score"]), 0.88)
            months = dict(zip(global_ranked["asset_id"], global_ranked["month"]))
            self.assertEqual(months["synth_plant_lagos"], "2024-07")
            self.assertEqual(months["synth_plant_sydney"], "2024-01")
            self.assertEqual(months["synth_plant_bay"], "2024-09")
            self.assertFalse(pd.isna(demo["high_or_extreme_share"].iloc[0]))
            self.assertFalse(pd.isna(demo["herfindahl_top_cells"].iloc[0]))
            readme = (run_root / "README.md").read_text(encoding="utf-8")
            self.assertIn("SYNTHETIC", readme)
            self.assertIn("not a customer book", readme)


if __name__ == "__main__":
    unittest.main()
