"""Overlay public power assets on v8 monthly burned-area scores.

CLI:
  python -m wildfire_risk.asset_exposure --predictions PATH \\
      --run-root artifacts/wildfire_risk_runs --scope global [--month 2024-07] [--region africa]

`--scope california` is the original HIFLD California demo (transmission lines and
substations from HIFLD FeatureServers, joined on the same 0.5° `grid.py` cells).
`--scope global` tiles the prediction CSV by continent and overlays OSM lines plus
WRI GPPD plants, with HIFLD as the North America fallback when global lines are missing.

This reports this month's burned-area probability. It does not simulate spread,
smoke, or dollar losses. The demo utility portfolio is a public-data book, not a
customer book. v8/v9 model artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from .asset_scale import (
        CALIFORNIA_BBOX,
        DEFAULT_ASSET_CACHE_DIR,
        DEMO_COLUMNS,
        GPPD_LICENSE,
        GPPD_URL,
        HIFLD_LICENSE,
        HIFLD_SUBSTATIONS_URL,
        HIFLD_TRANSMISSION_URL,
        OSM_POWER_PARQUET_LICENSE,
        OSM_POWER_PARQUET_URL,
        OSM_POWER_TMM_LICENSE,
        OSM_POWER_TMM_URL,
        RANKED_COLUMNS,
        SourceStatus,
        FailureCache,
        assign_inspect_rank,
        build_demo_portfolio,
        collect_region_assets,
        empty_demo_frame,
        empty_ranked_frame,
        iter_process_regions,
        load_gppd_plants,
        load_hifld_assets,
        load_world_bank_osm_lines,
        overlay_region_predictions,
        peak_month_for_region,
        require_predictions_file,
        stream_predictions_by_continent,
        synthetic_demo_inputs,
    )
    from .grid import GridSpec
    from .run_artifacts import create_run_layout, write_stage_manifest
except ImportError:  # pragma: no cover - allows direct script execution
    from asset_scale import (
        CALIFORNIA_BBOX,
        DEFAULT_ASSET_CACHE_DIR,
        DEMO_COLUMNS,
        GPPD_LICENSE,
        GPPD_URL,
        HIFLD_LICENSE,
        HIFLD_SUBSTATIONS_URL,
        HIFLD_TRANSMISSION_URL,
        OSM_POWER_PARQUET_LICENSE,
        OSM_POWER_PARQUET_URL,
        OSM_POWER_TMM_LICENSE,
        OSM_POWER_TMM_URL,
        RANKED_COLUMNS,
        SourceStatus,
        FailureCache,
        assign_inspect_rank,
        build_demo_portfolio,
        collect_region_assets,
        empty_demo_frame,
        empty_ranked_frame,
        iter_process_regions,
        load_gppd_plants,
        load_hifld_assets,
        load_world_bank_osm_lines,
        overlay_region_predictions,
        peak_month_for_region,
        require_predictions_file,
        stream_predictions_by_continent,
        synthetic_demo_inputs,
    )
    from grid import GridSpec
    from run_artifacts import create_run_layout, write_stage_manifest


LOGGER = logging.getLogger(__name__)

README_BODY = """# Wildfire asset exposure

This run overlays public power assets on Hephaelion v8 monthly burned-area scores.

**This is this month's burned-area probability.** The horizon is the calendar
month. The overlay does **not** simulate fire spread, smoke transport, or dollar
losses. A high score means the 0.5° cell that holds the asset ranked high for
burned area this month, not that a fire will reach the asset.

The `demo_utility_portfolio_*.csv` file is a **demo book built from public
plant and line records, not a customer book.** Do not treat it as an insured
portfolio, a utility's GIS, or a loss run.

## Model

- Scores come from the existing v8 global 0.5° monthly burned-area model
  (train 2020–2023, holdout 2024–2025). This overlay does not retrain v8 or v9.
- Join uses `wildfire_risk/grid.py` (same formula as `assign_points_to_grid`).
  Lines are sampled every 0.1°. ranking_score is S_max; length_weighted_score is
  S_len. score_range is not a confidence interval. See ASSET_EXPOSURE_MATH.md.
- Each continent keeps its own peak-fire month so Africa does not overwrite
  Australia. Pass `--month YYYY-MM` to force one month.

## Asset sources

Only real public files are used. If a download fails, the exact URL is recorded
in `coverage_report.csv` and that region continues without invented assets.

| Source | Role | License |
|---|---|---|
| World Bank OSM extract `osm_power_tmm.zip` | Global transmission lines | CC-BY-4.0 |
| OSM GeoParquet `power_line.parquet` | Global `power=line` fallback when the World Bank zip is blocked | ODbL 1.0, © OpenStreetMap contributors |
| WRI Global Power Plant Database | Generation sites (`asset_type=plant`) | CC-BY-4.0 |
| HIFLD transmission lines + substations | California demo, and North America fallback | U.S. public domain / HIFLD Open |

World Bank URL (previously HTTP 403 from this project):
{osm_tmm_url}

OSM GeoParquet fallback:
{osm_parquet_url}

GPPD:
{gppd_url}

HIFLD lines:
{hifld_lines_url}

HIFLD substations:
{hifld_subs_url}

## Ranked columns

See `wildfire_risk/ASSET_EXPOSURE_MATH.md` for numbered equations.

asset_id, asset_name, asset_type, region, cell_id, month, ranking_score (S_max),
length_weighted_score (S_len), predicted_probability, calibrated_probability,
risk_band, month_risk_percentile, n_cells, n_high_cells, n_extreme_cells,
extreme_cell_fraction, score_range (U, not a CI), score_std, inspect_rank,
reason, asset_source.

Demo book extra columns: high_or_extreme_share, herfindahl_top_cells, n_demo_assets.
Those are concentration of this public demo book, not customer TVaR.
"""


def coverage_frame(rows: list[SourceStatus]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["region", "source", "url", "status", "n_assets", "license", "note"]
        )
    return pd.DataFrame([row.to_dict() for row in rows])[
        ["region", "source", "url", "status", "n_assets", "license", "note"]
    ]


def write_readme(path: Path, scope: str, extra: str = "") -> Path:
    text = README_BODY.format(
        osm_tmm_url=OSM_POWER_TMM_URL,
        osm_parquet_url=OSM_POWER_PARQUET_URL,
        gppd_url=GPPD_URL,
        hifld_lines_url=HIFLD_TRANSMISSION_URL,
        hifld_subs_url=HIFLD_SUBSTATIONS_URL,
    )
    text += f"\n## This run\n\nScope: `{scope}`.\n"
    if extra:
        text += f"\n{extra.rstrip()}\n"
    path.write_text(text, encoding="utf-8")
    return path


def render_region_map(
    ranked: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: Path,
    region: str,
    month: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    if not predictions.empty and {"lon_center", "lat_center", "ranking_score"}.issubset(predictions.columns):
        background = predictions.drop_duplicates("cell_id")
        ax.scatter(
            background["lon_center"],
            background["lat_center"],
            c=background["ranking_score"],
            cmap="YlOrRd",
            s=8,
            marker="s",
            linewidths=0,
            alpha=0.55,
            vmin=0.0,
            vmax=max(float(background["ranking_score"].quantile(0.99)), 0.05),
        )
    if not ranked.empty:
        plot_df = ranked.merge(
            predictions[["cell_id", "lon_center", "lat_center"]].drop_duplicates("cell_id"),
            on="cell_id",
            how="left",
        )
        colors = {"monitor": "#1b4f72", "elevated": "#b9770e", "high": "#ca6f1e", "extreme": "#7b241c"}
        for band, group in plot_df.groupby("risk_band"):
            ax.scatter(
                group["lon_center"],
                group["lat_center"],
                c=colors.get(str(band), "#17202a"),
                s=18 if str(band) in {"high", "extreme"} else 10,
                marker="o",
                linewidths=0.2,
                edgecolors="white",
                label=str(band),
                zorder=3,
            )
        ax.legend(title="Asset risk band", loc="lower left", fontsize=8)
    ax.set_title(f"{region} power assets on this month's burned-area probability ({month})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.2, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _write_table(path: Path, frame: pd.DataFrame, columns: tuple[str, ...]) -> Path:
    if frame is None or frame.empty:
        out = pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
    else:
        out = frame.copy()
        for column in columns:
            if column not in out.columns:
                out[column] = pd.NA
        out = out.loc[:, list(columns)]
    out.to_csv(path, index=False)
    return path


def _write_ranked(path: Path, ranked: pd.DataFrame) -> Path:
    return _write_table(path, ranked, RANKED_COLUMNS)


def _write_demo(path: Path, demo: pd.DataFrame) -> Path:
    return _write_table(path, demo if demo is not None else empty_demo_frame(), DEMO_COLUMNS)


def run_california_exposure(
    predictions_path: str | Path,
    layout,
    cache_dir: Path,
    failure_cache: FailureCache,
    month: str | None = None,
    spec: GridSpec | None = None,
) -> dict[str, Any]:
    spec = spec or GridSpec()
    tile_dir = layout.features_dir / "prediction_tiles"
    paths = stream_predictions_by_continent(
        predictions_path,
        tile_dir,
        bbox=CALIFORNIA_BBOX,
    )
    na_path = paths.get("north_america")
    if na_path is None:
        pred_df = empty_ranked_frame()
        LOGGER.warning("No north_america prediction rows inside the California bbox")
        california_pred = pd.DataFrame()
    else:
        california_pred = pd.read_csv(na_path)
        lat_min, lat_max, lon_min, lon_max = CALIFORNIA_BBOX
        california_pred = california_pred[
            california_pred["lat_center"].between(lat_min, lat_max)
            & california_pred["lon_center"].between(lon_min, lon_max)
        ]
    hifld, reports = load_hifld_assets(
        CALIFORNIA_BBOX,
        failure_cache,
        spec=spec,
        region="california",
    )
    plants, plant_status = load_gppd_plants(cache_dir, failure_cache, spec=spec, bbox=CALIFORNIA_BBOX)
    plant_status.region = "california"
    reports.append(plant_status)
    assets = pd.concat([frame for frame in (hifld, plants) if frame is not None and not frame.empty], ignore_index=True)
    if assets.empty or california_pred.empty:
        ranked = empty_ranked_frame()
        chosen_month = month or ""
    else:
        assets["region"] = "california"
        ranked, chosen_month = overlay_region_predictions(
            california_pred, assets, region="california", month=month, spec=spec
        )
    ranked_path = _write_ranked(layout.reports_dir / "ranked_asset_exposure_california.csv", ranked)
    demo_path = _write_demo(layout.reports_dir / "demo_utility_portfolio_california.csv", build_demo_portfolio(ranked))
    coverage = coverage_frame(reports)
    coverage_path = layout.reports_dir / "coverage_report.csv"
    coverage.to_csv(coverage_path, index=False)
    map_path = render_region_map(
        ranked,
        california_pred,
        layout.maps_dir / f"asset_exposure_california_{chosen_month or 'no_month'}.png",
        region="california",
        month=chosen_month or "n/a",
    )
    readme_path = write_readme(
        layout.root / "README.md",
        scope="california",
        extra=(
            f"Peak-fire month used: {chosen_month or 'none (no prediction rows)'}.\n"
            "California uses HIFLD FeatureServers on the 0.5° grid. EPSG:3310 is not used.\n"
        ),
    )
    return {
        "ranked_paths": [str(ranked_path)],
        "demo_path": str(demo_path),
        "coverage_path": str(coverage_path),
        "map_paths": [str(map_path)],
        "readme_path": str(readme_path),
        "n_assets": int(len(ranked)),
        "month": chosen_month,
        "coverage": reports,
    }


def run_global_exposure(
    predictions_path: str | Path,
    layout,
    cache_dir: Path,
    failure_cache: FailureCache,
    month: str | None = None,
    region: str | None = None,
    spec: GridSpec | None = None,
) -> dict[str, Any]:
    spec = spec or GridSpec()
    tile_dir = layout.features_dir / "prediction_tiles"
    paths = stream_predictions_by_continent(
        predictions_path,
        tile_dir,
        region=region,
        month=None if month is None else month,
    )
    LOGGER.info("Continent tiles: %s", ", ".join(sorted(paths)) or "(none)")

    global_lines, line_status = load_world_bank_osm_lines(cache_dir, failure_cache, spec=spec)
    if line_status.status != "ok":
        LOGGER.warning("World Bank OSM zip unavailable (%s): %s", line_status.status, line_status.note)
        global_lines = pd.DataFrame()
    plants, plant_status = load_gppd_plants(cache_dir, failure_cache, spec=spec)
    reports: list[SourceStatus] = [line_status, plant_status]

    ranked_parts: list[pd.DataFrame] = []
    map_paths: list[Path] = []
    ranked_paths: list[Path] = []
    region_months: dict[str, str] = {}

    for name in iter_process_regions(paths, region=region):
        pred_df = pd.read_csv(paths[name])
        assets, region_reports = collect_region_assets(
            name,
            cache_dir,
            failure_cache,
            global_lines=global_lines if not global_lines.empty else pd.DataFrame(),
            global_line_status=line_status,
            plants=plants,
            spec=spec,
        )
        reports.extend(region_reports)
        if assets.empty:
            LOGGER.warning("No public assets for %s; recording source URLs and skipping invented scores", name)
            ranked = empty_ranked_frame()
            chosen_month = month or (peak_month_for_region(pred_df, month) if not pred_df.empty else "")
        else:
            ranked, chosen_month = overlay_region_predictions(
                pred_df, assets, region=name, month=month, spec=spec
            )
        region_months[name] = chosen_month
        ranked_path = _write_ranked(layout.reports_dir / f"ranked_asset_exposure_{name}.csv", ranked)
        ranked_paths.append(ranked_path)
        if not ranked.empty:
            ranked_parts.append(ranked)
        map_paths.append(
            render_region_map(
                ranked,
                pred_df[pred_df["month"].astype(str) == str(chosen_month)] if "month" in pred_df.columns else pred_df,
                layout.maps_dir / f"asset_exposure_{name}_{chosen_month or 'no_month'}.png",
                region=name,
                month=chosen_month or "n/a",
            )
        )

    global_ranked = (
        pd.concat(ranked_parts, ignore_index=True) if ranked_parts else empty_ranked_frame()
    )
    if not global_ranked.empty:
        global_ranked = assign_inspect_rank(global_ranked)
    global_path = _write_ranked(layout.reports_dir / "ranked_asset_exposure_global.csv", global_ranked)
    demo_path = _write_demo(
        layout.reports_dir / "demo_utility_portfolio_global.csv",
        build_demo_portfolio(global_ranked),
    )
    coverage = coverage_frame(reports)
    coverage_path = layout.reports_dir / "coverage_report.csv"
    coverage.to_csv(coverage_path, index=False)
    month_lines = "\n".join(f"- {name}: {value}" for name, value in region_months.items()) or "- none"
    readme_path = write_readme(
        layout.root / "README.md",
        scope="global",
        extra=(
            "Per-region peak-fire months (Africa does not overwrite Australia):\n"
            f"{month_lines}\n\n"
            f"World Bank OSM zip status: {line_status.status}. {line_status.note}\n"
            f"If that zip is HTTP 403, the failure is cached at {failure_cache.path} "
            "and is not retried per region. The ODbL OSM GeoParquet is used instead when it downloads.\n"
        ),
    )
    return {
        "ranked_paths": [str(path) for path in ranked_paths] + [str(global_path)],
        "demo_path": str(demo_path),
        "coverage_path": str(coverage_path),
        "map_paths": [str(path) for path in map_paths],
        "readme_path": str(readme_path),
        "n_assets": int(len(global_ranked)),
        "region_months": region_months,
        "coverage": reports,
    }


def write_synthetic_demo_run(
    run_root: str | Path,
    run_name: str = "wildfire-asset-exposure-global",
) -> dict[str, Any]:
    """Write a SYNTHETIC run folder with math columns. Not v8 scores. Predictions were absent."""
    layout = create_run_layout(run_root, run_name=run_name)
    predictions, assets = synthetic_demo_inputs()
    predictions = predictions.copy()
    try:
        from .evaluate import assign_continent
    except ImportError:  # pragma: no cover
        from evaluate import assign_continent
    predictions["continent"] = assign_continent(predictions["lat_center"], predictions["lon_center"])
    ranked_parts: list[pd.DataFrame] = []
    ranked_paths: list[Path] = []
    map_paths: list[Path] = []
    region_months: dict[str, str] = {}
    for region, group in assets.groupby("region", sort=False):
        pred = predictions[predictions["continent"] == region].copy()
        ranked, chosen_month = overlay_region_predictions(pred, group, region=str(region))
        region_months[str(region)] = chosen_month
        ranked_path = _write_ranked(layout.reports_dir / f"ranked_asset_exposure_{region}.csv", ranked)
        ranked_paths.append(ranked_path)
        if not ranked.empty:
            ranked_parts.append(ranked)
        month_pred = pred[pred["month"].astype(str) == str(chosen_month)]
        map_paths.append(
            render_region_map(
                ranked,
                month_pred,
                layout.maps_dir / f"asset_exposure_{region}_{chosen_month or 'no_month'}.png",
                region=str(region),
                month=chosen_month or "n/a",
            )
        )
    global_ranked = assign_inspect_rank(pd.concat(ranked_parts, ignore_index=True)) if ranked_parts else empty_ranked_frame()
    global_path = _write_ranked(layout.reports_dir / "ranked_asset_exposure_global.csv", global_ranked)
    demo = build_demo_portfolio(global_ranked)
    demo_path = _write_demo(layout.reports_dir / "demo_utility_portfolio_global.csv", demo)
    coverage = pd.DataFrame(
        [
            {
                "region": "global",
                "source": "synthetic_demo",
                "url": "",
                "status": "synthetic",
                "n_assets": int(len(global_ranked)),
                "license": "not a public GIS extract",
                "note": (
                    "SYNTHETIC. v8 validation_predictions.csv was not on disk. "
                    "These rows demonstrate S_max, S_len, U(a), N_ext, and demo-book H. "
                    "They are not burned-area forecasts."
                ),
            }
        ]
    )
    coverage_path = layout.reports_dir / "coverage_report.csv"
    coverage.to_csv(coverage_path, index=False)
    month_lines = "\n".join(f"- {name}: {value}" for name, value in region_months.items()) or "- none"
    readme_path = write_readme(
        layout.root / "README.md",
        scope="global",
        extra=(
            "**SYNTHETIC DEMO.** v8 `validation_predictions.csv` was absent. "
            "Do not treat these scores as this month's burned-area probability from v8. "
            "The columns implement ASSET_EXPOSURE_MATH.md eqs. (6)–(13) on labeled toy assets "
            "in Africa, Australia, and North America.\n\n"
            f"Per-region months used by the toy overlay:\n{month_lines}\n"
        ),
    )
    write_stage_manifest(
        layout=layout,
        stage_name="asset_exposure",
        config={
            "predictions_path": "SYNTHETIC (validation_predictions.csv absent)",
            "scope": "global",
            "synthetic": True,
            "grid": "wildfire_risk.grid.GridSpec 0.5 degree, not EPSG:3310",
            "horizon": "calendar month burned-area probability (formulas only; scores are synthetic)",
            "not_modeled": ["spread", "smoke", "dollar_losses", "customer_tvar"],
            "formulas": "ASSET_EXPOSURE_MATH.md eqs. (6)-(13); score_range is not a confidence interval",
            "demo_book": "synthetic public-style book, not a customer book; H is not TVaR",
        },
        artifacts={
            "ranked_csv": [str(path) for path in ranked_paths] + [str(global_path)],
            "demo_portfolio": str(demo_path),
            "coverage_report": str(coverage_path),
            "maps": [str(path) for path in map_paths],
            "readme": str(readme_path),
        },
        summary={
            "n_ranked_assets": int(len(global_ranked)),
            "region_months": region_months,
            "synthetic": True,
        },
    )
    return {
        "run_root": str(layout.root),
        "ranked_paths": [str(path) for path in ranked_paths] + [str(global_path)],
        "demo_path": str(demo_path),
        "coverage_path": str(coverage_path),
        "map_paths": [str(path) for path in map_paths],
        "readme_path": str(readme_path),
        "n_assets": int(len(global_ranked)),
        "region_months": region_months,
        "synthetic": True,
    }


def run_asset_exposure(
    predictions_path: str | Path,
    run_root: str | Path,
    scope: str = "global",
    month: str | None = None,
    region: str | None = None,
    run_name: str | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    predictions_path = require_predictions_file(predictions_path)
    if scope not in {"global", "california"}:
        raise ValueError("scope must be 'global' or 'california'")
    resolved_name = run_name or (
        "wildfire-asset-exposure-global" if scope == "global" else "wildfire-asset-exposure-california"
    )
    layout = create_run_layout(run_root, run_name=resolved_name)
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_ASSET_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    failure_cache = FailureCache.load(cache_dir / "source_fetch_failures.json")
    spec = GridSpec()

    if scope == "california":
        result = run_california_exposure(
            predictions_path,
            layout,
            cache_dir,
            failure_cache,
            month=month,
            spec=spec,
        )
    else:
        result = run_global_exposure(
            predictions_path,
            layout,
            cache_dir,
            failure_cache,
            month=month,
            region=region,
            spec=spec,
        )

    coverage_rows = [row.to_dict() if isinstance(row, SourceStatus) else row for row in result.get("coverage", [])]
    write_stage_manifest(
        layout=layout,
        stage_name="asset_exposure",
        config={
            "predictions_path": str(predictions_path),
            "scope": scope,
            "month": month,
            "region": region,
            "grid": "wildfire_risk.grid.GridSpec 0.5 degree, not EPSG:3310",
            "horizon": "calendar month burned-area probability",
            "asset_cache_dir": str(cache_dir),
            "not_modeled": ["spread", "smoke", "dollar_losses", "customer_tvar"],
            "demo_book": "public assets, not a customer book; H is demo-book Herfindahl, not TVaR",
            "formulas": "ASSET_EXPOSURE_MATH.md eqs. (6)-(13); score_range is not a confidence interval",
            "sources": {
                "osm_power_tmm": OSM_POWER_TMM_URL,
                "osm_power_tmm_license": OSM_POWER_TMM_LICENSE,
                "osm_parquet": OSM_POWER_PARQUET_URL,
                "osm_parquet_license": OSM_POWER_PARQUET_LICENSE,
                "gppd": GPPD_URL,
                "gppd_license": GPPD_LICENSE,
                "hifld_lines": HIFLD_TRANSMISSION_URL,
                "hifld_substations": HIFLD_SUBSTATIONS_URL,
                "hifld_license": HIFLD_LICENSE,
            },
        },
        artifacts={
            "ranked_csv": result.get("ranked_paths"),
            "demo_portfolio": result.get("demo_path"),
            "coverage_report": result.get("coverage_path"),
            "maps": result.get("map_paths"),
            "readme": result.get("readme_path"),
        },
        summary={
            "n_ranked_assets": result.get("n_assets"),
            "region_months": result.get("region_months"),
            "source_status": coverage_rows,
        },
    )
    result["run_root"] = str(layout.root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay public power assets on v8 monthly burned-area scores. "
            "Reports this month's burned-area probability. Does not simulate spread, smoke, or loss."
        )
    )
    parser.add_argument("--predictions", help="validation_predictions.csv from train_baseline.py")
    parser.add_argument("--run-root", default="artifacts/wildfire_risk_runs", help="Parent folder for a new run layout")
    parser.add_argument(
        "--scope",
        choices=("global", "california"),
        default="global",
        help="global continent tiles, or the original HIFLD California demo",
    )
    parser.add_argument("--month", help="Optional YYYY-MM. Default is each region's own peak-fire month")
    parser.add_argument("--region", help="Optional single continent for --scope global (e.g. africa)")
    parser.add_argument("--run-name", help="Folder name inside --run-root")
    parser.add_argument(
        "--cache-dir",
        help="Downloaded assets and the 403 failure cache (default datasets/wildfires/assets/)",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Write a labeled SYNTHETIC run with math columns. Does not invent v8 scores.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.synthetic_demo:
        result = write_synthetic_demo_run(
            run_root=args.run_root,
            run_name=args.run_name or "wildfire-asset-exposure-global",
        )
        print(f"Wrote SYNTHETIC asset exposure demo to {result['run_root']}")
        print(f"Ranked assets: {result.get('n_assets', 0)}")
        print(f"Coverage: {result.get('coverage_path')}")
        return
    if not args.predictions:
        parser.error("--predictions is required unless --synthetic-demo")
    result = run_asset_exposure(
        predictions_path=args.predictions,
        run_root=args.run_root,
        scope=args.scope,
        month=args.month,
        region=args.region,
        run_name=args.run_name,
        cache_dir=args.cache_dir,
    )
    print(f"Wrote asset exposure run to {result['run_root']}")
    print(f"Ranked assets: {result.get('n_assets', 0)}")
    print(f"Coverage: {result.get('coverage_path')}")


if __name__ == "__main__":
    main()
