from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "artifacts" / "wildfire_risk_runs"
WATCHLIST_PATH = Path(__file__).with_name("wildfire_watchlist.json")


def get_run_root() -> Path:
    configured = os.getenv("HEPHAELION_RUN_ROOT")
    root = Path(configured).expanduser() if configured else DEFAULT_RUN_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def list_run_directories(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    candidates = [path for path in run_root.iterdir() if path.is_dir()]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_run_directory(run_name: str | None) -> Path:
    run_root = get_run_root()
    runs = list_run_directories(run_root)
    if not runs:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No wildfire risk runs were found under `{run_root}`. "
                "Generate a run with wildfire_risk/build_labels.py, build_features.py, "
                "train_baseline.py, and render_maps.py first."
            ),
        )
    if run_name is None:
        return runs[0]

    selected = run_root / run_name
    if not selected.is_dir():
        raise HTTPException(status_code=404, detail=f"Run `{run_name}` was not found under `{run_root}`.")
    return selected


def load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_month_number(month: str | int) -> int:
    if isinstance(month, int):
        month_num = month
    else:
        value = month.strip()
        if len(value) == 7 and value[4] == "-":
            month_num = int(value[5:7])
        else:
            month_num = int(value)
    if month_num < 1 or month_num > 12:
        raise HTTPException(status_code=422, detail="Month must be in the range 1-12 or formatted as YYYY-MM.")
    return month_num


def isoformat_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def load_watchlist() -> list[dict[str, Any]]:
    payload = load_json(WATCHLIST_PATH)
    if isinstance(payload, list):
        return payload
    return []


def attach_signal_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    sorted_rows = sorted(rows, key=lambda item: item["predicted_probability"], reverse=True)
    total = max(len(sorted_rows), 1)
    enriched: list[dict[str, Any]] = []
    for idx, row in enumerate(sorted_rows, start=1):
        new_row = dict(row)
        if new_row.get("risk_rank") is None:
            new_row["risk_rank"] = idx
        relative_score = 100.0 * (1.0 - ((idx - 1.0) / total))
        if new_row.get("relative_risk_score") is None:
            new_row["relative_risk_score"] = round(relative_score, 2)
        percentile = max(min(relative_score / 100.0, 1.0), 0.0)
        if new_row.get("risk_percentile") is None:
            new_row["risk_percentile"] = round(percentile, 4)
        if percentile >= 0.95:
            band = "extreme"
        elif percentile >= 0.8:
            band = "high"
        elif percentile >= 0.5:
            band = "elevated"
        else:
            band = "monitor"
        if not new_row.get("risk_band"):
            new_row["risk_band"] = band
        enriched.append(new_row)
    return enriched


def infer_presentation_guidance(metrics: dict[str, Any]) -> dict[str, Any]:
    guidance = metrics.get("presentation_guidance")
    if isinstance(guidance, dict) and guidance:
        return guidance

    classification = metrics.get("classification", {})
    precision_at_threshold = classification.get("precision_at_0_5")
    brier_score = classification.get("brier_score")
    if precision_at_threshold is None or brier_score is None:
        return {}
    if precision_at_threshold >= 0.25 and brier_score <= 0.15:
        warning = "Probabilities may be used with caution, but ranked presentation is still preferred."
    else:
        warning = (
            "Use this run as a ranked wildfire signal. Raw probabilities should not be marketed as literal "
            "event probabilities without additional calibration work."
        )
    return {
        "recommended_mode": "ranked_signal",
        "recommended_score_field": "relative_risk_score",
        "warning": warning,
    }


def build_run_summary(run_dir: Path) -> dict[str, Any]:
    models_dir = run_dir / "models"
    maps_dir = run_dir / "maps"
    reports_dir = run_dir / "reports"

    metrics_path = models_dir / "metrics.json"
    signal_profile_path = models_dir / "signal_profile.json"
    predictions_path = models_dir / "validation_predictions.csv"
    top_risk_path = maps_dir / "top_risk_cells.csv"

    map_files = sorted(maps_dir.glob("wildfire_risk_month_*.png"))
    manifests = sorted(reports_dir.glob("*_manifest.json"))
    metrics = load_json(metrics_path)

    return {
        "run_name": run_dir.name,
        "run_root": str(run_dir),
        "updated_at_utc": isoformat_utc(run_dir),
        "artifacts": {
            "metrics": str(metrics_path) if metrics_path.exists() else None,
            "signal_profile": str(signal_profile_path) if signal_profile_path.exists() else None,
            "validation_predictions": str(predictions_path) if predictions_path.exists() else None,
            "top_risk_cells": str(top_risk_path) if top_risk_path.exists() else None,
            "map_images": [
                {
                    "month_num": int(path.stem.split("_")[-1]),
                    "path": str(path),
                    "url": f"/static/wildfire/{run_dir.name}/maps/{path.name}",
                }
                for path in map_files
            ],
            "manifests": [str(path) for path in manifests],
        },
        "metrics": metrics,
        "presentation_guidance": infer_presentation_guidance(metrics),
    }


def build_top_cells_from_rows(rows: list[dict[str, str]], month_num: int, limit: int) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        row_month = row.get("month_num")
        if row_month is None:
            continue
        if int(float(row_month)) != month_num:
            continue
        filtered.append(
            {
                "cell_id": row.get("cell_id"),
                "lat_center": float(row["lat_center"]) if row.get("lat_center") else None,
                "lon_center": float(row["lon_center"]) if row.get("lon_center") else None,
                "predicted_probability": float(
                    row.get("predicted_probability")
                    or row.get("predicted_burned_fraction")
                    or 0.0
                ),
                "predicted_burned_fraction": float(row["predicted_burned_fraction"])
                if row.get("predicted_burned_fraction")
                else None,
                "month": row.get("month"),
                "risk_rank": int(row["risk_rank"]) if row.get("risk_rank") else None,
                "relative_risk_score": float(row["relative_risk_score"]) if row.get("relative_risk_score") else None,
                "risk_percentile": float(row["risk_percentile"]) if row.get("risk_percentile") else None,
                "risk_band": row.get("risk_band"),
            }
        )

    filtered = attach_signal_columns(filtered)
    return filtered[:limit]


def get_top_cells(run_dir: Path, month_num: int, limit: int) -> tuple[list[dict[str, Any]], str]:
    maps_top_risk_path = run_dir / "maps" / "top_risk_cells.csv"
    if maps_top_risk_path.exists():
        rows = load_csv_rows(maps_top_risk_path)
        return build_top_cells_from_rows(rows, month_num=month_num, limit=limit), "maps/top_risk_cells.csv"

    validation_path = run_dir / "models" / "validation_predictions.csv"
    if validation_path.exists():
        rows = load_csv_rows(validation_path)
        return build_top_cells_from_rows(rows, month_num=month_num, limit=limit), "models/validation_predictions.csv"

    raise HTTPException(
        status_code=404,
        detail=f"Run `{run_dir.name}` does not contain `top_risk_cells.csv` or `validation_predictions.csv`.",
    )


def cell_matches_watchlist(cell: dict[str, Any], watch_item: dict[str, Any]) -> bool:
    lat = cell.get("lat_center")
    lon = cell.get("lon_center")
    if lat is None or lon is None:
        return False
    return (
        watch_item["lat_min"] <= lat <= watch_item["lat_max"]
        and watch_item["lon_min"] <= lon <= watch_item["lon_max"]
    )


def build_watchlist_matches(top_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for watch_item in load_watchlist():
        covered_cells = [cell for cell in top_cells if cell_matches_watchlist(cell, watch_item)]
        if not covered_cells:
            continue
        best_cell = max(covered_cells, key=lambda item: item["predicted_probability"])
        matches.append(
            {
                "symbol": watch_item["symbol"],
                "name": watch_item["name"],
                "sector": watch_item["sector"],
                "theme": watch_item["theme"],
                "region": watch_item["region"],
                "channels": watch_item["channels"],
                "matched_cells": len(covered_cells),
                "risk_band": best_cell.get("risk_band"),
                "relative_risk_score": best_cell.get("relative_risk_score"),
                "predicted_probability": best_cell.get("predicted_probability"),
                "sample_cell": {
                    "lat_center": best_cell.get("lat_center"),
                    "lon_center": best_cell.get("lon_center"),
                },
            }
        )
    return sorted(
        matches,
        key=lambda item: (
            item.get("relative_risk_score") or 0.0,
            item.get("predicted_probability") or 0.0,
        ),
        reverse=True,
    )


def build_investor_summary(run_dir: Path) -> dict[str, Any]:
    summary = build_run_summary(run_dir)
    metrics = summary.get("metrics", {})
    classification = metrics.get("classification", {})
    return {
        "run_name": summary["run_name"],
        "updated_at_utc": summary["updated_at_utc"],
        "presentation_guidance": summary.get("presentation_guidance", {}),
        "proof_points": {
            "roc_auc": classification.get("roc_auc"),
            "pr_auc": classification.get("pr_auc"),
            "precision_top_10pct": classification.get("precision_top_10pct", classification.get("precision_top_decile")),
            "capture_top_10pct": classification.get("capture_top_10pct", classification.get("recall_top_decile")),
            "precision_at_0_5": classification.get("precision_at_0_5"),
            "brier_score": classification.get("brier_score"),
        },
        "top_features": metrics.get("top_features", [])[:10],
        "watchlist_coverage": [item["symbol"] for item in load_watchlist()],
    }


app = FastAPI(
    title="Hephaelion Wildfire Risk MVP API",
    version="0.1.0",
    description=(
        "Minimal backend for serving monthly wildfire risk outputs produced by the `wildfire_risk` pipeline."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/wildfire", StaticFiles(directory=str(get_run_root())), name="wildfire-static")


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "hephaelion-wildfire-risk-mvp",
        "docs": "/docs",
        "health": "/health",
        "investor_summary": "/api/v1/wildfire/investor/summary",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    run_root = get_run_root()
    runs = list_run_directories(run_root)
    return {
        "status": "ok",
        "service": "hephaelion-wildfire-risk-mvp",
        "run_root": str(run_root),
        "available_runs": len(runs),
        "latest_run": runs[0].name if runs else None,
    }


@app.get("/api/v1/wildfire/runs")
def list_runs() -> dict[str, Any]:
    run_root = get_run_root()
    runs = list_run_directories(run_root)
    return {
        "run_root": str(run_root),
        "count": len(runs),
        "runs": [build_run_summary(run_dir) for run_dir in runs],
    }


@app.get("/api/v1/wildfire/risk/summary")
def wildfire_risk_summary(
    run_name: str | None = Query(default=None, description="Optional run folder name. Defaults to the latest run."),
) -> dict[str, Any]:
    run_dir = resolve_run_directory(run_name)
    return build_run_summary(run_dir)


@app.get("/api/v1/wildfire/risk/monthly")
def wildfire_monthly_risk(
    month: str = Query(description="Month number like `7` or date string like `2024-07`."),
    run_name: str | None = Query(default=None, description="Optional run folder name. Defaults to the latest run."),
    limit: int = Query(default=20, ge=1, le=200, description="Maximum number of top-risk cells to return."),
) -> dict[str, Any]:
    run_dir = resolve_run_directory(run_name)
    month_num = to_month_number(month)
    top_cells, source = get_top_cells(run_dir, month_num=month_num, limit=limit)
    map_image_path = run_dir / "maps" / f"wildfire_risk_month_{month_num:02d}.png"

    summary = build_run_summary(run_dir)
    return {
        "run_name": run_dir.name,
        "month_num": month_num,
        "top_cells_source": source,
        "map_image_url": f"/static/wildfire/{run_dir.name}/maps/{map_image_path.name}"
        if map_image_path.exists()
        else None,
        "top_cells_count": len(top_cells),
        "top_cells": top_cells,
        "metrics": summary.get("metrics", {}),
        "presentation_guidance": summary.get("presentation_guidance", {}),
    }


@app.get("/api/v1/wildfire/investor/summary")
def investor_summary(
    run_name: str | None = Query(default=None, description="Optional run folder name. Defaults to the latest run."),
) -> dict[str, Any]:
    run_dir = resolve_run_directory(run_name)
    return build_investor_summary(run_dir)


@app.get("/api/v1/wildfire/investor/watchlist")
def investor_watchlist(
    month: str = Query(description="Month number like `7` or date string like `2024-07`."),
    run_name: str | None = Query(default=None, description="Optional run folder name. Defaults to the latest run."),
    limit: int = Query(default=20, ge=1, le=200, description="Maximum number of top-risk cells to evaluate."),
) -> dict[str, Any]:
    run_dir = resolve_run_directory(run_name)
    month_num = to_month_number(month)
    top_cells, source = get_top_cells(run_dir, month_num=month_num, limit=limit)
    matches = build_watchlist_matches(top_cells)
    return {
        "run_name": run_dir.name,
        "month_num": month_num,
        "top_cells_source": source,
        "watchlist_match_count": len(matches),
        "watchlist_matches": matches,
        "presentation_guidance": build_run_summary(run_dir).get("presentation_guidance", {}),
    }
