from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "wildfire-risk-run"


@dataclass(frozen=True)
class RunLayout:
    root: Path
    labels_dir: Path
    features_dir: Path
    models_dir: Path
    maps_dir: Path
    reports_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def create_run_layout(output_root: str | Path, run_name: str | None = None) -> RunLayout:
    output_root = Path(output_root)
    resolved_name = slugify(run_name) if run_name else f"wildfire-risk-{utc_timestamp_slug()}"
    run_root = output_root / resolved_name
    labels_dir = run_root / "labels"
    features_dir = run_root / "features"
    models_dir = run_root / "models"
    maps_dir = run_root / "maps"
    reports_dir = run_root / "reports"
    for path in (labels_dir, features_dir, models_dir, maps_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)
    return RunLayout(
        root=run_root,
        labels_dir=labels_dir,
        features_dir=features_dir,
        models_dir=models_dir,
        maps_dir=maps_dir,
        reports_dir=reports_dir,
    )


def write_json(data: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return output_path


def write_stage_manifest(
    layout: RunLayout,
    stage_name: str,
    config: dict[str, Any],
    artifacts: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "stage": stage_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_layout": layout.to_dict(),
        "config": config,
        "artifacts": artifacts,
    }
    if summary:
        payload["summary"] = summary
    return write_json(payload, layout.reports_dir / f"{slugify(stage_name)}_manifest.json")
