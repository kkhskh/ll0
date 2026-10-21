"""Wildfire risk package. Imports are lazy so `python -m wildfire_risk.train_baseline` does not load twice."""

from typing import Any

__all__ = [
    "GridSpec",
    "LabelSpec",
    "FIRST_WAVE_SOURCES",
    "build_feature_table",
    "build_label_table",
    "build_predictor_table",
    "create_global_grid",
    "create_month_index",
    "export_source_inventory",
    "render_monthly_risk_maps",
    "train_baseline_models",
]


def __getattr__(name: str) -> Any:
    if name in {"GridSpec", "LabelSpec", "create_global_grid", "create_month_index"}:
        from . import grid

        return getattr(grid, name)
    if name in {"FIRST_WAVE_SOURCES", "export_source_inventory"}:
        from . import data_sources

        return getattr(data_sources, name)
    if name == "build_label_table":
        from .build_labels import build_label_table

        return build_label_table
    if name == "build_feature_table":
        from .build_features import build_feature_table

        return build_feature_table
    if name == "build_predictor_table":
        from .build_predictor_table import build_predictor_table

        return build_predictor_table
    if name == "train_baseline_models":
        from .train_baseline import train_baseline_models

        return train_baseline_models
    if name == "render_monthly_risk_maps":
        from .render_maps import render_monthly_risk_maps

        return render_monthly_risk_maps
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
