"""Scored, uncertain, inspectable ranking of public assets on monthly cell scores.

Formulas are written in `ASSET_EXPOSURE_MATH.md`. This module is the
implementation, not a second set of definitions.

Product claim: this month's burned-area probability overlaid on public assets.
Not spread, not smoke, not dollar loss. `U(a)` is a score range, not a
confidence interval. Demo-book `H` is not customer TVaR.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

HOT_BANDS: frozenset[str] = frozenset({"high", "extreme"})

RANKED_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "asset_name",
    "asset_type",
    "region",
    "cell_id",
    "month",
    "ranking_score",
    "length_weighted_score",
    "predicted_probability",
    "calibrated_probability",
    "risk_band",
    "month_risk_percentile",
    "n_cells",
    "n_high_cells",
    "n_extreme_cells",
    "extreme_cell_fraction",
    "score_range",
    "score_std",
    "inspect_rank",
    "reason",
    "asset_source",
)

CONCENTRATION_COLUMNS: tuple[str, ...] = (
    "high_or_extreme_share",
    "herfindahl_top_cells",
    "n_demo_assets",
)

DEMO_COLUMNS: tuple[str, ...] = RANKED_COLUMNS + CONCENTRATION_COLUMNS


def empty_ranked_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in RANKED_COLUMNS})


def empty_demo_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in DEMO_COLUMNS})


def reason_for_asset_type(asset_type: str) -> str:
    if asset_type == "transmission_line":
        return (
            "S_max is the highest ranking_score cell among 0.1-degree samples; "
            "S_len is the sample-weighted mean. This is this month's burned-area "
            "probability, not spread, smoke, or loss. score_range is not a confidence interval."
        )
    return (
        "Asset sits in this cell. calibrated_probability is v8 isotonic p_c, a ranking "
        "aid, not a decision-grade probability. This is this month's burned-area "
        "probability, not spread, smoke, or loss."
    )


def _as_weight_map(cell_ids: Iterable[str], cell_weights: object | None) -> dict[str, float]:
    ids = [str(value) for value in cell_ids if value is not None and str(value) != ""]
    if isinstance(cell_weights, dict):
        return {str(key): float(value) for key, value in cell_weights.items() if str(key) in set(ids)}
    return {cell_id: 1.0 for cell_id in ids}


def expand_asset_cells(assets: pd.DataFrame) -> pd.DataFrame:
    """One row per (asset, cell) with sample weight L(a,c)."""
    working = assets.copy()
    if "cell_ids" not in working.columns:
        if "cell_id" in working.columns:
            working["cell_ids"] = working["cell_id"].map(lambda value: [value] if pd.notna(value) else [])
        else:
            raise ValueError("Assets need cell_ids or cell_id")
    rows: list[dict] = []
    weight_col = "cell_weights" if "cell_weights" in working.columns else None
    for record in working.itertuples(index=False):
        cell_ids = list(getattr(record, "cell_ids") or [])
        raw_weights = getattr(record, weight_col) if weight_col else None
        weights = _as_weight_map(cell_ids, raw_weights)
        unique_ids = list(dict.fromkeys(str(cell) for cell in cell_ids if cell is not None and str(cell) != ""))
        payload = record._asdict() if hasattr(record, "_asdict") else dict(zip(working.columns, record))
        for cell_id in unique_ids:
            row = {key: value for key, value in payload.items() if key not in {"cell_ids", "cell_weights", "cell_id"}}
            row["cell_id"] = cell_id
            row["sample_weight"] = float(weights.get(cell_id, 1.0))
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["asset_id", "cell_id", "sample_weight"])
    return pd.DataFrame(rows)


def aggregate_cell_scores(joined: pd.DataFrame) -> pd.DataFrame:
    """Implement equations (6)–(10) of ASSET_EXPOSURE_MATH.md.

    `joined` must have asset_id, cell_id, ranking_score, sample_weight, and
    optionally predicted_probability, risk_band, month_risk_percentile, month.
    """
    if joined is None or joined.empty:
        return empty_ranked_frame()
    required = {"asset_id", "cell_id", "ranking_score"}
    missing = required - set(joined.columns)
    if missing:
        raise ValueError(f"Joined table missing {sorted(missing)}")
    work = joined.copy()
    work["ranking_score"] = pd.to_numeric(work["ranking_score"], errors="coerce")
    work = work.dropna(subset=["ranking_score", "asset_id", "cell_id"])
    if "sample_weight" not in work.columns:
        work["sample_weight"] = 1.0
    work["sample_weight"] = pd.to_numeric(work["sample_weight"], errors="coerce").fillna(1.0).clip(lower=0.0)
    work["cell_id"] = work["cell_id"].astype(str)
    if work.empty:
        return empty_ranked_frame()

    def _one_asset(group: pd.DataFrame) -> pd.Series:
        unique = group.drop_duplicates("cell_id").reset_index(drop=True)
        scores = unique["ranking_score"].to_numpy(dtype=float)
        weights = unique["sample_weight"].to_numpy(dtype=float)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            weights = np.ones_like(scores)
            weight_sum = float(len(scores))
        s_max = float(scores.max())
        s_min = float(scores.min())
        s_len = float(np.dot(scores, weights) / weight_sum)
        # Weighted std of s_c; 0 for a single cell. Not a confidence interval.
        if len(scores) <= 1:
            s_std = 0.0
        else:
            s_std = float(np.sqrt(np.dot(weights, (scores - s_len) ** 2) / weight_sum))
        bands = unique["risk_band"].astype(str) if "risk_band" in unique.columns else pd.Series([""] * len(unique))
        n_high = int((bands == "high").sum())
        n_extreme = int((bands == "extreme").sum())
        n_cells = int(len(unique))
        n_ext = int(bands.isin(HOT_BANDS).sum())
        max_row = unique.loc[unique["ranking_score"].idxmax()]
        p_max = max_row["predicted_probability"] if "predicted_probability" in unique.columns else np.nan
        perc = max_row["month_risk_percentile"] if "month_risk_percentile" in unique.columns else np.nan
        band = max_row["risk_band"] if "risk_band" in unique.columns else ""
        month = max_row["month"] if "month" in unique.columns else pd.NA
        return pd.Series(
            {
                "cell_id": str(max_row["cell_id"]),
                "ranking_score": s_max,
                "length_weighted_score": s_len,
                "predicted_probability": p_max,
                "calibrated_probability": p_max,
                "risk_band": band,
                "month_risk_percentile": perc,
                "month": month,
                "n_cells": n_cells,
                "n_high_cells": n_high,
                "n_extreme_cells": n_extreme,
                "extreme_cell_fraction": float(n_ext / n_cells) if n_cells else 0.0,
                "score_range": s_max - s_min,
                "score_std": s_std,
            }
        )

    rows: list[dict] = []
    meta_cols = [column for column in ("asset_name", "asset_type", "region", "asset_source") if column in work.columns]
    for asset_id, group in work.groupby("asset_id", sort=False):
        metrics = _one_asset(group)
        payload = {"asset_id": asset_id, **metrics.to_dict()}
        first = group.iloc[0]
        for column in meta_cols:
            payload[column] = first[column]
        rows.append(payload)
    scored = pd.DataFrame(rows)
    if "asset_name" not in scored.columns:
        scored["asset_name"] = scored["asset_id"]
    if "asset_type" not in scored.columns:
        scored["asset_type"] = "unknown"
    if "region" not in scored.columns:
        scored["region"] = "global"
    if "asset_source" not in scored.columns:
        scored["asset_source"] = "unknown"
    scored["reason"] = scored["asset_type"].map(reason_for_asset_type)
    return assign_inspect_rank(scored)


def assign_inspect_rank(ranked: pd.DataFrame) -> pd.DataFrame:
    """Equation (11): inspect S_max first, then N_ext (high+extreme), then U(a)."""
    if ranked.empty:
        return empty_ranked_frame()
    out = ranked.copy()
    n_ext = (
        pd.to_numeric(out.get("n_high_cells", 0), errors="coerce").fillna(0)
        + pd.to_numeric(out.get("n_extreme_cells", 0), errors="coerce").fillna(0)
    )
    order = np.lexsort(
        (
            -pd.to_numeric(out["score_range"], errors="coerce").fillna(0.0).to_numpy(),
            -n_ext.to_numpy(),
            -pd.to_numeric(out["ranking_score"], errors="coerce").fillna(0.0).to_numpy(),
        )
    )
    ranks = np.empty(len(out), dtype=int)
    ranks[order] = np.arange(1, len(out) + 1)
    out["inspect_rank"] = ranks
    for column in RANKED_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, list(RANKED_COLUMNS)].sort_values("inspect_rank", kind="mergesort").reset_index(drop=True)


def portfolio_concentration(ranked: pd.DataFrame) -> dict[str, float]:
    """Equations (12)–(13). Public demo book only. Not customer TVaR."""
    if ranked is None or ranked.empty:
        return {
            "high_or_extreme_share": float("nan"),
            "herfindahl_top_cells": float("nan"),
            "n_demo_assets": 0.0,
        }
    n = int(len(ranked))
    bands = ranked["risk_band"].astype(str) if "risk_band" in ranked.columns else pd.Series([""] * n)
    share = float(bands.isin(HOT_BANDS).mean())
    counts = ranked.groupby("cell_id").size().to_numpy(dtype=float)
    herfindahl = float(np.sum((counts / n) ** 2)) if n else float("nan")
    return {
        "high_or_extreme_share": share,
        "herfindahl_top_cells": herfindahl,
        "n_demo_assets": float(n),
    }


def attach_concentration(demo: pd.DataFrame) -> pd.DataFrame:
    stats = portfolio_concentration(demo)
    out = demo.copy()
    for key, value in stats.items():
        out[key] = value
    return out
