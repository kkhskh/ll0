from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .run_artifacts import create_run_layout, write_stage_manifest
except ImportError:  # pragma: no cover - allows direct script execution
    from run_artifacts import create_run_layout, write_stage_manifest


def read_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "month" not in df.columns:
        raise ValueError("Prediction table must include a `month` column")
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").astype(str)
    df["month_num"] = pd.to_datetime(df["month"]).dt.month
    return df


def choose_score_column(df: pd.DataFrame, value_col: str) -> str:
    if value_col in df.columns and value_col not in {"risk_percentile", "month_risk_percentile"}:
        return value_col
    for candidate in ("ranking_score", "raw_predicted_probability", "predicted_probability"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Prediction table needs ranking_score or predicted_probability to render maps")


def add_month_percentiles(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.copy()
    out["month_risk_percentile"] = out.groupby("month_num", sort=False)[score_col].rank(
        method="average",
        pct=True,
    )
    return out


def render_monthly_risk_maps(
    predictions_path: str | Path,
    output_dir: str | Path | None = None,
    value_col: str = "month_risk_percentile",
    run_root: str | Path | None = None,
    run_name: str | None = None,
    absolute: bool = False,
) -> list[Path]:
    df = read_predictions(predictions_path)
    score_col = choose_score_column(df, value_col)
    required = {"lat_center", "lon_center", score_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {sorted(missing)}")

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_dir = layout.maps_dir
    elif output_dir is None:
        raise ValueError("Either output_dir or run_root must be provided")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_cols = [
        column
        for column in ["month_num", "cell_id", "lat_center", "lon_center"]
        if column in df.columns
    ]
    climatology = (
        df.groupby(group_cols, as_index=False)[score_col]
        .mean()
        .sort_values(["month_num", "lat_center", "lon_center"])
    )
    if absolute:
        plot_col = score_col
        color_label = "Predicted wildfire probability"
        title_kind = "probability"
    else:
        climatology = add_month_percentiles(climatology, score_col)
        plot_col = "month_risk_percentile"
        color_label = "Within-month risk percentile"
        title_kind = "percentile rank"

    rendered_paths: list[Path] = []
    for month_num, month_df in climatology.groupby("month_num"):
        fig, ax = plt.subplots(figsize=(14, 6))
        scatter = ax.scatter(
            month_df["lon_center"],
            month_df["lat_center"],
            c=month_df[plot_col],
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            s=14,
            marker="s",
            linewidths=0,
        )
        ax.set_title(f"Wildfire risk {title_kind} for month {int(month_num):02d}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.grid(True, linewidth=0.2, alpha=0.5)
        fig.colorbar(scatter, ax=ax, label=color_label)
        fig.tight_layout()

        output_path = output_dir / f"wildfire_risk_month_{int(month_num):02d}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        rendered_paths.append(output_path)

    top_risk = (
        climatology.sort_values(plot_col, ascending=False)
        .groupby("month_num", as_index=False)
        .head(20)
        .reset_index(drop=True)
    )
    top_risk["risk_rank"] = top_risk.groupby("month_num").cumcount() + 1
    percentile_col = "month_risk_percentile" if "month_risk_percentile" in top_risk.columns else plot_col
    top_risk["risk_percentile"] = top_risk[percentile_col]
    top_risk["relative_risk_score"] = (pd.to_numeric(top_risk["risk_percentile"], errors="coerce") * 100.0).clip(
        0.0,
        100.0,
    )
    top_risk["risk_band"] = pd.cut(
        top_risk["risk_percentile"],
        bins=[-0.01, 0.5, 0.8, 0.95, 1.01],
        labels=["monitor", "elevated", "high", "extreme"],
    ).astype(str)
    top_risk_path = output_dir / "top_risk_cells.csv"
    top_risk.to_csv(top_risk_path, index=False)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="render_maps",
            config={
                "predictions_path": str(predictions_path),
                "value_col": value_col,
                "score_col": score_col,
                "plot_col": plot_col,
                "absolute": absolute,
            },
            artifacts={
                "map_images": [str(path) for path in rendered_paths],
                "top_risk_cells": str(top_risk_path),
            },
            summary={"rendered_map_count": int(len(rendered_paths)), "output_dir": str(output_dir)},
        )
    return rendered_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Render climatological monthly wildfire risk maps")
    parser.add_argument("--predictions", required=True, help="Validation prediction CSV from train_baseline.py")
    parser.add_argument("--output-dir", help="Directory for PNG maps")
    parser.add_argument("--value-col", default="month_risk_percentile")
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Color raw probabilities with vmin=0 vmax=1 instead of within-month percentiles.",
    )
    parser.add_argument("--run-root", help="Optional root directory for a Colab-friendly run layout")
    parser.add_argument("--run-name", help="Optional run folder name inside --run-root")
    args = parser.parse_args()

    rendered_paths = render_monthly_risk_maps(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        value_col=args.value_col,
        run_root=args.run_root,
        run_name=args.run_name,
        absolute=args.absolute,
    )
    output_dir = rendered_paths[0].parent if rendered_paths else args.output_dir
    print(f"Rendered {len(rendered_paths)} monthly maps into {output_dir}")


if __name__ == "__main__":
    main()
