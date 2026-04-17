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


def render_monthly_risk_maps(
    predictions_path: str | Path,
    output_dir: str | Path | None = None,
    value_col: str = "predicted_probability",
    run_root: str | Path | None = None,
    run_name: str | None = None,
) -> list[Path]:
    df = read_predictions(predictions_path)
    required = {"lat_center", "lon_center", value_col}
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

    climatology = (
        df.groupby(["month_num", "lat_center", "lon_center"], as_index=False)[value_col]
        .mean()
        .sort_values(["month_num", "lat_center", "lon_center"])
    )

    rendered_paths: list[Path] = []
    for month_num, month_df in climatology.groupby("month_num"):
        fig, ax = plt.subplots(figsize=(14, 6))
        scatter = ax.scatter(
            month_df["lon_center"],
            month_df["lat_center"],
            c=month_df[value_col],
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            s=14,
            marker="s",
            linewidths=0,
        )
        ax.set_title(f"Wildfire Risk Climatology for Month {int(month_num):02d}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.grid(True, linewidth=0.2, alpha=0.5)
        fig.colorbar(scatter, ax=ax, label="Predicted wildfire probability")
        fig.tight_layout()

        output_path = output_dir / f"wildfire_risk_month_{int(month_num):02d}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        rendered_paths.append(output_path)

    top_risk = (
        climatology.sort_values(value_col, ascending=False)
        .groupby("month_num", as_index=False)
        .head(20)
        .reset_index(drop=True)
    )
    top_risk_path = output_dir / "top_risk_cells.csv"
    top_risk.to_csv(top_risk_path, index=False)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="render_maps",
            config={
                "predictions_path": str(predictions_path),
                "value_col": value_col,
            },
            artifacts={
                "map_images": [str(path) for path in rendered_paths],
                "top_risk_cells": str(top_risk_path),
            },
            summary={"rendered_map_count": int(len(rendered_paths))},
        )
    return rendered_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Render climatological monthly wildfire risk maps")
    parser.add_argument("--predictions", required=True, help="Validation prediction CSV from train_baseline.py")
    parser.add_argument("--output-dir", help="Directory for PNG maps")
    parser.add_argument("--value-col", default="predicted_probability")
    parser.add_argument("--run-root", help="Optional root directory for a Colab-friendly run layout")
    parser.add_argument("--run-name", help="Optional run folder name inside --run-root")
    args = parser.parse_args()

    rendered_paths = render_monthly_risk_maps(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        value_col=args.value_col,
        run_root=args.run_root,
        run_name=args.run_name,
    )
    print(f"Rendered {len(rendered_paths)} monthly maps into {args.output_dir}")


if __name__ == "__main__":
    main()
