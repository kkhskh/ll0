from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

try:
    from .evaluate import format_continent_table, save_metrics, score_continents
    from .run_artifacts import create_run_layout, write_stage_manifest
except ImportError:  # pragma: no cover - allows direct script execution
    from evaluate import format_continent_table, save_metrics, score_continents
    from run_artifacts import create_run_layout, write_stage_manifest


LOGGER = logging.getLogger(__name__)


def score_prediction_regions(
    predictions_path: str | Path,
    output_dir: str | Path | None = None,
    run_root: str | Path | None = None,
    run_name: str | None = None,
) -> dict:
    LOGGER.info("Reading predictions %s", predictions_path)
    df = pd.read_csv(predictions_path)
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").astype(str)
    LOGGER.info("Loaded %s rows", f"{len(df):,}")

    rows = score_continents(df)
    table = format_continent_table(rows)
    LOGGER.info("Continent scores:\n%s", table)

    layout = None
    if run_root is not None:
        layout = create_run_layout(run_root, run_name=run_name)
        output_dir = layout.reports_dir
    elif output_dir is None:
        output_dir = Path(predictions_path).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "predictions_path": str(predictions_path),
        "n_rows": int(len(df)),
        "note": (
            "Ranking metrics use ranking_score. Brier and mean predicted probability use "
            "isotonic predicted_probability. Continents are coarse lat/lon boxes."
        ),
        "regions": rows,
    }
    json_path = save_metrics(payload, output_dir / "continent_metrics.json")
    csv_path = output_dir / "continent_metrics.csv"
    pd.DataFrame(
        [
            {key: value for key, value in row.items() if key not in {"classification", "classification_raw"}}
            for row in rows
        ]
    ).to_csv(csv_path, index=False)

    if layout is not None:
        write_stage_manifest(
            layout=layout,
            stage_name="score_regions",
            config={"predictions_path": str(predictions_path)},
            artifacts={"continent_metrics_json": str(json_path), "continent_metrics_csv": str(csv_path)},
            summary={"regions": [row["region"] for row in rows]},
        )
    return {"table": table, "rows": rows, "json_path": json_path, "csv_path": csv_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score wildfire holdout predictions by continent")
    parser.add_argument("--predictions", required=True, help="validation_predictions.csv from train_baseline.py")
    parser.add_argument("--output-dir", help="Where to write continent_metrics.json")
    parser.add_argument("--run-root", help="Optional Colab run root")
    parser.add_argument("--run-name", help="Optional run folder name inside --run-root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    results = score_prediction_regions(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        run_root=args.run_root,
        run_name=args.run_name,
    )
    print(results["table"])
    print(json.dumps({"json_path": str(results["json_path"]), "csv_path": str(results["csv_path"])}, indent=2))


if __name__ == "__main__":
    main()
