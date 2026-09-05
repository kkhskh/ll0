# Hephaelion Wildfire Risk MVP Backend

This backend is the first product spine for the monthly wildfire risk MVP. It serves run outputs already produced by the `wildfire_risk/` pipeline instead of retraining models inside the API.

## What It Serves

- `GET /health`
- `GET /api/v1/wildfire/runs`
- `GET /api/v1/wildfire/risk/summary`
- `GET /api/v1/wildfire/risk/monthly?month=7`
- Static map images under `/static/wildfire/...`

## Expected Run Layout

The API expects run folders created by the `wildfire_risk` pipeline, for example:

```text
artifacts/wildfire_risk_runs/<run-name>/
  labels/
  features/
  models/
    metrics.json
    validation_predictions.csv
  maps/
    wildfire_risk_month_01.png
    ...
    wildfire_risk_month_12.png
    top_risk_cells.csv
  reports/
    train_baseline_manifest.json
    render_maps_manifest.json
```

If `maps/top_risk_cells.csv` is missing, the API falls back to `models/validation_predictions.csv`.

## Install

```bash
python -m pip install -r backend/requirements.txt
```

## Run

By default the API looks for runs under:

```text
artifacts/wildfire_risk_runs
```

You can override that with `HEPHAELION_RUN_ROOT`:

```bash
export HEPHAELION_RUN_ROOT=/absolute/path/to/wildfire/runs
python -m uvicorn backend.app:app --reload --port 8000
```

## Example Requests

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/v1/wildfire/runs
curl "http://127.0.0.1:8000/api/v1/wildfire/risk/summary"
curl "http://127.0.0.1:8000/api/v1/wildfire/risk/monthly?month=7"
curl "http://127.0.0.1:8000/api/v1/wildfire/risk/monthly?run_name=my-run&month=2024-08&limit=10"
```

## Suggested MVP Flow

1. Run `wildfire_risk/build_labels.py`
2. Run `wildfire_risk/build_features.py`
3. Run `wildfire_risk/train_baseline.py`
4. Run `wildfire_risk/render_maps.py`
5. Start this API
6. Point the frontend or a notebook at the monthly risk endpoints
