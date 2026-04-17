# MCD64A1 Ingestion

The monthly wildfire-risk pipeline can now use a directory of downloaded `MCD64A1` HDF tiles directly.

## What the builder expects

- Input path: a folder containing files like `MCD64A1.A2020001.h10v04.061....hdf`
- Product: `MCD64A1` Collection `061`
- Burn signal used: the `Burn Date` layer
  - `> 0` means burned
  - `0` means unburned land
  - `-1` means missing data
  - `-2` means water

The current implementation aggregates burned pixels into the coarse global grid and computes:

- `burned_area_km2`
- `burned_pixel_count`
- `burned_fraction`
- `binary_risk_label`

## Example

```bash
python wildfire_risk/build_labels.py \
  --input "/Users/shkh/Downloads/MCD64A1_061-20260417_045030" \
  --input-format mcd64a1-hdf \
  --output "artifacts/mcd64a1_labels_2020_2025.csv" \
  --start-month 2020-01 \
  --end-month 2025-12 \
  --resolution-deg 0.5 \
  --burned-fraction-threshold 0.0001
```

## Current practical use

Your downloaded folder currently covers `2020-01` through `2026-02`, which is enough to:

- build the real label ingestion pipeline now
- train a first baseline on recent years
- backfill older years later once the ingestion path is proven

## Notes

- This path uses `pyhdf` to read NASA HDF4 files.
- It uses the embedded MODIS sinusoidal tile metadata and converts burned pixel centers into lat/lon before assigning them to the wildfire-risk grid.
- The implementation counts only burned pixels for aggregation, which keeps processing manageable even for global monthly folders.
