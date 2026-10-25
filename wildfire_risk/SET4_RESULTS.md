# Set 4

Set 4 checks whether the working model holds up one season at a time, and puts every source tried so far in one table.

The working model is v8: burn history, ERA5-Land without wind, and lagged FIRMS. Train 2020–2023, holdout 2024–2025. v8 itself is unchanged by this check.

## Keep rule

A layer stays only if a continent that matters moves by at least 0.003 ROC or 0.005 PR-AUC. Capture@10% of 0.005 counts the same way. Global ROC is not enough, because Africa is about 40% of the fires.

## Sources

Each layer was attached onto v3 and trained on its own. v4–v7 were not stacked. v8 is v3 plus lagged FIRMS. Ranking numbers use the raw score.

PR-AUC:

| Region | v3 | v4 land cover | v5 terrain | v6 population | v7 roads | v8 FIRMS |
|---|---:|---:|---:|---:|---:|---:|
| Global | 0.7311 | 0.7319 | 0.7314 | 0.7317 | 0.7311 | 0.7363 |
| Africa | 0.8827 | 0.8826 | 0.8827 | 0.8825 | 0.8828 | 0.8857 |
| Asia | 0.5120 | 0.5144 | 0.5136 | 0.5130 | 0.5121 | 0.5187 |
| Australia | 0.6465 | 0.6461 | 0.6468 | 0.6479 | 0.6465 | 0.6516 |
| Europe | 0.5081 | 0.5088 | 0.5077 | 0.5079 | 0.5071 | 0.5100 |
| North America | 0.4838 | 0.4852 | 0.4847 | 0.4856 | 0.4836 | 0.4885 |
| South America | 0.7191 | 0.7201 | 0.7189 | 0.7189 | 0.7190 | 0.7259 |

ROC:

| Region | v3 | v4 land cover | v5 terrain | v6 population | v7 roads | v8 FIRMS |
|---|---:|---:|---:|---:|---:|---:|
| Global | 0.9632 | 0.9641 | 0.9633 | 0.9636 | 0.9633 | 0.9645 |
| Africa | 0.9620 | 0.9620 | 0.9621 | 0.9619 | 0.9620 | 0.9629 |
| Asia | 0.9224 | 0.9268 | 0.9227 | 0.9236 | 0.9227 | 0.9271 |
| Australia | 0.8848 | 0.8849 | 0.8852 | 0.8860 | 0.8848 | 0.8861 |
| Europe | 0.9174 | 0.9224 | 0.9179 | 0.9190 | 0.9180 | 0.9222 |
| North America | 0.9236 | 0.9250 | 0.9242 | 0.9255 | 0.9242 | 0.9266 |
| South America | 0.9116 | 0.9129 | 0.9115 | 0.9118 | 0.9116 | 0.9129 |

Capture@10%, v3 against v8:

| Region | v3 | v8 |
|---|---:|---:|
| Global | 0.8139 | 0.8161 |
| Africa | 0.5292 | 0.5299 |
| Asia | 0.7088 | 0.7176 |
| Australia | 0.4373 | 0.4384 |
| Europe | 0.7051 | 0.7117 |
| North America | 0.6913 | 0.6996 |
| South America | 0.4703 | 0.4722 |

Calls: land cover, terrain, population, and roads drop. Lagged FIRMS stays. Asia, Australia, and South America clear the PR bar. North America capture rises by 0.0083. Africa was already near the ceiling.

Calibration is isotonic, fit on the holdout path, and is separate from the ranking score. v3 Brier 0.0268, mean probability 0.0610, fire rate 0.0569. v8 Brier 0.0265, mean probability 0.0603, ECE 0.0049. v8 isotonic ROC stays 0.9645 and isotonic PR-AUC is 0.7320.

## Yearly refit

The holdout score is one fit. This check fits again for each test year:

| Test year | Trains on |
|---|---|
| 2022 | 2020–2021 |
| 2023 | 2020–2022 |
| 2024 | 2020–2023 |
| 2025 | 2020–2024 |

2024 uses the same training years as v8, but it is a new fit scored on 2024 alone. 2025 is allowed to see 2024. Same-month FIRMS and the ERA5 wind family stay out.

The Colab run writes `reports/yearly_backtest.csv` and `reports/yearly_backtest.json` under `wildfire-risk-mvp-2020-2025-noleak-v8-yearly`. It does not write a new holdout model and does not overwrite v8. Scores in that table are raw ranking probabilities, by continent.

```bash
python wildfire_risk/train_baseline.py \
  --features /content/drive/MyDrive/Hephaelion/runs/wildfire-risk-mvp-2020-2025-noleak-v8/features/feature_table.csv \
  --run-root /content/drive/MyDrive/Hephaelion/runs \
  --run-name wildfire-risk-mvp-2020-2025-noleak-v8-yearly \
  --yearly-backtest-only \
  --backtest-start-year 2022 \
  --backtest-end-year 2025 \
  --n-estimators 100 \
  --drop-era5-family wind \
  --skip-regressor
```
