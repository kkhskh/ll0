# Set 4

Set 4 checked whether the working model holds up one season at a time, and collected every source tried so far in one table.

The working model stays v8: burn history, ERA5-Land without wind, and lagged FIRMS. Train 2020–2023, holdout 2024–2025. The yearly fits were a check. They were not saved over v8.

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

Each test year is a new fit on every earlier year. Same-month FIRMS and the ERA5 wind family stay out. Scores are raw ranking probabilities. 2024 uses the same training years as v8 and is scored on 2024 alone. 2025 is allowed to see 2024.

| Test year | Trains on | Rows in the fit | Fires in the test year | Fire rate |
|---|---|---:|---:|---:|
| 2022 | 2020–2021 | 2,264,520 | 62,131 | 0.0549 |
| 2023 | 2020–2022 | 3,396,780 | 67,973 | 0.0600 |
| 2024 | 2020–2023 | 4,529,040 | 67,913 | 0.0600 |
| 2025 | 2020–2024 | 5,661,300 | 61,006 | 0.0539 |

Each test year is 1,132,260 rows. The `other` region is ignored.

ROC:

| Region | 2022 | 2023 | 2024 | 2025 | v8 pooled |
|---|---:|---:|---:|---:|---:|
| Global | 0.9621 | 0.9616 | 0.9648 | 0.9647 | 0.9645 |
| Africa | 0.9639 | 0.9647 | 0.9645 | 0.9623 | 0.9629 |
| Asia | 0.9197 | 0.9205 | 0.9262 | 0.9266 | 0.9271 |
| Australia | 0.8946 | 0.8728 | 0.8931 | 0.8832 | 0.8861 |
| Europe | 0.8855 | 0.9221 | 0.9250 | 0.9179 | 0.9222 |
| North America | 0.9219 | 0.9157 | 0.9226 | 0.9326 | 0.9266 |
| South America | 0.9107 | 0.9156 | 0.9144 | 0.9147 | 0.9129 |

PR-AUC:

| Region | 2022 | 2023 | 2024 | 2025 | v8 pooled |
|---|---:|---:|---:|---:|---:|
| Global | 0.7200 | 0.7356 | 0.7448 | 0.7266 | 0.7363 |
| Africa | 0.8807 | 0.8890 | 0.8904 | 0.8801 | 0.8857 |
| Asia | 0.4663 | 0.5127 | 0.5292 | 0.4999 | 0.5187 |
| Australia | 0.6421 | 0.6690 | 0.6529 | 0.6483 | 0.6516 |
| Europe | 0.3222 | 0.4620 | 0.5283 | 0.4990 | 0.5100 |
| North America | 0.4569 | 0.5094 | 0.4992 | 0.4837 | 0.4885 |
| South America | 0.7343 | 0.7444 | 0.7518 | 0.7035 | 0.7259 |

Capture@10%:

| Region | 2022 | 2023 | 2024 | 2025 | v8 pooled |
|---|---:|---:|---:|---:|---:|
| Global | 0.8095 | 0.8030 | 0.8122 | 0.8241 | 0.8161 |
| Africa | 0.5533 | 0.5297 | 0.5165 | 0.5413 | 0.5299 |
| Asia | 0.6945 | 0.7038 | 0.7041 | 0.7366 | 0.7176 |
| Australia | 0.5059 | 0.4050 | 0.4554 | 0.4210 | 0.4384 |
| Europe | 0.5919 | 0.7319 | 0.7204 | 0.7030 | 0.7117 |
| North America | 0.6914 | 0.6635 | 0.6876 | 0.7253 | 0.6996 |
| South America | 0.4537 | 0.4645 | 0.4416 | 0.5099 | 0.4722 |

## What the years say

Global ROC stays between 0.9616 and 0.9648. v8’s pooled 0.9645 was not one season. 2024 alone, trained on 2020–2023, is 0.9648. 2025, trained through 2024, is 0.9647.

PR-AUC follows how much burned. 2024 has 67,913 fires and global PR-AUC 0.7448. 2025 has 61,006 fires and PR-AUC 0.7266, with the best capture of the four years at 0.8241. The pooled v8 PR-AUC of 0.7363 sits between those two years.

Africa stays near the ceiling in every year. Australia stays the weak high-fire continent. Its lowest ROC is 0.8728 in 2023, when Australia’s fire rate rose to 0.1891 (6,788 fires) and capture fell to 0.4050. South America and Asia keep a flat ROC. Their 2025 PR-AUC drop lines up with fewer fires: South America 14,655 to 11,107, Asia 11,913 to 10,081.

Europe in 2022 is the miss. PR-AUC 0.3222 and capture 0.5919, with two training years and 2,012 fires. From 2023 on, Europe is back near v8, PR-AUC 0.46 to 0.53.

## Where this leaves the model

v8 remains the working model. Set 4 is the evidence that the same skill shows up in each season from 2022 through 2025.

The table is on Drive at `runs/wildfire-risk-mvp-2020-2025-noleak-v8-yearly/reports/yearly_backtest.csv`.

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
