# Set 5

Set 5 is aimed at the two misses Set 4 separated.

Europe in 2022 is a short-record miss. Australia in 2023 is a heavy-year miss that a longer record did not remove. They take different fixes. v8 stays the working model while those fixes are measured against it.

## Europe in 2022

Train 2020–2021, test 2022: ROC 0.8855, PR-AUC 0.3222, capture@10% 0.5919, 2,012 fires, fire rate 0.0373.

2023 had fewer fires (1,686) and a lower fire rate (0.0313), and the score recovered to ROC 0.9221, PR-AUC 0.4620, capture 0.7319 once 2022 was in the fit. 2024 and 2025, with the longer fit, stay near PR-AUC 0.50–0.53 and capture about 0.70. A quiet year did not cause the 2022 drop.

Europe is about 3% of the fires in a year. The loss is dominated by Africa. Two European summers are a thin sample next to 25,000 African fires. The features that carry the model are last month’s burned fraction, the previous 12 months, and lagged monthly weather. Those light up cells that burned in 2020 or 2021. 2022’s bad European season spread into places those two summers did not burn, including the Atlantic side of France. 2021’s large Greek fires were already in the training set, so the miss is a season in a different part of the box, not an empty history.

The same Europe box is a wide mix. South of 46°N is Iberia, southern France, Italy, and the Balkans. North of 46°N runs through Germany, Scandinavia, Ukraine, and west to Moscow. The Africa box is applied first and claims every cell at or below 37.5°N, so Seville, Faro, the southern Peloponnese, Crete, and Cyprus are scored as Africa. Their fires never enter the Europe number, and they are too few to move Africa.

## Where that miss applies

It applies to a region-year with all three of these:

- a few percent of global fires, so the fit barely spends splits on it
- a season that burns cells the training summers left quiet, so the lag features stay near zero
- a training window of about two seasons

That is the 2022 Europe row. It is the number to ignore when quoting the model.

## v8 holdout, split

Scored on the existing v8 predictions, 2024–2025. Continent totals are unchanged. `europe_south` is below 46°N. `europe_north` is 46°N and above. `africa_fringe` is the strip the Africa box claims.

| Region | Rows | Fires | Fire rate | ROC | PR-AUC | Capture@10% |
|---|---:|---:|---:|---:|---:|---:|
| Europe | 107,832 | 4,533 | 0.0420 | 0.9222 | 0.5100 | 0.7117 |
| europe_south | 27,456 | 1,543 | 0.0562 | 0.8775 | 0.3943 | 0.5509 |
| europe_north | 80,376 | 2,990 | 0.0372 | 0.9357 | 0.5756 | 0.7940 |
| africa_fringe | 4,680 | 125 | 0.0267 | 0.8880 | 0.3063 | 0.6480 |

The north holds the published Europe number up. ROC 0.9357, PR-AUC 0.5756, capture 0.7940, and 2,990 of the 4,533 fires. The south is the weak piece: ROC 0.8775, PR-AUC 0.3943, capture 0.5509, close to Australia. The fringe is 125 fires. It is missing from the Europe score and too small to move Africa.

Training through 2023 raised the mixed Europe score from the 2022 miss to PR-AUC 0.51. It left the Mediterranean south at 0.39. More seasons are aimed at that south. The headline recovery is not a fix there.

The same pattern applies to the first year a district burns off its recent map. Gironde in 2022 is the prototype. It also applies to any later refit that goes back to a two-year window.

It does not apply to Africa, South America, or the global ROC. Those held in 2022. It is a different miss from Australia in 2023: that year was trained on 2020–2022, Australia’s ROC still fell to 0.8728, and capture fell to 0.4050. More seasons did not fix the heavy Australian year.

## Order of work

1. Split the Europe score. Done. The weak piece is `europe_south`, not the north.
2. Daily fire-weather, rolled up to the month and lagged, attached onto the current table. Judge it on Australia, North America, and `europe_south`. The month row is hot-dry days (at least 30 C, at most 40% humidity, under 1 mm of rain), the Canadian fire-weather index, and the worst 7-day mean of that index. Same-month values are dropped on attach. 2019 is included so January 2020 has a prior year. v8 stays in place. The new fit is v9.
3. Add MCD64A1, ERA5-Land, and lagged FIRMS for 2010–2019, with 2009 present so the first year’s lag is real. Holdout stays 2024–2025. The question is whether `europe_south` moves once more Mediterranean seasons are in the fit.
4. A finer grid last, and only for Australia or western North America.

```bash
python -m wildfire_risk.score_regions \
  --predictions /content/drive/MyDrive/Hephaelion/runs/wildfire-risk-mvp-2020-2025-noleak-v8/models/validation_predictions.csv \
  --run-root /content/drive/MyDrive/Hephaelion/runs \
  --run-name wildfire-risk-mvp-2020-2025-noleak-v8-europe
```
