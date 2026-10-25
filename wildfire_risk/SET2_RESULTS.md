# Set 2 results

Set 2 added one static layer at a time onto the working model and retrained. None of the four layers earned a place. The working model is still v3.

## Working model

v3 is `wildfire-risk-mvp-2020-2025-noleak-v3`.

- Labels: MCD64A1 monthly burned area on a 0.5° land grid.
- Features: burn history plus ERA5-Land, with the wind family dropped.
- Split: train 2020–2023, holdout 2024–2025. LightGBM, 100 trees, isotonic calibration.
- Holdout: 2,264,520 rows, 128,919 fires, fire rate 0.0569.
- Ranking metrics below use `ranking_score`. Brier and mean probability use the isotonic probability.

v3 holdout, global: ROC 0.9632, PR-AUC 0.7311, capture@10% 0.8139, Brier 0.0268, mean probability 0.0610.

| Region | ROC | PR-AUC | Fires |
|---|---:|---:|---:|
| Africa | 0.9620 | 0.8827 | 52,006 |
| Asia | 0.9224 | 0.5120 | 21,994 |
| Australia | 0.8848 | 0.6465 | 11,655 |
| Europe | 0.9174 | 0.5081 | 4,533 |
| North America | 0.9236 | 0.4838 | 12,894 |
| South America | 0.9116 | 0.7191 | 25,762 |

Australia is the weak high-fire continent. Africa is about 40% of the fires and is nearly saturated, so it pulls the global ROC up. The `other` region is 731,088 rows and 75 fires; it is ignored.

Each Set 2 run was attached onto the v3 feature table. The layers were not stacked on each other.

## Keep rule

A layer stays only if a continent that matters moves by at least 0.003 ROC or 0.005 PR-AUC. Capture@10% of 0.005 counts the same way. Global ROC is not enough, because Africa dominates it.

## What was run

| Run | Layer | What was joined | Cells with data | Call |
|---|---|---|---:|---|
| v4 | MCD12Q1 land cover | IGBP class fractions, product year joined to the next calendar year | 94,737 | Drop |
| v5 | NASADEM terrain | Elevation and slope, western North America only (`-130,25,-100,52`) | 2,853 | Drop |
| v6 | GPWv4 population | 2020 density at 2.5 arc minutes, one static value for 2020–2025 | 70,123 | Drop |
| v7 | gROADS | Road length and density from the global geodatabase | 46,636 | Drop |

## v4 — land cover

Global ROC 0.9641, PR-AUC 0.7319, capture@10% 0.8138. That is +0.0009 ROC and +0.0008 PR against v3.

| Region | v4 ROC / PR | Change vs v3 |
|---|---|---|
| Africa | 0.9620 / 0.8826 | 0 / −0.0001 |
| Asia | 0.9268 / 0.5144 | +0.0044 / +0.0024 |
| Australia | 0.8849 / 0.6461 | +0.0001 / −0.0004 |
| Europe | 0.9224 / 0.5088 | +0.0050 / +0.0007 |
| North America | 0.9250 / 0.4852 | +0.0014 / +0.0014 |
| South America | 0.9129 / 0.7201 | +0.0013 / +0.0010 |

Australia, South America, and North America miss the bar. Asia and Europe clear the ROC bar. The top feature was `landcover_land_cover_valid_pixel_count` (importance 197), which is how many pixels in the cell NASA could classify: a land-versus-ice mask. Forest and savanna fractions were used, and they ranked well below that mask. The Asia and Europe ROC gains track that mask, not vegetation type. Dropped.

## v5 — terrain

Western North America only. Cells outside that box stayed unjoined and filled as 0, with `has_terrain` marking real terrain. Global ROC 0.9633, PR-AUC 0.7314.

| Region | v5 ROC / PR | Change vs v3 |
|---|---|---|
| Africa | 0.9621 / 0.8827 | +0.0001 / 0 |
| Asia | 0.9227 / 0.5136 | +0.0003 / +0.0016 |
| Australia | 0.8852 / 0.6468 | +0.0004 / +0.0003 |
| Europe | 0.9179 / 0.5077 | +0.0005 / −0.0004 |
| North America | 0.9242 / 0.4847 | +0.0006 / +0.0009 |
| South America | 0.9115 / 0.7189 | −0.0001 / −0.0002 |

North America was the only continent this layer could have helped. It moved +0.0006 ROC and +0.0009 PR. Elevation and slope were not in the top 25 features. At 0.5°, slope does not separate a burned cell from its neighbors. Dropped.

## v6 — population

Global ROC 0.9636, PR-AUC 0.7317, capture@10% 0.8144.

| Region | v6 ROC / PR | Change vs v3 |
|---|---|---|
| Africa | 0.9619 / 0.8825 | −0.0001 / −0.0002 |
| Asia | 0.9236 / 0.5130 | +0.0012 / +0.0010 |
| Australia | 0.8860 / 0.6479 | +0.0012 / +0.0014 |
| Europe | 0.9190 / 0.5079 | +0.0016 / −0.0002 |
| North America | 0.9255 / 0.4856 | +0.0019 / +0.0018 |
| South America | 0.9118 / 0.7189 | +0.0002 / −0.0002 |

The largest move is North America, +0.0019 ROC and +0.0018 PR. Australia is +0.0012 / +0.0014. Both are short of the bar. Dropped.

## v7 — roads

Global ROC 0.9633, PR-AUC 0.7311, capture@10% 0.8138. PR matches v3 at four decimals.

| Region | v7 ROC / PR | Change vs v3 |
|---|---|---|
| Africa | 0.9620 / 0.8828 | 0 / +0.0001 |
| Asia | 0.9227 / 0.5121 | +0.0003 / +0.0001 |
| Australia | 0.8848 / 0.6465 | 0 / 0 |
| Europe | 0.9180 / 0.5071 | +0.0006 / −0.0010 |
| North America | 0.9242 / 0.4836 | +0.0006 / −0.0002 |
| South America | 0.9116 / 0.7190 | 0 / −0.0001 |

`roads_length_km` landed in the top features at importance 76. The holdout ranking did not change. Australia is identical to v3 at four decimals. Dropped.

## Where this leaves the model

v3 remains the working model: burn history plus ERA5 without wind. Land cover, terrain, population, and roads at monthly 0.5° did not move the continents that the global score hides.

The v4–v7 artifacts stay on Drive under `runs/wildfire-risk-mvp-2020-2025-noleak-v4` through `v7`. They are recorded negative results.

Set 3 added lagged FIRMS onto v3 and kept it. The working model is v8. Same-month detections stay out of the features. Set 4 refits v8 once per year, from 2022 through 2025, and writes that table separately. See SET4_RESULTS.md.
