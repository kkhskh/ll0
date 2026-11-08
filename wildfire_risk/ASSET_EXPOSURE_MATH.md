# Mathematics of the worldwide asset overlay

This note writes the formulas that turn a GIS join into a scored, uncertain,
inspectable ranking. The product claim is exact and does not grow with the
algebra:

**This is this month's burned-area probability overlaid on public power assets.**
It is not fire spread, not hourly wind, not smoke, and not insured loss.

Implemented in `wildfire_risk/asset_scores.py` and called from
`wildfire_risk/asset_scale.py`. v8 and v9 are not retrained.

## Notation

| Symbol | Meaning |
|---|---|
| \(c\) | A 0.5° cell from `wildfire_risk/grid.py` |
| \(a\) | A public power asset (plant, substation, or transmission line) |
| \(C(a)\) | Cells that asset \(a\) intersects this month |
| \(s_c\) | `ranking_score` of cell \(c\) (raw LightGBM probability; v8 ranking field) |
| \(p_c\) | `predicted_probability` of cell \(c\) (isotonic-calibrated) |
| \(\mathrm{band}(c)\) | `risk_band` \(\in \{\mathrm{monitor},\mathrm{elevated},\mathrm{high},\mathrm{extreme}\}\) |
| \(L(a,c)\) | Length proxy of asset \(a\) in cell \(c\): count of 0.1° sample points (lines) or \(1\) (points) |
| \(N\) | Number of assets in the public demo book |
| \(n_c\) | Number of demo-book assets whose max-score cell is \(c\) |

v8 trains 2020–2023 and holds out 2024–2025. The overlay never invents \(s_c\)
or \(p_c\). Missing cells are dropped, not filled.

---

## 1. Overlay, not spread

Hennessy and Chester (2025) intersect California transmission lines, substations,
and generators with a fire-threat raster and count assets in high-threat cells.
That is a GIS overlay of a static hazard field on linear and point assets, not a
spread model.

Taylor and Roald (2022, arXiv:2110.07348) overlay USGS Wildland Fire Potential
Index rasters on transmission lines and then *aggregate along the line*. They
state two geographic aggregations, which we adopt with burned-area scores in
place of WFPI:

> “Maximum risk defines the wildfire risk \(r_{l,s}\) as the maximum wildfire
> potential value along the line.”
>
> “Cumulative risk metric defines the wildfire risk as the integral of the
> wildfire risk along the line segment.”

Pacific Gas and Electric Company (2023) uses its Wildfire Distribution Risk
Model (WDRM v3) and Wildfire Transmission Risk Model (WTRM v1) to
**risk-prioritize inspections**, vegetation work, and hardening. The overlay
here is an inspect-first ranking of public assets on this month's burned-area
probability. It is not PG&E's WDRM, not a PSPS protocol, and not an operability
assessment driven by forecast wind.

### What we refuse from the line-fault literature

Dian, Cheng, Ye, Wu, Luo, Wang, Hui, Zhou, Zou, Yu, and Gong (2019), IEEE
Access (open access), couple a cellular-automaton **spread** model to an air-gap
breakdown line-outage model. Equation (40) in that paper maps operating voltage
against flame/ion/smoke breakdown voltage to a trip-out probability. We do not
implement that. We have no hourly wind, no flame length, and no insulation
physics.

The CPUC / Office of Energy Infrastructure Safety decision on PG&E's 2023–2025
WMP (Resolution SPD-20) treats PSPS as a mitigation with its own outage
consequence, not as an output of a burned-area overlay. We do not simulate PSPS.

---

## 2. Cell scores from v8

For each land cell \(c\) and calendar month \(t\), v8 stores a ranking score
\(s_c\) and an isotonic probability \(p_c\). Ranking uses \(s_c\). Probability
language, when used, uses \(p_c\) and is caveated.

Niculescu-Mizil and Caruana (2005, ICML) show that isotonic regression can
correct any monotonic distortion of predicted probabilities, and that it
overfits when the calibration set is small. Guo, Pleiss, Sun, and Weinberger
(2017, ICML; arXiv:1706.04599) define calibration as

\[
\mathbb{P}(\hat Y = Y \mid \hat P = p) = p \qquad \forall p \in [0,1] \tag{1}
\]

and measure the gap with Expected Calibration Error. Naeini, Cooper, and
Hauskrecht (2015, AAAI) write the binned estimator we use in
`wildfire_risk/evaluate.py`:

\[
\mathrm{ECE} = \sum_{i=1}^{K} P(i)\,|o_i - e_i| \tag{2}
\]

where \(o_i\) is the observed positive rate in bin \(i\), \(e_i\) is the mean
predicted probability in that bin, and \(P(i)\) is the bin weight. Brier (1950)
is the mean squared error of probability forecasts:

\[
\mathrm{Brier} = \frac{1}{n}\sum_{j=1}^{n}(p_j - y_j)^2 \tag{3}
\]

v8 holdout (this repository, `SET4_RESULTS.md`): Brier \(0.0265\), ECE
\(0.0049\), isotonic mean probability \(0.0603\) against a fire rate of
\(0.0569\). Those numbers say \(p_c\) is a **ranking aid**, not a
decision-grade probability for de-energization, insurance pricing, or PSPS.
Kumar, Kang, Chakraborty, and Mukhopadhyay (2024, arXiv:2402.15163) show on
wildfire maps that ECE, PR-AUC, and MSE rank models differently; we keep
\(s_c\) for order and \(p_c\) only as a labeled extra column.

---

## 3. Asset–cell incidence

Points (WRI plants, HIFLD substations) occupy one cell:

\[
C(a) = \{c(a)\}, \qquad L(a,c(a)) = 1 \tag{4}
\]

Lines are sampled every \(0.1^\circ\) along the polyline in geographic
coordinates, then each sample is assigned with `assign_points_to_grid`. Let
\(m(a,c)\) be the number of sample points of \(a\) that fall in \(c\). Then

\[
L(a,c) = m(a,c), \qquad C(a) = \{c : m(a,c) > 0\} \tag{5}
\]

**Approximation.** \(0.1^\circ\) sampling on a \(0.5^\circ\) grid is not a
geodesic length. At the equator \(0.1^\circ\) is about \(11\,\mathrm{km}\); at
\(60^\circ\) latitude it is about \(5.5\,\mathrm{km}\) in longitude. Taylor and
Roald (2022) compute true GIS length \(\times\) raster value with
`TabulateIntersection`. We document the cheaper proxy and do not claim
survey-grade miles.

EPSG:3310 (California Albers) is not used. The join is the same 0.5° formula
everywhere, including California.

---

## 4. Asset scores

### (1) Max score — inspect-first

\[
S_{\max}(a) = \max_{c \in C(a)} s_c \tag{6}
\]

This is Taylor and Roald's maximum-risk aggregation (their eq. 4) with \(s_c\)
in place of WFPI. PG&E (2023) ranks inspection work by modeled wildfire risk;
\(S_{\max}\) is the inspect-first score: a line that nicks one extreme cell is
flagged even if most of its length is quiet. Stored as `ranking_score`. The
cell that attains the max is `cell_id`.

### (2) Length-weighted score — lines only

\[
S_{\mathrm{len}}(a) = \frac{\sum_{c \in C(a)} s_c\, L(a,c)}{\sum_{c \in C(a)} L(a,c)} \tag{7}
\]

For a point, \(S_{\mathrm{len}}(a) = S_{\max}(a)\). For a line, this is a
**normalized** length-weighted mean. Taylor and Roald's cumulative metric
(their eq. 5) is the unnormalized sum \(\sum r_i\), which grows with line
length because they fold length into an undergrounding budget. We normalize so
that a long quiet line does not outrank a short hot line on length alone.
Stored as `length_weighted_score`.

A line with 90% of samples in a low cell and a short dip into an extreme cell
therefore has \(S_{\max}\) in the extreme cell and \(S_{\mathrm{len}}\) pulled
down. Both numbers are shown. Neither is a spread probability.

### (3) Extreme-cell count and fraction

Let \(\mathcal{H} = \{\mathrm{high},\mathrm{extreme}\}\).

\[
N_{\mathrm{ext}}(a) = \bigl|\{c \in C(a) : \mathrm{band}(c) \in \mathcal{H}\}\bigr| \tag{8}
\]

\[
f_{\mathrm{ext}}(a) = \frac{N_{\mathrm{ext}}(a)}{|C(a)|} \tag{9}
\]

`n_high_cells` and `n_extreme_cells` are the two band counts separately.
`n_cells` is \(|C(a)|\). `extreme_cell_fraction` is \(f_{\mathrm{ext}}\).
Hennessy and Chester (2025) report the share of California lines, substations,
and plants in high-threat zones; \(f_{\mathrm{ext}}\) is the per-asset analogue
on this month's v8 bands, not Cal-Adapt threat classes.

### (4) Rank uncertainty from crossing several cells

\[
U(a) = \max_{c \in C(a)} s_c - \min_{c \in C(a)} s_c \tag{10}
\]

Stored as `score_range`. Optionally we also store the sample-weighted standard
deviation \(\sigma(a)\) of \(\{s_c\}\) as `score_std`. **This is not a
confidence interval.** No paper cited here justifies treating the range of
neighboring 0.5° monthly scores as a 95% CI on ignition, outage, or loss. It
only says: if \(U(a)\) is large, the inspect-first rank is sensitive to which
intersected cell is used. A point asset has \(U(a)=0\).

### Inspect rank

\[
\mathrm{inspect\_rank}(a) = \mathrm{rank}\bigl(-S_{\max}(a),\, -N_{\mathrm{ext}}(a),\, -U(a)\bigr) \tag{11}
\]

Ties break toward more extreme cells, then toward larger score range. The
reason string still says this is this month's burned-area probability, not
spread or loss.

### (6) Calibrated exceedance for points

For plants and substations, report \(p_c\) of the occupied cell as
`calibrated_probability`. For lines, report \(p_c\) of the \(S_{\max}\) cell
(inspect-first) and do not treat a length-weighted \(p\) as an exceedance
probability. Given v8 Brier/ECE above and Niculescu-Mizil and Caruana (2005),
\(p_c\) is a ranking aid, not a decision-grade probability.

---

## 5. Demo-book concentration (not customer TVaR)

Let the demo book be a public subset of ranked assets (named plants first). Let
\(N\) be its size and \(n_c\) the number of those assets whose max-score cell
is \(c\).

Share in high or extreme:

\[
\pi_{\mathcal{H}} = \frac{1}{N}\bigl|\{a : \mathrm{band}(c^\star(a)) \in \mathcal{H}\}\bigr| \tag{12}
\]

where \(c^\star(a)\) is the cell attaining \(S_{\max}(a)\). Stored as
`high_or_extreme_share`.

Herfindahl–Hirschman concentration over max cells:

\[
H = \sum_{c} \left(\frac{n_c}{N}\right)^2 \tag{13}
\]

Hirschman (1964) records the index; the U.S. Department of Justice and Federal
Trade Commission *Horizontal Merger Guidelines* (2010, §5.3) use \(H = \sum s_i^2\)
on market shares. Zvezdov (2019, SOA ERM symposium monograph) discusses
geospatial concentration and the back-allocation of VaR/TVaR onto locations.
The Institute and Faculty of Actuaries / Society of Actuaries catastrophe
modelling guidance defines TVaR as the mean loss above a percentile of a **loss**
distribution.

**We do not compute TVaR.** There is no dollar loss, no vulnerability curve, no
RMS/Verisk event set, and no customer book. \(H\) is concentration of the
**public demo book** on this month's max cells. It is not a customer TVaR, not
an AAL, and not an OEP/AEP curve.

---

## 6. Completeness bias of the asset layer

The ranking is only as complete as the asset file.

- Barrington-Leigh and Millard-Ball (2017, *PLOS ONE*): OSM roads were about
  83% complete globally in 2016, with a U-shaped relationship to density and
  large between-country gaps. That paper is roads, not power lines; it is the
  standard public evidence that OSM completeness is geographically biased.
- MapYourGrid (2025) estimates roughly 75% of world high-voltage circuit length
  is in OSM, with most of the gap in Asia (China, India). Higher voltage maps
  more completely than distribution.
- Byers, Friedrich, Hennig, Kressig, Li, Malaguzzi Valeri, and McCormick (2018,
  WRI technical note): the Global Power Plant Database is generation sites, not
  substations. We label those rows `asset_type=plant` and do not invent
  substations outside HIFLD.
- MapYourGrid (2025-09-18): about 25% of HIFLD U.S. line features cite OSM as a
  source. HIFLD is still US-only.
- Arderne, Zorn, Nicolas, and Koks (2020, *Scientific Data*): gridfinder
  **predicts** MV/LV lines from nighttime lights. We refuse that file. Predicted
  geometries are invented assets for this product.

World Bank `osm_power_tmm.zip` (CC-BY-4.0, 2016 OSM transmission extract)
returned HTTP 403 from this project. The ODbL GeoParquet
`power_line.parquet` (Geofabrik extracts; © OpenStreetMap contributors) returned
HTTP 200 and is the global line fallback, queried by continent bbox, cached
under `datasets/wildfires/assets/`. North America falls back to HIFLD if that
query fails.

There are **no OSM substations globally** in this overlay. Outside the United
States, substations are missing, not zero-risk.

---

## 7. Claims we refuse

| Claim | Why not |
|---|---|
| Fire spread to the asset | No FARSITE/CA model; Dian et al. (2019) is a different paper |
| Hourly wind / PSPS threshold | PG&E OA model uses forecast wind at structures; we have a month |
| Smoke or air-gap flashover | No flame/ion/smoke zones |
| Insured loss, AAL, TVaR, OEP | No vulnerability, no event set, no customer book |
| Decision-grade \(p_c\) | v8 ECE/Brier say ranking aid only |
| Complete global grid | OSM/HIFLD completeness is biased; GPPD is plants |

The reason string on every ranked row repeats: this is this month's burned-area
probability, not spread, smoke, or loss.

---

## References (all retrieved; none invented)

1. Eleanor M. Hennessy and Mikhail V. Chester (2025). Electric utility vulnerability to wildfires and post-fire debris flows in California. *Environmental Research: Infrastructure and Sustainability* 5:015019. https://doi.org/10.1088/2634-4505/adb90a
2. Sofia Taylor and Line A. Roald (2022). A framework for risk assessment and optimal line upgrade selection to mitigate wildfire risk. arXiv:2110.07348. https://arxiv.org/abs/2110.07348
3. Pacific Gas and Electric Company (2023). *2023–2025 Wildfire Mitigation Plan*, Revision 3 (public). https://www.pge.com/content/dam/pge/docs/outages-and-safety/outage-preparedness-and-support/pge-wmp-r3-092723.pdf
4. California Public Utilities Commission / Office of Energy Infrastructure Safety (2023). Final Resolution SPD-20, PG&E 2023–2025 WMP. https://www.cpuc.ca.gov/-/media/cpuc-website/divisions/safety-policy-division/documents/final-resolution-spd-20-pge-wmp-decision-2023-2025.pdf
5. Songyi Dian, Peng Cheng, Qiang Ye, Jirong Wu, Ruisen Luo, Chen Wang, Dafeng Hui, Ning Zhou, Dong Zou, Qin Yu, and Xiao-Feng Gong (2019). Integrating wildfires propagation prediction into early warning of electrical transmission line outages. *IEEE Access*. https://doi.org/10.1109/ACCESS.2019.2894141 — **open access. We cite it as a spread/outage model we do not implement.**
6. Alexandru Niculescu-Mizil and Rich Caruana (2005). Predicting good probabilities with supervised learning. *ICML* 2005. https://icml.cc/Conferences/2005/proceedings/papers/079_GoodProbabilities_NiculescuMizilCaruana.pdf
7. Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger (2017). On calibration of modern neural networks. *ICML* 2017. arXiv:1706.04599. https://arxiv.org/abs/1706.04599
8. Mahdi Pakdaman Naeini, Gregory F. Cooper, and Milos Hauskrecht (2015). Obtaining well calibrated probabilities using Bayesian binning. *AAAI*. https://ojs.aaai.org/index.php/AAAI/article/view/9602
9. Glenn W. Brier (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78(1):1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
10. Harshit Kumar, Beomseok Kang, Biswadeep Chakraborty, and Saibal Mukhopadhyay (2024). Has the deep neural network learned the stochastic process? A wildfire perspective. arXiv:2402.15163. https://arxiv.org/abs/2402.15163
11. Albert O. Hirschman (1964). The paternity of an index. *American Economic Review* 54(5):761–762. (JSTOR; paywalled. Formula used from the public DOJ/FTC guidelines below.)
12. U.S. Department of Justice and Federal Trade Commission (2010). *Horizontal Merger Guidelines*, §5.3 (HHI). https://www.justice.gov/atr/horizontal-merger-guidelines-08192010
13. Ivelin M. Zvezdov (2019). Geospatial metrics for insurance risk concentration and diversification. SOA ERM Symposium monograph. https://www.soa.org/globalassets/assets/files/resources/essays-monographs/2019-erm-symposium/mono-2019-erm-zvezdov.pdf
14. Institute and Faculty of Actuaries (public guidance, retrieved 2026). Catastrophe modelling guidance for non-catastrophe modellers (VaR/TVaR, OEP/AEP). https://web.actuaries.ie/sites/default/files/erm-resources/catastrophe_modelling_guidance_for_non_catastrophe_modellers.pdf
15. Christopher Barrington-Leigh and Adam Millard-Ball (2017). The world’s user-generated road map is more than 80% complete. *PLOS ONE* 12(8):e0180698. https://doi.org/10.1371/journal.pone.0180698
16. MapYourGrid (2025). High-voltage line length and OSM coverage (~75% globally). https://mapyourgrid.org/blog/20251128-LineLength/
17. MapYourGrid (2025-09-18). OSM as a source inside HIFLD (~25% of U.S. line features). https://mapyourgrid.org/blog/20250918-HIFLD/
18. Logan Byers, Johannes Friedrich, Roman Hennig, Aaron Kressig, Xinyue Li, Laura Malaguzzi Valeri, and Colin McCormick (2018). *A Global Database of Power Plants*. WRI technical note. https://www.wri.org/research/global-database-power-plants
19. C. Arderne, C. Zorn, C. Nicolas, and E. E. Koks (2020). Predictive mapping of the global power system using open data. *Scientific Data* 7:19. https://doi.org/10.1038/s41597-019-0347-4 — **cited as a predicted-line product we refuse to use as assets.**
20. Hephaelion v8 holdout metrics: `wildfire_risk/SET4_RESULTS.md` in this repository (Brier 0.0265, ECE 0.0049).

Paywalled items: Hirschman (1964) on JSTOR; some IEEE Xplore full texts (we used the open IEEE Access paper and the public arXiv of Taylor and Roald instead of guessing paywalled PDFs).
