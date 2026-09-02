# Independent validation — SeabedNet v5, Churchill corridor

Test set: GMRT multibeam cells (research cruises: Amundsen, Healy, Knorr, Armstrong, Merian — **not** CHS holdings) at pixels where CHS NONNA-100 has **no published sounding** and the model produced a fill (≤60 px from the nearest sounding). The model never saw these depths. Baselines: nearest published sounding; gravity-derived prior (the model's own input). Run: `indep_validate.py corridor_out`, 2026-08-31. 386685 cells, 17 blocks.

## Headline (shelf water 50–400 m, n = 109044)

| | MAE (m) | bias (m, + = model shallower) |
|---|---|---|
| **v5 model** | **4.9** | +3.0 |
| nearest published sounding | 13.4 | -0.3 |
| gravity prior | 5.5 | +1.3 |

By distance from the nearest published sounding (shelf water):

| distance | n | v5 | nearest | gravity | mean σ |
|---|---|---|---|---|---|
| 0–0.5 km | 37313 | 4.8 | 8.5 | 5.9 | 3.6 |
| 0.5–1 km | 31657 | 4.6 | 12.9 | 5.5 | 4.7 |
| 1–2 km | 27743 | 5.0 | 17.5 | 4.8 | 5.3 |
| 2–4 km | 12151 | 5.5 | 20.8 | 5.8 | 6.2 |
| 4–8 km | 180 | 2.7 | 3.7 | 5.1 | 8.3 |

By the model's own σ (shelf water, quintiles):

| σ range | n | v5 | gravity | nearest |
|---|---|---|---|---|
| 0.3–2.0 | 21810 | 4.1 | 3.0 | 5.5 |
| 2.0–3.0 | 21877 | 3.9 | 3.3 | 6.6 |
| 3.0–4.2 | 21833 | 4.0 | 3.8 | 7.1 |
| 4.2–6.6 | 21885 | 5.1 | 5.8 | 15.2 |
| 6.6–80.8 | 21810 | 7.1 | 11.6 | 32.7 |

## What this says, plainly

1. **The model beats copying the nearest sounding by 2–4×** and the gap widens with distance (5.5 m vs 20.8 m at 2–4 km). That is the claim the atlas makes, and it holds on data the model never saw.
2. **On these cells the model is only on par with the gravity prior** (4.9 vs 5.5 m). The model earns its keep only where σ is high (top quintile: 7.1 vs 11.6 m); where σ is low, gravity alone is slightly better. Reason: research-cruise swaths run through deep channels and basins where gravity works. The glacial shelf where gravity fails — the water that matters for Churchill — is exactly where cruises don't go and NONNA is dense, so it is under-represented here (n < 600 per block in the Hudson Bay blocks).
3. **Deep water (>1000 m, n = 265934): model 10.9 m vs gravity 6.3 m.** Off the shelf the model should defer to gravity. Fix: gate the completion by depth/σ (use gravity beyond the shelf break). Slope 400–1000 m is worst (model 29.5 m, bias -27.4).
4. **σ is under-confident by ~1.7× and heavy-tailed.** Shelf: |err| ≤ σ in 48% of cells (should be 68%), ≤ 2σ in 74% (should be 95%); the 95th percentile of |err|/σ is 4.5. Correlation σ vs |err| = 0.30. σ ranks risk correctly but its scale is wrong → isotonic recalibration on this set before any σ number goes on the page. The Thamesborg '98th percentile' statement is a rank, so it survives; the '48 m' does not.
5. **Coastal cells (<50 m, n = 5858, median depth -8 m, median 131 m from a sounding) are excluded from the headline.** There the model reads -20 m too deep — but so does the nearest published sounding (-39 m median), and the gravity prior reads shallower by +9 m. Three blocks, all at the shoreline; most likely GMRT's coastal grid, not measured swath. Unresolved — flagged, not hidden.

## Per block

| block | n | v5 | nearest | gravity | bias | median depth |
|---|---|---|---|---|---|---|
| b0025_0030 | 118 | 3.2 | 2.2 | 1.5 | +3.0 | -315 |
| b0026_0030 | 37 | 3.7 | 2.8 | 13.3 | +0.2 | -341 |
| b0026_0031 | 42 | 2.9 | 2.5 | 3.2 | +2.9 | -194 |
| b0027_0030 | 521 | 9.7 | 8.8 | 22.8 | +9.0 | -413 |
| b0027_0031 | 105 | 8.4 | 9.3 | 17.2 | +6.2 | -212 |
| b0028_0029 | 140 | 4.4 | 4.3 | 4.9 | +3.7 | -163 |
| b0029_0029 | 60 | 1.3 | 1.3 | 1.9 | +1.1 | -175 |
| b0034_0031 | 4268 | 21.3 | 30.5 | 11.5 | -19.4 | -7 |
| b0034_0042 | 93027 | 5.1 | 14.6 | 5.9 | +3.3 | -273 |
| b0035_0042 | 10268 | 1.9 | 6.5 | 1.3 | +1.4 | -188 |
| b0036_0043 | 5079 | 5.3 | 7.4 | 3.6 | -3.7 | -1006 |
| b0036_0044 | 212205 | 8.2 | 35.9 | 6.3 | +3.5 | -2483 |
| b0038_0040 | 29 | 19.9 | 22.3 | 8.9 | -19.9 | -2 |
| b0040_0043 | 1542 | 21.6 | 65.5 | 12.7 | -21.4 | -8 |
| b0040_0045 | 3497 | 7.7 | 26.6 | 4.3 | +4.7 | -3094 |
| b0041_0045 | 50725 | 25.0 | 45.5 | 6.6 | -12.8 | -2717 |
| b0042_0045 | 5022 | 5.4 | 6.6 | 5.3 | -1.1 | -199 |

## What is still missing

- **Temporal holdout on the shelf** (the real test): CHS Survey Index (1832–2016, CATZOC-dated) shows 19,471,411 of 34,281,587 corridor soundings (56.8%) lie outside every indexed polygon → surveyed after 2016. Retrain on in-index pixels only, score on the post-2016 ones. Needs one 5090 rental.
- σ recalibration (isotonic) — Spark, today.
- Depth gate (gravity beyond the shelf break) — Spark, today.
- Grounding hindcast benchmark (5 incidents) — after the retrain.
