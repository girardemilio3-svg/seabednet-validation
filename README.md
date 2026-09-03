# SeabedNet validation — Churchill corridor

Every number on https://girardemilio3-svg.github.io/churchill-corridor-atlas/ comes from a file in `results/`,
produced by a script in this repo. Nothing is retyped by hand (`build_atlas_v2.py` reads the JSON).

## What is tested

| Test | Script | Result file | What it shows |
|---|---|---|---|
| 1. Independent multibeam | `indep_validate.py` | `results/indep_validation_corridor_out.json`, `results/indep_cells_corridor_out.npy` (per-cell: model err, nearest-sounding err, gravity err, σ, distance km, depth) | v5 scored on GMRT/NCEI research-cruise depths (Amundsen, Healy, Knorr, Armstrong, Merian) at cells where CHS NONNA-100 has no sounding |
| 2. Temporal holdout | `index_coverage.py` → `v5_train.py` (env `V5_TEMPORAL`) → `temporal_eval.py` | `results/temporal_validation_v5_tiny_temporal.json` | model trained only on soundings inside the CHS Survey Index (1832–2016) scores the post-2016 soundings |
| σ calibration | `sigma_calibrate.py` | `results/sigma_calibration.json` | isotonic 68 % / 95 % map on Test 1 cells, shelf 50–400 m |
| Depth gate | `gate_corridor.py` | — | inferred cells < −400 m defer to the gravity prior |
| Keel-profile provenance | `profile_catzoc.py` | `results/route_profile_v2.json` | CHS survey year + CATZOC under every 2 km point of the route |
| Public forecast | `make_forecast.py` | `results/forecast_2026-09-02.csv`, `results/forecast_manifest.json` (SHA-256) | 1,314 predicted depths at unsounded cells; score it with any future survey |
| Hazard model | `build_hazard.py`, `train_hazard.py` | (in progress) | shallowest depth within 500 m from NONNA-10/100 pairs |
| Grounding hindcast | `hindcast.py` | (in progress) | pre-incident data only; percentile of the strike site in the hazard field |

`VALIDATION_REPORT.md` is the written summary, limits included.

## Reproduce
Data: CHS NONNA-100 / NONNA-10 (Open Government Licence – Canada) via the CHS GeoServer WCS;
GMRT GridServer `layer=topo-mask`; CHS Survey Index (DFO EGIS MapServer `chs_edh_survey_index`).
`python3 indep_validate.py corridor_out` etc. — each script has its usage line at the top.
Model weights are not included (v5-small 418 MB); ask.

SeabedNet · Montréal · 2026

## Benchmark: NONNA-Temporal-Churchill v1
`benchmark/` — the temporal-holdout split as a public benchmark: index rasters for 94 corridor blocks, 6,686,037 target cells (`targets_corridor.npz`), the exact WCS request to re-fetch NONNA-100, `score.py`, and `leaderboard.json` with the independent baselines and SeabedNet. Rules and citation in `benchmark/README.md`. Submit scores by issue or PR.

## The bet, graded monthly
`grade_forecast.py` re-fetches every corridor block holding a sealed forecast cell from the CHS WCS on the 1st of each month (`monthly_grade.sh`, cron on the SeabedNet workstation), scores any sealed cell that has since received a CHS sounding, and publishes the result to the atlas (`grades/latest.json`). The sealed file is hash-checked before every grade and never modified.
