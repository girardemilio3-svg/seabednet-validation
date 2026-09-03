# NONNA-Temporal-Churchill v1 — a temporal-holdout benchmark for bathymetric completion

**Task.** Given every CHS NONNA-100 sounding in a block that lies inside a CHS Survey Index polygon (surveyed 1832–2016), predict the depth at every sounding CHS published after the index closed (post-2016), restricted to cells within 60 cells (6 km) of a training sounding. 6,686,037 target cells over 82 blocks of the Churchill corridor (Hudson Bay, Hudson Strait, Labrador approaches, Franklin Strait). 100 m grid, EPSG:3857.

**Why temporal.** Random or geographic splits of a survey archive leak: adjacent soundings come from the same survey line. Splitting on survey date tests what a model does on the water that gets surveyed next, which is the only question anyone funds.

**Files.** `split/<block>.npz` — `year` (int16; 0 = post-index = TEST) and `zoc` (CATZOC code) rasters aligned to the NONNA-100 block grid. `targets_corridor.npz` — TEST cells (block id, row, col, depth, distance to nearest TRAIN cell). `blocks.json` — block ids, bounding boxes and the exact WCS request to re-fetch the NONNA-100 blocks (Open Government Licence – Canada; snapshot 2026-08-29; later snapshots contain more post-index soundings, which is fine: the TEST set is defined by `targets_corridor.npz`, not by whatever the archive holds later). `score.py` — the scorer.

**Rules.** Train on TRAIN cells (year > 0) of these blocks plus anything that is not NONNA post-2016 data (gravity, satellite, other archives); do not use any NONNA cell with year == 0, in any block, national or corridor. Report MAE, bias, by-distance, by-depth; report 1σ coverage if you output uncertainty. State what else you trained on.

**Leaderboard (v1, 2026-09-03).** See `leaderboard.json`. Independent baselines: nearest training sounding 18.9 m; SRTM15+ gravity 18.9 m; gravity trend + natural-neighbour residual 16.8 m; trend + inverse-distance residual 16.2 m; SeabedNet v5 small 13.3 m (74% inside 1σ). GEBCO scores 6.2 m but contains the test soundings through IBCAO v5 and is listed as a reference, not a competitor.

**Submit.** Open an issue or pull request on github.com/girardemilio3-svg/seabednet-validation with your `score.py` output and a one-paragraph method description; we add it to the leaderboard as submitted, with the independence flag you declare.

**Citation.** Girard, E. (2026). NONNA-Temporal-Churchill v1: a temporal-holdout benchmark for bathymetric completion. SeabedNet, Montréal. Data: Canadian Hydrographic Service NONNA-100 (OGL-Canada); CHS Survey Index (DFO).
