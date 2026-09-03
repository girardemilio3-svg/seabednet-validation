#!/usr/bin/env python3
"""Package the temporal-holdout benchmark: NONNA-Temporal-Churchill v1.

The split is defined by the CHS Survey Index (1832–2016): a NONNA-100 sounding inside any
indexed survey polygon is TRAIN (dated ≤ 2016); a sounding outside every polygon is TEST
(published after the index closed). Anyone can rebuild both sides from public data with the
index rasters shipped here; the test targets are shipped explicitly so scoring is exact.

benchmark/
  README.md                       rules, baselines, how to submit
  split/<block>.npz               year (int16, 0 = post-index) + zoc (int8) rasters, corridor 94 blocks
  targets_corridor.npz            block id, row, col, depth (float16) for every test cell within 60 px of a train sounding
  blocks.json                     block name -> bbox3857, shape; WCS request to re-fetch NONNA-100
  score.py                        scorer: takes a CSV/NPZ of predictions at target cells -> MAE, bias, by-distance, by-depth
  leaderboard.json                reference results (nearest sounding, gravity, trend+residual, GEBCO*, SeabedNet)
"""
import glob, json, os, shutil, numpy as np
from scipy import ndimage as ndi
os.makedirs("benchmark/split", exist_ok=True)
files = [l.strip() for l in open("corridor_blocks.txt") if l.strip()]
meta = {}; B, I, J, Z, D = [], [], [], [], []
for bi, f in enumerate(files):
    name = os.path.basename(f)[:-4]; ip = f"index_out/{name}.npz"
    if not os.path.exists(ip): continue
    d = np.load(f, allow_pickle=True); z = d["z"].astype("float32"); bb = [float(v) for v in d["bbox3857"]]; H, W = z.shape
    idx = np.load(ip); yr = idx["year"]; allk = np.isfinite(z); pre = allk & (yr > 0); post = allk & (yr == 0)
    shutil.copy(ip, f"benchmark/split/{name}.npz")
    meta[name] = dict(id=bi, bbox3857=bb, shape=[H, W], n_train=int(pre.sum()), n_test=int(post.sum()))
    if pre.sum() < 500 or post.sum() < 100: continue
    dist = ndi.distance_transform_edt(~pre); val = post & (dist <= 60)
    ii, jj = np.nonzero(val)
    B.append(np.full(len(ii), bi, np.int16)); I.append(ii.astype(np.int16)); J.append(jj.astype(np.int16)); Z.append(z[val].astype(np.float16)); D.append((dist[val]).astype(np.float16))
np.savez_compressed("benchmark/targets_corridor.npz", block=np.concatenate(B), row=np.concatenate(I), col=np.concatenate(J), depth=np.concatenate(Z), dist_px=np.concatenate(D))
json.dump(dict(blocks=meta, wcs=dict(url="https://nonna-geoserver.data.chs-shc.ca/geoserver/wcs", coverageId="nonna__NONNA 100 Coverage", request="GetCoverage version 2.0.1, format image/geotiff, subset x(xmin,xmax) y(ymin,ymax) in EPSG:3857", snapshot="2026-08-29", cleaning="values > 1e30, == 0, > 25 m -> NaN"),
               index_source="CHS Survey Index, DFO EGIS MapServer chs_edh_survey_index (7,078 polygons, 1832-2016); rasterized with latest survey winning"), open("benchmark/blocks.json", "w"), indent=1)
n = sum(len(b) for b in B)
T = json.load(open("temporal_validation_v5_small_temporal.json")); Tt = json.load(open("temporal_validation_v5_tiny_temporal.json")); BL = json.load(open("baselines.json")); GB = json.load(open("gebco_eval.json"))
lb = [dict(method="nearest train sounding", mae=BL["test2"]["all"]["nearest"], independent=True),
      dict(method="SRTM15+ V2.7 gravity prior", mae=BL["test2"]["all"]["gravity"], independent=True),
      dict(method="gravity trend + natural-neighbour residual", mae=BL["test2"]["all"]["trend_natural"], independent=True),
      dict(method="gravity trend + inverse-distance residual (k=12)", mae=BL["test2"]["all"]["trend_idw"], independent=True),
      dict(method="GEBCO 2024/25 (NCEI global mosaic) — contains the test soundings via IBCAO v5; reference only", mae=GB["test2"]["all"]["gebco"], independent=False),
      dict(method="SeabedNet v5 tiny (6.8M), trained on TRAIN only", mae=Tt["overall"]["mae_model"], independent=True),
      dict(method="SeabedNet v5 small (34.8M), trained on TRAIN only", mae=T["overall"]["mae_model"], frac_within_1sigma=T["overall"]["frac_within_1sigma"], independent=True)]
json.dump(dict(version="1.0", date="2026-09-03", n_targets=n, leaderboard=lb), open("benchmark/leaderboard.json", "w"), indent=1)
open("benchmark/score.py", "w").write('''#!/usr/bin/env python3
"""Score predictions on NONNA-Temporal-Churchill v1.
usage: python3 score.py predictions.npz   (arrays: block, row, col, pred [, sigma]) — same order/keys as targets_corridor.npz, or any subset matched on (block,row,col)."""
import sys, numpy as np, json
T = np.load("targets_corridor.npz"); P = np.load(sys.argv[1])
key = lambda a: a["block"].astype(np.int64)*10**8 + a["row"].astype(np.int64)*10**4 + a["col"].astype(np.int64)
kt, kp = key(T), key(P); order = np.argsort(kt); pos = np.searchsorted(kt[order], kp); ok = (pos < len(kt)) & (kt[order][np.minimum(pos, len(kt)-1)] == kp)
ti = order[pos[ok]]; pred = P["pred"][ok].astype(float); truth = T["depth"][ti].astype(float); dist = T["dist_px"][ti].astype(float)*0.1
e = pred - truth; out = dict(n=int(ok.sum()), coverage=float(ok.sum()/len(kt)), mae=float(np.abs(e).mean()), bias=float(e.mean()), rmse=float(np.sqrt((e**2).mean())))
out["by_distance_km"] = {f"{a}-{b}": float(np.abs(e[(dist>=a)&(dist<b)]).mean()) for a, b in [(0,.5),(.5,1),(1,2),(2,4),(4,8)] if ((dist>=a)&(dist<b)).any()}
out["by_depth_m"] = {f"{a}-{b}": float(np.abs(e[(-truth>=a)&(-truth<b)]).mean()) for a, b in [(0,20),(20,50),(50,100),(100,200),(200,400),(400,9999)] if ((-truth>=a)&(-truth<b)).any()}
if "sigma" in P: s = P["sigma"][ok].astype(float); out["frac_within_1sigma"] = float((np.abs(e) <= s).mean()); out["frac_within_2sigma"] = float((np.abs(e) <= 2*s).mean())
print(json.dumps(out, indent=1))
''')
open("benchmark/README.md", "w").write(f'''# NONNA-Temporal-Churchill v1 — a temporal-holdout benchmark for bathymetric completion

**Task.** Given every CHS NONNA-100 sounding in a block that lies inside a CHS Survey Index polygon (surveyed 1832–2016), predict the depth at every sounding CHS published after the index closed (post-2016), restricted to cells within 60 cells (6 km) of a training sounding. {n:,} target cells over {len(B)} blocks of the Churchill corridor (Hudson Bay, Hudson Strait, Labrador approaches, Franklin Strait). 100 m grid, EPSG:3857.

**Why temporal.** Random or geographic splits of a survey archive leak: adjacent soundings come from the same survey line. Splitting on survey date tests what a model does on the water that gets surveyed next, which is the only question anyone funds.

**Files.** `split/<block>.npz` — `year` (int16; 0 = post-index = TEST) and `zoc` (CATZOC code) rasters aligned to the NONNA-100 block grid. `targets_corridor.npz` — TEST cells (block id, row, col, depth, distance to nearest TRAIN cell). `blocks.json` — block ids, bounding boxes and the exact WCS request to re-fetch the NONNA-100 blocks (Open Government Licence – Canada; snapshot 2026-08-29; later snapshots contain more post-index soundings, which is fine: the TEST set is defined by `targets_corridor.npz`, not by whatever the archive holds later). `score.py` — the scorer.

**Rules.** Train on TRAIN cells (year > 0) of these blocks plus anything that is not NONNA post-2016 data (gravity, satellite, other archives); do not use any NONNA cell with year == 0, in any block, national or corridor. Report MAE, bias, by-distance, by-depth; report 1σ coverage if you output uncertainty. State what else you trained on.

**Leaderboard (v1, 2026-09-03).** See `leaderboard.json`. Independent baselines: nearest training sounding {BL["test2"]["all"]["nearest"]:.1f} m; SRTM15+ gravity {BL["test2"]["all"]["gravity"]:.1f} m; gravity trend + natural-neighbour residual {BL["test2"]["all"]["trend_natural"]:.1f} m; trend + inverse-distance residual {BL["test2"]["all"]["trend_idw"]:.1f} m; SeabedNet v5 small {T["overall"]["mae_model"]:.1f} m ({T["overall"]["frac_within_1sigma"]*100:.0f}% inside 1σ). GEBCO scores {GB["test2"]["all"]["gebco"]:.1f} m but contains the test soundings through IBCAO v5 and is listed as a reference, not a competitor.

**Submit.** Open an issue or pull request on github.com/girardemilio3-svg/seabednet-validation with your `score.py` output and a one-paragraph method description; we add it to the leaderboard as submitted, with the independence flag you declare.

**Citation.** Girard, E. (2026). NONNA-Temporal-Churchill v1: a temporal-holdout benchmark for bathymetric completion. SeabedNet, Montréal. Data: Canadian Hydrographic Service NONNA-100 (OGL-Canada); CHS Survey Index (DFO).
''')
print("BENCHMARK_DONE targets", n, "blocks", len(B)); os.system("du -sh benchmark")
