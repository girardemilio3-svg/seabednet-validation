#!/usr/bin/env python3
"""Era audit: how wrong each survey era's soundings are, measured by CHS's own post-2016
re-surveys. For every post-2016 sounding cell (index year==0) within 1 km of a dated
pre-2016 sounding, the error of that nearest old sounding vs the modern value, stratified
by the old survey's decade. National, all blocks with index rasters. -> era_audit.json"""
import glob, json, os, numpy as np
from scipy import ndimage as ndi
DEC = [(0,1900),(1900,1940),(1940,1950),(1950,1960),(1960,1970),(1970,1980),(1980,1990),(1990,2000),(2000,2010),(2010,2017)]
acc = {d: [] for d in DEC}
nblocks = 0
for f in sorted(glob.glob("tiles_nat/*.npz")):
    ip = f"index_out/{os.path.basename(f)}"
    if not os.path.exists(ip): continue
    z = np.load(f, allow_pickle=True)["z"].astype("float32")
    yr = np.load(ip)["year"]
    k = np.isfinite(z); pre = k & (yr > 0); post = k & (yr == 0)
    if pre.sum() < 500 or post.sum() < 500: continue
    dist, idx = ndi.distance_transform_edt(~pre, return_indices=True)
    sel = post & (dist <= int(__import__("os").environ.get("MAXPX","10")))
    if sel.sum() < 100: continue
    ii, jj = np.nonzero(sel)
    oi, oj = idx[0][ii, jj], idx[1][ii, jj]
    err = z[oi, oj] - z[ii, jj]        # old sounding vs modern truth
    oyr = yr[oi, oj]
    dep = z[ii, jj]
    shallow = dep > -100               # where it matters
    for a, b in DEC:
        m = (oyr >= a) & (oyr < b) & shallow
        if m.any(): acc[(a, b)].append(err[m])
    nblocks += 1
out = {}
for (a, b), lst in acc.items():
    if not lst: continue
    e = np.concatenate(lst)
    if len(e) < 500: continue
    out[f"{a}-{b}"] = dict(n=int(len(e)), median_abs_err=float(np.median(np.abs(e))), p90_abs_err=float(np.quantile(np.abs(e), .9)),
                           median_err=float(np.median(e)), pct_shallower_than_charted_5m=float((e < -5).mean()*100))
json.dump(dict(rule="post-2016 sounding cells within 1 km of a dated pre-2016 sounding; water shallower than 100 m; error = old sounding minus modern value; negative = reality shallower than the old chart", n_blocks=nblocks, by_survey_decade=out), open("era_audit.json", "w"), indent=1)
for k2, v in out.items(): print(f"{k2}: n={v['n']:8d} median|err| {v['median_abs_err']:5.1f} m  p90 {v['p90_abs_err']:6.1f} m  shallower-than-charted>5m {v['pct_shallower_than_charted_5m']:4.1f}%")
print("ERA_DONE")
