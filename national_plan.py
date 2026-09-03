#!/usr/bin/env python3
"""National sigma-ranked survey plan: top-20 highest-uncertainty boxes across all NONNA
blocks (0.5x0.5 deg boxes, inferred water only, calibrated sigma68), priced at the corridor
assumptions (40 km2/day; C$2,600/km2 CHS-contract rate to C$183k/day icebreaker rate).
-> national_plan.json"""
import glob, json, math, os, numpy as np
from v5_data import lat_of_y, lon_of_x
cal = json.load(open("sigma_calibration.json")); g = np.array(cal["sigma_raw_grid"]); s68 = np.array(cal["sigma68"])
boxes = {}
for f in sorted(glob.glob("national_v5_out/*.npz")):
    d = np.load(f, allow_pickle=True)
    if "sigma" not in d.files and "sigma_raw" not in d.files: continue
    s = (d["sigma_raw"] if "sigma_raw" in d.files else d["sigma"]).astype("float32")
    c = d["complete"].astype("float32"); k = d["known"].astype(bool); bb = d["bbox3857"]
    H, W = s.shape; x0, x1 = min(bb[0],bb[2]), max(bb[0],bb[2]); y0, y1 = min(bb[1],bb[3]), max(bb[1],bb[3])
    fill = np.isfinite(c) & ~k & (s > 0)
    if fill.sum() < 100: continue
    ii, jj = np.nonzero(fill)
    lons = np.degrees((x0 + (jj+0.5)/W*(x1-x0))/6378137.0)
    lats = np.degrees(2*np.arctan(np.exp((y1 - (ii+0.5)/H*(y1-y0))/6378137.0)) - np.pi/2)
    sc = np.interp(s[fill], g, s68)
    res_km2 = ((x1-x0)/W*np.cos(np.radians(lats))/1000.0)**2
    bx = (np.floor(lons*2)/2); by = (np.floor(lats*2)/2)
    for b_lo, b_la, sv, a in zip(bx, by, sc, res_km2):
        kbox = (float(b_lo), float(b_la)); e = boxes.setdefault(kbox, [0.0, 0.0, 0.0])
        e[0] += a; e[1] += sv*a; e[2] = max(e[2], float(sv))
rank = []
for (blo, bla), (area, mass, peak) in boxes.items():
    if area < 50: continue
    rank.append(dict(lon=blo, lat=bla, area_km2=round(area), mean_sigma68=round(mass/area, 1), peak_sigma68=round(peak, 1), burden=round(mass)))
rank.sort(key=lambda r: -r["burden"])
top = rank[:20]
RATE = 40.0
for r in top:
    r["ship_days"] = round(r["area_km2"]/RATE, 1)
    r["cost_low_MCAD"] = round(r["area_km2"]*2600/1e6, 1); r["cost_high_MCAD"] = round(r["ship_days"]*183000/1e6, 1)
tot = dict(area_km2=round(sum(r["area_km2"] for r in top)), ship_days=round(sum(r["ship_days"] for r in top)),
           cost_low_MCAD=round(sum(r["cost_low_MCAD"] for r in top)), cost_high_MCAD=round(sum(r["cost_high_MCAD"] for r in top)))
json.dump(dict(assumptions="0.5 deg boxes; inferred water only; calibrated sigma68; 40 km2/day; C$2,600/km2 (CHS Sedna contract rate) to C$183k/day (CCG charter)", top20=top, total=tot), open("national_plan.json", "w"), indent=1)
print(json.dumps(tot)); [print(f"{r['lat']:5.1f}N {abs(r['lon']):6.1f}W  {r['area_km2']:6d} km2  meanS {r['mean_sigma68']:5.1f}  days {r['ship_days']:6.1f}  C${r['cost_low_MCAD']}-{r['cost_high_MCAD']}M") for r in top[:10]]
print("NATPLAN_DONE")
