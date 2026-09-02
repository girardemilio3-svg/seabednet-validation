#!/usr/bin/env python3
"""Falsifiable public forecast: predicted depths at cells with NO published sounding,
along the found Churchill channel + 1,000 random inferred cells across the corridor.
Anyone with a future survey can score it. forecast_<date>.csv + sha256."""
import csv, glob, hashlib, json, math, datetime, numpy as np
from scipy import ndimage as ndi
from v5_data import lat_of_y, lon_of_x
R = 6378137.0
xl = lambda lo: R*math.radians(lo); yl = lambda la: R*math.log(math.tan(math.pi/4+math.radians(la)/2))
rng = np.random.default_rng(20260902)
blocks = {}
for f in sorted(glob.glob("corridor_out_v2/*.npz")):
    d = np.load(f, allow_pickle=True); bb = d["bbox3857"]
    blocks[f] = (min(bb[0],bb[2]), max(bb[0],bb[2]), min(bb[1],bb[3]), max(bb[1],bb[3]))
rows = []
def add(f, i, j, kind):
    d = np.load(f, allow_pickle=True); c = d["complete"].astype(float); k = d["known"]; s68 = d["sigma68"].astype(float); s95 = d["sigma95"].astype(float)
    x0, x1, y0, y1 = blocks[f]; H, W = c.shape
    if k[i, j] or not np.isfinite(c[i, j]): return False
    lon = lon_of_x(x0 + (j+0.5)/W*(x1-x0)); lat = lat_of_y(y1 - (i+0.5)/H*(y1-y0))
    rows.append(dict(lon=round(lon, 5), lat=round(lat, 5), depth_m=round(c[i, j], 1), sigma68_m=round(s68[i, j], 1), sigma95_m=round(s95[i, j], 1), kind=kind, gated=bool(d["gated"][i, j])))
    return True
path = json.load(open("corridor_found.json"))["path"]
for lo, la in path:
    x, y = xl(lo), yl(la)
    for f, (x0, x1, y0, y1) in blocks.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            d = np.load(f, allow_pickle=True); H, W = d["known"].shape
            add(f, int((y1-y)/(y1-y0)*(H-1)), int((x-x0)/(x1-x0)*(W-1)), "channel"); break
files = list(blocks)
while sum(r["kind"] == "random" for r in rows) < 1000:
    f = files[rng.integers(len(files))]; d = np.load(f, allow_pickle=True); k = d["known"]; c = d["complete"]
    cand = np.argwhere(~k & np.isfinite(c.astype(float)))
    if len(cand) == 0: continue
    i, j = cand[rng.integers(len(cand))]; add(f, i, j, "random")
date = datetime.date.today().isoformat()
fn = f"forecast_{date}.csv"
with open(fn, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["lon", "lat", "depth_m", "sigma68_m", "sigma95_m", "kind", "gated"]); w.writeheader(); w.writerows(rows)
h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
json.dump(dict(file=fn, sha256=h, n=len(rows), n_channel=sum(r["kind"]=="channel" for r in rows), date=date,
               model="SeabedNet v5-small corridor completion + depth gate + isotonic σ (2026-09-02)",
               rule="A future survey scores this file: MAE, and the share of soundings inside the 68 % / 95 % bands. Anything outside CHS NONNA-100 as of 2026-08-29 counts."),
          open("forecast_manifest.json", "w"), indent=1)
print(fn, len(rows), "rows; channel", sum(r["kind"]=="channel" for r in rows), "sha256", h[:16], "FORECAST_DONE")
