#!/usr/bin/env python3
"""Grade the sealed forecast against whatever CHS has published since the seal.

Re-fetches from the CHS NONNA-100 WCS every corridor block that holds a sealed cell, finds
cells that were unsounded on 2026-08-29 (the archive snapshot in tiles_nat/) and now carry a
sounding, and scores the sealed prediction: MAE, bias, share inside the 68 % / 95 % bands,
against the nearest-sounding and gravity baselines. Also reports how much the archive moved
(new soundings per block). Runs from cron; publishes nothing itself.
-> grades/grade_<date>.json, grades/latest.json     usage: python3 grade_forecast.py [--dry]
"""
import csv, glob, hashlib, io, json, math, os, sys, time, datetime, numpy as np, requests, tifffile
from scipy import ndimage as ndi
from v5_data import GravityPrior, lat_of_y, lon_of_x
R = 6378137.0
def x_of_lon(lo): return R*math.radians(lo)
def y_of_lat(la): return R*math.log(math.tan(math.pi/4+math.radians(la)/2))
BASE = "https://nonna-geoserver.data.chs-shc.ca/geoserver/wcs"
SEALED = "forecast_2026-09-02.csv"; SEALED_SHA = "4d3f5dc8a5c2b50f3f72fba95384581fc778d6d453a3eec46aa2dc383c66be45"
os.makedirs("grades", exist_ok=True); os.makedirs("nonna_refetch", exist_ok=True)
assert hashlib.sha256(open(SEALED, "rb").read()).hexdigest() == SEALED_SHA, "sealed file has changed — refuse to grade"
rows = list(csv.DictReader(open(SEALED)))
grav = GravityPrior()
blocks = {}
for f in sorted(glob.glob("tiles_nat/*.npz")):
    bb = np.load(f, allow_pickle=True)["bbox3857"]; blocks[f] = (min(bb[0],bb[2]), max(bb[0],bb[2]), min(bb[1],bb[3]), max(bb[1],bb[3]))
# assign sealed cells to blocks
cells = {}
for r in rows:
    x, y = x_of_lon(float(r["lon"])), y_of_lat(float(r["lat"]))
    for f, (x0, x1, y0, y1) in blocks.items():
        if x0 <= x <= x1 and y0 <= y <= y1: cells.setdefault(f, []).append((r, x, y)); break
today = datetime.date.today().isoformat()
def refetch(f, bbox):
    fn = f"nonna_refetch/{today}_{os.path.basename(f)[:-4]}.tif"
    if os.path.exists(fn): return tifffile.imread(fn).astype("float32")
    x0, x1, y0, y1 = bbox
    p = {"service": "WCS", "version": "2.0.1", "request": "GetCoverage", "coverageId": "nonna__NONNA 100 Coverage", "format": "image/geotiff", "subset": [f"x({x0},{x1})", f"y({y0},{y1})"]}
    for k in range(3):
        try:
            rq = requests.get(BASE, params=p, timeout=240)
            if rq.status_code == 200 and "xml" not in rq.headers.get("Content-Type", ""):
                open(fn, "wb").write(rq.content); z = tifffile.imread(io.BytesIO(rq.content)).astype("float32"); return z
        except Exception: pass
        time.sleep(10)
    return None
scored = []; moved = {}
for f, lst in cells.items():
    old = np.load(f, allow_pickle=True)["z"].astype("float32"); H, W = old.shape; x0, x1, y0, y1 = blocks[f]
    new = refetch(f, blocks[f])
    if new is None: moved[os.path.basename(f)] = "fetch failed"; continue
    new[new > 1e30] = np.nan; new[new == 0] = np.nan; new[new > 25] = np.nan
    if new.shape != old.shape:   # resample defensively
        gy = np.clip((np.arange(H)/(H-1)*(new.shape[0]-1)).astype(int), 0, new.shape[0]-1); gx = np.clip((np.arange(W)/(W-1)*(new.shape[1]-1)).astype(int), 0, new.shape[1]-1); new = new[np.ix_(gy, gx)]
    ok, nk = np.isfinite(old), np.isfinite(new)
    moved[os.path.basename(f)] = dict(new_soundings=int((nk & ~ok).sum()), removed=int((ok & ~nk).sum()), known_before=int(ok.sum()), known_now=int(nk.sum()))
    if not (nk & ~ok).any(): continue
    lons = np.array([lon_of_x(v) for v in np.linspace(x0, x1, W)]); lats = np.array([lat_of_y(v) for v in np.linspace(y1, y0, H)])
    dist_px, idx = ndi.distance_transform_edt(~ok, return_indices=True)
    for r, x, y in lst:
        j = int((x-x0)/(x1-x0)*(W-1)); i = int((y1-y)/(y1-y0)*(H-1))
        if not ok[i, j] and nk[i, j]:
            truth = float(new[i, j]); pred = float(r["depth_m"]); s68 = float(r["sigma68_m"]); s95 = float(r["sigma95_m"])
            nn = float(old[idx[0][i, j], idx[1][i, j]]); g = float(grav.sample(np.array([lons[j]]), np.array([lats[i]]))[0])
            scored.append(dict(lon=r["lon"], lat=r["lat"], kind=r["kind"], predicted=pred, truth=truth, err=pred-truth, in68=abs(pred-truth) <= s68, in95=abs(pred-truth) <= s95, nearest_err=nn-truth, gravity_err=g-truth, block=os.path.basename(f)[:-4]))
out = dict(graded_on=today, sealed_file=SEALED, sealed_sha256=SEALED_SHA, n_sealed=len(rows), n_scored=len(scored),
           archive_movement=moved, cells=scored)
if scored:
    e = np.array([c["err"] for c in scored]); en = np.array([c["nearest_err"] for c in scored]); eg = np.array([c["gravity_err"] for c in scored])
    out["score"] = dict(mae=float(np.abs(e).mean()), bias=float(e.mean()), inside_68=float(np.mean([c["in68"] for c in scored])), inside_95=float(np.mean([c["in95"] for c in scored])),
                        mae_nearest_sounding=float(np.abs(en).mean()), mae_gravity=float(np.abs(eg).mean()))
json.dump(out, open(f"grades/grade_{today}.json", "w"), indent=1, default=float); json.dump(out, open("grades/latest.json", "w"), indent=1, default=float)
tot_new = sum(m["new_soundings"] for m in moved.values() if isinstance(m, dict))
print(f"GRADE {today}: blocks {len(cells)}, new soundings in those blocks {tot_new:,}, sealed cells now sounded {len(scored)} of {len(rows)}", out.get("score", "— no sealed cell resolved yet"))
print("GRADE_DONE")
