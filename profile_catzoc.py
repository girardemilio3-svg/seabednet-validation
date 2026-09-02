#!/usr/bin/env python3
"""Keel profile v2: every 2 km point on the Churchill route gets the CHS Survey Index
provenance (survey year + CATZOC of the latest indexed survey covering the pixel, or
"post-2016 multibeam" if the sounding lies outside every indexed polygon), the distance
to the nearest published sounding, calibrated σ (68/95 %) from sigma_calibration.json,
and a trust grade:  A = published sounding  B = inferred, ≤1 km from a sounding
C = inferred, 1–6 km  D = no inference (beyond 6 km).  -> route_profile_v2.json
"""
import glob, json, math, numpy as np
from scipy import ndimage as ndi
from v5_data import lat_of_y, lon_of_x
R = 6378137.0
def x_of_lon(lo): return R*math.radians(lo)
def y_of_lat(la): return R*math.log(math.tan(math.pi/4+math.radians(la)/2))
cal = json.load(open("sigma_calibration.json"))
g = np.array(cal["sigma_raw_grid"]); s68 = np.array(cal["sigma68"]); s95 = np.array(cal["sigma95"])
ZOCN = {0: None, 1: "A1", 2: "A2", 3: "B", 4: "C", 5: "D", 6: "U"}
blocks = []
for f in sorted(glob.glob("corridor_out/*.npz")):
    bb = np.load(f, allow_pickle=True)["bbox3857"]
    blocks.append((f, min(bb[0], bb[2]), max(bb[0], bb[2]), min(bb[1], bb[3]), max(bb[1], bb[3])))
cache = {}
def load(f):
    if f not in cache:
        d = np.load(f, allow_pickle=True); k = d["known"].astype(bool)
        name = f.split("/")[-1]
        idx = np.load(f"index_out/{name}")
        dist = ndi.distance_transform_edt(~k)
        cache[f] = dict(k=k, yr=idx["year"], zoc=idx["zoc"], dist=dist, H=k.shape[0], W=k.shape[1])
    return cache[f]
prof = json.load(open("route_profile.json"))
out = []; grades = {"A": 0, "B": 0, "C": 0, "D": 0}
for p in prof:
    x, y = x_of_lon(p["lon"]), y_of_lat(p["lat"])
    hit = [b for b in blocks if b[1] <= x <= b[2] and b[3] <= y <= b[4]]
    q = dict(p)
    if hit:
        f, x0, x1, y0, y1 = hit[0]; d = load(f)
        j = int((x-x0)/(x1-x0)*(d["W"]-1)); i = int((y1-y)/(y1-y0)*(d["H"]-1))
        res_km = (x1-x0)/d["W"]*math.cos(math.radians(p["lat"]))/1000
        dk = float(d["dist"][i, j]*res_km)
        yr = int(d["yr"][i, j]); zc = ZOCN.get(int(d["zoc"][i, j]))
        q["dist_km"] = round(dk, 2)
        if d["k"][i, j]:
            q["survey"] = f"{yr} · CATZOC {zc}" if yr > 0 else "post-2016 multibeam (outside CHS Survey Index)"
            q["survey_year"] = yr if yr > 0 else 2017; q["catzoc"] = zc if yr > 0 else "A1/A2 (modern)"
        else:
            q["survey"] = None
    else:
        dk = None
    if p["prov"] == "surveyed": gr = "A"
    elif p["prov"] == "inferred": gr = "B" if (dk is not None and dk <= 1.0) else "C"
    else: gr = "D"
    q["grade"] = gr; grades[gr] += 1
    if p["sigma"] and p["sigma"] > 0:
        q["sigma68"] = round(float(np.interp(p["sigma"], g, s68)), 1); q["sigma95"] = round(float(np.interp(p["sigma"], g, s95)), 1)
    out.append(q)
json.dump(out, open("route_profile_v2.json", "w"))
n = len(out)
print({k: f"{v} ({v/n*100:.0f}%)" for k, v in grades.items()})
yrs = [q["survey_year"] for q in out if q.get("survey_year")]
old = sum(1 for y in yrs if y < 1980)
print(f"surveyed points {len(yrs)}: pre-1980 surveys {old} ({old/max(1,len(yrs))*100:.0f}%), median survey year {int(np.median(yrs))}")
from collections import Counter
print("CATZOC on surveyed points:", Counter(q.get("catzoc") for q in out if q.get("catzoc")).most_common())
print("PROFILE_V2_DONE")
