#!/usr/bin/env python3
"""Apply the danger-report model nationally: for every block, score chart-safe cells on a 500 m grid (every 5th cell) inside
shipping-relevant water, then rank 0.5-degree boxes by expected reported dangers per ship-day. The model is refit on all dated
notices (both years) with the same features as danger_model.py. -> danger_score/<block>.npz (score on the 500 m grid), national_plan_v3.json"""
import csv, glob, json, math, os, numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from v5_data import GravityPrior
R = 6378137.0; STRIDE = 5
def merc(lon, lat): return R*math.radians(lon), R*math.log(math.tan(math.pi/4 + math.radians(lat)/2))
rows = list(csv.DictReader(open("danger_features.csv"))); FEATS = [c for c in rows[0].keys() if c not in ("y", "year", "category", "block")]
f = lambda r, k: float(r[k]) if r[k] not in ("", "nan", "None") else np.nan
Rr = [r for r in rows if f(r, "complete_m") < -21]; A = np.array([[f(r, k) for k in FEATS] for r in Rr], dtype=np.float32); y = np.array([int(r["y"]) for r in Rr])
clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0).fit(A, y); print("model fit on", len(Rr), "rows,", int(y.sum()), "positives", flush=True)
import re
D = open("map/data.js").read(); route = json.loads(re.search(r"\"route\":\s*(\[\[.*?\]\])", D).group(1)); comm = json.loads(re.search(r"\"communities\":\s*(\[.*?\])\s*,\s*\"plan\"", D, re.S).group(1))
nrc = []
for L in (0, 1):
    for ft in json.load(open(f"planetary/routes/nrcan_layer{L}.geojson"))["features"]:
        g = ft["geometry"]; lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for ln in lines:
            for a, b in zip(ln[:-1], ln[1:]):
                n = max(2, int(math.hypot(a[0]-b[0], a[1]-b[1])/0.05) + 1)
                for t in np.linspace(0, 1, n): nrc.append(merc(a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1])))
tree_r = cKDTree([merc(p[1], p[0]) for p in route]); tree_c = cKDTree([merc(c["lon"], c["lat"]) for c in comm]); tree_n = cKDTree(nrc)
G = GravityPrior(); os.makedirs("danger_score", exist_ok=True); boxes = {}; ix = {k: i for i, k in enumerate(FEATS)}
for fpath in sorted(glob.glob("hazard_nat_v2/*.npz")):
    name = os.path.basename(fpath); cf = f"national_v5_out/{name}"
    if not os.path.exists(cf): continue
    hd = np.load(fpath, allow_pickle=True); P = hd["p105"].astype("float32"); S = hd["shoal"].astype("float32"); bb = hd["bbox3857"]
    cd = np.load(cf, allow_pickle=True); C = cd["complete"].astype("float32"); K = cd["known"].astype(bool); SG = cd["sigma"].astype("float32")
    Z = np.load(f"tiles_nat/{name}", allow_pickle=True)["z"].astype("float32")
    if P.shape != C.shape or Z.shape != C.shape: continue
    H, W = P.shape; x0, x1 = min(bb[0], bb[2]), max(bb[0], bb[2]); y0, y1 = min(bb[1], bb[3]), max(bb[1], bb[3])
    kk = np.isfinite(Z)
    if kk.sum() > 500:   # gravity-anchor sanity (skip inland lakes)
        yi, xi = np.nonzero(kk); sel = np.random.default_rng(0).choice(len(yi), min(4000, len(yi)), replace=False)
        glon = np.degrees((x0 + (xi[sel]+0.5)/W*(x1-x0))/R); glat = np.degrees(2*np.arctan(np.exp((y1 - (yi[sel]+0.5)/H*(y1-y0))/R)) - np.pi/2)
        if np.median(np.abs(G.sample(glon, glat) - Z[yi[sel], xi[sel]])) > 30: continue
    dist, nidx = ndi.distance_transform_edt(~K, return_indices=True); NZ = Z[nidx[0], nidx[1]]
    a = np.load(f"aux_out/{name}") if os.path.exists(f"aux_out/{name}") else None
    L = a["land"].astype("float32") if a is not None else np.full((H, W), np.nan, np.float32); Lm = np.isfinite(L) & (L > 0.5)
    L0 = np.where(Lm, L, 0.0); Lmax = ndi.maximum_filter(L0, size=11); cdist = ndi.distance_transform_edt(~Lm) if Lm.any() else np.full((H, W), 9999.0)
    s2 = a["s2"].astype("float32")/255.0 if a is not None else np.zeros((3, H, W), np.float32)
    b = np.load(f"aux_s1/{name}") if os.path.exists(f"aux_s1/{name}") else None
    VV = b["vv"].astype("float32") if b is not None else np.full((H, W), np.nan, np.float32); VH = b["vh"].astype("float32") if b is not None else np.full((H, W), np.nan, np.float32)
    Pmax = ndi.maximum_filter(np.nan_to_num(P, nan=0.0), size=11)
    ii, jj = np.mgrid[0:H:STRIDE, 0:W:STRIDE]; ii = ii.ravel(); jj = jj.ravel()
    lons = np.degrees((x0 + (jj+0.5)/W*(x1-x0))/R); lats = np.degrees(2*np.arctan(np.exp((y1 - (ii+0.5)/H*(y1-y0))/R)) - np.pi/2)
    safe = np.isfinite(C[ii, jj]) & (C[ii, jj] < -21) & (dist[ii, jj] <= 60) & np.isfinite(P[ii, jj])
    if safe.sum() < 20: continue
    q = np.c_[x0 + (jj+0.5)/W*(x1-x0), y1 - (ii+0.5)/H*(y1-y0)]; scale = 1/np.cos(np.radians(lats))
    rel = safe & ((tree_r.query(q)[0]/scale < 30000) | (tree_c.query(q)[0]/scale < 15000) | (tree_n.query(q)[0]/scale < 30000))
    if rel.sum() < 20: continue
    I = ii[rel]; J = jj[rel]
    X = np.stack([Pmax[I, J], S[I, J], C[I, J], SG[I, J], NZ[I, J], dist[I, J]*0.1, L0[I, J], Lmax[I, J], Lm[I, J].astype(np.float32), s2[0, I, J], s2[1, I, J], s2[2, I, J],
                  np.where(np.isfinite(VV[I, J]), VV[I, J], -30.0), np.where(np.isfinite(VH[I, J]), VH[I, J], -40.0), np.isfinite(VV[I, J]).astype(np.float32), G.sample(lons[rel], lats[rel]), lats[rel], np.minimum(cdist[I, J], 999)], 1).astype(np.float32)
    sc = clf.predict_proba(X)[:, 1]
    np.savez_compressed(f"danger_score/{name}", i=I.astype(np.int16), j=J.astype(np.int16), score=sc.astype(np.float16), bbox3857=bb)
    area = (STRIDE*(x1-x0)/W*np.cos(np.radians(lats[rel]))/1000.0)**2   # km2 represented by each 500 m sample
    bx = np.floor(lons[rel]*2)/2; by = np.floor(lats[rel]*2)/2
    for blo, bla, ar, s_ in zip(bx, by, area, sc):
        e = boxes.setdefault((float(blo), float(bla)), [0.0, 0.0, 0]); e[0] += ar; e[1] += s_*ar; e[2] += 1
    print(name, int(rel.sum()), "cells, mean score %.3f" % sc.mean(), flush=True)
rank = []
for (blo, bla), (area, mass, n) in boxes.items():
    if area < 25: continue
    days = area/40.0; rank.append(dict(lon=blo, lat=bla, area_km2=round(area), cells=int(n), expected_reports_km2=round(mass, 2), ship_days=round(days, 1), reports_per_ship_day=round(mass/days, 3)))
rank.sort(key=lambda r: -r["reports_per_ship_day"]); top = rank[:20]
out = dict(method="danger-report model (gradient boosting on hazard field + soundings + coast + elevation + imagery + radar + gravity; year-split validated AUC 0.93) applied on a 500 m grid to chart-safe water within 6 km of a sounding; boxes ranked by expected reported-danger area per ship-day",
           relevance="within 30 km of the Churchill route or an NRCan shipping route / Arctic sea route, or 15 km of a sealift community; inland lakes excluded", top20=top,
           total=dict(area_km2=round(sum(r["area_km2"] for r in top)), ship_days=round(sum(r["ship_days"] for r in top)), cost_low_MCAD=round(sum(r["area_km2"] for r in top)*2600/1e6), cost_high_MCAD=round(sum(r["ship_days"] for r in top)*183000/1e6)), n_boxes=len(rank))
json.dump(out, open("national_plan_v3.json", "w"), indent=1); print("PLAN_V3_DONE", out["total"])
for r in top[:10]: print(f"  {r['lat']:5.1f}N {abs(r['lon']):6.1f}W area {r['area_km2']:5d} km2 expected-report km2 {r['expected_reports_km2']:6.2f} per ship-day {r['reports_per_ship_day']:.3f}")
