#!/usr/bin/env python3
"""Score every hazard-field shoal candidate against crowdsourced pings, memory-light: pings are pre-filtered per day file to a
500 m grid around the candidates (with one-cell dilation) before any tree is built. For each candidate with >= 5 pings within 300 m:
ping depth quantiles, number of platforms, and a verdict: SUPPORTED if the 10th-percentile ping depth <= predicted shoal + 3 m,
REFUTED if it is >= 60 % of the chart depth, else ambiguous. -> candidates_vs_csb.csv"""
import pandas as pd, numpy as np, glob, math, os
from scipy.spatial import cKDTree
R = 6378137.0; G = 500.0
def merc(lon, lat): return R*np.radians(lon), R*np.log(np.tan(np.pi/4 + np.radians(lat)/2))
C = []
for f in sorted(glob.glob("shoal_candidates_*.csv")):
    c = pd.read_csv(f); c["file"] = os.path.basename(f); C.append(c[["file", "lat", "lon", "predicted_shoal_m", "mean_map_depth_m"]])
C = pd.concat(C, ignore_index=True); cx, cy = merc(C.lon.values, C.lat.values); C["x"] = cx; C["y"] = cy
keys = set()
for x, y in zip(cx, cy):
    kx, ky = int(x//G), int(y//G)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1): keys.add((kx+dx, ky+dy))
KX = np.array([k[0] for k in keys]); KY = np.array([k[1] for k in keys]); key_set = set(zip(KX.tolist(), KY.tolist()))
print("candidates", len(C), "grid keys", len(keys), flush=True)
kept = []
for f in sorted(glob.glob("recon_csb/csb_canada/*/*.parquet")):
    try: d = pd.read_parquet(f, columns=["LON", "LAT", "DEPTH", "PLATFORM_NAME"])
    except Exception: continue
    d = d[(d.DEPTH > 0.5) & (d.DEPTH < 1500)]
    if not len(d): continue
    x, y = merc(d.LON.values, d.LAT.values); kx = (x//G).astype(np.int64); ky = (y//G).astype(np.int64)
    m = np.fromiter(((a, b) in key_set for a, b in zip(kx.tolist(), ky.tolist())), bool, len(d))
    if m.any(): kept.append(pd.DataFrame(dict(x=x[m].astype(np.float64), y=y[m].astype(np.float64), depth=d.DEPTH.values[m].astype(np.float32), plat=d.PLATFORM_NAME.values[m].astype(str))))
P = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=["x", "y", "depth", "plat"]); print("pings near candidates", len(P), flush=True)
tree = cKDTree(np.c_[P.x.values, P.y.values]) if len(P) else None; out = []
for r in C.itertuples():
    if tree is None: break
    sc = math.cos(math.radians(r.lat)); idx = tree.query_ball_point([r.x, r.y], 300/sc)
    if len(idx) < 5: continue
    dep = P.depth.values[idx]; q10 = float(np.quantile(dep, 0.1)); pred = abs(float(r.predicted_shoal_m)); chart = abs(float(r.mean_map_depth_m))
    verdict = "SUPPORTED" if q10 <= pred + 3 else ("REFUTED" if q10 >= 0.6*chart else "ambiguous")
    out.append(dict(file=r.file, lat=round(r.lat, 5), lon=round(r.lon, 5), predicted_m=pred, chart_m=chart, n_pings=len(idx), n_platforms=int(len(set(P.plat.values[idx]))), ping_min=round(float(dep.min()), 1), ping_q10=round(q10, 1), ping_median=round(float(np.median(dep)), 1), verdict=verdict))
o = pd.DataFrame(out); o.to_csv("candidates_vs_csb.csv", index=False)
print("SCORED", len(o), "candidates with pings |", o.verdict.value_counts().to_dict() if len(o) else "none", flush=True)
if len(o): print(o.sort_values("ping_q10").head(15).to_string(index=False))
