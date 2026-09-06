#!/usr/bin/env python3
"""Reported-dangers hindcast: Canadian Coast Guard NAVWARNs in the danger categories (Shallow Depth Confirmed /
Reported, Shoal, Uncharted Rock, Submerged Object) vs the national hazard field (hazard_nat, 2026-08-29 NONNA snapshot).
For each notice position: hazard P(shallowest<10.5 m) (max within 500 m), its percentile among apparently-safe water
(mean map < -21 m, within 6 km of a sounding) inside 25 km, completed depth, nearest published sounding (distance, depth),
survey year at the point. 'blind' = no published sounding within 300 m of the position.
in: recon_navwarn/danger_points.csv  out: navwarn_hindcast.csv / navwarn_hindcast.json"""
import csv, glob, json, math, os, numpy as np
from scipy import ndimage as ndi
from scipy.stats import binomtest
import math as _m
R = 6378137.0
def x_of_lon(lo): return R*_m.radians(lo)
def y_of_lat(la): return R*_m.log(_m.tan(_m.pi/4+_m.radians(la)/2))
pts = list(csv.DictReader(open("recon_navwarn/danger_points.csv")))
blocks = []
HZD = os.environ.get("HZ_DIR", "hazard_nat"); TAG = os.environ.get("TAG", "")
for f in sorted(glob.glob(f"{HZD}/*.npz")):
    bb = np.load(f, allow_pickle=True)["bbox3857"]; blocks.append((os.path.basename(f), min(bb[0], bb[2]), max(bb[0], bb[2]), min(bb[1], bb[3]), max(bb[1], bb[3])))
def pct_rank(field, mask, i, j, r, ii, jj):
    loc = mask & ((ii-i)**2 + (jj-j)**2 <= r*r) & np.isfinite(field)
    if loc.sum() < 50 or not np.isfinite(field[i, j]): return None, int(loc.sum())
    return float((field[loc] < field[i, j]).mean()*100), int(loc.sum())
bypt = {}
for p in pts:
    x, y = x_of_lon(float(p["lon"])), y_of_lat(float(p["lat"]))
    hit = [b for b in blocks if b[1] <= x <= b[2] and b[3] <= y <= b[4]]
    if hit: bypt.setdefault(hit[0][0], []).append((p, x, y, hit[0]))
out = []; print(f"{len(pts)} points, {sum(len(v) for v in bypt.values())} inside hazard blocks ({len(bypt)} blocks)", flush=True)
for name, items in bypt.items():
    hd = np.load(f"{HZD}/{name}", allow_pickle=True); P = hd["p105"].astype("float32"); S = hd["shoal"].astype("float32")
    cd = np.load(f"national_v5_out/{name}", allow_pickle=True); C = cd["complete"].astype("float32"); K = cd["known"].astype(bool)
    SG = cd["sigma"].astype("float32") if "sigma" in cd else np.full_like(C, np.nan)
    Z = np.load(f"tiles_nat/{name}", allow_pickle=True)["z"].astype("float32"); YR = np.load(f"index_out/{name}")["year"]
    if P.shape != C.shape or Z.shape != C.shape: continue
    H, W = P.shape; ii, jj = np.mgrid[0:H, 0:W]
    dist_px, nidx = ndi.distance_transform_edt(~K, return_indices=True); nearest_z = Z[nidx[0], nidx[1]]
    for p, x, y, (nm, x0, x1, y0, y1) in items:
        j = int((x-x0)/(x1-x0)*(W-1)); i = int((y1-y)/(y1-y0)*(H-1)); lat = float(p["lat"])
        res_m = (x1-x0)/W*math.cos(math.radians(lat)); r25 = int(25000/res_m)
        near = (ii-i)**2 + (jj-j)**2 <= 25
        safe = (dist_px <= 60) & np.isfinite(C) & (C < -21)
        pm = P[near & np.isfinite(P)]; pmax = float(pm.max()) if pm.size else None
        pr, n = pct_rank(P, safe, i, j, r25, ii, jj)
        pr_nn, _ = pct_rank(nearest_z, safe, i, j, r25, ii, jj)      # baseline: shallowness of nearest sounding (higher z = shallower)
        pr_sg, _ = pct_rank(SG, safe, i, j, r25, ii, jj)
        yr_near = YR[near & np.isfinite(Z)]; yr = int(yr_near.max()) if yr_near.size else None
        out.append(dict(p, block=nm, hazard_p=None if not np.isfinite(P[i, j]) else round(float(P[i, j]), 3), hazard_p_500m=None if pmax is None else round(pmax, 3),
                        shoal_model_m=None if not np.isfinite(S[i, j]) else round(float(S[i, j]), 1), complete_m=None if not np.isfinite(C[i, j]) else round(float(C[i, j]), 1),
                        nearest_sounding_km=round(float(dist_px[i, j]*res_m/1000), 2), nearest_sounding_m=None if not np.isfinite(nearest_z[i, j]) else round(float(nearest_z[i, j]), 1),
                        survey_year_500m=yr, in_domain=bool(dist_px[i, j] <= 60), map_safe=bool(np.isfinite(C[i, j]) and C[i, j] < -21),
                        pct_hazard=None if pr is None else round(pr, 1), pct_nearest=None if pr_nn is None else round(pr_nn, 1), pct_sigma=None if pr_sg is None else round(pr_sg, 1), n_safe_cells=n))
    print(name, len(items), flush=True)
with open(f"navwarn_hindcast{TAG}.csv", "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
def summ(rows, label):
    v = [r["pct_hazard"] for r in rows if r["pct_hazard"] is not None]
    if not v: print(f"{label}: no scorable points"); return
    k90 = sum(x >= 90 for x in v); k75 = sum(x >= 75 for x in v); med = float(np.median(v))
    nn = [r["pct_nearest"] for r in rows if r["pct_nearest"] is not None]; sg = [r["pct_sigma"] for r in rows if r["pct_sigma"] is not None]
    print(f"{label}: n={len(v)} median pct {med:.0f} | >=90th {k90} ({k90/len(v)*100:.0f}%, binomial p={binomtest(k90, len(v), 0.10, alternative='greater').pvalue:.2e}) | >=75th {k75} ({k75/len(v)*100:.0f}%) "
          f"| baselines median: nearest-sounding {np.median(nn) if nn else float('nan'):.0f}, sigma {np.median(sg) if sg else float('nan'):.0f}")
    return dict(n=len(v), median=med, ge90=k90, ge75=k75, p_ge90=binomtest(k90, len(v), 0.10, alternative='greater').pvalue)
dep = lambda r: float(r["depth_m"]) if r["depth_m"] not in ("", None, "None") else None
S = {}
S["all_scorable"] = summ(out, "ALL danger points in domain")
S["map_safe"] = summ([r for r in out if r["map_safe"]], "map says safe (<-21 m) at the position")
S["map_safe_blind"] = summ([r for r in out if r["map_safe"] and r["nearest_sounding_km"] >= 0.3], "map safe AND no sounding within 300 m (blind)")
S["shallow_reported"] = summ([r for r in out if dep(r) is not None and dep(r) <= 21], "reported depth <= 21 m")
S["shallow_reported_map_safe"] = summ([r for r in out if dep(r) is not None and dep(r) <= 21 and r["map_safe"]], "reported <=21 m AND map safe")
S["shallow_reported_map_safe_blind"] = summ([r for r in out if dep(r) is not None and dep(r) <= 21 and r["map_safe"] and r["nearest_sounding_km"] >= 0.3], "reported <=21 m AND map safe AND blind")
S["arctic"] = summ([r for r in out if r["series"].startswith("NW-A")], "Arctic notices (NW-A)")
S["arctic_map_safe"] = summ([r for r in out if r["series"].startswith("NW-A") and r["map_safe"]], "Arctic AND map safe")
S["by_category"] = {c: summ([r for r in out if r["category"] == c and r["map_safe"]], f"category {c} (map safe)") for c in sorted(set(r["category"] for r in out))}
json.dump(dict(summary=S, n_points=len(pts), n_in_blocks=len(out), source="CCG NAVWARN search (nis.ccg-gcc.gc.ca), active notices 2026-09-06", hazard=f"{HZD} (NONNA 2026-08-29 snapshot)"), open(f"navwarn_hindcast{TAG}.json", "w"), indent=1, default=float)
print("NAVWARN_HINDCAST_DONE")
