#!/usr/bin/env python3
"""ICESat-2 photon check of the sealed claims. For each claim: all ATL03 photons (any confidence) in a 1.2 km box, 2019-2026.
Per track-date: the sea surface is the photon mode near the local mean sea level; a bottom return is a second peak 1.5-40 m below it,
at least 5 sigma above the deep-water background rate. Reports, per claim: number of passes, passes with a bottom return, the
shallowest and median bottom depth seen, and whether anything shallower than 10 m appears within the box.
-> is2_claims.json, is2_claims/<claim>.npz (photon subsets)"""
import json, math, os, time, numpy as np, pandas as pd
from sliderule import sliderule, icesat2
sliderule.init("slideruleearth.io", verbose=False)
d = pd.read_csv("shoal_list_v2_2026-09-06.csv"); os.makedirs("is2_claims", exist_ok=True); out = []
for i, r in d.iterrows():
    dl = 0.6/111.0; dn = 0.6/(111.0*abs(math.cos(math.radians(r.lat))))
    poly = [{"lon": r.lon-dn, "lat": r.lat-dl}, {"lon": r.lon+dn, "lat": r.lat-dl}, {"lon": r.lon+dn, "lat": r.lat+dl}, {"lon": r.lon-dn, "lat": r.lat+dl}, {"lon": r.lon-dn, "lat": r.lat-dl}]
    parms = {"poly": poly, "srt": 0, "cnf": 0, "len": 20.0, "res": 10.0, "pass_invalid": True, "t0": "2019-06-01T00:00:00Z", "t1": "2026-09-01T00:00:00Z"}
    rec = dict(claim=int(i), region=r.region, lat=float(r.lat), lon=float(r.lon), predicted_shoal_m=float(r.predicted_shoal_m), chart_m=float(r.mean_map_depth_m), photons=0, passes=0, passes_with_bottom=0, bottom_depths_m=[], shallowest_bottom_m=None, note="")
    try:
        t = time.time(); g = icesat2.atl03sp(parms); rec["photons"] = int(len(g)); rec["fetch_s"] = round(time.time()-t)
    except Exception as e: rec["note"] = f"fetch error: {str(e)[:80]}"; out.append(rec); print(i, r.region, rec["note"], flush=True); continue
    if len(g) < 200: rec["note"] = "no photons"; out.append(rec); print(i, r.region, "no photons", flush=True); continue
    g = g.reset_index(); tcol = "time" if "time" in g.columns else g.columns[0]
    g["date"] = pd.to_datetime(g[tcol]).dt.date.astype(str); g["trk"] = g["date"] + "_" + g["pair"].astype(str) + "_" + g.get("track", g.get("rgt", 0)).astype(str) if "pair" in g else g["date"]
    np.savez_compressed(f"is2_claims/claim_{i:02d}.npz", h=g["height"].values.astype(np.float32), lat=g["geometry"].y.values.astype(np.float64), lon=g["geometry"].x.values.astype(np.float64), trk=g["trk"].values.astype(str))
    for trk, s in g.groupby("trk"):
        h = s["height"].values.astype(np.float32); h = h[np.isfinite(h)]
        if len(h) < 300: continue
        # surface: mode of the height histogram (0.25 m bins) within the central 60 m of the distribution
        lo, hi = np.percentile(h, [5, 95]); bins = np.arange(lo, hi + 0.25, 0.25); cnt, edges = np.histogram(h, bins)
        if cnt.max() < 30: continue
        surf = edges[np.argmax(cnt)] + 0.125
        sub = h[(h < surf - 1.5) & (h > surf - 40)]
        if len(sub) < 50: rec["passes"] += 1; continue
        b2 = np.arange(surf - 40, surf - 1.5 + 0.25, 0.25); c2, e2 = np.histogram(sub, b2)
        bg = c2[(e2[:-1] < surf - 25)]; bg_mu = bg.mean() if len(bg) else 0.0; bg_sd = max(bg.std() if len(bg) else 1.0, 1.0)
        peak = int(np.argmax(c2)); z = (c2[peak] - bg_mu)/bg_sd
        rec["passes"] += 1
        if z >= 5 and c2[peak] >= 8:
            depth = float(surf - (e2[peak] + 0.125)); rec["passes_with_bottom"] += 1; rec["bottom_depths_m"].append(round(depth, 1))
    if rec["bottom_depths_m"]:
        rec["shallowest_bottom_m"] = min(rec["bottom_depths_m"]); rec["median_bottom_m"] = float(np.median(rec["bottom_depths_m"]))
    out.append(rec); print(i, r.region, f"photons {rec['photons']} passes {rec['passes']} bottom-passes {rec['passes_with_bottom']} depths {rec['bottom_depths_m'][:6]}", flush=True)
json.dump(out, open("is2_claims.json", "w"), indent=1)
n_b = sum(1 for r in out if r["passes_with_bottom"]); print(f"claims with any bottom return: {n_b} of {len(out)}"); print("IS2_CLAIMS_DONE")
