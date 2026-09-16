#!/usr/bin/env python3
"""Fetch ATL03 photon subsets over deep-water CONTROL sites (chart > 150 m, no bottom return possible) with exactly the same
request as is2_claims.py, so the v2 detector's false-positive rate can be measured. -> is2_controls/ctl_XX.npz"""
import json, math, os, time, numpy as np, pandas as pd
from sliderule import sliderule, icesat2
sliderule.init("slideruleearth.io", verbose=False); os.makedirs("is2_controls", exist_ok=True)
SITES = [("Lancaster Sound", 74.20, -85.00, 400), ("Beaufort slope", 70.60, -134.00, 300), ("Davis Strait", 66.00, -58.00, 600), ("Hudson Strait centre", 62.30, -70.50, 300), ("Baffin Bay", 72.00, -66.00, 800), ("Foxe Channel", 65.20, -80.00, 200)]
out = []
for i, (name, lat, lon, depth) in enumerate(SITES):
    dl = 0.6/111.0; dn = 0.6/(111.0*abs(math.cos(math.radians(lat))))
    poly = [{"lon": lon-dn, "lat": lat-dl}, {"lon": lon+dn, "lat": lat-dl}, {"lon": lon+dn, "lat": lat+dl}, {"lon": lon-dn, "lat": lat+dl}, {"lon": lon-dn, "lat": lat-dl}]
    parms = {"poly": poly, "srt": 0, "cnf": 0, "len": 20.0, "res": 10.0, "pass_invalid": True, "t0": "2019-06-01T00:00:00Z", "t1": "2026-09-01T00:00:00Z"}
    t = time.time()
    try: g = icesat2.atl03sp(parms)
    except Exception as e: print(name, "fetch error", str(e)[:100], flush=True); continue
    if g is None or len(g) < 200: print(name, "no photons", flush=True); out.append(dict(site=name, lat=lat, lon=lon, chart_m=depth, photons=0)); continue
    g = g.reset_index(); tcol = "time" if "time" in g.columns else g.columns[0]
    g["date"] = pd.to_datetime(g[tcol]).dt.date.astype(str); g["trk"] = g["date"] + "_" + g["pair"].astype(str) + "_" + g.get("track", g.get("rgt", 0)).astype(str) if "pair" in g else g["date"]
    np.savez_compressed(f"is2_controls/ctl_{i:02d}.npz", h=g["height"].values.astype(np.float32), lat=g["geometry"].y.values.astype(np.float64), lon=g["geometry"].x.values.astype(np.float64), trk=g["trk"].values.astype(str))
    out.append(dict(site=name, lat=lat, lon=lon, chart_m=depth, photons=int(len(g)), passes=int(g["trk"].nunique()))); print(name, "photons", len(g), "passes", g["trk"].nunique(), f"{time.time()-t:.0f}s", flush=True)
json.dump(out, open("is2_controls.json", "w"), indent=1); print("CONTROLS_DONE")
