#!/usr/bin/env python3
"""Re-analyse the cached ICESat-2 photons per claim with a stricter bottom-return detector:
surface = mode of heights; search window surf-40 .. surf-3 m (the 3 m guard excludes the surface return's tail and subsurface
scattering); a bottom return is a histogram peak (0.25 m bins) that is >= 5 sigma above the deep background (surf-40..-25),
>= 3x the window median, not in the first metre of the window, and a local maximum over +-1 m.
-> is2_claims_strict.json"""
import glob, json, numpy as np, pandas as pd
d = pd.read_csv("shoal_list_v2_2026-09-06.csv"); out = []
for f in sorted(glob.glob("is2_claims/claim_*.npz")):
    i = int(f[-6:-4]); r = d.iloc[i]; z = np.load(f, allow_pickle=True); h_all = z["h"]; trk = z["trk"]
    rec = dict(claim=i, region=r.region, lat=float(r.lat), lon=float(r.lon), predicted_shoal_m=float(r.predicted_shoal_m), chart_m=float(r.mean_map_depth_m), photons=int(len(h_all)), passes=0, passes_with_bottom=0, bottom_depths_m=[], surface_widths=[])
    for t in np.unique(trk):
        h = h_all[trk == t]; h = h[np.isfinite(h)]
        if len(h) < 300: continue
        lo, hi = np.percentile(h, [5, 95]); bins = np.arange(lo, hi + 0.25, 0.25); cnt, edges = np.histogram(h, bins)
        if cnt.max() < 30: continue
        surf = edges[np.argmax(cnt)] + 0.125; rec["passes"] += 1
        near = h[(h > surf - 2) & (h < surf + 2)]; rec["surface_widths"].append(round(float(np.std(near)), 2))
        sub = h[(h < surf - 3.0) & (h > surf - 40)]
        if len(sub) < 40: continue
        b2 = np.arange(surf - 40, surf - 3.0 + 0.25, 0.25); c2, e2 = np.histogram(sub, b2)
        bg = c2[(e2[:-1] < surf - 25)]; bg_mu = bg.mean() if len(bg) else 0.0; bg_sd = max(bg.std() if len(bg) else 1.0, 1.0); med = max(np.median(c2), 1.0)
        for k in range(4, len(c2)-4):
            if c2[k] >= 8 and (c2[k]-bg_mu)/bg_sd >= 5 and c2[k] >= 3*med and c2[k] == c2[k-4:k+5].max():
                depth = float(surf - (e2[k] + 0.125)); rec["passes_with_bottom"] += 1; rec["bottom_depths_m"].append(round(depth, 1)); break
    if rec["bottom_depths_m"]: rec["shallowest_bottom_m"] = min(rec["bottom_depths_m"]); rec["median_bottom_m"] = float(np.median(rec["bottom_depths_m"]))
    out.append(rec); print(i, r.region[:28], f"passes {rec['passes']} bottom {rec['passes_with_bottom']} depths {rec['bottom_depths_m'][:8]}")
json.dump(out, open("is2_claims_strict.json", "w"), indent=1)
print("claims with a strict bottom return:", sum(1 for r in out if r["passes_with_bottom"]), "of", len(out)); print("IS2_STRICT_DONE")
