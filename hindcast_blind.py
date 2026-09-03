#!/usr/bin/env python3
"""Blind hindcast + sensitivity.
Blindness: for each grounding, ALL soundings within R km of the site are removed from the
model input (R = 10, 25), on top of the date filter; the hazard model itself was trained with
every tile group within 0.4 deg of the sites excluded. Sensitivity: the strike-site hazard
percentile is recomputed for comparison windows of 10/25/50 km and "apparently safe" filters
of 1.5x/2x/3x draft on the mean map. Also the base-rate / skill statistic.
usage: HZ_CKPT=hazard_small.pt HZ_SIZE=small python3 hindcast_blind.py -> hindcast_blind.json"""
import json, math, os, numpy as np, torch
from scipy import ndimage as ndi
from scipy.stats import norm, binom
import hindcast as H          # reuses INCIDENTS, infer(), models, blocks (module-level code runs the plain hindcast once)
DEV = "cuda"
R = 6378137.0
res = {}
for inc in H.INCIDENTS:
    x, y = H.x_of_lon(inc["lon"]), H.y_of_lat(inc["lat"])
    hit = [b for b in H.blocks if b[1] <= x <= b[2] and b[3] <= y <= b[4]]
    if not hit: continue
    f, x0, x1, y0, y1 = hit[0]
    d = np.load(f, allow_pickle=True); z = d["z"].astype("float32"); Hh, W = z.shape
    idx = np.load(f"index_out/{os.path.basename(f)}")["year"]; allk = np.isfinite(z)
    pre0 = allk & (idx > 0) & (idx < inc["year"]) if inc["year"] <= 2016 else allk
    lons = np.array([H.lon_of_x(v) for v in np.linspace(x0, x1, W)]); lats = np.array([H.lat_of_y(v) for v in np.linspace(y1, y0, Hh)])
    G = H.grav.sample(np.tile(lons, (Hh, 1)), np.tile(lats[:, None], (1, W))).astype("float32")
    j = int((x-x0)/(x1-x0)*(W-1)); i = int((y1-y)/(y1-y0)*(Hh-1))
    res_m = (x1-x0)/W*math.cos(math.radians(inc["lat"]))
    ii, jj = np.mgrid[0:Hh, 0:W]; dist2 = (ii-i)**2 + (jj-j)**2
    out = {}
    for Rkm in [0, 10, 25]:
        pre = pre0 & ~(dist2 <= (Rkm*1000/res_m)**2) if Rkm else pre0
        if pre.sum() < 500: out[f"mask_{Rkm}km"] = dict(status="too few soundings left"); continue
        zin = np.where(pre, z, np.nan); known = pre.astype(np.float32)
        dist_px = ndi.distance_transform_edt(~pre)
        mu_s, sg_s = H.infer(H.hz, zin, known, G); mu_m, sg_m = H.infer(H.mean_net, zin, known, G)
        draft = inc["draft"]; p = 1 - norm.cdf((-draft - mu_s)/np.maximum(sg_s, 0.5))
        near = dist2 <= 25; pv = p[near & np.isfinite(p)]
        rec = dict(dist_to_input_sounding_km=round(float(dist_px[i, j]*res_m/1000), 2), p_shoal_at_site=float(pv.max()) if pv.size else None,
                   shoal_depth_at_site=float(mu_s[i, j]) if np.isfinite(mu_s[i, j]) else None, mean_sigma_at_site=float(sg_m[i, j]) if np.isfinite(sg_m[i, j]) else None, ranks={})
        for win in [10, 25, 50]:
            for mult in [1.5, 2, 3]:
                safe = (dist_px <= 120) & np.isfinite(mu_m) & (mu_m < -mult*draft)
                pr, n = H.pct_rank(p, safe, i, j, int(win*1000/res_m))
                rec["ranks"][f"win{win}km_safe{mult}x"] = dict(percentile=pr, n=n)
        sig_pr, _ = H.pct_rank(sg_m, (dist_px <= 120) & np.isfinite(mu_m) & (mu_m < -2*draft), i, j, int(25000/res_m))
        rec["mean_sigma_percentile_25km"] = sig_pr
        out[f"mask_{Rkm}km"] = rec
        print(f"{inc['name']:20s} mask {Rkm:2d} km  dist {rec['dist_to_input_sounding_km']:5.1f} km  P {rec['p_shoal_at_site']}  pct(25km,2x) {rec['ranks']['win25km_safe2x']['percentile']}  pct(10km) {rec['ranks']['win10km_safe2x']['percentile']}  pct(50km) {rec['ranks']['win50km_safe2x']['percentile']}", flush=True)
    res[inc["name"]] = out
# skill statistic: under the null, a site lands above the 90th percentile with p=0.10
def hits(mask, key="win25km_safe2x", names=None):
    v = [res[n][mask]["ranks"][key]["percentile"] for n in (names or res) if "ranks" in res[n].get(mask, {})]
    v = [x for x in v if x is not None]; k = sum(1 for x in v if x >= 90); return k, len(v)
stats = {}
for mask in ["mask_0km", "mask_10km", "mask_25km"]:
    k, n = hits(mask); k4, n4 = hits(mask, names=["Thamesborg", "Akademik Ioffe", "Clipper Adventurer", "Hanseatic"])
    stats[mask] = dict(hits_all=[k, n], p_binomial_all=float(binom.sf(k-1, n, 0.1)) if n else None,
                       hits_uncharted4=[k4, n4], p_binomial_uncharted4=float(binom.sf(k4-1, n4, 0.1)) if n4 else None)
json.dump(dict(sites=res, skill=stats), open("hindcast_blind.json", "w"), indent=1, default=float)
print(json.dumps(stats, indent=1)); print("BLIND_DONE")
