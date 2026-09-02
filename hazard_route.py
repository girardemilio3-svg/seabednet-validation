#!/usr/bin/env python3
"""Hazard along the Churchill route + corridor hazard map.
For every 2 km route point: mean-map depth (atlas), predicted shallowest-within-500 m,
P(shoal < 10.5 m). Counts the km of route that the mean map calls safe (> 2x draft) but
the hazard field flags (P > 5 %). Renders hazard_corridor.png (+ _web.jpg).
-> route_hazard.json, hazard_corridor.json"""
import glob, json, math, numpy as np
from v5_data import lat_of_y, lon_of_x
R = 6378137.0
xl = lambda lo: R*math.radians(lo); yl = lambda la: R*math.log(math.tan(math.pi/4+math.radians(la)/2))
blocks = {}
for f in sorted(glob.glob("hazard_out/*.npz")):
    bb = np.load(f, allow_pickle=True)["bbox3857"]; blocks[f] = (min(bb[0],bb[2]), max(bb[0],bb[2]), min(bb[1],bb[3]), max(bb[1],bb[3]))
cache = {}
def get(f):
    if f not in cache:
        d = np.load(f, allow_pickle=True); cache[f] = dict(shoal=d["shoal"].astype(float), sig=d["shoal_sigma"].astype(float), p=d["p105"].astype(float))
    return cache[f]
prof = json.load(open("route_profile_v2.json")); out = []
for p in prof:
    x, y = xl(p["lon"]), yl(p["lat"]); q = dict(p)
    for f, (x0, x1, y0, y1) in blocks.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            d = get(f); H, W = d["shoal"].shape; i = int((y1-y)/(y1-y0)*(H-1)); j = int((x-x0)/(x1-x0)*(W-1))
            if np.isfinite(d["shoal"][i, j]):
                q["shoal"] = round(d["shoal"][i, j], 1); q["shoal_sigma"] = round(d["sig"][i, j], 1); q["p_shoal"] = round(d["p"][i, j], 4)
            break
    out.append(q)
json.dump(out, open("route_hazard.json", "w"))
has = [q for q in out if "p_shoal" in q]
safe_mean = [q for q in has if q.get("depth") is not None and -q["depth"] > 21]
flag = [q for q in safe_mean if q["p_shoal"] > 0.05]
stats = dict(n_points=len(out), n_with_hazard=len(has), km_step=2.0,
             km_mean_map_safe=len(safe_mean)*2.0, km_flagged=len(flag)*2.0,
             worst=sorted([(q["km"], q["lat"], q["lon"], q.get("depth"), q["shoal"], q["p_shoal"], q.get("grade")) for q in flag], key=lambda t: -t[5])[:10],
             median_gap_m=float(np.median([q["shoal"] - q["depth"] for q in has if q.get("depth") is not None])))
# corridor-wide cell stats
tot = 0; f5 = 0; f20 = 0
for f in blocks:
    d = get(f); v = np.isfinite(d["shoal"]); tot += int(v.sum()); f5 += int((d["p"][v] > 0.05).sum()); f20 += int((d["p"][v] > 0.2).sum())
stats.update(corridor_cells=tot, frac_p_gt_5pct=f5/max(tot,1), frac_p_gt_20pct=f20/max(tot,1))
json.dump(stats, open("hazard_corridor.json", "w"), indent=1)
print(json.dumps({k: v for k, v in stats.items() if k != "worst"}, indent=1)); print("worst:", stats["worst"][:5])
# ---- render
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
xs = [b[0] for b in blocks.values()] + [b[1] for b in blocks.values()]; ys = [b[2] for b in blocks.values()] + [b[3] for b in blocks.values()]
X0, X1, Y0, Y1 = min(xs), max(xs), min(ys), max(ys)
Wc = 2400; Hc = int(Wc*(Y1-Y0)/(X1-X0)); canvas = np.full((Hc, Wc), np.nan, np.float32)
for f, (x0, x1, y0, y1) in blocks.items():
    d = get(f); H, W = d["p"].shape
    c0, c1 = int((x0-X0)/(X1-X0)*Wc), int((x1-X0)/(X1-X0)*Wc); r0, r1 = int((Y1-y1)/(Y1-Y0)*Hc), int((Y1-y0)/(Y1-Y0)*Hc)
    if c1-c0 < 2 or r1-r0 < 2: continue
    im = np.array(Image.fromarray(np.nan_to_num(d["p"], nan=-1).astype("float32")).resize((c1-c0, r1-r0), Image.BILINEAR))
    im[im < 0] = np.nan; canvas[r0:r1, c0:c1] = np.where(np.isfinite(im), im, canvas[r0:r1, c0:c1])
fig = plt.figure(figsize=(16, 16*Hc/Wc), facecolor="#04060c"); ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("#04060c")
ax.imshow(np.clip(canvas, 0, 0.5), cmap="inferno", vmin=0, vmax=0.5, interpolation="nearest")
rx = [(xl(q["lon"])-X0)/(X1-X0)*Wc for q in prof]; ry = [(Y1-yl(q["lat"]))/(Y1-Y0)*Hc for q in prof]
ax.plot(rx, ry, color="#6aa9e0", lw=0.9, alpha=0.9)
for q in flag: ax.plot((xl(q["lon"])-X0)/(X1-X0)*Wc, (Y1-yl(q["lat"]))/(Y1-Y0)*Hc, "o", ms=4, mfc="none", mec="#ff5d4d", mew=0.8)
ax.set_xticks([]); ax.set_yticks([])
fig.text(0.01, 0.985, "THE HAZARD FIELD — P(shallowest point within 500 m < 10.5 m draft), every cell of the corridor", color="#d8e0ee", fontsize=13, va="top", family="monospace")
fig.text(0.01, 0.955, f"black = safe · yellow = >50% · blue line = route · red rings = route km the mean map calls safe (>21 m) but the hazard field flags (P>5%): {stats['km_flagged']:.0f} km of {stats['km_mean_map_safe']:.0f} km", color="#7c8aa6", fontsize=10.5, va="top", family="monospace")
fig.savefig("hazard_corridor.png", dpi=110, facecolor="#04060c")
Image.open("hazard_corridor.png").convert("RGB").save("hazard_corridor_web.jpg", quality=82)
print("HAZARD_ROUTE_DONE")
