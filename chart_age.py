#!/usr/bin/env python3
"""The age of Canada's chart: national mosaic of survey year per sounded cell + the
national statistics. -> chart_age.json, chart_age.png/_web.jpg"""
import glob, json, math, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
xs = []; ys = []
metas = []
tot = {"n":0, "pre1960":0, "pre1980":0, "pre2000":0, "post":0}
for f in sorted(glob.glob("tiles_nat/*.npz")):
    ip = f"index_out/{os.path.basename(f)}"
    if not os.path.exists(ip): continue
    bb = np.load(f, allow_pickle=True)["bbox3857"]
    metas.append((f, ip, min(bb[0],bb[2]), max(bb[0],bb[2]), min(bb[1],bb[3]), max(bb[1],bb[3])))
    xs += [min(bb[0],bb[2]), max(bb[0],bb[2])]; ys += [min(bb[1],bb[3]), max(bb[1],bb[3])]
X0, X1, Y0, Y1 = min(xs), max(xs), min(ys), max(ys)
Wc = 3000; Hc = int(Wc*(Y1-Y0)/(X1-X0))
canvas = np.zeros((Hc, Wc), np.int16)
for f, ip, x0, x1, y0, y1 in metas:
    z = np.load(f, allow_pickle=True)["z"]; yr = np.load(ip)["year"].astype(np.int16)
    k = np.isfinite(z)
    y_eff = np.where(k, np.where(yr == 0, 2020, yr), 0).astype(np.int16)
    tot["n"] += int(k.sum()); tot["pre1960"] += int((k & (yr > 0) & (yr < 1960)).sum())
    tot["pre1980"] += int((k & (yr > 0) & (yr < 1980)).sum()); tot["pre2000"] += int((k & (yr > 0) & (yr < 2000)).sum())
    tot["post"] += int((k & (yr == 0)).sum())
    c0, c1 = int((x0-X0)/(X1-X0)*Wc), int((x1-X0)/(X1-X0)*Wc); r0, r1 = int((Y1-y1)/(Y1-Y0)*Hc), int((Y1-y0)/(Y1-Y0)*Hc)
    if c1-c0 < 2 or r1-r0 < 2: continue
    im = np.array(Image.fromarray(y_eff, mode="I;16").resize((c1-c0, r1-r0), Image.NEAREST)).astype(np.int16)
    reg = canvas[r0:r1, c0:c1]; canvas[r0:r1, c0:c1] = np.where(im > 0, im, reg)
pct = lambda v: round(100*v/max(tot["n"],1), 1)
stats = dict(sounded_cells=tot["n"], pct_pre1960=pct(tot["pre1960"]), pct_pre1980=pct(tot["pre1980"]),
             pct_pre2000=pct(tot["pre2000"]), pct_post2016_modern=pct(tot["post"]))
json.dump(stats, open("chart_age.json", "w"), indent=1); print(stats)
bounds = [1800, 1940, 1960, 1980, 2000, 2017, 2030]
colors = ["#8c1d1d", "#c0392b", "#e67e22", "#e8b24a", "#7f9db9", "#2e7dd1"]
cmap = ListedColormap(colors); norm = BoundaryNorm(bounds, cmap.N)
fig = plt.figure(figsize=(18, 18*Hc/Wc), facecolor="#04060c"); ax = fig.add_axes([0, 0, 1, 0.95]); ax.set_facecolor("#04060c")
masked = np.ma.masked_where(canvas == 0, canvas)
ax.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
fig.text(0.01, 0.985, "THE AGE OF CANADA'S CHART — year of the latest survey under every sounded cell (CHS NONNA + CHS Survey Index)", color="#d8e0ee", fontsize=15, va="top", family="monospace")
fig.text(0.01, 0.962, f"dark red <1940 · red 1940-60 · orange 1960-80 · amber 1980-2000 · grey-blue 2000-16 · blue post-2016 multibeam   |   {stats['pct_pre1980']}% of sounded cells predate 1980 · {stats['pct_post2016_modern']}% are post-2016", color="#7c8aa6", fontsize=11.5, va="top", family="monospace")
fig.savefig("chart_age.png", dpi=110, facecolor="#04060c")
Image.open("chart_age.png").convert("RGB").save("chart_age_web.jpg", quality=82)
print("CHARTAGE_DONE")
