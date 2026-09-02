#!/usr/bin/env python3
"""Rasterize the CHS Survey Index (1832-2016, with CATZOC + dates) onto the NONNA
block grids. Output per block: year of latest indexed survey per pixel (0 = not in
any indexed polygon => surveyed after the index era, i.e. modern OPP multibeam),
and CATZOC grade code. Reports the size of the temporal test set.

usage: INDEX_OUT=index_out python3 index_coverage.py blocks.txt   -> $INDEX_OUT/<block>.npz
"""
import json, math, os, sys, numpy as np
from PIL import Image, ImageDraw
from v5_data import lat_of_y, lon_of_x
R = 6378137.0
def y_of_lat(la): return R*math.log(math.tan(math.pi/4+math.radians(la)/2))
def x_of_lon(lo): return R*math.radians(lo)

J = json.load(open("chs_survey_index.geojson"))["features"]
ZOC = {"zone of confidence A1": 1, "zone of confidence A2": 2, "zone of confidence B": 3,
       "zone of confidence C": 4, "zone of confidence D": 5}
polys = []
for f in J:
    p = f["properties"]; y = p["SURSTA"] or ""
    try: yr = int(y[:4])
    except: yr = 0
    if yr < 1800: yr = 0
    g = f["geometry"]
    rings = g["coordinates"] if g["type"] == "Polygon" else [r for pg in g["coordinates"] for r in pg]
    outer = [rings[0]] if g["type"] == "Polygon" else [pg[0] for pg in g["coordinates"]]
    lons = [c[0] for r in outer for c in r]; lats = [c[1] for r in outer for c in r]
    polys.append((min(lons), max(lons), min(lats), max(lats), yr, ZOC.get(p["CATZOC"], 6), outer))
polys.sort(key=lambda t: t[4])   # draw oldest first so the latest survey wins per pixel

OUT = os.environ.get("INDEX_OUT", "index_out"); os.makedirs(OUT, exist_ok=True)
files = [l.strip() for l in open(sys.argv[1]) if l.strip()]
tot_known = tot_modern = 0
for f in files:
    d = np.load(f, allow_pickle=True); z = d["z"]; bb = d["bbox3857"]; H, W = z.shape
    known = np.isfinite(z)
    x0, x1 = min(bb[0], bb[2]), max(bb[0], bb[2]); y0, y1 = min(bb[1], bb[3]), max(bb[1], bb[3])
    lo0, lo1, la0, la1 = lon_of_x(x0), lon_of_x(x1), lat_of_y(y0), lat_of_y(y1)
    yr_img = Image.new("I", (W, H), 0); zoc_img = Image.new("I", (W, H), 0)
    dy = ImageDraw.Draw(yr_img); dz = ImageDraw.Draw(zoc_img)
    for (a, b, c, e, yr, zoc, outer) in polys:
        if b < lo0 or a > lo1 or e < la0 or c > la1: continue
        for ring in outer:
            pts = [((x_of_lon(lo)-x0)/(x1-x0)*(W-1), (y1-y_of_lat(la))/(y1-y0)*(H-1)) for lo, la in ring]
            if len(pts) >= 3:
                dy.polygon(pts, fill=int(yr)); dz.polygon(pts, fill=int(zoc))
    yr_a = np.array(yr_img, dtype=np.int16); zoc_a = np.array(zoc_img, dtype=np.int8)
    modern = known & (yr_a == 0)
    tot_known += known.sum(); tot_modern += modern.sum()
    np.savez_compressed(f"{OUT}/{os.path.basename(f)}", year=yr_a, zoc=zoc_a)
    print(f"{os.path.basename(f)}: known {known.sum():8d}  in-index {(known & (yr_a>0)).sum():8d}  post-index(modern) {modern.sum():8d}", flush=True)
print(f"TOTAL known px {tot_known:,}  post-index (temporal test set) {tot_modern:,}  = {tot_modern/max(tot_known,1)*100:.1f}%")
print("INDEX_DONE")
