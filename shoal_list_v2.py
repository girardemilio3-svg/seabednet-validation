#!/usr/bin/env python3
"""Shoal List v2 candidates: same scan as shoal_list.py but every candidate also gets its distance to the
GSHHG full-resolution coastline, and the list is drawn only from candidates >= COAST_KM from any land.
Writes shoal_candidates_v2.csv (all candidates with coast distance) and prints the offshore survivors.
env: COAST_KM (default 2.0), PMIN (default 0.8)"""
import csv, glob, math, os, numpy as np, shapefile
from scipy import ndimage as ndi
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from v5_data import lat_of_y, lon_of_x, GravityPrior
COAST_KM = float(os.environ.get("COAST_KM", 2.0)); PMIN = float(os.environ.get("PMIN", 0.6)); SH_MIN = float(os.environ.get("SH_MIN", -21)); SH_MAX = float(os.environ.get("SH_MAX", 1.0)); OUT = os.environ.get("OUT", "shoal_candidates_v2.csv")
polys = []
sf = shapefile.Reader("planetary/gshhg/GSHHS_shp/f/GSHHS_f_L1")
for shp in sf.iterShapes():
    b = shp.bbox
    if b[2] < -145 or b[0] > -45 or b[3] < 40 or b[1] > 82: continue
    parts = list(shp.parts) + [len(shp.points)]
    for a, c in zip(parts[:-1], parts[1:]):
        p = Polygon(shp.points[a:c]); polys.append(p if p.is_valid else p.buffer(0))
tree = STRtree(polys); print("coast polys", len(polys), flush=True)
def coast_km(lon, lat):
    pt = Point(lon, lat); idx = tree.nearest(pt); g = polys[idx]
    if g.contains(pt): return 0.0
    # local-metric distance: scale lon by cos(lat)
    from shapely.affinity import scale
    return g.exterior.distance(pt) * 111.0 * math.sqrt((math.cos(math.radians(lat))**2 + 1)/2)
G = GravityPrior(); rows = []
HZD = os.environ.get("HZ_DIR", "hazard_nat")
for hf in sorted(glob.glob(f"{HZD}/*.npz")):
    name = os.path.basename(hf); cf = f"national_v5_out/{name}"
    if not os.path.exists(cf): cf = f"corridor_out_v2/{name}"
    if not os.path.exists(cf): continue
    hd = np.load(hf, allow_pickle=True); p = hd["p105"].astype("float32"); shoal = hd["shoal"].astype("float32"); bb = hd["bbox3857"]
    cd = np.load(cf, allow_pickle=True); c = cd["complete"].astype("float32"); k = cd["known"].astype(bool)
    z = np.load(f"tiles_nat/{name}", allow_pickle=True)["z"].astype("float32"); H, W = p.shape
    if c.shape != p.shape: continue
    kk = np.isfinite(z)
    if kk.sum() > 500:
        yi, xi = np.nonzero(kk); sel = np.random.default_rng(0).choice(len(yi), min(4000, len(yi)), replace=False)
        x0b, x1b = min(bb[0], bb[2]), max(bb[0], bb[2]); y0b, y1b = min(bb[1], bb[3]), max(bb[1], bb[3])
        lons_s = np.degrees((x0b + (xi[sel]+0.5)/W*(x1b-x0b))/6378137.0); lats_s = np.degrees(2*np.arctan(np.exp((y1b - (yi[sel]+0.5)/H*(y1b-y0b))/6378137.0)) - np.pi/2)
        if np.median(np.abs(G.sample(lons_s, lats_s) - z[yi[sel], xi[sel]])) > 30: continue
    dist_px = ndi.distance_transform_edt(~k)
    charted_shallow = ndi.maximum_filter(np.where(np.isfinite(z), -z, 0), size=11) > 0
    charted_shallow &= ndi.maximum_filter(np.where(np.isfinite(z), z, -1e9), size=11) > -21
    cand = (np.isfinite(p) & (p >= PMIN) & np.isfinite(c) & (c < -21) & (c > -150) & (dist_px <= 60) & ~charted_shallow
            & np.isfinite(shoal) & (shoal > SH_MIN) & (shoal < SH_MAX) & ((shoal - c) < 100))
    if not cand.any(): continue
    lab, nl = ndi.label(cand); x0, x1 = min(bb[0], bb[2]), max(bb[0], bb[2]); y0, y1 = min(bb[1], bb[3]), max(bb[1], bb[3])
    for li in range(1, nl+1):
        m = lab == li
        if m.sum() < 3 or m.sum() > 300: continue
        pm = np.where(m, p, 0); i, j = np.unravel_index(np.argmax(pm), pm.shape)
        ys, xs = np.nonzero(m); n_rows, n_cols = len(set(ys.tolist())), len(set(xs.tolist()))
        lon = lon_of_x(x0 + (j+0.5)/W*(x1-x0)); lat = lat_of_y(y1 - (i+0.5)/H*(y1-y0))
        rows.append(dict(lat=round(lat, 5), lon=round(lon, 5), p_peak=round(float(p[i, j]), 3), cells=int(m.sum()), hazard_mass=round(float(p[m].sum()), 1),
                         predicted_shoal_m=round(float(shoal[i, j]), 1), mean_map_depth_m=round(float(c[i, j]), 1), km_to_sounding=round(float(dist_px[i, j])*0.1, 1),
                         coast_km=round(coast_km(lon, lat), 2), n_rows=n_rows, n_cols=n_cols, edge_px=int(min(i, j, H-1-i, W-1-j)), block=name[:-4]))
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
import collections
print(f"candidates {len(rows)}; coast>=1km {sum(r['coast_km']>=1 for r in rows)}; >=2km {sum(r['coast_km']>=2 for r in rows)}; >=5km {sum(r['coast_km']>=5 for r in rows)}")
off = [r for r in rows if r["coast_km"] >= COAST_KM]; off.sort(key=lambda r: (-round(r["p_peak"], 2), -r["hazard_mass"]))
sh = np.array([r["predicted_shoal_m"] for r in off]); print("offshore shoal-depth histogram (m):", np.histogram(sh, bins=[-21, -15, -10, -5, -2, 1.1])[0].tolist() if len(sh) else "none")
for r in off[:40]: print(f"  {r['lat']:.3f}N {abs(r['lon']):.3f}W P={r['p_peak']:.2f} shoal {r['predicted_shoal_m']} map {r['mean_map_depth_m']} coast {r['coast_km']} km cells {r['cells']} {r['block']}")
print("V2_DONE")
