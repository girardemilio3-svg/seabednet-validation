#!/usr/bin/env python3
"""Value-of-information survey plan (v2): rank 0.5-degree boxes by the EXPECTED NUMBER OF KEEL-DEPTH HAZARDS a survey would
find per ship-day, not by uncertainty. Hazard = P(shallowest point within 500 m < 10.5 m) from the v2 hazard head, counted only
on cells the chart calls navigable (completed depth < -21 m) with no published sounding within 500 m (a hazard nobody has seen).
Shipping relevance: cells within 30 km of the Churchill route, within 15 km of a sealift community, or within 30 km of a
Coast Guard reported-danger notice (recon_navwarn/danger_points.csv), i.e. water ships actually use.
Cost: 40 km2/ship-day; C$2,600/km2 (CHS contract) to C$183k/day (icebreaker charter), as in national_plan.py.
-> national_plan_v2.json"""
import csv, glob, json, math, os, numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from v5_data import lat_of_y, lon_of_x
R = 6378137.0
def merc(lon, lat): return R*math.radians(lon), R*math.log(math.tan(math.pi/4 + math.radians(lat)/2))
route = json.load(open("map/data.js").read().split("\"route\": ")[1].split("], \"shoals\"")[0] + "]") if False else None
import re
D = open("map/data.js").read(); route = json.loads(re.search(r"\"route\":\s*(\[\[.*?\]\])", D).group(1)); comm = json.loads(re.search(r"\"communities\":\s*(\[.*?\])\s*,\s*\"plan\"", D, re.S).group(1))
pts = [merc(p[1], p[0]) for p in route] + [merc(c["lon"], c["lat"]) for c in comm]
nrc = []
for L in (0, 1):
    for ft in json.load(open(f"planetary/routes/nrcan_layer{L}.geojson"))["features"]:
        g = ft["geometry"]; lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for ln in lines:
            for a, b in zip(ln[:-1], ln[1:]):
                n = max(2, int(math.hypot(a[0]-b[0], a[1]-b[1])/0.05) + 1)
                for t in np.linspace(0, 1, n): nrc.append(merc(a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1])))
print("NRCan route vertices (densified)", len(nrc))
tree_r = cKDTree([merc(p[1], p[0]) for p in route]); tree_c = cKDTree([merc(c["lon"], c["lat"]) for c in comm]); tree_n = cKDTree(nrc)
from v5_data import GravityPrior; G = GravityPrior()
boxes = {}
for f in sorted(glob.glob("hazard_nat_v2/*.npz")):
    name = os.path.basename(f); cf = f"national_v5_out/{name}"
    if not os.path.exists(cf): continue
    hd = np.load(f, allow_pickle=True); p = hd["p105"].astype("float32"); bb = hd["bbox3857"]
    cd = np.load(cf, allow_pickle=True); c = cd["complete"].astype("float32"); k = cd["known"].astype(bool)
    if p.shape != c.shape: continue
    H, W = p.shape; x0, x1 = min(bb[0], bb[2]), max(bb[0], bb[2]); y0, y1 = min(bb[1], bb[3]), max(bb[1], bb[3])
    z = np.load(f"tiles_nat/{name}", allow_pickle=True)["z"]; kk = np.isfinite(z)
    if kk.sum() > 500:                                   # gravity-anchor sanity: skip inland lakes / fjord heads where the physics prior is invalid
        yi, xi = np.nonzero(kk); sel = np.random.default_rng(0).choice(len(yi), min(4000, len(yi)), replace=False)
        glon = np.degrees((x0 + (xi[sel]+0.5)/W*(x1-x0))/R); glat = np.degrees(2*np.arctan(np.exp((y1 - (yi[sel]+0.5)/H*(y1-y0))/R)) - np.pi/2)
        if np.median(np.abs(G.sample(glon, glat) - z[yi[sel], xi[sel]])) > 30: continue
    dist = ndi.distance_transform_edt(~k)
    unseen = np.isfinite(p) & np.isfinite(c) & (c < -21) & (dist > 5) & (dist <= 60)
    if unseen.sum() < 100: continue
    ii, jj = np.nonzero(unseen)
    xs = x0 + (jj+0.5)/W*(x1-x0); ys = y1 - (ii+0.5)/H*(y1-y0)
    lat = np.degrees(2*np.arctan(np.exp(ys/R)) - np.pi/2); lon = np.degrees(xs/R)
    scale = 1/np.cos(np.radians(lat))                     # mercator metres -> ground metres
    q = np.c_[xs, ys]
    dr = tree_r.query(q)[0]/scale; dc = tree_c.query(q)[0]/scale; dn = tree_n.query(q)[0]/scale
    rel = (dr < 30000) | (dc < 15000) | (dn < 30000)   # dn = NRCan Canadian shipping routes / Arctic sea routes
    if not rel.any(): continue
    area = ((x1-x0)/W*np.cos(np.radians(lat[rel]))/1000.0)**2       # km2 per cell
    pv = p[ii[rel], jj[rel]]
    bx = np.floor(lon[rel]*2)/2; by = np.floor(lat[rel]*2)/2
    for blo, bla, a, pp in zip(bx, by, area, pv):
        e = boxes.setdefault((float(blo), float(bla)), [0.0, 0.0, 0]); e[0] += a; e[1] += pp*a; e[2] += 1
rank = []
for (blo, bla), (area, mass, n) in boxes.items():
    if area < 25: continue
    days = area/40.0; exp_haz = mass/0.01   # expected hazard cells (each cell ~0.01 km2 at 100 m) with P summed over km2 -> hazards
    rank.append(dict(lon=blo, lat=bla, area_km2=round(area), unseen_cells=int(n), expected_hazard_km2=round(mass, 2), ship_days=round(days, 1), hazards_per_ship_day=round(mass/days, 3)))
rank.sort(key=lambda r: -r["hazards_per_ship_day"]); top = rank[:20]
tot_area = sum(r["area_km2"] for r in top); tot_days = sum(r["ship_days"] for r in top)
out = dict(method="expected keel-depth hazard area (sum of P(shoal<10.5 m) over unseen navigable cells) per ship-day, within shipping-relevant water",
           relevance="within 30 km of the Churchill route or an NRCan Canadian shipping route / Arctic sea route, or 15 km of a sealift community; blocks where the gravity prior fails (inland lakes) excluded", hazard_field="hazard_nat_v2 (coast-aware head, 2026-09-06)",
           top20=top, total=dict(area_km2=round(tot_area), ship_days=round(tot_days), cost_low_MCAD=round(tot_area*2600/1e6), cost_high_MCAD=round(tot_days*183000/1e6), expected_hazard_km2=round(sum(r["expected_hazard_km2"] for r in top), 1)),
           n_boxes_considered=len(rank))
json.dump(out, open("national_plan_v2.json", "w"), indent=1)
print("boxes", len(rank), "| top20 area", out["total"]["area_km2"], "km2, ship-days", out["total"]["ship_days"], "C$", out["total"]["cost_low_MCAD"], "-", out["total"]["cost_high_MCAD"], "M; expected hazard km2", out["total"]["expected_hazard_km2"])
for r in top[:12]: print(f"  {r['lat']:5.1f}N {abs(r['lon']):6.1f}W  area {r['area_km2']:5d} km2  hazard-km2 {r['expected_hazard_km2']:6.2f}  per ship-day {r['hazards_per_ship_day']:.3f}")
print("PLAN_V2_DONE")
