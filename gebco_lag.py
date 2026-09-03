#!/usr/bin/env python3
"""GEBCO lag audit: where the global chart product disagrees with Canada's own archive.
At every NONNA-sounded cell of the corridor blocks (GEBCO tiles cached), measure
GEBCO - NONNA. Report: fraction and km2 of sounded water where the global grid errs by
>10 m / >20 m and is SHALLOW-BIASED vs DEEP-BIASED (deep-biased = dangerous), and the same
along the 2 km route corridor. -> gebco_lag.json"""
import glob, json, math, os, numpy as np, tifffile
from v5_data import lat_of_y, lon_of_x
R=6378137.0
rows_tot = dict(n=0, gt10=0, gt20=0, deep10=0, km2=0.0, deep_km2=0.0)
route = json.load(open("route_profile_v2.json"))
rpts = [(R*math.radians(p["lon"]), R*math.log(math.tan(math.pi/4+math.radians(p["lat"])/2))) for p in route]
route_rows = []
for gt in sorted(glob.glob("gebco_cache/*.tif")):
    name = os.path.basename(gt)[:-4]
    f = f"tiles_nat/{name}.npz"
    if not os.path.exists(f): continue
    d = np.load(f, allow_pickle=True); z = d["z"].astype("float32"); bb = d["bbox3857"]; H, W = z.shape
    t = tifffile.TiffFile(gt); pg = t.pages[0]
    try:
        tp = pg.tags["ModelTiepointTag"].value; sc = pg.tags["ModelPixelScaleTag"].value
    except Exception: continue
    g = pg.asarray().astype("float32"); gh, gw = g.shape
    gx0, gy0, sx, sy = float(tp[3]), float(tp[4]), float(sc[0]), float(sc[1])
    x0, x1 = min(bb[0],bb[2]), max(bb[0],bb[2]); y0, y1 = min(bb[1],bb[3]), max(bb[1],bb[3])
    k = np.isfinite(z); ii, jj = np.nonzero(k)
    if len(ii) == 0: continue
    lons = np.degrees((x0 + (jj+0.5)/W*(x1-x0))/R)
    lats = np.degrees(2*np.arctan(np.exp((y1 - (ii+0.5)/H*(y1-y0))/R)) - np.pi/2)
    gx = np.clip(((lons-gx0)/sx).astype(int), 0, gw-1); gy = np.clip(((gy0-lats)/sy).astype(int), 0, gh-1)
    diff = g[gy, gx] - z[ii, jj]
    # only water NONNA calls shallower than 200 m (where it matters) and GEBCO calls water
    sel = (z[ii, jj] > -200) & (g[gy, gx] < 0)
    diff = diff[sel]
    if len(diff) == 0: continue
    res_km2 = ( (x1-x0)/W * math.cos(math.radians(float(np.mean(lats)))) / 1000.0 )**2
    rows_tot["n"] += len(diff); rows_tot["km2"] += len(diff)*res_km2
    rows_tot["gt10"] += int((np.abs(diff) > 10).sum()); rows_tot["gt20"] += int((np.abs(diff) > 20).sum())
    rows_tot["deep10"] += int((diff < -10).sum()); rows_tot["deep_km2"] += float((diff < -10).sum()*res_km2)
    # route points inside this block
    zi = np.where(k, z, np.nan)
    for rx, ry in rpts:
        if x0 <= rx <= x1 and y0 <= ry <= y1:
            j2 = int((rx-x0)/(x1-x0)*(W-1)); i2 = int((y1-ry)/(y1-y0)*(H-1))
            if np.isfinite(zi[i2, j2]):
                lo2 = math.degrees(rx/R); la2 = math.degrees(2*math.atan(math.exp(ry/R))-math.pi/2)
                gj = min(gw-1, max(0, int((lo2-gx0)/sx))); gi = min(gh-1, max(0, int((gy0-la2)/sy)))
                route_rows.append(float(g[gi, gj] - zi[i2, j2]))
rr = np.array(route_rows)
out = dict(scope="corridor blocks with cached GEBCO tiles; NONNA-sounded cells shallower than 200 m; GEBCO = NCEI global mosaic (GEBCO 2024/25)",
           n_cells=rows_tot["n"], km2=round(rows_tot["km2"]), pct_err_gt10=round(100*rows_tot["gt10"]/max(rows_tot["n"],1), 1),
           pct_err_gt20=round(100*rows_tot["gt20"]/max(rows_tot["n"],1), 1),
           pct_deep_biased_gt10=round(100*rows_tot["deep10"]/max(rows_tot["n"],1), 1),
           km2_deep_biased_gt10=round(rows_tot["deep_km2"]),
           route=dict(n_points=len(rr), pct_gt10=round(100*float((np.abs(rr)>10).mean()),1) if len(rr) else None,
                      pct_deep_gt10=round(100*float((rr<-10).mean()),1) if len(rr) else None,
                      worst_deep_m=float(rr.min()) if len(rr) else None))
json.dump(out, open("gebco_lag.json", "w"), indent=1)
print(json.dumps(out, indent=1)); print("GEBCO_LAG_DONE")
