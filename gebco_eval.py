#!/usr/bin/env python3
"""GEBCO (NCEI DEM_global_mosaic, GEBCO-based) sampled at the Test 1 and Test 2 target cells
with correct georeferencing (the ImageServer export forces square pixels, so the returned
raster is taller than the request; use its tiepoint + pixel scale). -> gebco_eval.json
Caveat carried everywhere: GEBCO ingests NONNA-100 (via IBCAO) and NCEI multibeam; it is a
reference, not an independent baseline."""
import glob, json, math, os, time, numpy as np, requests, tifffile
from scipy import ndimage as ndi
from v5_data import lat_of_y, lon_of_x
NCEI = "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/exportImage"
os.makedirs("gebco_cache", exist_ok=True)
def fetch(name, lo0, lo1, la0, la1):
    fn = f"gebco_cache/{name}.tif"
    if not os.path.exists(fn):
        for k in range(3):
            r = requests.get(NCEI, params=dict(bbox=f"{lo0},{la0},{lo1},{la1}", bboxSR=4326, imageSR=4326, size="2000,2000", format="tiff", pixelType="F32",
                                               interpolation="RSP_BilinearInterpolation", f="image"), timeout=300)
            if r.status_code == 200 and r.content[:2] in (b"II", b"MM"): open(fn, "wb").write(r.content); break
            time.sleep(5)
        else: return None
    t = tifffile.TiffFile(fn); p = t.pages[0]
    tp = p.tags["ModelTiepointTag"].value; sc = p.tags["ModelPixelScaleTag"].value
    return p.asarray().astype("float32"), float(tp[3]), float(tp[4]), float(sc[0]), float(sc[1])
def sample(gb, lons, lats):
    g, x0, y0, sx, sy = gb; gh, gw = g.shape
    gx = np.clip(((lons - x0)/sx).astype(int), 0, gw-1); gy = np.clip(((y0 - lats)/sy).astype(int), 0, gh-1)
    return g[gy, gx]
out = {}
# ---- Test 1
rows = []
for f in sorted(glob.glob("corridor_out/*.npz")):
    name = os.path.basename(f)[:-4]; mfn = f"gmrt_cache/{name}_mask.tif"
    if not os.path.exists(mfn): continue
    d = np.load(f, allow_pickle=True); c = d["complete"].astype("float32"); k = d["known"].astype(bool); bb = d["bbox3857"]; H, W = c.shape
    lo0, lo1 = lon_of_x(min(bb[0],bb[2])), lon_of_x(max(bb[0],bb[2])); la0, la1 = lat_of_y(min(bb[1],bb[3])), lat_of_y(max(bb[1],bb[3]))
    m = tifffile.imread(mfn).astype("float32"); gh, gw = m.shape
    lons = np.linspace(lo0, lo1, W); ymerc = np.linspace(max(bb[1],bb[3]), min(bb[1],bb[3]), H); lats = np.array([lat_of_y(v) for v in ymerc])
    gx = np.clip(((lons-lo0)/(lo1-lo0)*(gw-1)).astype(int), 0, gw-1); gy = np.clip(((la1-lats)/(la1-la0)*(gh-1)).astype(int), 0, gh-1)
    T = m[np.ix_(gy, gx)]
    dist_px = ndi.distance_transform_edt(~k)
    val = np.isfinite(T) & (T < -2) & ~k & np.isfinite(c) & (dist_px <= 60)
    if val.sum() < 20: continue
    gb = fetch(name, lo0, lo1, la0, la1)
    if gb is None: continue
    vy, vx = np.nonzero(val); G = sample(gb, lons[vx], lats[vy])
    rows.append(np.stack([c[val]-T[val], G-T[val], T[val]], 1))
    print(f"T1 {name}: n={val.sum():6d} model {np.abs(rows[-1][:,0]).mean():5.1f} gebco {np.abs(rows[-1][:,1]).mean():5.1f}", flush=True)
A = np.concatenate(rows); em, eg, dep = A.T
def st(sel): return dict(n=int(sel.sum()), model=float(np.abs(em[sel]).mean()), gebco=float(np.abs(eg[sel]).mean()), gebco_bias=float(eg[sel].mean()))
out["test1"] = dict(shelf_50_400=st((-dep >= 50) & (-dep < 400)), all=st(np.ones(len(A), bool)),
                    by_depth={f"{a}-{b}": st((-dep >= a) & (-dep < b)) for a, b in [(0,50),(50,400),(400,1000),(1000,99999)]})
print("TEST1", json.dumps(out["test1"]["shelf_50_400"]), flush=True)
# ---- Test 2 (full target set, same rule as temporal_eval: post-2016 within 60 px of a pre-2016 sounding)
rows = []
for f in [l.strip() for l in open("corridor_blocks.txt") if l.strip()]:
    name = os.path.basename(f)[:-4]; ip = f"index_out/{name}.npz"
    if not os.path.exists(ip): continue
    d = np.load(f, allow_pickle=True); z = d["z"].astype("float32"); bb = d["bbox3857"]; H, W = z.shape
    yr = np.load(ip)["year"]; allk = np.isfinite(z); pre = allk & (yr > 0); post = allk & (yr == 0)
    if pre.sum() < 500 or post.sum() < 100: continue
    lo0, lo1 = lon_of_x(min(bb[0],bb[2])), lon_of_x(max(bb[0],bb[2])); la0, la1 = lat_of_y(min(bb[1],bb[3])), lat_of_y(max(bb[1],bb[3]))
    lons = np.linspace(lo0, lo1, W); ymerc = np.linspace(max(bb[1],bb[3]), min(bb[1],bb[3]), H); lats = np.array([lat_of_y(v) for v in ymerc])
    dist_px = ndi.distance_transform_edt(~pre); val = post & (dist_px <= 60)
    if val.sum() < 50: continue
    gb = fetch(name, lo0, lo1, la0, la1)
    if gb is None: continue
    vy, vx = np.nonzero(val); G = sample(gb, lons[vx], lats[vy])
    res_m = abs(bb[2]-bb[0])/W*math.cos(math.radians(float(lats.mean())))
    rows.append(np.stack([G-z[val], dist_px[val]*res_m/1000, z[val]], 1))
    print(f"T2 {name}: n={val.sum():7d} gebco {np.abs(rows[-1][:,0]).mean():5.1f} bias {rows[-1][:,0].mean():+5.1f}", flush=True)
B = np.concatenate(rows); eg, dk, dep = B.T
def st2(sel): return dict(n=int(sel.sum()), gebco=float(np.abs(eg[sel]).mean()), gebco_bias=float(eg[sel].mean()))
out["test2"] = dict(all=st2(np.ones(len(B), bool)), by_distance={f"{a}-{b}": st2((dk >= a) & (dk < b)) for a, b in [(0,.5),(.5,1),(1,2),(2,4),(4,8)]},
                    by_depth={f"{a}-{b}": st2((-dep >= a) & (-dep < b)) for a, b in [(0,20),(20,50),(50,100),(100,200),(200,400),(400,99999)]})
out["caveat"] = "GEBCO 2024/2025 grid (NCEI DEM_global_mosaic) ingests CHS NONNA-100 through IBCAO v5 and NCEI multibeam through GMRT; on Test 2 it has seen the post-2016 soundings and on Test 1 the cruise data. Reference, not baseline."
json.dump(out, open("gebco_eval.json", "w"), indent=1)
print("TEST2", json.dumps(out["test2"]["all"])); print("GEBCO_DONE")
