#!/usr/bin/env python3
"""Stronger baselines for Test 1 (independent multibeam) and Test 2 (temporal holdout):
  - trend+residual interpolation: gravity prior as trend, residual (sounding - gravity) at
    the input soundings interpolated to the target cell by Delaunay natural-neighbour-like
    linear interpolation (scipy griddata 'linear'); plus IDW-kriging-like inverse-distance^2
    on the 12 nearest residuals (KD-tree). This is what a hydrographer would do to fill a gap.
  - GEBCO (NCEI DEM_global_mosaic, GEBCO-based) sampled at the target cells. NOT independent:
    GEBCO ingests NONNA-100 (via IBCAO) and NCEI multibeam; reported with that caveat.
  - gravity-leakage check: SRTM15+ error on Test 1 cells vs on NONNA-sounded shelf cells of
    the same blocks.
Recomputes the target sets with the same rules as indep_validate.py / temporal_eval.py.
-> baselines.json
usage: python3 baselines.py
"""
import glob, json, math, os, time, numpy as np, requests, tifffile
from scipy import ndimage as ndi
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from v5_data import GravityPrior, lat_of_y, lon_of_x
grav = GravityPrior(); os.makedirs("gebco_cache", exist_ok=True)
NCEI = "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/exportImage"
rng = np.random.default_rng(0)
MAXPTS = 250_000     # subsample input soundings for the Delaunay (speed); residual field is smooth

def gebco(name, lo0, lo1, la0, la1, W, H):
    fn = f"gebco_cache/{name}.tif"
    if not os.path.exists(fn):
        for k in range(3):
            r = requests.get(NCEI, params=dict(bbox=f"{lo0},{la0},{lo1},{la1}", bboxSR=4326, imageSR=4326, size=f"{min(W,2000)},{min(H,2000)}",
                                               format="tiff", pixelType="F32", interpolation="RSP_BilinearInterpolation", f="image"), timeout=300)
            if r.status_code == 200 and r.content[:2] in (b"II", b"MM"): open(fn, "wb").write(r.content); break
            time.sleep(5)
        else: return None
    g = tifffile.imread(fn).astype("float32"); gh, gw = g.shape
    gy = np.clip((np.arange(H)/(H-1)*(gh-1)).astype(int), 0, gh-1); gx = np.clip((np.arange(W)/(W-1)*(gw-1)).astype(int), 0, gw-1)
    return g[np.ix_(gy, gx)]

def interp_baselines(zin, known, G, val):
    """returns (linear natural-neighbour residual interp, IDW-12 residual interp) at val cells"""
    ky, kx = np.nonzero(known); res = (zin - G)[known]
    if len(ky) > MAXPTS:
        sel = rng.choice(len(ky), MAXPTS, replace=False); ky, kx, res = ky[sel], kx[sel], res[sel]
    vy, vx = np.nonzero(val)
    lin = griddata(np.c_[ky, kx], res, np.c_[vy, vx], method="linear")
    tree = cKDTree(np.c_[ky, kx]); d, i = tree.query(np.c_[vy, vx], k=12)
    w = 1.0/np.maximum(d, 0.5)**2; idw = (res[i]*w).sum(1)/w.sum(1)
    Gv = G[val]
    return Gv + np.where(np.isfinite(lin), lin, idw), Gv + idw

out = dict(test1=dict(), test2=dict(), leakage=dict())
# ------------------------------------------------------------------ Test 1
rows = []; leak_test = []; leak_known = []
for f in sorted(glob.glob("corridor_out/*.npz")):
    name = os.path.basename(f)[:-4]; mfn = f"gmrt_cache/{name}_mask.tif"
    if not os.path.exists(mfn): continue
    d = np.load(f, allow_pickle=True); c = d["complete"].astype("float32"); k = d["known"].astype(bool); bb = d["bbox3857"]; H, W = c.shape
    z = np.load(f"tiles_nat/{name}.npz", allow_pickle=True)["z"].astype("float32")
    lo0, lo1 = lon_of_x(min(bb[0],bb[2])), lon_of_x(max(bb[0],bb[2])); la0, la1 = lat_of_y(min(bb[1],bb[3])), lat_of_y(max(bb[1],bb[3]))
    m = tifffile.imread(mfn).astype("float32"); gh, gw = m.shape
    lons = np.linspace(lo0, lo1, W); lats = np.linspace(la1, la0, H)
    gx = np.clip(((lons-lo0)/(lo1-lo0)*(gw-1)).astype(int), 0, gw-1); gy = np.clip(((la1-lats)/(la1-la0)*(gh-1)).astype(int), 0, gh-1)
    T = m[np.ix_(gy, gx)]
    dist_px, idx = ndi.distance_transform_edt(~k, return_indices=True)
    fill = np.isfinite(c) & ~k & (dist_px <= 60)
    val = np.isfinite(T) & (T < -2) & ~k & fill
    if val.sum() < 20: continue
    G = grav.sample(np.tile(lons, (H,1)), np.tile(lats[:,None], (1,W))).astype("float32")
    zin = np.where(k, z, np.nan)
    nat, idw = interp_baselines(zin, k, G, val)
    gb = gebco(name, lo0, lo1, la0, la1, W, H)
    em = c[val]-T[val]; en = c[idx[0], idx[1]][val]-T[val]; eg = G[val]-T[val]; enat = nat-T[val]; eidw = idw-T[val]
    egb = (gb[val]-T[val]) if gb is not None else np.full(val.sum(), np.nan)
    rows.append(np.stack([em, en, eg, enat, eidw, egb, T[val]], 1))
    # leakage: gravity error at NONNA-known shelf cells of the same block (not in GMRT test)
    ks = k & (z < -50) & (z > -400)
    if ks.sum() > 100:
        sel = rng.choice(np.flatnonzero(ks), min(20000, ks.sum()), replace=False)
        leak_known.append(np.abs(G.ravel()[sel] - z.ravel()[sel]))
    print(f"T1 {name}: n={val.sum():6d} model {np.abs(em).mean():5.1f} nn {np.abs(en).mean():5.1f} grav {np.abs(eg).mean():5.1f} nat {np.abs(enat).mean():5.1f} idw {np.abs(eidw).mean():5.1f} gebco {np.nanmean(np.abs(egb)):5.1f}", flush=True)
A = np.concatenate(rows); em, en, eg, enat, eidw, egb, dep = A.T
sh = (-dep >= 50) & (-dep < 400)
def st(sel):
    return dict(n=int(sel.sum()), model=float(np.abs(em[sel]).mean()), nearest=float(np.abs(en[sel]).mean()), gravity=float(np.abs(eg[sel]).mean()),
                trend_natural=float(np.abs(enat[sel]).mean()), trend_idw=float(np.abs(eidw[sel]).mean()), gebco=float(np.nanmean(np.abs(egb[sel]))))
out["test1"]["shelf_50_400"] = st(sh); out["test1"]["all"] = st(np.ones(len(A), bool))
out["test1"]["by_depth"] = {f"{a}-{b}": st((-dep >= a) & (-dep < b)) for a, b in [(0,50),(50,400),(400,1000),(1000,99999)]}
out["leakage"] = dict(srtm15_vintage="SRTM15+ V2.7 (Tozer et al.; NetCDF dated 2024-11 in planetary/SRTM15_V2.7.nc)",
                      gravity_mae_on_test1_shelf_cells=float(np.abs(eg[sh]).mean()),
                      gravity_mae_on_nonna_sounded_shelf_cells_same_blocks=float(np.concatenate(leak_known).mean()),
                      n_sounded_cells=int(sum(len(x) for x in leak_known)))
np.save("baseline_cells_test1.npy", A.astype("float32"))
print("TEST1", json.dumps(out["test1"]["shelf_50_400"]), "\nLEAK", json.dumps(out["leakage"]), flush=True)
# ------------------------------------------------------------------ Test 2 (temporal): same target rule as temporal_eval.py (model fill assumed wherever dist<=60)
rows = []
files = [l.strip() for l in open("corridor_blocks.txt") if l.strip()]
for f in files:
    name = os.path.basename(f)[:-4]; ip = f"index_out/{name}.npz"
    if not os.path.exists(ip): continue
    d = np.load(f, allow_pickle=True); z = d["z"].astype("float32"); bb = d["bbox3857"]; H, W = z.shape
    yr = np.load(ip)["year"]; allk = np.isfinite(z); pre = allk & (yr > 0); post = allk & (yr == 0)
    if pre.sum() < 500 or post.sum() < 100: continue
    lo0, lo1 = lon_of_x(min(bb[0],bb[2])), lon_of_x(max(bb[0],bb[2])); la0, la1 = lat_of_y(min(bb[1],bb[3])), lat_of_y(max(bb[1],bb[3]))
    lons = np.linspace(lo0, lo1, W); lats = np.linspace(la1, la0, H)
    G = grav.sample(np.tile(lons, (H,1)), np.tile(lats[:,None], (1,W))).astype("float32")
    dist_px, idx = ndi.distance_transform_edt(~pre, return_indices=True)
    val = post & (dist_px <= 60)
    if val.sum() < 50: continue
    if val.sum() > 400_000:      # cap per block for the Delaunay query; unbiased random subset
        sub = np.zeros_like(val); sel = rng.choice(np.flatnonzero(val), 400_000, replace=False); sub.ravel()[sel] = True; val = sub
    zin = np.where(pre, z, np.nan)
    nat, idw = interp_baselines(zin, pre, G, val)
    gb = gebco(name, lo0, lo1, la0, la1, W, W)
    en = zin[idx[0], idx[1]][val]-z[val]; eg = G[val]-z[val]; enat = nat-z[val]; eidw = idw-z[val]
    egb = (gb[val]-z[val]) if gb is not None else np.full(val.sum(), np.nan)
    res_m = abs(bb[2]-bb[0])/W*math.cos(math.radians(float(lats.mean())))
    rows.append(np.stack([en, eg, enat, eidw, egb, dist_px[val]*res_m/1000, z[val]], 1))
    print(f"T2 {name}: n={val.sum():6d} nn {np.abs(en).mean():5.1f} grav {np.abs(eg).mean():5.1f} nat {np.abs(enat).mean():5.1f} idw {np.abs(eidw).mean():5.1f} gebco {np.nanmean(np.abs(egb)):5.1f}", flush=True)
B = np.concatenate(rows); en, eg, enat, eidw, egb, dk, dep = B.T
def st2(sel):
    return dict(n=int(sel.sum()), nearest=float(np.abs(en[sel]).mean()), gravity=float(np.abs(eg[sel]).mean()),
                trend_natural=float(np.abs(enat[sel]).mean()), trend_idw=float(np.abs(eidw[sel]).mean()), gebco=float(np.nanmean(np.abs(egb[sel]))))
out["test2"]["all"] = st2(np.ones(len(B), bool))
out["test2"]["by_distance"] = {f"{a}-{b}": st2((dk >= a) & (dk < b)) for a, b in [(0,.5),(.5,1),(1,2),(2,4),(4,8)]}
out["test2"]["by_depth"] = {f"{a}-{b}": st2((-dep >= a) & (-dep < b)) for a, b in [(0,20),(20,50),(50,100),(100,200),(200,400),(400,99999)]}
out["test2"]["note"] = "targets: post-2016 soundings within 60 px of a pre-2016 sounding, per-block random cap 400k; model MAE from temporal_validation_v5_small_temporal.json on the uncapped set"
np.save("baseline_cells_test2.npy", B.astype("float32"))
json.dump(out, open("baselines.json", "w"), indent=1)
print("TEST2", json.dumps(out["test2"]["all"])); print("BASELINES_DONE")
