#!/usr/bin/env python3
"""Test 3 — ICESat-2 ATL24 laser bathymetry vs SeabedNet, shallow corridor water.

ATL24 (NASA, 2025) gives refraction-corrected seafloor orthometric heights from the ATL03
photon cloud, independent of NONNA, SRTM15+ and GMRT. Datum: ATL24 is orthometric (geoid);
NONNA is chart datum. Per block we fit ONE constant offset on cells where NONNA has a
sounding (median of laser − sounding), then score at cells with NO published sounding where
the model produced a fill. Baselines: nearest sounding, gravity prior.
Cell rule: >= 8 bathy photons (confidence >= 0.6) in a 100 m cell, photon spread (MAD) <= 1.5 m,
laser depth <= −2 m after offset. Boxes: shallow water along the corridor.
usage: python3 icesat2_eval.py -> icesat2_cells.npy, icesat2_validation.json
"""
import glob, json, math, os, time, numpy as np
import sliderule
from scipy import ndimage as ndi
from v5_data import GravityPrior, lat_of_y, lon_of_x
R = 6378137.0
xl = lambda lo: R*math.radians(lo); yl = lambda la: R*math.log(math.tan(math.pi/4+math.radians(la)/2))
BOXES = [  # lon0, lat0, lon1, lat1 — shallow shipping water with NONNA coverage
    ("churchill_approach", -94.6, 58.5, -92.2, 59.4),
    ("nelson_hayes",       -93.2, 56.9, -91.4, 57.9),
    ("chesterfield",       -94.5, 63.1, -90.6, 64.2),
    ("rankin_whale_cove",  -93.2, 62.0, -91.0, 63.1),
    ("southampton_coral",  -86.5, 63.8, -82.5, 65.0),
    ("hudson_strait_south",-72.5, 60.9, -69.0, 62.2),
    ("big_island_kimmirut",-70.5, 62.2, -67.5, 63.1),
    ("frobisher",          -68.8, 62.6, -66.0, 63.8),
    ("ungava_south",       -70.0, 58.4, -65.5, 59.9),
    ("simpson_strait",     -97.9, 68.3, -95.2, 68.8),
    ("gjoa_rae",           -96.2, 68.6, -93.0, 69.2),
    ("franklin_tasmania",  -97.6, 70.8, -95.6, 71.7),
]
grav = GravityPrior(); os.makedirs("atl24_cache", exist_ok=True)
sliderule.init("slideruleearth.io", verbose=False)
blocks = []
for f in sorted(glob.glob("tiles_nat/*.npz")):
    bb = np.load(f, allow_pickle=True)["bbox3857"]
    blocks.append((f, min(bb[0],bb[2]), max(bb[0],bb[2]), min(bb[1],bb[3]), max(bb[1],bb[3])))
def harvest(name, lo0, la0, lo1, la1):
    fn = f"atl24_cache/{name}.npz"
    if os.path.exists(fn):
        d = np.load(fn); return d["lon"], d["lat"], d["h"]
    poly = [{"lon": lo0, "lat": la0}, {"lon": lo1, "lat": la0}, {"lon": lo1, "lat": la1}, {"lon": lo0, "lat": la1}, {"lon": lo0, "lat": la0}]
    lons = []; lats = []; hs = []
    for t0, t1 in [("2018-10-01", "2021-01-01"), ("2021-01-01", "2023-06-01"), ("2023-06-01", "2026-08-01")]:
        for attempt in range(2):
            try:
                g = sliderule.run("atl24x", {"poly": poly, "t0": t0+"T00:00:00Z", "t1": t1+"T00:00:00Z", "atl24": {"class_ph": ["bathymetry"], "confidence_threshold": 0.6}})
                if g is not None and len(g):
                    lons.append(g.geometry.x.values); lats.append(g.geometry.y.values); hs.append(g["ortho_h"].values)
                break
            except Exception as e:
                print(f"  {name} {t0}: {repr(e)[:90]}"); time.sleep(20)
    if not lons: return None, None, None
    lon = np.concatenate(lons); lat = np.concatenate(lats); h = np.concatenate(hs)
    np.savez_compressed(fn, lon=lon, lat=lat, h=h.astype("float32"))
    return lon, lat, h
rows = []; summary = {}
for name, lo0, la0, lo1, la1 in BOXES:
    lon, lat, h = harvest(name, lo0, la0, lo1, la1)
    if lon is None or len(lon) < 100: print(f"{name}: no photons"); continue
    print(f"{name}: {len(lon):,} bathy photons", flush=True)
    x = R*np.radians(lon); y = R*np.log(np.tan(np.pi/4 + np.radians(lat)/2))
    used = 0
    for f, x0, x1, y0, y1 in blocks:
        sel = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if sel.sum() < 100: continue
        d = np.load(f, allow_pickle=True); z = d["z"].astype("float32"); H, W = z.shape
        cf = f.replace("tiles_nat", "corridor_out_v2")
        if not os.path.exists(cf): cf = f.replace("tiles_nat", "national_v5_out")
        if not os.path.exists(cf): continue
        cd = np.load(cf, allow_pickle=True); c = cd["complete"].astype("float32")
        s68 = cd["sigma68"].astype("float32") if "sigma68" in cd.files else cd["sigma"].astype("float32")
        k = np.isfinite(z)
        j = np.clip(((x[sel]-x0)/(x1-x0)*(W-1)).astype(int), 0, W-1); i = np.clip(((y1-y[sel])/(y1-y0)*(H-1)).astype(int), 0, H-1)
        cellid = i.astype(np.int64)*W + j
        order = np.argsort(cellid); cid = cellid[order]; hv = h[sel][order]
        cut = np.searchsorted(cid, np.unique(cid))
        uid = np.unique(cid)
        med = np.array([np.median(hv[a:b]) for a, b in zip(cut, list(cut[1:])+[len(hv)])])
        cnt = np.diff(list(cut)+[len(hv)])
        mad = np.array([np.median(np.abs(hv[a:b]-m)) for (a, b), m in zip(zip(cut, list(cut[1:])+[len(hv)]), med)])
        good = (cnt >= int(os.environ.get("MINPH","8"))) & (mad <= float(os.environ.get("MAXMAD","1.5")))
        if good.sum() < 30: continue
        uid = uid[good]; med = med[good]
        ii = (uid // W).astype(int); jj = (uid % W).astype(int)
        # per-block datum offset from sounded cells
        kn = k[ii, jj]
        if kn.sum() < 20: continue
        off = float(np.median(med[kn] - z[ii[kn], jj[kn]]))
        if abs(off) > float(os.environ.get("MAXOFF", "1e9")): continue   # datum offset must be tidal-scale; else photons are not seafloor
        laser = med - off
        deep_enough = (laser <= -3) & (laser >= float(os.environ.get("MINLASER","-1e9")))
        dist_px, idx = ndi.distance_transform_edt(~k, return_indices=True)
        fillok = np.isfinite(c[ii, jj]) & ~kn & (dist_px[ii, jj] <= 60) & deep_enough
        # gravity + lonlat for these cells
        lons_c = np.degrees((x0 + (jj+0.5)/W*(x1-x0))/R)
        lats_c = np.degrees(2*np.arctan(np.exp((y1 - (ii+0.5)/H*(y1-y0))/R)) - np.pi/2)
        G = grav.sample(lons_c, lats_c)
        nn = z[idx[0][ii, jj], idx[1][ii, jj]]
        # scoring rows on no-sounding cells
        for w in np.nonzero(fillok)[0]:
            rows.append((c[ii[w], jj[w]]-laser[w], nn[w]-laser[w], G[w]-laser[w], s68[ii[w], jj[w]], dist_px[ii[w], jj[w]]*0.1, laser[w]))
        # laser vs NONNA agreement on sounded cells (sanity + NONNA check)
        agree = np.abs(med[kn]-off - z[ii[kn], jj[kn]])
        summary.setdefault(name, {}).setdefault("blocks", []).append(dict(block=os.path.basename(f)[:-4], offset_m=round(off, 2), n_cells=int(good.sum()), n_sounded=int(kn.sum()), n_scored=int(fillok.sum()), laser_vs_nonna_mae=float(agree.mean())))
        used += int(fillok.sum())
    print(f"  -> scored cells so far from {name}: {used}", flush=True)
A = np.array(rows, dtype="float32")
if len(A):
    np.save(os.environ.get("CELLS_OUT","icesat2_cells.npy"), A)
    em, en, eg, s6, dk, dep = A.T
    def st(sel): return dict(n=int(sel.sum()), model=float(np.abs(em[sel]).mean()), nearest=float(np.abs(en[sel]).mean()), gravity=float(np.abs(eg[sel]).mean()), bias=float(em[sel].mean()), within_68=float((np.abs(em[sel]) <= s6[sel]).mean()))
    out = dict(source="ICESat-2 ATL24 (NASA, release 006) via SlideRule atl24x; bathymetry photons, confidence >= 0.6; cells: >=8 photons, MAD <= 1.5 m; per-block datum offset fitted on NONNA-sounded cells",
               overall=st(np.ones(len(A), bool)),
               by_depth={f"{a}-{b}": st((-dep >= a) & (-dep < b)) for a, b in [(2,5),(5,10),(10,20),(20,40)] if ((-dep >= a) & (-dep < b)).sum() >= 30},
               by_distance={f"{a}-{b}": st((dk >= a) & (dk < b)) for a, b in [(0,.5),(.5,1),(1,2),(2,6)] if ((dk >= a) & (dk < b)).sum() >= 30},
               boxes=summary)
    json.dump(out, open(os.environ.get("JSON_OUT","icesat2_validation.json"), "w"), indent=1)
    print(json.dumps(out["overall"]), flush=True)
    for k2, v in out["by_depth"].items(): print("depth", k2, v)
else:
    json.dump(dict(status="no scorable cells", boxes=summary), open(os.environ.get("JSON_OUT","icesat2_validation.json"), "w"), indent=1)
print("ICESAT2_DONE")
