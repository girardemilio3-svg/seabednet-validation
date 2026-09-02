#!/usr/bin/env python3
"""Independent validation of v5 completion against GMRT multibeam (research-cruise
swaths: Amundsen/Healy/Knorr/... — not CHS holdings).

Criterion for a validation cell: GMRT topo-mask has a measured depth (water, <0),
NONNA has NO sounding there (model never saw it), and the model produced an inferred
value within the 6 km fill discipline.  Baselines: nearest published sounding, gravity
prior.  Stratified by distance to the nearest NONNA sounding.  σ calibration curve.

usage: python3 indep_validate.py corridor_out   (or national_v5_out)
"""
import glob, json, math, os, sys, time, numpy as np, requests, tifffile
from scipy import ndimage as ndi
from v5_data import GravityPrior, lat_of_y, lon_of_x

SRC = sys.argv[1] if len(sys.argv) > 1 else "corridor_out"
CACHE = "gmrt_cache"; os.makedirs(CACHE, exist_ok=True)
GMRT = "https://www.gmrt.org/services/GridServer"
BINS = [0, 0.5, 1, 2, 4, 8, 1e9]
grav = GravityPrior()

def fetch_mask(name, lo0, lo1, la0, la1):
    fn = f"{CACHE}/{name}_mask.tif"
    if os.path.exists(fn) and os.path.getsize(fn) > 1000: return fn
    for attempt in range(3):
        try:
            r = requests.get(GMRT, params=dict(minlatitude=la0, maxlatitude=la1, minlongitude=lo0,
                             maxlongitude=lo1, format="geotiff", resolution="high", layer="topo-mask"),
                             timeout=300)
            if r.status_code == 200 and r.content[:2] in (b"II", b"MM"):
                open(fn, "wb").write(r.content); time.sleep(1.5); return fn
            print("  gmrt", r.status_code, r.content[:120]); time.sleep(10)
        except Exception as e:
            print("  gmrt err", e); time.sleep(15)
    return None

rows = []   # per-cell records: err_model, err_nn, err_grav, sigma, dist_km, depth, block
summary = {}
files = sorted(glob.glob(f"{SRC}/*.npz"))
for n, f in enumerate(files):
    name = os.path.basename(f)[:-4]
    d = np.load(f, allow_pickle=True)
    c = d["complete"].astype("float32"); k = d["known"].astype(bool)
    s = d["sigma"].astype("float32"); bb = d["bbox3857"]
    H, W = c.shape
    lo0, lo1 = lon_of_x(min(bb[0],bb[2])), lon_of_x(max(bb[0],bb[2]))
    la0, la1 = lat_of_y(min(bb[1],bb[3])), lat_of_y(max(bb[1],bb[3]))
    fn = fetch_mask(name, lo0, lo1, la0, la1)
    if fn is None: print(name, "no gmrt"); continue
    try: m = tifffile.imread(fn).astype("float32")
    except Exception as e: print(name, "bad tif", e); os.remove(fn); continue
    gh, gw = m.shape
    lons = np.linspace(lo0, lo1, W); lats = np.linspace(la1, la0, H)
    gx = np.clip(((lons-lo0)/(lo1-lo0)*(gw-1)).astype(int), 0, gw-1)
    gy = np.clip(((la1-lats)/(la1-la0)*(gh-1)).astype(int), 0, gh-1)
    T = m[np.ix_(gy, gx)]
    dist_px, idx = ndi.distance_transform_edt(~k, return_indices=True)
    res_m = abs(bb[2]-bb[0])/W * math.cos(math.radians((la0+la1)/2))   # mercator -> ground
    fill = np.isfinite(c) & ~k & (dist_px <= 60)
    val = np.isfinite(T) & (T < -2) & ~k & fill
    if val.sum() < 20: print(f"{n:3d} {name}: {val.sum()} val cells"); continue
    nn = c[idx[0], idx[1]]
    G = grav.sample(np.tile(lons, (H,1)), np.tile(lats[:,None], (1,W))).astype("float32")
    # GMRT cells are ~100 m; block cells ~150 m: compare model to GMRT directly
    em = c[val]-T[val]; en = nn[val]-T[val]; eg = G[val]-T[val]
    rows.append(np.stack([em, en, eg, s[val], dist_px[val]*res_m/1000, T[val]], 1).astype("float32"))
    summary[name] = dict(n=int(val.sum()), mae_model=float(np.abs(em).mean()), mae_nn=float(np.abs(en).mean()),
                         mae_grav=float(np.abs(eg).mean()), bias_model=float(em.mean()), med_depth=float(np.median(T[val])))
    print(f"{n:3d} {name}: n={val.sum():6d} model {np.abs(em).mean():5.1f} nn {np.abs(en).mean():5.1f} grav {np.abs(eg).mean():5.1f} bias {em.mean():+5.1f} depth~{np.median(T[val]):.0f}", flush=True)

if not rows: raise SystemExit("no validation cells")
A = np.concatenate(rows); np.save(f"indep_cells_{SRC}.npy", A)
em, en, eg, sg, dk, dep = A.T
out = dict(source=SRC, n_cells=int(len(A)), n_blocks=len(summary),
           overall=dict(mae_model=float(np.abs(em).mean()), mae_nn=float(np.abs(en).mean()),
                        mae_grav=float(np.abs(eg).mean()), bias_model=float(em.mean()),
                        rmse_model=float(np.sqrt((em**2).mean())), median_abs_model=float(np.median(np.abs(em))),
                        sigma_corr=float(np.corrcoef(sg, np.abs(em))[0,1])),
           strata=[], calib=[], blocks=summary)
for a, b in zip(BINS[:-1], BINS[1:]):
    sel = (dk >= a) & (dk < b)
    if sel.sum() < 50: continue
    out["strata"].append(dict(km=[a, min(b, 99)], n=int(sel.sum()), mae_model=float(np.abs(em[sel]).mean()),
                              mae_nn=float(np.abs(en[sel]).mean()), mae_grav=float(np.abs(eg[sel]).mean()),
                              bias=float(em[sel].mean()), mean_sigma=float(sg[sel].mean())))
# σ calibration: deciles of σ -> observed |err| quantiles
q = np.quantile(sg, np.linspace(0, 1, 11))
for a, b in zip(q[:-1], q[1:]):
    sel = (sg >= a) & (sg <= b)
    if sel.sum() < 30: continue
    out["calib"].append(dict(sigma_lo=float(a), sigma_hi=float(b), n=int(sel.sum()),
                             abs_err_p50=float(np.median(np.abs(em[sel]))), abs_err_p90=float(np.quantile(np.abs(em[sel]), .9)),
                             frac_within_1sigma=float((np.abs(em[sel]) <= sg[sel]).mean())))
json.dump(out, open(f"indep_validation_{SRC}.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n_cells", "n_blocks", "overall")}, indent=1))
for r in out["strata"]: print(r)
print("INDEP_DONE")
