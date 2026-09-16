#!/usr/bin/env python3
"""Independent test of the national map against crowdsourced bathymetry (IHO DCDB): CSB points binned to the 100 m block grid
(median depth per cell, >= 3 pings), compared with the completed depth where the map had NO sounding (known == False), and with
the nearest-sounding baseline. CSB depths are raw (no tide/draft/sound-speed correction), so expect a few metres of bias; the
comparison is by depth band and region. usage: python3 csb_eval.py <map_dir> <parquet glob> -> csb_eval_<map_dir>.json"""
import sys, glob, json, math, os, numpy as np, pandas as pd
from scipy import ndimage as ndi
MAP = sys.argv[1]; files = sorted(glob.glob(sys.argv[2])); R = 6378137.0
def merc(lon, lat): return R*np.radians(lon), R*np.log(np.tan(np.pi/4 + np.radians(lat)/2))
bb = {}
for f in glob.glob(f"{MAP}/*.npz"):
    b = np.load(f, allow_pickle=True)["bbox3857"]; bb[os.path.basename(f)] = (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3]))
names = list(bb); X0 = np.array([bb[n][0] for n in names]); Y0 = np.array([bb[n][1] for n in names]); X1 = np.array([bb[n][2] for n in names]); Y1 = np.array([bb[n][3] for n in names])
pts = []
for f in files:
    try: d = pd.read_parquet(f, columns=["LON", "LAT", "DEPTH", "PLATFORM_NAME"])
    except Exception: continue
    d = d[(d.DEPTH > 1) & (d.DEPTH < 1500) & d.LAT.between(41, 84) & d.LON.between(-142, -50)]
    x, y = merc(d.LON.values, d.LAT.values); pts.append(pd.DataFrame(dict(x=x, y=y, z=-d.DEPTH.values.astype("float32"), plat=d.PLATFORM_NAME.values)))
P = pd.concat(pts); print("csb points", len(P), flush=True)
# assign block by bbox (blocks are disjoint tiles)
bi = np.full(len(P), -1); 
for i in range(len(names)):
    m = (P.x.values >= X0[i]) & (P.x.values < X1[i]) & (P.y.values >= Y0[i]) & (P.y.values < Y1[i]); bi[m] = i
P["blk"] = bi; P = P[P.blk >= 0]; print("in map blocks", len(P), "blocks", P.blk.nunique(), flush=True)
rows = []
for i, g in P.groupby("blk"):
    n = names[i]; a = np.load(f"{MAP}/{n}", allow_pickle=True); C = a["complete"].astype("float32"); K = a["known"].astype(bool); H, W = C.shape
    x0, y0, x1, y1 = bb[n]; jj = ((g.x.values - x0)/(x1 - x0)*W).astype(int); ii = ((y1 - g.y.values)/(y1 - y0)*H).astype(int)
    ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W); ii, jj, z = ii[ok], jj[ok], g.z.values[ok]
    cell = pd.DataFrame(dict(i=ii, j=jj, z=z)).groupby(["i", "j"]).agg(z=("z", "median"), n=("z", "size")).reset_index(); cell = cell[cell.n >= 3]
    if len(cell) == 0: continue
    if "z" in np.load(f"tiles_nat/{n}", allow_pickle=True).files: Z = np.load(f"tiles_nat/{n}", allow_pickle=True)["z"].astype("float32")
    else: continue
    dist, idx = ndi.distance_transform_edt(~K, return_indices=True); NZ = Z[idx[0], idx[1]]
    I, J = cell.i.values, cell.j.values; c = C[I, J]; k = K[I, J]; nz = NZ[I, J]; dk = dist[I, J]*0.1
    lat = np.degrees(2*np.arctan(np.exp(((y1 - (I+0.5)/H*(y1-y0)))/R)) - np.pi/2)
    for q in range(len(cell)):
        if not np.isfinite(c[q]): continue
        rows.append((n, float(cell.z.values[q]), float(c[q]), bool(k[q]), float(nz[q]), float(dk[q]), float(lat[q]), int(cell.n.values[q])))
D = pd.DataFrame(rows, columns=["block", "csb", "model", "known", "nearest", "dist_km", "lat", "npings"]); D.to_parquet(f"csb_eval_{os.path.basename(MAP)}.parquet", index=False)
def stats(s):
    if len(s) < 50: return None
    em = s.model - s.csb; en = s.nearest - s.csb
    return dict(n=int(len(s)), mae_model=round(float(np.mean(np.abs(em))), 2), mae_nearest=round(float(np.mean(np.abs(en))), 2), bias_model=round(float(np.mean(em)), 2), bias_nearest=round(float(np.mean(en)), 2), median_abs_model=round(float(np.median(np.abs(em))), 2), median_abs_nearest=round(float(np.median(np.abs(en))), 2))
U = D[~D.known]; out = dict(map=MAP, n_cells=int(len(D)), n_unsounded=int(len(U)), all_unsounded=stats(U), sounded_cells_model_vs_csb=stats(D[D.known]))
out["by_depth"] = {f"{lo}-{hi}": stats(U[(-U.csb >= lo) & (-U.csb < hi)]) for lo, hi in ((0, 20), (20, 50), (50, 200), (200, 5000))}
out["by_distance_km"] = {f"{lo}-{hi}": stats(U[(U.dist_km >= lo) & (U.dist_km < hi)]) for lo, hi in ((0, 1), (1, 3), (3, 6))}
out["by_region"] = {"BC (lon<-120)": None, "Atlantic (lon>-70, lat<56)": None, "Arctic/Hudson (lat>=56, lon>-120)": None}
blk_lon = {n: math.degrees((bb[n][0] + bb[n][2])/2/R) for n in names}; U = U.assign(lon=U.block.map(blk_lon))
out["by_region"] = {"BC": stats(U[U.lon < -120]), "Atlantic": stats(U[(U.lon > -70) & (U.lat < 56)]), "Arctic_Hudson": stats(U[(U.lat >= 56) & (U.lon > -120)])}
json.dump(out, open(f"csb_eval_{os.path.basename(MAP)}.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "by_distance_km"}, indent=1))
