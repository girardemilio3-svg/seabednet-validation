#!/usr/bin/env python3
"""Crowdsourced-bathymetry test with platform QC. Per ping: map cell (known sounding Z, completed C, nearest-sounding NZ, distance).
QC: for each platform, offset = median(csb - Z) over pings on sounded cells, spread = MAD; keep platforms with >= 300 sounded pings,
spread <= 3 m and |offset| <= 10 m; subtract the offset. Then unsounded cells = median corrected depth per 100 m cell (>= 3 pings).
Metrics are robust (median |err|, share within 5 m and 10 m). usage: python3 csb_eval2.py <map_dir> "<parquet glob>" """
import sys, glob, json, math, os, numpy as np, pandas as pd
from scipy import ndimage as ndi
MAP = sys.argv[1]; files = sorted(glob.glob(sys.argv[2])); R = 6378137.0
def merc(lon, lat): return R*np.radians(lon), R*np.log(np.tan(np.pi/4 + np.radians(lat)/2))
bb = {os.path.basename(f): (lambda b: (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])))(np.load(f, allow_pickle=True)["bbox3857"]) for f in glob.glob(f"{MAP}/*.npz")}
names = list(bb); X0 = np.array([bb[n][0] for n in names]); Y0 = np.array([bb[n][1] for n in names]); X1 = np.array([bb[n][2] for n in names]); Y1 = np.array([bb[n][3] for n in names])
pts = []
for f in files:
    try: d = pd.read_parquet(f, columns=["LON", "LAT", "DEPTH", "PLATFORM_NAME"])
    except Exception: continue
    d = d[(d.DEPTH > 1) & (d.DEPTH < 1500) & d.LAT.between(41, 84) & d.LON.between(-142, -50)]
    x, y = merc(d.LON.values, d.LAT.values); pts.append(pd.DataFrame(dict(x=x, y=y, z=-d.DEPTH.values.astype("float32"), plat=d.PLATFORM_NAME.values.astype(str))))
P = pd.concat(pts, ignore_index=True); del pts; print("pings", len(P), flush=True)
bi = np.full(len(P), -1, np.int16)
for i in range(len(names)):
    m = (P.x.values >= X0[i]) & (P.x.values < X1[i]) & (P.y.values >= Y0[i]) & (P.y.values < Y1[i]); bi[m] = i
P["blk"] = bi; P = P[P.blk >= 0].reset_index(drop=True); print("in blocks", len(P), flush=True)
recs = []
for i, g in P.groupby("blk"):
    n = names[i]; a = np.load(f"{MAP}/{n}", allow_pickle=True); C = a["complete"].astype("float32"); K = a["known"].astype(bool); H, W = C.shape
    Z = np.load(f"tiles_nat/{n}", allow_pickle=True)["z"].astype("float32"); x0, y0, x1, y1 = bb[n]
    jj = ((g.x.values - x0)/(x1 - x0)*W).astype(int); ii = ((y1 - g.y.values)/(y1 - y0)*H).astype(int); ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
    dist, idx = ndi.distance_transform_edt(~K, return_indices=True); NZ = Z[idx[0], idx[1]]
    I, J = ii[ok], jj[ok]; lat = np.degrees(2*np.arctan(np.exp(((y1 - (I+0.5)/H*(y1-y0)))/R)) - np.pi/2)
    recs.append(pd.DataFrame(dict(blk=np.int16(i), i=I.astype(np.int16), j=J.astype(np.int16), csb=g.z.values[ok], plat=g.plat.values[ok], Z=Z[I, J], C=C[I, J], known=K[I, J], NZ=NZ[I, J], dkm=(dist[I, J]*0.1).astype(np.float32), lat=lat.astype(np.float32))))
D = pd.concat(recs, ignore_index=True); del recs, P; print("pings on map cells", len(D), flush=True)
# platform QC on sounded cells
s = D[D.known & np.isfinite(D.Z)]; e = s.csb - s.Z
q = s.assign(e=e).groupby("plat").e.agg(n="size", offset="median", mad=lambda v: float(np.median(np.abs(v - np.median(v)))))
good = q[(q.n >= 300) & (q.mad <= 3.0) & (q.offset.abs() <= 10)]; print("platforms", len(q), "kept", len(good), "| pings kept share", round(float(D.plat.isin(good.index).mean()), 3), flush=True)
D = D[D.plat.isin(good.index)].copy(); D["csbc"] = D.csb - D.plat.map(good.offset).astype("float32")
U = D[~D.known & np.isfinite(D.C)]
cell = U.groupby(["blk", "i", "j"]).agg(csb=("csbc", "median"), n=("csbc", "size"), C=("C", "first"), NZ=("NZ", "first"), dkm=("dkm", "first"), lat=("lat", "first")).reset_index(); cell = cell[cell.n >= 3]
cell["lon"] = [math.degrees((bb[names[b]][0] + bb[names[b]][2])/2/R) for b in cell.blk]
cell.to_parquet(f"csb_eval2_{os.path.basename(MAP)}.parquet", index=False)
def st(c):
    if len(c) < 50: return None
    em = np.abs(c.C - c.csb); en = np.abs(c.NZ - c.csb)
    return dict(n=int(len(c)), median_model=round(float(np.median(em)), 2), median_nearest=round(float(np.median(en)), 2), within5_model=round(float(np.mean(em <= 5)), 3), within5_nearest=round(float(np.mean(en <= 5)), 3), within10_model=round(float(np.mean(em <= 10)), 3), within10_nearest=round(float(np.mean(en <= 10)), 3), bias_model=round(float(np.median(c.C - c.csb)), 2))
out = dict(map=MAP, platforms_total=int(len(q)), platforms_kept=int(len(good)), unsounded_cells=int(len(cell)), all=st(cell),
           by_depth={f"{lo}-{hi}": st(cell[(-cell.csb >= lo) & (-cell.csb < hi)]) for lo, hi in ((0, 20), (20, 50), (50, 200), (200, 5000))},
           by_distance_km={f"{lo}-{hi}": st(cell[(cell.dkm >= lo) & (cell.dkm < hi)]) for lo, hi in ((0, 0.5), (0.5, 1), (1, 3), (3, 6))},
           by_region=dict(BC=st(cell[cell.lon < -120]), Atlantic=st(cell[(cell.lon > -70) & (cell.lat < 56)]), Arctic_Hudson=st(cell[(cell.lat >= 56) & (cell.lon > -120)])),
           sounded_noise_floor=dict(median_abs=round(float(np.median(np.abs((s.csb - s.plat.map(good.offset)).dropna() - s.Z[s.plat.isin(good.index)]))), 2)))
json.dump(out, open(f"csb_eval2_{os.path.basename(MAP)}.json", "w"), indent=1); print(json.dumps(out, indent=1))
