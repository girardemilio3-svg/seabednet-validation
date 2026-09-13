#!/usr/bin/env python3
"""SeabedNet: complete a depth grid from your own soundings.

Runs the SeabedNet ensemble (masked-completion U-Nets trained on the CHS NONNA archive with a gravity anchor) on a grid of
soundings you provide, and writes the completed depth, its 1-sigma uncertainty, and the provenance mask. Works on CPU
(slow, ~minutes per 200 km block) or GPU.

Input: a NumPy .npz with  z  (float32 [H, W], depth in metres NEGATIVE below sea level, NaN where you have no sounding)
       and  bbox3857  ([x0, y0, x1, y1] in EPSG:3857 metres) — the same layout as NONNA-100 blocks used in this repo,
       OR a GeoTIFF of soundings in EPSG:3857 (NaN/nodata = no sounding) via --tif.
Output: <out>.npz with complete, sigma, known, bbox3857, and (with --tif) <out>.tif (2 bands: depth, sigma).

usage: python3 seabednet_complete.py --in block.npz --out block_completed.npz [--models v5_small.pt,v5_small_ctlall.pt] [--cpu]
       python3 seabednet_complete.py --tif soundings.tif --out completed
Requires: torch, numpy, scipy (and rasterio for --tif); the gravity prior planetary/gravity_prior_canada.npz (Canada window).
Members that need the winter-radar channel are skipped unless aux_s1/<name>.npz exists for the block."""
import argparse, math, os, sys, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp"); ap.add_argument("--tif"); ap.add_argument("--out", required=True)
ap.add_argument("--models", default="v5_small.pt,v5_small_ctlall.pt,v5_small_s1all.pt"); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--bw", type=int, default=8); ap.add_argument("--tta", type=int, default=4, help="test-time augmentation folds: 1, 4 or 8")
A = ap.parse_args()
import torch
from scipy import ndimage as ndi
from v5_data import GravityPrior, lat_of_y, lon_of_x
from v5_model import V5, normalize

def tta_folds(n):   # 1: identity; 4: the four flips; 8: all rotations x flips
    return [(0, False)] if n <= 1 else ([(0, False), (0, True), (2, False), (2, True)] if n == 4 else [(k, fl) for k in range(4) for fl in (False, True)])
def tta_forward(net, xin, ridx, n):
    mus, lvs = [], []
    for k, fl in tta_folds(n):
        xa = torch.rot90(xin, k, (2, 3)); xa = torch.flip(xa, (3,)) if fl else xa
        m_, l_ = net(xa, ridx); m_ = torch.flip(m_, (3,)) if fl else m_; l_ = torch.flip(l_, (3,)) if fl else l_
        mus.append(torch.rot90(m_, -k, (2, 3))); lvs.append(torch.rot90(l_, -k, (2, 3)))
    if n <= 1: return mus[0], lvs[0]
    return torch.stack(mus).float().mean(0), torch.log(torch.stack(lvs).float().exp().mean(0))
DEV = "cpu" if A.cpu or not torch.cuda.is_available() else "cuda"; P = 256; S = 128
if A.tif:
    import rasterio
    with rasterio.open(A.tif) as src:
        z = src.read(1).astype("float32"); nd = src.nodata
        if nd is not None: z[z == nd] = np.nan
        b = src.bounds; bbox = np.array([b.left, b.bottom, b.right, b.top]); crs = src.crs; tf = src.transform
    if crs is None or "3857" not in str(crs): sys.exit("--tif must be in EPSG:3857")
else:
    d = np.load(A.inp, allow_pickle=True); z = d["z"].astype("float32"); bbox = np.array(d["bbox3857"], dtype=float); tf = None
H, W = z.shape; known = np.isfinite(z).astype(np.float32)
x0, x1 = min(bbox[0], bbox[2]), max(bbox[0], bbox[2]); y0, y1 = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])
grav = GravityPrior(); lons = np.array([lon_of_x(v) for v in np.linspace(x0, x1, W)]); lats = np.array([lat_of_y(v) for v in np.linspace(y1, y0, H)])
G = grav.sample(np.tile(lons, (H, 1)), np.tile(lats[:, None], (1, W))).astype("float32")
Hp = (H//S+3)*S; Wp = (W//S+3)*S
zp = np.full((Hp, Wp), np.nan, np.float32); zp[:H, :W] = z; kp = np.zeros((Hp, Wp), np.float32); kp[:H, :W] = known
gp = np.zeros((Hp, Wp), np.float32); gp[:H, :W] = G; gp[H:, :] = G[-1:, :].mean(); gp[:, W:] = gp[:, W-1:W]
s1 = None; name = os.path.basename(A.inp or A.tif).rsplit(".", 1)[0] + ".npz"
if os.path.exists(f"aux_s1/{name}"):
    b = np.load(f"aux_s1/{name}"); v = b["vv"].astype(np.float32); h = b["vh"].astype(np.float32); m = np.isfinite(v)
    s1 = np.zeros((3, Hp, Wp), np.float32); s1[0, :H, :W] = np.where(m, (v+20)/20, 0); s1[1, :H, :W] = np.where(np.isfinite(h), (h+30)/20, 0); s1[2, :H, :W] = m
win = np.outer(np.hanning(P), np.hanning(P)) + 1e-3
coords = [(i, j) for i in range(0, Hp-P+1, S) for j in range(0, Wp-P+1, S) if kp[i:i+P, j:j+P].sum() >= 400]
members = []
for ck_path in A.models.split(","):
    if not os.path.exists(ck_path): print("skip (missing)", ck_path); continue
    needs_s1 = "s1" in os.path.basename(ck_path)
    if needs_s1 and s1 is None: print("skip (no radar channel for this block)", ck_path); continue
    net = V5("small", in_ch=3 + (3 if needs_s1 else 0)).to(DEV); ck = torch.load(ck_path, map_location=DEV, weights_only=False); net.load_state_dict(ck["net"]); net.eval()
    accm = np.zeros((Hp, Wp)); accs = np.zeros((Hp, Wp)); wacc = np.zeros((Hp, Wp))
    with torch.no_grad():
        for b0 in range(0, len(coords), A.bw):
            cb = coords[b0:b0+A.bw]; t = lambda a: torch.tensor(np.stack(a))[:, None].float().to(DEV)
            dt = t([np.nan_to_num(zp[i:i+P, j:j+P]) for i, j in cb]); kt = t([kp[i:i+P, j:j+P] for i, j in cb]); gt = t([gp[i:i+P, j:j+P] for i, j in cb])
            dn, gn, mu0, sd0 = normalize(dt, kt, gt); xin = torch.cat([dn*kt, kt, gn], 1)
            if needs_s1: xin = torch.cat([xin, torch.tensor(np.stack([s1[:, i:i+P, j:j+P] for i, j in cb])).float().to(DEV)], 1)
            mu, lv = tta_forward(net, xin, torch.zeros(len(cb), device=DEV, dtype=torch.long), A.tta)
            est = (mu.float()*sd0 + mu0)[:, 0].cpu().numpy(); sig = (torch.exp(0.5*lv.float())*sd0)[:, 0].cpu().numpy()
            for q, (i, j) in enumerate(cb): accm[i:i+P, j:j+P] += est[q]*win; accs[i:i+P, j:j+P] += sig[q]*win; wacc[i:i+P, j:j+P] += win
    est = np.where(wacc > 0, accm/np.maximum(wacc, 1e-6), np.nan)[:H, :W]; sig = np.where(wacc > 0, accs/np.maximum(wacc, 1e-6), np.nan)[:H, :W]
    members.append((est, sig)); print("member", ck_path, "done")
if not members: sys.exit("no model could be run")
complete = np.nanmean(np.stack([m[0] for m in members]), 0); sigma = np.sqrt(np.nanmean(np.stack([m[1]**2 for m in members]), 0))
dist = ndi.distance_transform_edt(~(known > 0)); complete[dist > 60] = np.nan; sigma[dist > 60] = np.nan     # 6 km fill discipline
complete[complete > -4] = np.nan                                                                              # predicted land masked
np.savez_compressed(A.out if A.out.endswith(".npz") else A.out + ".npz", complete=complete.astype("float32"), sigma=sigma.astype("float32"), known=known.astype(bool), bbox3857=bbox, members=np.int8(len(members)))
if A.tif:
    with rasterio.open(A.out + ".tif", "w", driver="GTiff", height=H, width=W, count=2, dtype="float32", crs="EPSG:3857", transform=tf, nodata=np.nan) as dst:
        dst.write(complete.astype("float32"), 1); dst.write(sigma.astype("float32"), 2)
print("done:", A.out, "| members", len(members), "| completed cells", int(np.isfinite(complete).sum()), "| known", int(known.sum()))
