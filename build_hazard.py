#!/usr/bin/env python3
"""Extreme-depth (hazard) dataset from NONNA-10 / NONNA-100 pairs.

For every 100 m cell covered by a NONNA-10 tile: input = 100 m mean depth (what a
NONNA-100 archive shows), target = the SHALLOWEST 10 m sounding within 500 m of the cell
centre (what a keel meets). Tiles are mosaicked 4x4 into 800x800 100 m canvases so the
v5 sampler (P=256) can draw patches. Output tiles_hz/g<I>_<J>.npz: z (mean100), zshoal,
cov (fraction of the 500 m disc surveyed at 10 m), bbox3857.
"""
import glob, os, re, numpy as np
from scipy import ndimage as ndi
R = 101                       # 10 m px -> ~500 m radius window
os.makedirs("tiles_hz", exist_ok=True)
files = sorted(glob.glob("tiles10/*.npz"))
groups = {}
for f in files:
    i, j = map(int, re.findall(r"b(\d+)_(\d+)", f)[0]); groups.setdefault((i//4, j//4), []).append((i, j, f))
print(len(files), "tiles ->", len(groups), "groups", flush=True)
done = 0
for (gi, gj), lst in sorted(groups.items()):
    out = f"tiles_hz/g{gi:04d}_{gj:04d}.npz"
    if os.path.exists(out): continue
    bb = None; tile_w = None
    Z = np.full((800, 800), np.nan, np.float32); S = np.full((800, 800), np.nan, np.float32); C = np.zeros((800, 800), np.float32)
    x0 = y0 = None
    for i, j, f in lst:
        d = np.load(f, allow_pickle=True); z = d["z"].astype(np.float32); b = d["bbox3857"]
        if z.shape != (2000, 2000): continue
        fin = np.isfinite(z)
        if fin.sum() < 1000: continue
        # 100 m mean (needs >= 30 of 100 px)
        zb = z.reshape(200, 10, 200, 10); fb = fin.reshape(200, 10, 200, 10)
        cnt = fb.sum((1, 3)); mean = np.where(cnt >= 30, np.nansum(np.where(fb, zb, 0), (1, 3)) / np.maximum(cnt, 1), np.nan)
        # shallowest within 500 m: max of (negative) depths, NaN -> -inf
        zmax = ndi.maximum_filter(np.where(fin, z, -np.inf), size=R, mode="constant", cval=-np.inf)
        cov = ndi.uniform_filter(fin.astype(np.float32), size=R, mode="constant", cval=0.0)
        shoal = zmax[5::10, 5::10]; cv = cov[5::10, 5::10]
        shoal = np.where(np.isfinite(shoal) & (cv >= 0.5), shoal, np.nan)
        # place in canvas: tile (i,j) -> row (i - 4*gi), col (j - 4*gj); tile indices increase with y? infer from bbox
        r, c = (i - 4*gi)*200, (j - 4*gj)*200
        Z[r:r+200, c:c+200] = mean; S[r:r+200, c:c+200] = shoal; C[r:r+200, c:c+200] = cv
        if bb is None: bb = np.array(b, float); tile_w = abs(b[2]-b[0]); ij0 = (i, j)
        else:
            bb[0] = min(bb[0], b[0], b[2]); bb[2] = max(bb[2], b[0], b[2]); bb[1] = min(bb[1], b[1], b[3]); bb[3] = max(bb[3], b[1], b[3])
    if bb is None or np.isfinite(S).sum() < 5000: continue
    # canvas bbox = full 4x4 extent regardless of which tiles exist
    i0, j0 = 4*gi, 4*gj
    b0 = np.load(lst[0][2], allow_pickle=True)["bbox3857"]; ii, jj = lst[0][0], lst[0][1]
    # tile bbox: x grows with j, y ... check sign with two tiles if possible
    xs = b0[0] - (jj - j0)*tile_w; xe = xs + 4*tile_w
    ytop = max(b0[1], b0[3]) + (ii - i0)*tile_w   # assume row index grows downward (north->south)
    ybot = ytop - 4*tile_w
    np.savez_compressed(out, z=Z, zshoal=S, cov=C, bbox3857=np.array([xs, ybot, xe, ytop]))
    done += 1
    if done % 50 == 0: print(done, "groups written", flush=True)
print("HAZARD_BUILD_DONE", done)
