#!/usr/bin/env python3
"""10 m relief tiles (z11-13) — fast version: iterate NONNA-10 tiles, touch only the web tiles each one covers,
sample from the tile itself plus its overlapping neighbours (vectorised bbox filter), fall back to the 100 m
completion then the gravity prior. -> map/tiles/terrain/{11..13}/..."""
import glob, math, os, numpy as np
from PIL import Image
from v5_data import lat_of_y, lon_of_x, GravityPrior
R = 6378137.0; W0 = 2*math.pi*R; ORG = -math.pi*R; G = GravityPrior()
def tb_of(z, x, y):
    s = W0/2**z; return ORG + x*s, (math.pi*R) - (y+1)*s, ORG + (x+1)*s, (math.pi*R) - y*s
def norm(bb): return (min(bb[0],bb[2]), min(bb[1],bb[3]), max(bb[0],bb[2]), max(bb[1],bb[3]))
files = []; B = []
for f in sorted(glob.glob("tiles10/*.npz")):
    bb = norm(np.load(f, allow_pickle=True)["bbox3857"]); lo = lon_of_x(bb[0]); la = lat_of_y(bb[1])
    if -102 <= lo <= -52 and 53 <= la <= 74.5: files.append(f); B.append(bb)
B = np.array(B)   # N x 4
c100 = [(f, norm(np.load(f, allow_pickle=True)["bbox3857"])) for f in sorted(glob.glob("corridor_out_v2/*.npz"))]
C100 = np.array([b for _, b in c100])
print(f"{len(files)} NONNA-10 tiles", flush=True)
cache = {}
def arr(f, key):
    if f not in cache:
        if len(cache) > 24: cache.pop(next(iter(cache)))
        cache[f] = np.load(f, allow_pickle=True)[key].astype("float32")
    return cache[f]
def sample(items, key, tb, n=256):
    x0, y0, x1, y1 = tb
    xs = x0 + (np.arange(n)+0.5)/n*(x1-x0); ys = y1 - (np.arange(n)+0.5)/n*(y1-y0)
    out = np.full((n, n), np.nan, np.float32)
    for f, (bx0, by0, bx1, by1) in items:
        a = arr(f, key); H, W = a.shape
        jj = ((xs-bx0)/(bx1-bx0)*W).astype(int); ii = ((by1-ys)/(by1-by0)*H).astype(int)
        okx = (jj >= 0) & (jj < W); oky = (ii >= 0) & (ii < H)
        if not (okx.any() and oky.any()): continue
        sub = a[np.ix_(np.clip(ii, 0, H-1), np.clip(jj, 0, W-1))]
        m = oky[:, None] & okx[None, :] & np.isfinite(sub) & ~np.isfinite(out); out[m] = sub[m]
    return out
def enc(v, tb):
    x0, y0, x1, y1 = tb; n = v.shape[0]
    if (~np.isfinite(v)).any():
        xs = x0 + (np.arange(n)+0.5)/n*(x1-x0); ys = y1 - (np.arange(n)+0.5)/n*(y1-y0)
        lons = np.degrees(xs/R); lats = np.degrees(2*np.arctan(np.exp(ys/R)) - np.pi/2)
        g = G.sample(np.tile(lons, (n, 1)), np.tile(lats[:, None], (1, n))); v = np.where(np.isfinite(v), v, np.minimum(g, 0.0))
    e = np.clip((v + 10000.0)/0.1, 0, 2**24-1).astype(np.uint32)
    return np.dstack([(e >> 16) & 255, (e >> 8) & 255, e & 255]).astype(np.uint8)
def overlapping(BB, tb):
    x0, y0, x1, y1 = tb
    return np.nonzero(~((BB[:, 2] < x0) | (BB[:, 0] > x1) | (BB[:, 3] < y0) | (BB[:, 1] > y1)))[0]
done = set(); n_written = 0
for z in (11, 12, 13):
    s = W0/2**z
    for i, f in enumerate(files):
        bx0, by0, bx1, by1 = B[i]
        xa, xb = int((bx0-ORG)//s), int((bx1-ORG)//s); ya, yb = int((math.pi*R - by1)//s), int((math.pi*R - by0)//s)
        for x in range(xa, xb+1):
            for y in range(ya, yb+1):
                if (z, x, y) in done: continue
                done.add((z, x, y)); tb = tb_of(z, x, y)
                idx = overlapping(B, tb)
                v = sample([(files[k], B[k]) for k in idx], "z", tb)
                cov = float(np.isfinite(v).mean())
                if cov < 0.02: continue
                if cov < 1.0:
                    k2 = overlapping(C100, tb); v100 = sample([c100[k] for k in k2], "complete", tb); m = ~np.isfinite(v); v[m] = v100[m]
                d = f"map/tiles/terrain/{z}/{x}"; os.makedirs(d, exist_ok=True)
                Image.fromarray(enc(v, tb), "RGB").save(f"{d}/{y}.png", optimize=True); n_written += 1
        if i % 200 == 0: print(f"z{z} tile {i}/{len(files)}: {n_written} written", flush=True)
    print(f"z{z} done: {n_written} written", flush=True)
print("TILES10_DONE", n_written); os.system("du -sh map/tiles/terrain")
