#!/usr/bin/env python3
"""Web-mercator tile pyramids from the block grids, no GDAL.
  terrain/{z}/{x}/{y}.png  Terrain-RGB (Mapbox encoding) of the completed corridor seabed (corridor_out_v2 'complete')
  hazard/{z}/{x}/{y}.png   RGBA colormap of P(shoal < 10.5 m) from hazard_nat (corridor blocks), transparent where no data
  age/{z}/{x}/{y}.png      RGBA decade colormap of survey year from index_out (national), transparent where no sounding
Blocks are EPSG:3857 rasters with bbox3857; tiles are sampled nearest-neighbour from the block mosaic.
usage: python3 make_tiles.py  -> map/tiles/{terrain,hazard,age}/..."""
import glob, json, math, os, numpy as np
from PIL import Image
R = 6378137.0; W0 = 2*math.pi*R; ORG = -math.pi*R
def tile_bounds(z, x, y):
    s = W0/2**z; return ORG + x*s, (math.pi*R) - (y+1)*s, ORG + (x+1)*s, (math.pi*R) - y*s   # x0,y0,x1,y1 (mercator)
def load_blocks(kind):
    out = []
    if kind == "terrain":
        for f in sorted(glob.glob("corridor_out_v2/*.npz")):
            d = np.load(f, allow_pickle=True); out.append((d["bbox3857"], d["complete"].astype("float32")))
    elif kind == "hazard":
        for f in sorted(glob.glob("hazard_nat/*.npz")):
            name = os.path.basename(f)
            if not os.environ.get("HZ_ALL") and not os.path.exists(f"corridor_out_v2/{name}"): continue
            d = np.load(f, allow_pickle=True); p = d["p105"].astype("float32")
            cf = f"corridor_out_v2/{name}" if os.path.exists(f"corridor_out_v2/{name}") else f"national_v5_out/{name}"
            if os.path.exists(cf):
                c = np.load(cf, allow_pickle=True)["complete"].astype("float32")
                if c.shape == p.shape: p = np.where(np.isfinite(c) & (c < -21), p, np.nan)   # draw hazard only where the mean map calls the water navigable: the surprises
            out.append((d["bbox3857"], p))
    else:
        for f in sorted(glob.glob("tiles_nat/*.npz")):
            ip = f"index_out/{os.path.basename(f)}"
            if not os.path.exists(ip): continue
            d = np.load(f, allow_pickle=True); z = d["z"]; yr = np.load(ip)["year"].astype("float32")
            v = np.where(np.isfinite(z), np.where(yr == 0, 2020, yr), np.nan).astype("float32"); out.append((d["bbox3857"], v))
    norm = []
    for bb, a in out:
        norm.append((min(bb[0],bb[2]), min(bb[1],bb[3]), max(bb[0],bb[2]), max(bb[1],bb[3]), a))
    return norm
def sample(blocks, x0, y0, x1, y1, n=256):
    xs = x0 + (np.arange(n)+0.5)/n*(x1-x0); ys = y1 - (np.arange(n)+0.5)/n*(y1-y0)
    out = np.full((n, n), np.nan, np.float32); any_ = False
    for bx0, by0, bx1, by1, a in blocks:
        if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1: continue
        H, W = a.shape
        jj = ((xs-bx0)/(bx1-bx0)*W).astype(int); ii = ((by1-ys)/(by1-by0)*H).astype(int)
        okx = (jj >= 0) & (jj < W); oky = (ii >= 0) & (ii < H)
        if not (okx.any() and oky.any()): continue
        sub = a[np.ix_(np.clip(ii, 0, H-1), np.clip(jj, 0, W-1))]
        m = oky[:, None] & okx[None, :] & np.isfinite(sub)
        out[m] = sub[m]; any_ = any_ or bool(m.any())
    return out if any_ else None
from v5_data import GravityPrior
_G = GravityPrior()
def enc_terrain(v, tb=None):
    if tb is not None and (~np.isfinite(v)).any():   # no data -> gravity-derived depth (smooth), never a wall at sea level
        x0, y0, x1, y1 = tb; n = v.shape[0]
        xs = x0 + (np.arange(n)+0.5)/n*(x1-x0); ys = y1 - (np.arange(n)+0.5)/n*(y1-y0)
        lons = np.degrees(xs/R); lats = np.degrees(2*np.arctan(np.exp(ys/R)) - np.pi/2)
        g = _G.sample(np.tile(lons, (n, 1)), np.tile(lats[:, None], (1, n)))
        v = np.where(np.isfinite(v), v, np.minimum(g, 0.0))
    v = np.where(np.isfinite(v), v, 0.0)
    e = np.clip((v + 10000.0)/0.1, 0, 2**24-1).astype(np.uint32)
    return np.dstack([(e >> 16) & 255, (e >> 8) & 255, e & 255]).astype(np.uint8)
INFERNO = [(0,(0,0,4)),(0.15,(40,11,84)),(0.3,(101,21,110)),(0.45,(159,42,99)),(0.6,(212,72,66)),(0.75,(245,125,21)),(0.9,(250,193,39)),(1.0,(252,255,164))]
def cmap(v, lo, hi, stops):
    t = np.clip((v-lo)/(hi-lo), 0, 1); rgb = np.zeros(v.shape+(3,), np.float32)
    for (t0, c0), (t1, c1) in zip(stops[:-1], stops[1:]):
        m = (t >= t0) & (t <= t1); f = ((t-t0)/(t1-t0))[m][:, None] if t1 > t0 else 0
        rgb[m] = np.array(c0) + f*(np.array(c1)-np.array(c0))
    return rgb.astype(np.uint8)
def enc_hazard(v):
    ok = np.isfinite(v); rgb = cmap(np.nan_to_num(v), 0, 0.5, INFERNO)
    a = np.where(ok, np.clip((np.nan_to_num(v) - 0.25)/0.35*255, 0, 255), 0).astype(np.uint8)   # transparent below P=0.25, opaque at P>=0.6: only real flags are drawn
    return np.dstack([rgb, a])
AGE = [(1800,(140,29,29)),(1940,(192,57,43)),(1960,(230,126,34)),(1980,(232,178,74)),(2000,(127,157,185)),(2017,(46,125,209)),(2020,(46,125,209))]
def enc_age(v):
    ok = np.isfinite(v); rgb = np.zeros(v.shape+(3,), np.uint8)
    for (y0, c0), (y1, c1) in zip(AGE[:-1], AGE[1:]):
        m = ok & (v >= y0) & (v < y1); rgb[m] = c0
    m = ok & (v >= 2020); rgb[m] = AGE[-1][1]
    return np.dstack([rgb, np.where(ok, 235, 0).astype(np.uint8)])
import sys
ALL = [("terrain", (5, 10), enc_terrain, "RGB"), ("hazard", (int(os.environ.get("HZ_ZMIN", 5)), int(os.environ.get("HZ_ZMAX", 10))), enc_hazard, "RGBA"), ("age", (3, 8), enc_age, "RGBA")]
JOBS = [j for j in ALL if len(sys.argv) < 2 or j[0] in sys.argv[1:]]
os.makedirs("map/tiles", exist_ok=True)
for kind, (zmin, zmax), enc, mode in JOBS:
    blocks = load_blocks(kind); X0 = min(b[0] for b in blocks); Y0 = min(b[1] for b in blocks); X1 = max(b[2] for b in blocks); Y1 = max(b[3] for b in blocks)
    n = 0
    for z in range(zmin, zmax+1):
        s = W0/2**z
        xa, xb = int((X0-ORG)//s), int((X1-ORG)//s); ya, yb = int((math.pi*R - Y1)//s), int((math.pi*R - Y0)//s)
        for x in range(xa, xb+1):
            for y in range(ya, yb+1):
                tb = tile_bounds(z, x, y)
                v = sample(blocks, *tb)
                if v is None:
                    if kind != "terrain": continue
                    if not any(not (b[2] < tb[0] or b[0] > tb[2] or b[3] < tb[1] or b[1] > tb[3]) for b in blocks): continue
                    v = np.full((256, 256), np.nan, np.float32)
                d = f"map/tiles/{kind}/{z}/{x}"; os.makedirs(d, exist_ok=True)
                Image.fromarray(enc(v, tb) if kind == "terrain" else enc(v), mode).save(f"{d}/{y}.png", optimize=True); n += 1
        print(f"{kind} z{z}: {n} tiles so far", flush=True)
    json.dump(dict(kind=kind, minzoom=zmin, maxzoom=zmax, bounds=[X0, Y0, X1, Y1]), open(f"map/tiles/{kind}/meta.json", "w"))
print("TILES_DONE"); os.system("du -sh map/tiles/*")
