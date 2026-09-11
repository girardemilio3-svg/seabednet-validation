#!/usr/bin/env python3
"""v5 corpus: unified multi-resolution patch sampler.

Sources (scanned at construction, so it grows as harvests land):
  tiles/        43 named 100 m regional tiles   (res=100)
  tiles_nat/    437 national 100 m blocks       (res=100)
  tiles10/      national 10 m blocks            (res=10)
Each sample: dict(depth[P,P], known[P,P], gravity[P,P], res_m, lonlat_center).
Gravity channel: bilinear lookup into planetary/gravity_prior_canada.npz.
Holdouts (all resolutions, by geography): placentia, juandefuca, frobisher
boxes — any patch whose center falls inside a holdout bbox is excluded from
train and reserved for eval.
"""
import glob, math, os, numpy as np

R = 6378137.0
def lat_of_y(y): return math.degrees(2*math.atan(math.exp(y/R))-math.pi/2)
def lon_of_x(x): return math.degrees(x/R)
def to3857(lon, lat):
    return R*math.radians(lon), R*math.log(math.tan(math.pi/4+math.radians(lat)/2))

HOLDOUT_BBOXES = [  # lon0,lat0,lon1,lat1 (matches original holdout tiles)
    (-54.6, 47.0, -53.8, 47.7),    # placentia
    (-124.5, 48.2, -123.5, 48.6),  # juandefuca
    (-68.5, 62.9, -67.0, 63.6),    # frobisher
]

class GravityPrior:
    def __init__(s, path=None):
        path = path or os.environ.get("V5_PRIOR", "planetary/gravity_prior_canada.npz")   # V5_PRIOR=planetary/gravity_prior_canada_nosid.npz -> leakage-free anchor
        d = np.load(path)
        s.z = d["z"].astype("float32")
        s.lon0, s.lon1 = d["lon"]; s.lat0, s.lat1 = d["lat"]
        s.H, s.W = s.z.shape
    def sample(s, lons, lats):
        """Bilinear sample gravity depth at arrays of lon/lat."""
        x = (lons - s.lon0)/(s.lon1-s.lon0)*(s.W-1)
        y = (lats - s.lat0)/(s.lat1-s.lat0)*(s.H-1)
        x = np.clip(x, 0, s.W-1.001); y = np.clip(y, 0, s.H-1.001)
        x0 = x.astype(int); y0 = y.astype(int)
        fx = x-x0; fy = y-y0
        z = (s.z[y0, x0]*(1-fx)*(1-fy) + s.z[y0, x0+1]*fx*(1-fy)
             + s.z[y0+1, x0]*(1-fx)*fy + s.z[y0+1, x0+1]*fx*fy)
        return z

class Corpus:
    def __init__(s, P=256, seed=0, preload=True):
        s.P = P; s.rng = np.random.default_rng(seed)
        s.grav = GravityPrior()
        s.entries = []      # (path, res_m, kind)
        # V5_CORPUS="tiles_nat:100:3857,tiles10:10:3857" overrides the default corpus
        spec = os.environ.get("V5_CORPUS", "tiles_nat:100:3857,tiles10:10:3857,tiles:100:lonlat")
        for item in spec.split(","):
            dr, res, kind = item.split(":")
            for f in sorted(glob.glob(f"{dr}/*.npz")):
                s.entries.append((f, float(res), kind))
        # V5_TEMPORAL="tiles_nat=index_out,tiles10=index_out10": blank every sounding that lies
        # outside all CHS Survey Index polygons (i.e. surveyed after 2016) -> pre-2016 corpus
        s.temporal = {}
        for item in filter(None, os.environ.get("V5_TEMPORAL", "").split(",")):
            dr, idx = item.split("="); s.temporal[dr] = idx
        if s.temporal: s.n_blanked = 0
        s.n10 = sum(1 for e in s.entries if e[1] == 10.0)
        s.cache = {}; s.aux = {}; s.use_aux = os.environ.get("V5_AUX", "0") == "1"; s.n_aux = 0
        s.vpos = {}; s.fast = os.environ.get("V5_FASTSAMPLE", "0") == "1"   # indexed sampling (no rejection loop)
        s.use_grav = os.environ.get("V5_GRAV", "0") == "1"   # raw marine gravity anomaly + vertical gradient as channels
        if s.use_grav: s.gravA = GravityPrior("planetary/grav_canada.npz"); s.curvA = GravityPrior("planetary/curv_canada.npz")
        s.use_s1 = os.environ.get("V5_S1", "0") == "1"; s.s1 = {}   # winter Sentinel-1 VV/VH (dB) channels from aux_s1/
        if preload:                       # RAM-cache: float16 arrays + bbox
            import time; t0 = time.time(); tot = 0
            for f, res, kind in s.entries:
                d = np.load(f, allow_pickle=True)
                z = s._load_z(f, d["z"]).astype(np.float16)
                bb = d["bbox3857"] if kind == "3857" else d["bbox"]
                s.cache[f] = (z, np.array(bb, dtype=np.float64))
                tot += z.nbytes
                if s.use_aux:
                    ap = f"aux_out/{os.path.basename(f)}"
                    if os.path.exists(ap) and kind == "3857" and res == 100.0:
                        a = np.load(ap); s.aux[f] = (a["land"].astype(np.float16), a["s2"], a["s2ok"]); s.n_aux += 1; tot += a["s2"].nbytes + a["land"].nbytes//2
                if s.use_s1:
                    sp = f"aux_s1/{os.path.basename(f)}"
                    if os.path.exists(sp) and kind == "3857" and res == 100.0:
                        b = np.load(sp); s.s1[f] = (b["vv"], b["vh"]); tot += b["vv"].nbytes*2
            print(f"corpus preloaded: {tot/1e9:.1f} GB in RAM ({time.time()-t0:.0f}s)")
        print(f"corpus: {len(s.entries)} files ({s.n10} at 10 m)" + (f"; aux channels for {s.n_aux} blocks" if s.use_aux else ""))

    def _load_z(s, f, z):
        idx = s.temporal.get(os.path.dirname(f))
        if idx is None: return z
        p = f"{idx}/{os.path.basename(f)}"
        if not os.path.exists(p): return np.full_like(z, np.nan)   # no index raster -> unusable
        post = np.isfinite(z) & (np.load(p)["year"] == 0)
        z = z.copy(); z[post] = np.nan; s.n_blanked += int(post.sum())
        return z

    def _bbox_lonlat(s, bb, kind):
        if kind == "3857":
            x0, y0, x1, y1 = bb
            return lon_of_x(x0), lat_of_y(y0), lon_of_x(x1), lat_of_y(y1)
        return float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])

    def _in_holdout(s, lon, lat):
        return any(a <= lon <= c and b <= lat <= d for a, b, c, d in HOLDOUT_BBOXES)

    def _valid_positions(s, f, z, min_valid):
        """top-left (i, j) at stride 8 whose P x P window has >= min_valid known fraction; cached per file and threshold"""
        key = (f, round(min_valid, 2))
        if key in s.vpos: return s.vpos[key]
        P = s.P; H, W = z.shape; k = np.isfinite(z).astype(np.float32)
        c = np.zeros((H+1, W+1), np.float64); c[1:, 1:] = np.cumsum(np.cumsum(k, 0), 1)
        ii = np.arange(0, H-P+1, 8); jj = np.arange(0, W-P+1, 8)
        frac = (c[ii[:, None]+P, jj[None, :]+P] - c[ii[:, None], jj[None, :]+P] - c[ii[:, None]+P, jj[None, :]] + c[ii[:, None], jj[None, :]])/(P*P)
        ok = np.argwhere(frac >= min_valid); pos = np.stack([ii[ok[:, 0]], jj[ok[:, 1]]], 1) if len(ok) else np.zeros((0, 2), int)
        s.vpos[key] = pos; return pos

    def sample(s, min_valid=0.35, want_res=None, holdout=False, tries=400):
        P = s.P
        for _ in range(tries):
            f, res, kind = s.entries[s.rng.integers(len(s.entries))]
            if want_res and res != want_res: continue
            if f in s.cache:
                zc, bb = s.cache[f]
            else:
                d = np.load(f, allow_pickle=True)
                zc = s._load_z(f, d["z"]); bb = d["bbox3857"] if kind == "3857" else d["bbox"]
            z = zc
            H, W = z.shape
            if H < P or W < P: continue
            if s.fast:
                pos = s._valid_positions(f, z, min_valid)
                if len(pos) == 0: continue
                i, j = pos[s.rng.integers(len(pos))]; i = int(min(i + s.rng.integers(8), H-P)); j = int(min(j + s.rng.integers(8), W-P))
            else:
                i, j = s.rng.integers(H-P), s.rng.integers(W-P)
            p = z[i:i+P, j:j+P].astype(np.float32)
            known = np.isfinite(p)
            if known.mean() < min_valid: continue
            lo0, la0, lo1, la1 = s._bbox_lonlat(bb, kind)
            clon = lo0 + (j+P/2)/W*(lo1-lo0)
            clat = la1 - (i+P/2)/H*(la1-la0)
            if s._in_holdout(clon, clat) != holdout: continue
            # gravity channel for the patch
            lons = np.linspace(lo0+(j)/W*(lo1-lo0), lo0+(j+P)/W*(lo1-lo0), P)
            lats = np.linspace(la1-(i)/H*(la1-la0), la1-(i+P)/H*(la1-la0), P)
            LO, LA = np.meshgrid(lons, lats)
            grav = s.grav.sample(LO, LA)
            out = dict(depth=np.nan_to_num(p), known=known.astype(np.float32),
                        gravity=grav.astype(np.float32), res_m=res,
                        center=(clon, clat))
            if s.use_aux or s.use_grav or s.use_s1:
                ch = [s.aux_patch(f, i, j)] if s.use_aux else []
                if s.use_grav: ch.append(np.stack([s.gravA.sample(LO, LA)/50.0, s.curvA.sample(LO, LA)/100.0]).astype(np.float32))
                if s.use_s1: ch.append(s.s1_patch(f, i, j))
                out["aux"] = np.concatenate(ch, 0)
            return out
        return None

    def aux_patch(s, f, i, j):
        """5 channels: land elevation (clipped, /100 m), land flag, imagery R,G,B (/255); zeros + flags off when unavailable."""
        P = s.P; a = np.zeros((5, P, P), np.float32)
        if f in s.aux:
            land, s2, ok = s.aux[f]; L = land[i:i+P, j:j+P].astype(np.float32); m = np.isfinite(L) & (L > 0.5)
            a[0] = np.where(m, np.clip(L, 0, 500)/100.0, 0.0); a[1] = m
            a[2:5] = s2[:, i:i+P, j:j+P].astype(np.float32)/255.0 * ok[i:i+P, j:j+P]
        return a

    def s1_patch(s, f, i, j):
        """3 channels: VV dB (+20)/20, VH dB (+30)/20, valid flag; zeros when unavailable."""
        P = s.P; a = np.zeros((3, P, P), np.float32)
        if f in s.s1:
            vv, vh = s.s1[f]; v = vv[i:i+P, j:j+P].astype(np.float32); h = vh[i:i+P, j:j+P].astype(np.float32); m = np.isfinite(v)
            a[0] = np.where(m, (v + 20.0)/20.0, 0.0); a[1] = np.where(np.isfinite(h), (h + 30.0)/20.0, 0.0); a[2] = m
        return a

if __name__ == "__main__":
    c = Corpus()
    for r in [100.0, 10.0]:
        x = c.sample(want_res=r)
        if x:
            print(f"res {r:5.0f} m  depth[{x['depth'].min():7.1f},{x['depth'].max():6.1f}] "
                  f"known {x['known'].mean()*100:4.0f}%  gravity[{x['gravity'].min():7.0f},"
                  f"{x['gravity'].max():6.0f}]  @ {x['center'][0]:.2f},{x['center'][1]:.2f}")
        else:
            print(f"res {r}: no sample yet (harvest still filling)")
