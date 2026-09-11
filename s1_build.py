#!/usr/bin/env python3
"""Winter Sentinel-1 backscatter channel per 100 m block: median of VV and VH (dB) over Feb-Apr scenes (2 winters),
read at block resolution from the Planetary Computer RTC COGs (overviews). Grounded/landfast ice and ice ridges over shoals
have a distinct winter signature. -> aux_s1/<block>.npz {vv, vh (float16 dB, NaN = no data), n}
usage: python3 s1_build.py [tiles_nat]   env: S1_WORKERS (4), S1_MAX_SCENES (16)"""
import glob, os, sys, time, math, numpy as np, planetary_computer as pc, pystac_client, rasterio
from concurrent.futures import ThreadPoolExecutor
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from v5_data import lat_of_y, lon_of_x
SRC = sys.argv[1] if len(sys.argv) > 1 else "tiles_nat"; OUT = "aux_s1"; os.makedirs(OUT, exist_ok=True)
MAXS = int(os.environ.get("S1_MAX_SCENES", 16))
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
def one(f):
    out = f"{OUT}/{os.path.basename(f)}"
    if os.path.exists(out): return "have"
    d = np.load(f, allow_pickle=True); bb = d["bbox3857"]; H, W = d["z"].shape
    x0, x1 = min(bb[0], bb[2]), max(bb[0], bb[2]); y0, y1 = min(bb[1], bb[3]), max(bb[1], bb[3])
    bbox = [lon_of_x(x0), lat_of_y(y0), lon_of_x(x1), lat_of_y(y1)]
    items = []
    for dt in ("2023-02-01/2023-04-20", "2024-02-01/2024-04-20"):
        try: items += list(cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=dt, limit=100).items())
        except Exception: pass
    if not items:
        np.savez_compressed(out, vv=np.full((H, W), np.nan, np.float16), vh=np.full((H, W), np.nan, np.float16), n=np.zeros((H, W), np.uint8), bbox3857=bb); return "no scenes"
    items = items[:MAXS]; acc = {"vv": [], "vh": []}
    for it in items:
        for pol in ("vv", "vh"):
            if pol not in it.assets: continue
            try:
                with rasterio.open(it.assets[pol].href) as src:
                    b = transform_bounds("EPSG:4326", src.crs, *bbox); a = src.read(1, window=from_bounds(*b, src.transform), out_shape=(H, W), boundless=True, fill_value=np.nan).astype(np.float32)
                a = np.where(a > 0, 10*np.log10(np.maximum(a, 1e-6)), np.nan); acc[pol].append(a)
            except Exception: pass
    if not acc["vv"]:
        np.savez_compressed(out, vv=np.full((H, W), np.nan, np.float16), vh=np.full((H, W), np.nan, np.float16), n=np.zeros((H, W), np.uint8), bbox3857=bb); return "no reads"
    vv = np.nanmedian(np.stack(acc["vv"]), 0); vh = np.nanmedian(np.stack(acc["vh"]), 0) if acc["vh"] else np.full((H, W), np.nan, np.float32); n = np.isfinite(np.stack(acc["vv"])).sum(0).astype(np.uint8)
    np.savez_compressed(out, vv=vv.astype(np.float16), vh=vh.astype(np.float16), n=n, bbox3857=bb)
    return f"ok scenes {len(items)} cover {np.isfinite(vv).mean()*100:.0f}%"
if __name__ == "__main__":
    files = sorted(glob.glob(f"{SRC}/*.npz")); corr = set(os.path.basename(p) for p in glob.glob("corridor_out_v2/*.npz")); files.sort(key=lambda p: (os.path.basename(p) not in corr, p)); t0 = time.time()
    with ThreadPoolExecutor(int(os.environ.get("S1_WORKERS", 4))) as ex:
        for k, (f, r) in enumerate(zip(files, ex.map(one, files))): print(f"{k+1}/{len(files)} {os.path.basename(f)} {r} ({time.time()-t0:.0f}s)", flush=True)
    print("S1_DONE")
