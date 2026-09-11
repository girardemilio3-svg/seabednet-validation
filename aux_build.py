#!/usr/bin/env python3
"""Auxiliary input channels per 100 m block, aligned to the block grid (EPSG:3857 bbox3857, same H×W as z):
  land  float32  Copernicus GLO-30 DEM elevation (m) averaged to the block grid; NaN where no DEM tile (open sea)
  s2    uint8    [3,H,W] Sentinel-2 cloudless 2020 RGB (EOX WMTS z10 tiles, ~70 m at 63N) resampled to the block grid
  s2ok  uint8    1 where an imagery tile was available
Reads the DEM tiles straight from the public S3 bucket through GDAL (only the overview needed for 100 m is fetched).
usage: python3 aux_build.py [tiles_nat] -> aux_out/<block>.npz   env: AUX_WORKERS (default 6)"""
import glob, io, math, os, sys, time, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from PIL import Image
from v5_data import lat_of_y, lon_of_x
SRC = sys.argv[1] if len(sys.argv) > 1 else "tiles_nat"; OUT = "aux_out"; os.makedirs(OUT, exist_ok=True)
Z10 = 10; TILE_CACHE = "map/tiles_sat10"; os.makedirs(TILE_CACHE, exist_ok=True)
DEM = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM/Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM.tif"
missing_dem = set()
def dem_name(lat, lon):
    return ("N%02d" % lat if lat >= 0 else "S%02d" % -lat), ("W%03d" % -lon if lon < 0 else "E%03d" % lon)
def build_dem(x0, y0, x1, y1, W, H):
    lo0, lo1 = lon_of_x(x0), lon_of_x(x1); la0, la1 = lat_of_y(y0), lat_of_y(y1)
    acc = np.full((H, W), np.nan, np.float32); tf = from_bounds(x0, y0, x1, y1, W, H)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2"):
        for la in range(int(math.floor(la0)), int(math.floor(la1)) + 1):
            for lo in range(int(math.floor(lo0)), int(math.floor(lo1)) + 1):
                a, b = dem_name(la, lo); url = DEM.format(lat=a, lon=b)
                if url in missing_dem: continue
                try:
                    with rasterio.open(url) as src:
                        with WarpedVRT(src, crs="EPSG:3857", transform=tf, width=W, height=H, resampling=Resampling.average, nodata=np.nan) as vrt:
                            t = vrt.read(1, out_dtype="float32")
                    m = np.isfinite(t); acc[m] = t[m]
                except rasterio.errors.RasterioIOError:
                    missing_dem.add(url)
    return acc
def tile_xy(lon, lat, z):
    n = 2**z; return int((lon+180)/360*n), int((1 - math.log(math.tan(math.radians(lat)) + 1/math.cos(math.radians(lat)))/math.pi)/2*n)
def fetch_tile(z, x, y):
    f = f"{TILE_CACHE}/{z}/{y}/{x}.jpg"
    if os.path.exists(f) and os.path.getsize(f) > 0: return f
    os.makedirs(os.path.dirname(f), exist_ok=True)
    for k in range(3):
        r = subprocess.run(["curl", "-s", "-m", "60", "-o", f, "-w", "%{http_code}", f"https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg"], capture_output=True, text=True)
        if r.stdout == "200" and os.path.getsize(f) > 0: return f
        time.sleep(1)
    return None
def build_s2(x0, y0, x1, y1, W, H):
    R = 6378137.0; W0 = 2*math.pi*R; s = W0/2**Z10; ORG = -math.pi*R
    xa, xb = int((x0-ORG)//s), int((x1-ORG)//s); ya, yb = int((math.pi*R - y1)//s), int((math.pi*R - y0)//s)
    mosaic = np.zeros(((yb-ya+1)*256, (xb-xa+1)*256, 3), np.uint8); ok = np.zeros(((yb-ya+1)*256, (xb-xa+1)*256), np.uint8)
    coords = [(x, y) for x in range(xa, xb+1) for y in range(ya, yb+1)]
    with ThreadPoolExecutor(8) as ex: files = list(ex.map(lambda c: fetch_tile(Z10, c[0], c[1]), coords))
    for (x, y), f in zip(coords, files):
        if not f: continue
        try: im = np.asarray(Image.open(f).convert("RGB"))
        except Exception: continue
        i, j = (y-ya)*256, (x-xa)*256; mosaic[i:i+256, j:j+256] = im; ok[i:i+256, j:j+256] = 1
    # crop the mosaic to the block bbox and resample to the block grid (both web-mercator: pure scaling)
    mx0, my1 = ORG + xa*s, math.pi*R - ya*s
    px = s/256; c0 = int(round((x0-mx0)/px)); c1 = int(round((x1-mx0)/px)); r0 = int(round((my1-y1)/px)); r1 = int(round((my1-y0)/px))
    crop = mosaic[max(0, r0):r1, max(0, c0):c1]; cok = ok[max(0, r0):r1, max(0, c0):c1]
    rgb = np.asarray(Image.fromarray(crop).resize((W, H), Image.BILINEAR)); okr = np.asarray(Image.fromarray(cok*255).resize((W, H), Image.NEAREST)) > 0
    return np.moveaxis(rgb, -1, 0).copy(), okr.astype(np.uint8)
def one(f):
    out = f"{OUT}/{os.path.basename(f)}"
    if os.path.exists(out): return "have"
    d = np.load(f, allow_pickle=True); bb = d["bbox3857"]; H, W = d["z"].shape
    x0, x1 = min(bb[0], bb[2]), max(bb[0], bb[2]); y0, y1 = min(bb[1], bb[3]), max(bb[1], bb[3])
    land = build_dem(x0, y0, x1, y1, W, H); s2, s2ok = build_s2(x0, y0, x1, y1, W, H)
    np.savez_compressed(out, land=land.astype(np.float32), s2=s2, s2ok=s2ok, bbox3857=bb)
    return f"ok land {np.isfinite(land).mean()*100:.0f}% s2 {s2ok.mean()*100:.0f}%"
if __name__ == "__main__":
    files = sorted(glob.glob(f"{SRC}/*.npz")); t0 = time.time()
    # corridor blocks first (they gate the experiment), then the rest of the country
    corr = set(os.path.basename(p) for p in glob.glob("corridor_out_v2/*.npz")); files.sort(key=lambda p: (os.path.basename(p) not in corr, p))
    with ThreadPoolExecutor(int(os.environ.get("AUX_WORKERS", 6))) as ex:
        for k, (f, r) in enumerate(zip(files, ex.map(one, files))):
            print(f"{k+1}/{len(files)} {os.path.basename(f)} {r} ({time.time()-t0:.0f}s)", flush=True)
    print("AUX_DONE")
