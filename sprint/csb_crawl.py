#!/usr/bin/env python3
"""Crawl the IHO DCDB crowdsourced-bathymetry bucket day by day (newest first), keep only points in Canadian waters
(lat > 41, lon -142..-50), append to recon_csb/csb_canada/<year>.parquet, delete the raw files. Resumable via done_days.txt."""
import os, subprocess, glob, datetime, sys, pandas as pd
os.chdir("/home/fenexpertai/seabednet/recon_csb"); os.makedirs("csb_canada", exist_ok=True); os.makedirs("raw", exist_ok=True)
done = set(open("done_days.txt").read().split()) if os.path.exists("done_days.txt") else set()
start = datetime.date(2026, 9, 13); end = datetime.date(2017, 1, 1); d = start
while d >= end:
    key = d.strftime("%Y/%m/%d")
    if key in done: d -= datetime.timedelta(days=1); continue
    for f in glob.glob("raw/*"): os.remove(f)
    subprocess.run(["aws", "s3", "cp", "--no-sign-request", "--recursive", "--quiet", f"s3://noaa-dcdb-bathymetry-pds/csb/csv/{key}/", "raw/"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    keep = []; n = 0
    for f in glob.glob("raw/*.csv"):
        try: x = pd.read_csv(f, usecols=["LON", "LAT", "DEPTH", "TIME", "PLATFORM_NAME", "PROVIDER", "UNIQUE_ID"])
        except Exception: continue
        n += len(x); k = x[(x.LAT > 41) & (x.LON > -142) & (x.LON < -50)]
        if len(k): keep.append(k)
    if keep:
        k = pd.concat(keep); k["day"] = key; os.makedirs(f"csb_canada/{d.year}", exist_ok=True)
        for col in ("PLATFORM_NAME", "PROVIDER", "UNIQUE_ID"): k[col] = k[col].fillna("").map(str)
        for col in ("LON", "LAT", "DEPTH"): k[col] = pd.to_numeric(k[col], errors="coerce").astype("float64")
        k["TIME"] = k["TIME"].map(str); k = k.reset_index(drop=True)
        try: k.to_parquet(f"csb_canada/{d.year}/{d.strftime('%m-%d')}.parquet", index=False)
        except Exception as e: open("crawl.log", "a").write(f"{key} PARQUET_FAIL {e}\n"); k.to_csv(f"csb_canada/{d.year}/{d.strftime('%m-%d')}.csv.gz", index=False)
    open("done_days.txt", "a").write(key + "\n"); open("crawl.log", "a").write(f"{key} files={len(glob.glob('raw/*.csv'))} points={n} kept={sum(len(x) for x in keep)}\n")
    d -= datetime.timedelta(days=1)
open("crawl.log", "a").write("CRAWL_DONE\n")
