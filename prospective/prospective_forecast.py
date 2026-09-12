#!/usr/bin/env python3
"""Seal a PROSPECTIVE danger-report forecast. Every chart-safe 500 m cell scored by the danger-report model (danger_score/, built
from national_v5_out + hazard_nat_v2 + winter radar + exposure cues) is written with its lon/lat, score and national percentile,
then hashed and OpenTimestamped. Scoring rule fixed here, before any outcome is known (prospective_score.py):
  eligible outcomes = Coast Guard NAVWARN danger notices (Shallow Depth Confirmed/Reported, Shoal, Uncharted Rock, Submerged
  Object, sandspit/newly) with message id > CUTOFF_ID and issue date > SEAL_DATE, geolocated by the same parser as the hindcasts;
  a notice is 'inside the forecast' if a scored cell lies within 1 km; metric = share of inside-forecast notices whose nearest cell
  is in the national top decile (chance = 10%) and top percentile (chance = 1%), with a Wilson interval.
Writes forecast_danger_<date>.csv, forecast_danger_<date>.json (cutoff, thresholds, sha256), .ots proof."""
import csv, glob, hashlib, json, os, re, subprocess, datetime, zipfile, numpy as np, requests
def shape_of(path, key):
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        v = np.lib.format.read_magic(f); return (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(f)[0]
R = 6378137.0; today = datetime.date.today().isoformat()
# cutoff = highest NAVWARN id visible on the live search right now
t = requests.get("https://nis.ccg-gcc.gc.ca/public/rest/messages/en/search?page=0&maxHits=50", headers={"User-Agent": "Mozilla/5.0 (research; seabednet)"}, timeout=60).text
ids = [int(x) for x in re.findall(r"message/(\d+)", t)]; cutoff = max(ids)
rows = []
for f in sorted(glob.glob("danger_score/*.npz")):
    name = os.path.basename(f); a = np.load(f, allow_pickle=True); H, W = shape_of(f"hazard_nat_v2/{name}", "p105")
    x0, y0, x1, y1 = a["bbox3857"]; x0, x1 = min(x0, x1), max(x0, x1); y0, y1 = min(y0, y1), max(y0, y1)
    ii = a["i"].astype(float); jj = a["j"].astype(float); sc = a["score"].astype(float)
    lon = np.degrees((x0 + (jj+0.5)/W*(x1-x0))/R); lat = np.degrees(2*np.arctan(np.exp((y1 - (ii+0.5)/H*(y1-y0))/R)) - np.pi/2)
    rows += list(zip([name[:-4]]*len(sc), a["i"].tolist(), a["j"].tolist(), np.round(lon, 5).tolist(), np.round(lat, 5).tolist(), sc.tolist()))
sc = np.array([r[5] for r in rows]); order = np.argsort(-sc); rank = np.empty(len(sc), int); rank[order] = np.arange(len(sc))
pct = 100.0*(1 - rank/len(sc))                       # 100 = highest score
out = f"forecast_danger_{today}.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["block", "i", "j", "lon", "lat", "score", "percentile"])
    for k in order: w.writerow([*rows[k][:5], f"{rows[k][5]:.5f}", f"{pct[k]:.3f}"])
sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
meta = dict(sealed=today, cutoff_id=cutoff, n_cells=len(sc), n_blocks=len(set(r[0] for r in rows)), top_decile_threshold=float(np.quantile(sc, 0.9)), top_percentile_threshold=float(np.quantile(sc, 0.99)),
            model="danger_model2 recipe refit on all dated notices (danger_apply.py), inputs national_v5_out + hazard_nat_v2 + aux_s1 + exposure cues", sha256=sha, file=out,
            rule="notice eligible iff id > cutoff_id and date > sealed; inside iff a scored cell within 1 km; hit iff nearest cell percentile >= 90 (chance 10%) / >= 99 (chance 1%)")
json.dump(meta, open(out.replace(".csv", ".json"), "w"), indent=1)
subprocess.run(["ots", "stamp", out], check=False); print(json.dumps(meta, indent=1))
