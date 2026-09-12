#!/usr/bin/env python3
"""Score the sealed prospective danger forecast against Coast Guard NAVWARN danger notices issued AFTER the seal (rule fixed in
prospective_forecast.py). usage: python3 prospective_score.py forecast_danger_<date>.json [--min-id N]  (min-id overrides the
cutoff for dry runs). Caches fetched messages in recon_navwarn/prospective_msgs.jsonl. -> prospective_score_<date>.json"""
import csv, html, json, math, os, re, sys, time, requests, numpy as np
from scipy.spatial import cKDTree
meta = json.load(open(sys.argv[1])); cutoff = meta["cutoff_id"]; sealed = meta["sealed"]
if "--min-id" in sys.argv: cutoff = int(sys.argv[sys.argv.index("--min-id")+1]); sealed = "0000-00-00"
R = 6378137.0; S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0 (research; seabednet)"; B = "https://nis.ccg-gcc.gc.ca/public/rest/messages/en/"
def merc(lon, lat): return R*math.radians(lon), R*math.log(math.tan(math.pi/4 + math.radians(lat)/2))
# 1. ids issued after the cutoff (live search, newest first) + cancelled ones (they cancel within weeks in the Arctic)
ids = set()
for status in ("", "&status=CANCELLED"):
    for page in range(0, 400):
        t = S.get(B + f"search?page={page}&maxHits=50{status}", timeout=60).text; found = [int(x) for x in re.findall(r"message/(\d+)", t)]
        if not found: break
        ids |= {i for i in found if i > cutoff}
        if min(found) <= cutoff: break
ids = sorted(ids); print("candidate ids after cutoff", len(ids), flush=True)
cache = "recon_navwarn/prospective_msgs.jsonl"; have = {}
if os.path.exists(cache):
    for line in open(cache): m = json.loads(line); have[m["id"]] = m
pos = re.compile(r"(\d{2})\s*[°]?\s*(\d{2}(?:\.\d+)?)\s*'?\s*N\s*,?\s*(\d{3})\s*[°]?\s*(\d{2}(?:\.\d+)?)\s*'?\s*W", re.I); kw = re.compile(r"^(Shallow Depth (Confirmed|Reported)|Shoal\b|Uncharted Rock|Submerged Object|Sandspit|Obstruction \(Other\) Submerged)")
with open(cache, "a") as fh:
    for mid in ids:
        if mid in have: continue
        for k in range(3):
            try:
                t = S.get(B + f"message/{mid}", timeout=60).text; break
            except Exception: time.sleep(2); t = ""
        mt = html.unescape(re.sub(r"<[^>]+>", " ", t)); mt = re.sub(r"\s+", " ", mt); i = mt.find("NAVWARN details Navigational Warnings"); mt = mt[i:] if i >= 0 else mt; j = mt.find("Date Modified"); mt = mt[:j] if j > 0 else mt
        js = re.search(r"var data = (\{.*?\});", t); coords = []
        if js:
            try:
                for f in json.loads(js.group(1))["features"]: coords += [list(c) for c in f["coordinates"]]
            except Exception: pass
        m = dict(id=mid, text=mt[:6000], coords=coords); have[mid] = m; fh.write(json.dumps(m) + "\n")
pts = []
for mid in ids:
    t = have[mid]["text"]
    if not t: continue
    g = lambda k, stop: (re.search(k + r"\s+(.*?)\s+" + stop, t) or [None, None])[1]
    sid = g("ID", "Date"); date = (g("Date", "(?:UTC|Status)") or "")[:10]; cat = g("Categories", "Description"); desc = g("Description", "Position") or ""
    if not cat or not kw.match(cat.strip()) or date <= sealed: continue
    found = list(pos.finditer(desc))
    if found:
        for mm in found: pts.append(dict(id=mid, series=sid, date=date, category=cat, lat=float(mm.group(1)) + float(mm.group(2))/60, lon=-(float(mm.group(3)) + float(mm.group(4))/60)))
    elif have[mid]["coords"]: pts.append(dict(id=mid, series=sid, date=date, category=cat, lat=have[mid]["coords"][0][1], lon=have[mid]["coords"][0][0]))
print("eligible danger points", len(pts), flush=True)
# 2. forecast cells
F = list(csv.DictReader(open(meta["file"]))); xy = np.array([merc(float(r["lon"]), float(r["lat"])) for r in F]); pc = np.array([float(r["percentile"]) for r in F]); tree = cKDTree(xy)
res = []
for p in pts:
    d, k = tree.query(merc(p["lon"], p["lat"])); d_km = d*math.cos(math.radians(p["lat"]))/1000
    res.append(dict(**p, inside=bool(d_km <= 1.0), nearest_km=round(d_km, 2), percentile=round(float(pc[k]), 2) if d_km <= 1.0 else None))
ins = [r for r in res if r["inside"]]; n = len(ins); h10 = sum(r["percentile"] >= 90 for r in ins); h1 = sum(r["percentile"] >= 99 for r in ins)
def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k/n; d = 1 + z*z/n; c = p + z*z/(2*n); s = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); return [round((c-s)/d, 3), round((c+s)/d, 3)]
out = dict(sealed=meta["sealed"], scored=time.strftime("%Y-%m-%d"), cutoff_id=cutoff, n_notice_points=len(pts), n_inside_forecast=n, n_unique_messages_inside=len(set(r["id"] for r in ins)),
           top_decile_hits=h10, top_decile_share=round(h10/n, 3) if n else None, top_decile_ci95=wilson(h10, n), chance_decile=0.10,
           top_percentile_hits=h1, top_percentile_share=round(h1/n, 3) if n else None, top_percentile_ci95=wilson(h1, n), chance_percentile=0.01,
           median_percentile=round(float(np.median([r["percentile"] for r in ins])), 1) if n else None)
# per-message (a notice with several positions counts once: its best-scored position)
msg = {}
for r in ins: msg[r["id"]] = max(msg.get(r["id"], -1), r["percentile"])
m = len(msg); m10 = sum(v >= 90 for v in msg.values()); m1 = sum(v >= 99 for v in msg.values())
out.update(n_messages_inside=m, msg_top_decile_hits=m10, msg_top_decile_share=round(m10/m, 3) if m else None, msg_top_decile_ci95=wilson(m10, m), msg_top_percentile_hits=m1, msg_top_percentile_share=round(m1/m, 3) if m else None, msg_top_percentile_ci95=wilson(m1, m), points=res)
json.dump(out, open(f"prospective_score_{meta['sealed']}.json", "w"), indent=1); print({k: v for k, v in out.items() if k != "points"})
