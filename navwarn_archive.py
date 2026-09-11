#!/usr/bin/env python3
"""Fetch every cancelled NAVWARN (ids from recon_navwarn/cancelled_ids.json), parse category / date / positions / depths exactly
as the live set was parsed, and write recon_navwarn/cancelled_danger_points.csv. Resumable: messages are cached in
recon_navwarn/cancelled_msgs.jsonl."""
import json, re, html, os, time, csv, requests
from concurrent.futures import ThreadPoolExecutor
B = "https://nis.ccg-gcc.gc.ca/public/rest/messages/en/message/"
ids = json.load(open("recon_navwarn/cancelled_ids.json")); cache = "recon_navwarn/cancelled_msgs.jsonl"
have = set()
if os.path.exists(cache):
    for line in open(cache):
        try: have.add(json.loads(line)["id"])
        except Exception: pass
todo = [i for i in ids if i not in have]; print("ids", len(ids), "cached", len(have), "to fetch", len(todo), flush=True)
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0 (research; seabednet)"
def fetch(mid):
    for k in range(3):
        try:
            t = S.get(B + mid, timeout=60).text
            mt = html.unescape(re.sub(r"<[^>]+>", " ", t)); mt = re.sub(r"\s+", " ", mt)
            i = mt.find("NAVWARN details Navigational Warnings"); mt = mt[i:] if i >= 0 else mt
            j = mt.find("Date Modified"); mt = mt[:j] if j > 0 else mt
            js = re.search(r"var data = (\{.*?\});", t); coords = []
            if js:
                try:
                    for f in json.loads(js.group(1))["features"]: coords += [list(c) for c in f["coordinates"]]
                except Exception: pass
            return dict(id=mid, text=mt[:6000], coords=coords)
        except Exception: time.sleep(1)
    return dict(id=mid, text="", coords=[])
t0 = time.time(); n = 0
with open(cache, "a") as fh, ThreadPoolExecutor(12) as ex:
    for m in ex.map(fetch, todo):
        fh.write(json.dumps(m) + "\n"); n += 1
        if n % 5000 == 0: fh.flush(); print(f"fetched {n}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
print("FETCH_DONE", f"{time.time()-t0:.0f}s", flush=True)
# parse
pos = re.compile(r"(\d{2})\s*[°]?\s*(\d{2}(?:\.\d+)?)\s*'?\s*N\s*,?\s*(\d{3})\s*[°]?\s*(\d{2}(?:\.\d+)?)\s*'?\s*W", re.I); num = r"(\d+(?:\.\d+)?)"
def depth_after(s):
    m = re.search(r"(?:least depth\s*(?:of)?\s*|depth of\s*|of\s*|,\s*|at a depth of\s*|)" + num + r"\s*(m\b|metres|meters|mètres)", s, re.I)
    if m: return float(m.group(1))
    m = re.search(num + r"\s*fathoms?(?:\s*,?\s*(?:and\s*)?" + num + r"\s*feet)?", s, re.I)
    if m: return float(m.group(1))*1.8288 + (float(m.group(2))*0.3048 if m.group(2) else 0.0)
    m = re.search(num + r"\s*feet", s, re.I)
    if m: return float(m.group(1))*0.3048
    return None
rows = []; kw = re.compile(r"shoal|shallow|uncharted|rock|submerged object|sandspit|newly", re.I)
for line in open(cache):
    m = json.loads(line); t = m["text"]
    if not t: continue
    g = lambda k, stop: (re.search(k + r"\s+(.*?)\s+" + stop, t) or [None, None])[1]
    sid = g("ID", "Date"); date = (g("Date", "(?:UTC|Status)") or "")[:10]; area = g("Areas", "Categories"); cat = g("Categories", "Description"); desc = g("Description", "Position") or ""
    if not cat or not kw.search(cat): continue
    found = list(pos.finditer(desc))
    if found:
        for k, mm in enumerate(found):
            lat = float(mm.group(1)) + float(mm.group(2))/60; lon = -(float(mm.group(3)) + float(mm.group(4))/60)
            seg = desc[mm.end(): found[k+1].start() if k+1 < len(found) else mm.end()+80]
            rows.append(dict(id=m["id"], series=sid, date=date, category=cat, area=area, lat=round(lat, 5), lon=round(lon, 5), depth_m=depth_after(seg), src="desc"))
    elif m["coords"]:
        rows.append(dict(id=m["id"], series=sid, date=date, category=cat, area=area, lat=round(m["coords"][0][1], 5), lon=round(m["coords"][0][0], 5), depth_m=depth_after(desc), src="json"))
with open("recon_navwarn/cancelled_danger_points.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
from collections import Counter
print("cancelled danger points:", len(rows), "| with depth:", sum(1 for r in rows if r["depth_m"] is not None), "| Arctic:", sum(1 for r in rows if (r["series"] or "").startswith("NW-A")), "| years:", sorted(Counter(r["date"][:4] for r in rows).items()))
print("ARCHIVE_DONE")
