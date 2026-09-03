#!/usr/bin/env python3
"""Sealed strike-point predictions for the four Arctic groundings known by place name only.

For each incident, take the search area implied by the sources, and output the cell the
hazard model considers most dangerous for that vessel's draft among water the mean map calls
navigable: the model's answer to "where in <place> did it ground?" Sealed before the TSB
occurrence positions are obtained; graded verbatim against them when they arrive.
usage: HZ_CKPT=hazard_small.pt HZ_SIZE=small python3 predict_strikes.py -> strike_predictions_<date>.csv/.json + OTS
"""
import csv, datetime, hashlib, json, math, os, numpy as np, torch
from scipy import ndimage as ndi
from scipy.stats import norm
import hindcast as H
INCIDENTS = [
 dict(name="Nanny 2010", date="2010-09-01", place="Simpson Strait, eastern approach, near Gjoa Haven",
      lon0=-97.9, lon1=-95.3, lat0=68.35, lat1=68.75, draft=9.2,
      note="laden shuttle tanker; sources: 'eastern approach to Simpson Strait', '~50 km southwest of Gjoa Haven'"),
 dict(name="Mokami 2010", date="2010-08-08", place="Pangnirtung harbour approach, Cumberland Sound",
      lon0=-66.4, lon1=-65.2, lat0=66.0, lat1=66.4, draft=5.2,
      note="grounded at low tide in the harbour approach; refloated on the tide"),
 dict(name="Dorsch 2012", date="2012-10-24", place="Regina Narrows, Chesterfield Inlet",
      lon0=-94.2, lon1=-92.6, lat0=63.5, lat1=64.1, draft=4.6,
      note="cited in TSB M14C0219; shuttle tanker, day before the Nanny grounding"),
 dict(name="Rosaire A. Desgagnes 2025", date="2025-08-23", place="Arctic Bay / Adams Sound approach (position disputed: Shipfax says Arctic Bay; other summaries say Pelly Bay)",
      lon0=-85.6, lon1=-84.4, lat0=72.9, lat1=73.2, draft=7.5,
      note="TSB M25C0219 open; sealift to Nanisivik supports the Arctic Bay reading; if TSB says Pelly Bay this prediction is simply wrong and will be reported as such"),
]
rows = []
for inc in INCIDENTS:
    # collect candidate blocks intersecting the search box
    xs = [H.x_of_lon(inc["lon0"]), H.x_of_lon(inc["lon1"])]; ys = [H.y_of_lat(inc["lat0"]), H.y_of_lat(inc["lat1"])]
    best = None
    for (f, x0, x1, y0, y1) in H.blocks:
        if x1 < min(xs) or x0 > max(xs) or y1 < min(ys) or y0 > max(ys): continue
        d = np.load(f, allow_pickle=True); z = d["z"].astype("float32"); Hh, W = z.shape
        known = np.isfinite(z).astype("float32")
        if known.sum() < 500: continue
        lons = np.array([H.lon_of_x(v) for v in np.linspace(x0, x1, W)]); lats = np.array([H.lat_of_y(v) for v in np.linspace(y1, y0, Hh)])
        LO = np.tile(lons, (Hh, 1)); LA = np.tile(lats[:, None], (1, W))
        G = H.grav.sample(LO, LA).astype("float32")
        zin = np.where(known > 0, z, np.nan)
        mu_s, sg_s = H.infer(H.hz, zin, known, G); mu_m, _ = H.infer(H.mean_net, zin, known, G)
        dist_px = ndi.distance_transform_edt(known == 0)
        draft = inc["draft"]
        p = 1 - norm.cdf((-draft - mu_s)/np.maximum(sg_s, 0.5))
        box = (LO >= inc["lon0"]) & (LO <= inc["lon1"]) & (LA >= inc["lat0"]) & (LA <= inc["lat1"])
        # keep clear of charted shoreline/intertidal: no NONNA value shallower than -2 m within ~500 m
        nearshore = ndi.distance_transform_edt(~(np.nan_to_num(z, nan=-1e9) > -2)) <= 5
        # candidate: inside box, inference valid, mean map navigable, away from charted shore,
        # and the predicted shallowest point is groundable water (between surface and the keel), not land
        cand = (box & np.isfinite(p) & np.isfinite(mu_m) & (mu_m < -(draft + 2)) & (dist_px <= 120)
                & ~nearshore & (mu_s > -(draft + 3)) & (mu_s < 1.0))
        if cand.sum() < 10: continue
        pc = np.where(cand, p, -1)
        i, j = np.unravel_index(np.argmax(pc), pc.shape)
        rec = dict(pmax=float(p[i, j]), lon=float(LO[i, j]), lat=float(LA[i, j]), shoal=float(mu_s[i, j]), mean=float(mu_m[i, j]),
                   block=os.path.basename(f)[:-4], n_cand=int(cand.sum()),
                   pct=float((p[cand] < p[i, j]).mean()*100))
        if best is None or rec["pmax"] > best["pmax"]: best = rec
    if best is None:
        rows.append(dict(name=inc["name"], date=inc["date"], place=inc["place"], status="no candidate cells (outside harvest or inference envelope)")); print(inc["name"], "no candidates"); continue
    rows.append(dict(name=inc["name"], date=inc["date"], place=inc["place"], draft_assumed_m=inc["draft"],
                     predicted_lat=round(best["lat"], 5), predicted_lon=round(best["lon"], 5),
                     p_shoal_lt_draft=round(best["pmax"], 4), predicted_shoal_depth_m=round(best["shoal"], 1),
                     mean_map_depth_m=round(best["mean"], 1), candidates_in_search_area=best["n_cand"], note=inc["note"]))
    print(f"{inc['name']:28s} -> {best['lat']:.4f}N {abs(best['lon']):.4f}W  P(shoal<{inc['draft']}m)={best['pmax']:.2f}  shoal {best['shoal']:.1f} m  mean-map {best['mean']:.1f} m  ({best['n_cand']} candidates)", flush=True)
date = datetime.date.today().isoformat()
fn = f"strike_predictions_{date}.csv"
with open(fn, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) | rows[-1].keys() if False else sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
json.dump(dict(file=fn, sha256=h, date=date, model="SeabedNet hazard-small (34.8M) + v5-small mean model; inputs: full published NONNA-100 archive (2026-08-29 snapshot)",
               rule="Graded verbatim against TSB occurrence positions when obtained (ATIP request drafted 2026-09-03). A prediction counts as a hit if the TSB position lies within 2 km; the Rosaire case is void if TSB places it outside the stated search area.",
               incidents=rows), open("strike_predictions_manifest.json", "w"), indent=1)
print("sealed", fn, "sha256", h[:16]); print("STRIKES_DONE")
