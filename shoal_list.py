#!/usr/bin/env python3
"""The Shoal List: suspected uncharted keel-depth shoals in Canadian shipping water.

Scan the national hazard field (hazard_nat/, from the 34.8M shallowest-point model over all
NONNA-100 blocks) for cells where
  - the mean map says navigable: completed depth deeper than 21 m (2x a 10.5 m draft)
  - the hazard model disagrees: P(shallowest point within 500 m < 10.5 m) >= 0.5
  - the claim is in-envelope: within 6 km of a published sounding
  - and no published sounding within 500 m already shows < 21 m (else it is charted, not a discovery)
Cluster connected cells into candidate shoals, rank by hazard mass, keep the top 40, attach
the nearest named grounding-region/strait via a small gazetteer, seal with SHA-256 + OTS.
usage: python3 shoal_list.py -> shoal_list_<date>.csv / shoal_list_manifest.json
"""
import csv, datetime, glob, hashlib, json, math, os, numpy as np
from scipy import ndimage as ndi
from v5_data import lat_of_y, lon_of_x
GAZ = [("Simpson Strait", -97.0, 68.57), ("Chesterfield Inlet", -93.4, 63.85), ("Franklin Strait", -96.9, 71.0),
       ("Coronation Gulf", -112.0, 68.1), ("Gulf of Boothia", -91.0, 69.9), ("Hudson Strait", -70.5, 61.8),
       ("Foxe Basin", -79.0, 66.5), ("Frobisher Bay", -67.5, 63.2), ("Cumberland Sound", -65.8, 65.9),
       ("Hudson Bay, Churchill approach", -93.0, 58.9), ("Rankin Inlet approach", -92.0, 62.8),
       ("James Bay", -80.5, 53.0), ("Labrador coast", -60.5, 56.5), ("Ungava Bay", -67.5, 59.5),
       ("Lancaster Sound", -84.0, 74.1), ("Dease Strait", -107.0, 68.9), ("Queen Maud Gulf", -102.0, 68.4),
       ("Peel Sound", -96.0, 73.0), ("Bellot Strait", -91.9, 72.0), ("Cambridge Bay approach", -105.1, 69.0),
       ("Pond Inlet", -77.9, 72.7), ("Arctic Bay / Admiralty Inlet", -85.0, 73.0), ("Iqaluit approach", -68.4, 63.6)]
def nearest_place(lon, lat):
    d = [((lon-a)**2*math.cos(math.radians(lat))**2 + (lat-b)**2, n) for n, a, b in GAZ]
    dd, n = min(d); return n, math.sqrt(dd)*111.0
rows = []
for hf in sorted(glob.glob("hazard_nat/*.npz")):
    name = os.path.basename(hf)
    cf = f"national_v5_out/{name}"
    if not os.path.exists(cf): cf = f"corridor_out_v2/{name}"
    if not os.path.exists(cf): continue
    hd = np.load(hf, allow_pickle=True); p = hd["p105"].astype("float32"); shoal = hd["shoal"].astype("float32"); bb = hd["bbox3857"]
    cd = np.load(cf, allow_pickle=True); c = cd["complete"].astype("float32"); k = cd["known"].astype(bool)
    z = np.load(f"tiles_nat/{name}", allow_pickle=True)["z"].astype("float32")
    H, W = p.shape
    if c.shape != p.shape: continue
    dist_px = ndi.distance_transform_edt(~k)
    charted_shallow = ndi.maximum_filter(np.where(np.isfinite(z), -z, 0), size=11) > 0
    charted_shallow &= ndi.maximum_filter(np.where(np.isfinite(z), z, -1e9), size=11) > -21   # a sounding <21 m within ~500 m
    cand = np.isfinite(p) & (p >= 0.5) & np.isfinite(c) & (c < -21) & (dist_px <= 60) & ~charted_shallow
    if not cand.any(): continue
    lab, nl = ndi.label(cand)
    x0, x1 = min(bb[0], bb[2]), max(bb[0], bb[2]); y0, y1 = min(bb[1], bb[3]), max(bb[1], bb[3])
    for li in range(1, nl+1):
        m = lab == li
        if m.sum() < 3: continue                       # ≥ 3 cells ≈ 0.03 km²
        pm = np.where(m, p, 0); i, j = np.unravel_index(np.argmax(pm), pm.shape)
        lon = lon_of_x(x0 + (j+0.5)/W*(x1-x0)); lat = lat_of_y(y1 - (i+0.5)/H*(y1-y0))
        place, dkm = nearest_place(lon, lat)
        rows.append(dict(lat=round(lat, 5), lon=round(lon, 5), p_peak=round(float(p[i, j]), 3), cells=int(m.sum()),
                         hazard_mass=round(float(p[m].sum()), 1), predicted_shoal_m=round(float(shoal[i, j]), 1),
                         mean_map_depth_m=round(float(c[i, j]), 1), km_to_sounding=round(float(dist_px[i, j])*0.1, 1),
                         region=place, km_to_region_ref=round(dkm, 0), block=name[:-4]))
rows.sort(key=lambda r: -r["hazard_mass"])
top = rows[:40]
date = datetime.date.today().isoformat(); fn = f"shoal_list_{date}.csv"
with open(fn, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(top[0].keys())); w.writeheader(); w.writerows(top)
h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
json.dump(dict(file=fn, sha256=h, date=date, n_candidates_total=len(rows), n_sealed=len(top),
               criteria="mean map deeper than 21 m; P(shallowest point within 500 m < 10.5 m) >= 0.5; within 6 km of a published sounding; no published sounding shallower than 21 m within ~500 m; cluster >= 3 cells",
               model="SeabedNet hazard-small (34.8M) over the full NONNA-100 archive (snapshot 2026-08-29)",
               rule="Each entry is a falsifiable claim: a survey line over the position either finds water shallower than ~21 m within 500 m or it does not. Scored per entry; results published either way."),
          open("shoal_list_manifest.json", "w"), indent=1)
print(f"SHOAL_LIST: {len(rows)} candidate clusters nationally; sealed top {len(top)}; sha256 {h[:16]}")
for r in top[:12]: print(f"  {r['region']:32s} {r['lat']:.3f}N {abs(r['lon']):.3f}W  P={r['p_peak']:.2f} shoal~{r['predicted_shoal_m']} m mean-map {r['mean_map_depth_m']} m mass {r['hazard_mass']}")
print("SHOAL_DONE")
