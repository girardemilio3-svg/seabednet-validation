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
       ("Pond Inlet", -77.9, 72.7), ("Arctic Bay / Admiralty Inlet", -85.0, 73.0), ("Iqaluit approach", -68.4, 63.6),
       ("Scotian Shelf", -62.0, 44.0), ("Bay of Fundy", -66.0, 45.0), ("Georges Bank", -66.5, 41.5), ("Gulf of St. Lawrence", -62.5, 47.5),
       ("St. Lawrence estuary", -68.5, 48.8), ("Newfoundland south coast", -56.0, 47.2), ("Grand Banks", -50.0, 46.0), ("Strait of Belle Isle", -56.5, 51.5),
       ("Hecate Strait", -131.0, 53.5), ("Haida Gwaii", -132.0, 53.0), ("Vancouver Island west coast", -127.0, 49.5), ("Strait of Georgia", -123.5, 49.3),
       ("Juan de Fuca", -124.0, 48.3), ("Beaufort Sea", -134.0, 70.5), ("Amundsen Gulf", -122.0, 70.5), ("Rae Strait / King William Is.", -95.8, 69.2),
       ("Lake Superior", -87.0, 47.6), ("Lake Huron / Georgian Bay", -81.5, 45.0), ("Lake Erie", -81.0, 42.3), ("Lake Ontario", -77.5, 43.7), ("Lake Winnipeg", -97.0, 52.0)]
def nearest_place(lon, lat):
    d = [((lon-a)**2*math.cos(math.radians(lat))**2 + (lat-b)**2, n) for n, a, b in GAZ]
    dd, n = min(d); km = math.sqrt(dd)*111.0
    return (n if km < 200 else f"unnamed water near {lat:.1f}N {abs(lon):.1f}W"), km
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
    # gravity-anchor sanity: exclude inland lakes / elevated water where the physics prior is invalid
    from v5_data import GravityPrior
    global _GRV
    try: _GRV
    except NameError: _GRV = GravityPrior()
    kk = np.isfinite(z)
    if kk.sum() > 500:
        yi, xi = np.nonzero(kk)
        sel = np.random.default_rng(0).choice(len(yi), min(4000, len(yi)), replace=False)
        x0b, x1b = min(bb[0], bb[2]), max(bb[0], bb[2]); y0b, y1b = min(bb[1], bb[3]), max(bb[1], bb[3])
        lons_s = np.degrees((x0b + (xi[sel]+0.5)/W*(x1b-x0b))/6378137.0)
        lats_s = np.degrees(2*np.arctan(np.exp((y1b - (yi[sel]+0.5)/H*(y1b-y0b))/6378137.0)) - np.pi/2)
        gerr = np.median(np.abs(_GRV.sample(lons_s, lats_s) - z[yi[sel], xi[sel]]))
        if gerr > 30: continue
    dist_px = ndi.distance_transform_edt(~k)
    charted_shallow = ndi.maximum_filter(np.where(np.isfinite(z), -z, 0), size=11) > 0
    charted_shallow &= ndi.maximum_filter(np.where(np.isfinite(z), z, -1e9), size=11) > -21   # a sounding <21 m within ~500 m
    cand = (np.isfinite(p) & (p >= 0.8) & np.isfinite(c) & (c < -21) & (c > -150) & (dist_px <= 60) & ~charted_shallow
            & np.isfinite(shoal) & (shoal > -21) & (shoal < 1.0) & ((shoal - c) < 100))   # physically plausible pinnacle only
    if not cand.any(): continue
    lab, nl = ndi.label(cand)
    x0, x1 = min(bb[0], bb[2]), max(bb[0], bb[2]); y0, y1 = min(bb[1], bb[3]), max(bb[1], bb[3])
    for li in range(1, nl+1):
        m = lab == li
        if m.sum() < 3 or m.sum() > 300: continue      # 0.03–3 km²: a shoal, not a systematic artifact
        pm = np.where(m, p, 0); i, j = np.unravel_index(np.argmax(pm), pm.shape)
        lon = lon_of_x(x0 + (j+0.5)/W*(x1-x0)); lat = lat_of_y(y1 - (i+0.5)/H*(y1-y0))
        place, dkm = nearest_place(lon, lat)
        rows.append(dict(lat=round(lat, 5), lon=round(lon, 5), p_peak=round(float(p[i, j]), 3), cells=int(m.sum()),
                         hazard_mass=round(float(p[m].sum()), 1), predicted_shoal_m=round(float(shoal[i, j]), 1),
                         mean_map_depth_m=round(float(c[i, j]), 1), km_to_sounding=round(float(dist_px[i, j])*0.1, 1),
                         region=place, km_to_region_ref=round(dkm, 0), block=name[:-4]))
rows.sort(key=lambda r: (-round(r["p_peak"], 2), -r["hazard_mass"]))
# geographic diversity: at most 2 entries per 0.25-degree box, so one delta or shelf cannot fill the list
seen = {}; top = []
for r in rows:
    key = (round(r["lon"]*4)/4, round(r["lat"]*4)/4)
    if seen.get(key, 0) >= 2: continue
    seen[key] = seen.get(key, 0) + 1; top.append(r)
    if len(top) == 40: break
date = datetime.date.today().isoformat(); fn = f"shoal_list_{date}.csv"
with open(fn, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(top[0].keys())); w.writeheader(); w.writerows(top)
h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
json.dump(dict(file=fn, sha256=h, date=date, n_candidates_total=len(rows), n_sealed=len(top),
               criteria="mean map 21-150 m deep; P(shallowest point within 500 m < 10.5 m) >= 0.8; predicted shallowest point between the surface and 21 m with relief < 100 m; within 6 km of a published sounding; no published sounding shallower than 21 m within ~500 m; cluster 3-300 cells; blocks where the gravity prior fails (inland lakes) excluded; ranked by peak probability then size",
               model="SeabedNet hazard-small (34.8M) over the full NONNA-100 archive (snapshot 2026-08-29)",
               rule="Each entry is a falsifiable claim: a survey line over the position either finds water shallower than ~21 m within 500 m or it does not. Scored per entry; results published either way."),
          open("shoal_list_manifest.json", "w"), indent=1)
print(f"SHOAL_LIST: {len(rows)} candidate clusters nationally; sealed top {len(top)}; sha256 {h[:16]}")
for r in top[:12]: print(f"  {r['region']:32s} {r['lat']:.3f}N {abs(r['lon']):.3f}W  P={r['p_peak']:.2f} shoal~{r['predicted_shoal_m']} m mean-map {r['mean_map_depth_m']} m mass {r['hazard_mass']}")
print("SHOAL_DONE")
