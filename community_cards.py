#!/usr/bin/env python3
"""Community resupply-lane risk cards.

For each sealift community: a 15 km-radius approach zone (clipped to water cells present in
NONNA blocks). Reported per community: % of water cells with a published sounding, median
survey year of those soundings (CHS Survey Index; post-index = 2017+), % from surveys before
1980, CATZOC mix, and where the corridor hazard field covers it, the % of apparently-safe
water flagged P(shoal<10.5 m) > 5 %. -> community_cards.json
"""
import glob, json, math, os, numpy as np
from v5_data import lat_of_y, lon_of_x
R = 6378137.0
xl = lambda lo: R*math.radians(lo); yl = lambda la: R*math.log(math.tan(math.pi/4+math.radians(la)/2))
COMM = [("Churchill MB", -94.19, 58.77), ("Arviat", -94.06, 61.10), ("Whale Cove", -92.58, 62.18),
        ("Rankin Inlet", -92.08, 62.82), ("Chesterfield Inlet", -90.70, 63.34), ("Baker Lake (inlet route)", -93.5, 63.9),
        ("Coral Harbour", -83.17, 64.14), ("Naujaat", -86.25, 66.53), ("Kugaaruk", -89.83, 68.53),
        ("Gjoa Haven", -95.88, 68.63), ("Taloyoak", -93.53, 69.54), ("Kugluktuk", -115.10, 67.83),
        ("Cambridge Bay", -105.06, 69.12), ("Iqaluit", -68.51, 63.75), ("Kimmirut", -69.87, 62.85),
        ("Pangnirtung", -65.71, 66.15), ("Qikiqtarjuaq", -64.03, 67.56), ("Clyde River", -68.59, 70.47),
        ("Pond Inlet", -77.96, 72.70), ("Arctic Bay", -85.15, 73.03), ("Resolute", -94.83, 74.70),
        ("Sanikiluaq", -79.23, 56.54), ("Puvirnituq", -77.28, 60.04), ("Salluit", -75.63, 62.20),
        ("Deception Bay (Raglan)", -74.6, 62.13), ("Milne Inlet (Baffinland)", -80.88, 71.88)]
blocks = []
for f in sorted(glob.glob("tiles_nat/*.npz")):
    bb = np.load(f, allow_pickle=True)["bbox3857"]
    blocks.append((f, min(bb[0],bb[2]), max(bb[0],bb[2]), min(bb[1],bb[3]), max(bb[1],bb[3])))
ZOCN = {1: "A1", 2: "A2", 3: "B", 4: "C", 5: "D", 6: "U"}
cards = []
RAD = 15000.0
for name, lo, la in COMM:
    cx, cy = xl(lo), yl(la); m = math.cos(math.radians(la))
    tot_water = snd = pre80 = 0; yrs = []; zocs = {}; flag = safe = 0
    for f, x0, x1, y0, y1 in blocks:
        if cx + RAD/m < x0 or cx - RAD/m > x1 or cy + RAD/m < y0 or cy - RAD/m > y1: continue
        d = np.load(f, allow_pickle=True); z = d["z"].astype("float32"); H, W = z.shape
        ip = f"index_out/{os.path.basename(f)}"
        yr = np.load(ip)["year"] if os.path.exists(ip) else np.zeros_like(z, np.int16)
        zc = np.load(ip)["zoc"] if os.path.exists(ip) else np.zeros_like(z, np.int8)
        xs = np.linspace(x0, x1, W); ys = np.linspace(y1, y0, H)
        JJ, II = np.meshgrid(np.arange(W), np.arange(H))
        dist = np.sqrt((xs[JJ]-cx)**2 + (ys[II]-cy)**2) * m
        zone = dist <= RAD
        if not zone.any(): continue
        k = np.isfinite(z)
        # water = sounded cells + completed cells (proxy for water where we can say anything)
        cf = f.replace("tiles_nat", "corridor_out_v2")
        if not os.path.exists(cf): cf = f.replace("tiles_nat", "national_v5_out")
        comp = None
        if os.path.exists(cf):
            cd = np.load(cf, allow_pickle=True); comp = cd["complete"].astype("float32")
        water = zone & (k | (np.isfinite(comp) if comp is not None else False))
        tot_water += int(water.sum()); ksel = water & k
        snd += int(ksel.sum())
        y_in = yr[ksel]; y_eff = np.where(y_in == 0, 2017, y_in)
        yrs.extend(y_eff.tolist()); pre80 += int((y_eff < 1980).sum())
        for code, nm2 in ZOCN.items(): zocs[nm2] = zocs.get(nm2, 0) + int(((zc[ksel] == code)).sum())
        hf = f"hazard_out/{os.path.basename(f)}"
        if os.path.exists(hf) and comp is not None:
            hd = np.load(hf, allow_pickle=True); p = hd["p105"].astype("float32")
            sm = water & np.isfinite(comp) & (comp < -21) & np.isfinite(p)
            safe += int(sm.sum()); flag += int((p[sm] > 0.05).sum())
    if tot_water < 100: continue
    card = dict(community=name, lat=la, lon=lo, water_cells=tot_water,
                pct_sounded=round(100*snd/max(tot_water, 1), 1),
                median_survey_year=int(np.median(yrs)) if yrs else None,
                pct_pre1980=round(100*pre80/max(snd, 1), 1),
                catzoc={k2: v for k2, v in sorted(zocs.items()) if v})
    if safe > 200: card["pct_safe_water_flagged"] = round(100*flag/safe, 1)
    cards.append(card)
    print(f"{name:26s} sounded {card['pct_sounded']:5.1f}%  median yr {card['median_survey_year']}  pre-1980 {card['pct_pre1980']:5.1f}%" + (f"  hazard-flagged {card.get('pct_safe_water_flagged')}%" if 'pct_safe_water_flagged' in card else ""), flush=True)
json.dump(dict(radius_km=15, note="median_survey_year uses 2017 for soundings outside the 1832-2016 CHS Survey Index (i.e. modern)", cards=cards), open("community_cards.json", "w"), indent=1)
print("CARDS_DONE", len(cards))
