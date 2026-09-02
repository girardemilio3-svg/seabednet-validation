#!/usr/bin/env python3
"""Apply the validation findings to the corridor completion:
 - depth gate: inferred cells deeper than 400 m defer to the gravity prior (independent test:
   gravity 4.0-9.7 m vs model 20-31 m there); 300-400 m blends 50/50
 - calibrated σ (68 % / 95 %) from sigma_calibration.json (isotonic on independent cells)
 -> corridor_out_v2/<block>.npz (complete, sigma_raw, sigma68, sigma95, known, gated, bbox3857)"""
import glob, json, os, numpy as np
from v5_data import GravityPrior, lat_of_y, lon_of_x
cal = json.load(open("sigma_calibration.json")); g = np.array(cal["sigma_raw_grid"]); s68 = np.array(cal["sigma68"]); s95 = np.array(cal["sigma95"])
grav = GravityPrior(); os.makedirs("corridor_out_v2", exist_ok=True)
tot = dict(filled=0, gated=0, blended=0)
for f in sorted(glob.glob("corridor_out/*.npz")):
    if os.path.exists(f.replace("corridor_out", "corridor_out_v2")): continue
    d = np.load(f, allow_pickle=True); c = d["complete"].astype("float32"); k = d["known"].astype(bool); s = d["sigma"].astype("float32"); bb = d["bbox3857"]
    H, W = c.shape
    lons = np.array([lon_of_x(x) for x in np.linspace(min(bb[0],bb[2]), max(bb[0],bb[2]), W)]); lats = np.array([lat_of_y(y) for y in np.linspace(max(bb[1],bb[3]), min(bb[1],bb[3]), H)])
    G = grav.sample(np.tile(lons, (H,1)), np.tile(lats[:,None], (1,W))).astype("float32")
    fill = np.isfinite(c) & ~k
    deep = fill & (c < -400); mid = fill & (c >= -400) & (c < -300)
    c2 = c.copy(); c2[deep] = G[deep]; c2[mid] = 0.5*(c[mid] + G[mid])
    sig68 = np.where(fill, np.interp(s, g, s68), 0).astype("float16"); sig95 = np.where(fill, np.interp(s, g, s95), 0).astype("float16")
    sig68[deep] = 6.3; sig95[deep] = 15.0      # gravity-prior error in >1000 m independent cells (MAE 6.3 m)
    np.savez_compressed(f.replace("corridor_out", "corridor_out_v2"), complete=c2.astype("float16"), sigma_raw=s.astype("float16"),
                        sigma68=sig68, sigma95=sig95, known=k, gated=(deep|mid), bbox3857=bb)
    tot["filled"] += int(fill.sum()); tot["gated"] += int(deep.sum()); tot["blended"] += int(mid.sum())
print(tot, "GATE_DONE")
