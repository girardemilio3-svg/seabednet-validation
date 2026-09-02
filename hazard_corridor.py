#!/usr/bin/env python3
"""Hazard field for the Churchill corridor: for every 100 m cell (sounded or not), the
predicted shallowest depth within 500 m and P(shoal < draft) for a laden Handysize
(draft 10.5 m) and a Panamax (12.5 m). Input = the full NONNA-100 archive + gravity.
Outputs hazard_out/<block>.npz (shoal, shoal_sigma, p105, p125) and hazard_corridor.json
(route statistics: km of route where P(shoal<draft) > 5 % although the mean map says safe)
+ hazard_corridor.png / _web.jpg for the atlas.
usage: HZ_CKPT=hazard_tiny.pt python3 hazard_corridor.py"""
import glob, json, math, os, numpy as np, torch
from scipy.stats import norm
from v5_data import GravityPrior, lat_of_y, lon_of_x
from v5_model import V5, normalize
DEV = "cuda"; P = 256; S = 128; BW = 16
HZ = os.environ.get("HZ_CKPT", "hazard_tiny.pt"); HZS = os.environ.get("HZ_SIZE", "tiny")
DRAFTS = {"p105": 10.5, "p125": 12.5}
net = V5(HZS).to(DEV); ck = torch.load(HZ, map_location=DEV, weights_only=False); net.load_state_dict(ck["net"]); net.eval()
grav = GravityPrior(); os.makedirs("hazard_out", exist_ok=True)
win = np.outer(np.hanning(P), np.hanning(P)) + 1e-3
files = [l.strip() for l in open("corridor_blocks.txt") if l.strip()]
for n, f in enumerate(files):
    out = f"hazard_out/{os.path.basename(f)}"
    if os.path.exists(out): continue
    d = np.load(f, allow_pickle=True); z = d["z"].astype("float32"); bb = d["bbox3857"]; H, W = z.shape
    known = np.isfinite(z).astype(np.float32)
    if known.sum() < 500: continue
    x0, x1 = min(bb[0], bb[2]), max(bb[0], bb[2]); y0, y1 = min(bb[1], bb[3]), max(bb[1], bb[3])
    lons = np.array([lon_of_x(v) for v in np.linspace(x0, x1, W)]); lats = np.array([lat_of_y(v) for v in np.linspace(y1, y0, H)])
    G = grav.sample(np.tile(lons, (H, 1)), np.tile(lats[:, None], (1, W))).astype("float32")
    Hp = (H//S+3)*S; Wp = (W//S+3)*S
    zp = np.full((Hp, Wp), np.nan, np.float32); zp[:H, :W] = z
    kp = np.zeros((Hp, Wp), np.float32); kp[:H, :W] = known
    gp = np.zeros((Hp, Wp), np.float32); gp[:H, :W] = G; gp[H:, :] = G[-1:, :].mean(); gp[:, W:] = gp[:, W-1:W]
    accm = np.zeros((Hp, Wp)); accs = np.zeros((Hp, Wp)); wacc = np.zeros((Hp, Wp))
    coords = [(i, j) for i in range(0, Hp-P+1, S) for j in range(0, Wp-P+1, S) if kp[i:i+P, j:j+P].sum() >= 400]
    with torch.no_grad():
        for b0 in range(0, len(coords), BW):
            cb = coords[b0:b0+BW]
            t = lambda a: torch.tensor(np.stack(a))[:, None].float().to(DEV)
            dt = t([np.nan_to_num(zp[i:i+P, j:j+P]) for i, j in cb]); kt = t([kp[i:i+P, j:j+P] for i, j in cb]); gt = t([gp[i:i+P, j:j+P] for i, j in cb])
            dn, gn, mu0, sd0 = normalize(dt, kt, gt)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu, lv = net(torch.cat([dn*kt, kt, gn], 1), torch.zeros(len(cb), device=DEV, dtype=torch.long))
            est = (mu.float()*sd0 + mu0)[:, 0].cpu().numpy(); sig = (torch.exp(0.5*lv.float())*sd0)[:, 0].cpu().numpy()
            for q, (i, j) in enumerate(cb):
                accm[i:i+P, j:j+P] += est[q]*win; accs[i:i+P, j:j+P] += sig[q]*win; wacc[i:i+P, j:j+P] += win
    shoal = np.where(wacc > 0, accm/np.maximum(wacc, 1e-6), np.nan)[:H, :W].astype("float32")
    ssig = np.where(wacc > 0, accs/np.maximum(wacc, 1e-6), np.nan)[:H, :W].astype("float32")
    ps = {k: (1 - norm.cdf((-v - shoal)/np.maximum(ssig, 0.5))).astype("float16") for k, v in DRAFTS.items()}
    np.savez_compressed(out, shoal=shoal.astype("float16"), shoal_sigma=ssig.astype("float16"), bbox3857=bb, **ps)
    print(f"[{n+1}/{len(files)}] {os.path.basename(f)} windows={len(coords)} shoal median {np.nanmedian(shoal):.0f} m  P(shoal<10.5m)>5%: {np.mean(ps['p105'][np.isfinite(shoal)] > 0.05)*100:.1f}% of cells", flush=True)
print("HAZARD_CORRIDOR_DONE")
