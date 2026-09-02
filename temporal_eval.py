#!/usr/bin/env python3
"""Temporal holdout: a model trained ONLY on soundings inside the CHS Survey Index
(1832-2016) predicts the soundings CHS published after the index era (post-2016
multibeam). Input = pre-2016 soundings; target = post-2016 soundings; the model has
never seen a single target pixel. Baselines: nearest pre-2016 sounding; gravity prior.
Stratified by distance to the nearest pre-2016 sounding and by depth.

usage: V5_SIZE=tiny V5_CKPT=v5_tiny_temporal.pt python3 temporal_eval.py corridor_blocks.txt
"""
import json, math, os, sys, numpy as np, torch
from scipy import ndimage as ndi
from v5_data import GravityPrior, lat_of_y, lon_of_x
from v5_model import V5, normalize

DEV = "cuda"; P = 256; S = 128; BW = 16
SIZE = os.environ.get("V5_SIZE", "tiny"); CKPT = os.environ.get("V5_CKPT", f"v5_{SIZE}_temporal.pt")
TAG = os.environ.get("TAG", os.path.basename(CKPT)[:-3])
MAXPX = 60                      # same fill discipline as the atlas (≤ 60 px from a sounding)
BINS = [0, 0.5, 1, 2, 4, 8, 1e9]
net = V5(SIZE).to(DEV)
ck = torch.load(CKPT, map_location=DEV, weights_only=False)
net.load_state_dict(ck["net"]); net.eval()
RESIDUAL = os.environ.get("V5_RESIDUAL", "0") == "1"
grav = GravityPrior()
files = [l.strip() for l in open(sys.argv[1]) if l.strip()]
print(f"{CKPT} @ step {ck['step']} — {len(files)} blocks", flush=True)
win = np.outer(np.hanning(P), np.hanning(P)) + 1e-3

rows = []; blocks = {}
for n, f in enumerate(files):
    ip = f"index_out/{os.path.basename(f)}"
    if not os.path.exists(ip): continue
    d = np.load(f, allow_pickle=True); z = d["z"].astype("float32"); bbox = d["bbox3857"]
    yr = np.load(ip)["year"]
    H, W = z.shape
    allk = np.isfinite(z)
    pre = allk & (yr > 0); post = allk & (yr == 0)
    if pre.sum() < 500 or post.sum() < 100: continue
    known = pre.astype(np.float32); zin = np.where(pre, z, np.nan)
    x0, y0, x1, y1 = bbox
    lons = np.array([lon_of_x(x) for x in np.linspace(min(x0,x1), max(x0,x1), W)])
    lats = np.array([lat_of_y(y) for y in np.linspace(max(y0,y1), min(y0,y1), H)])
    G = grav.sample(np.tile(lons, (H,1)), np.tile(lats[:,None], (1,W))).astype("float32")
    Hp = (H//S+3)*S; Wp = (W//S+3)*S
    zp = np.full((Hp,Wp), np.nan, np.float32); zp[:H,:W] = zin
    kp = np.zeros((Hp,Wp), np.float32); kp[:H,:W] = known
    gp = np.zeros((Hp,Wp), np.float32); gp[:H,:W] = G
    gp[H:,:] = G[-1:,:].mean(); gp[:,W:] = gp[:,W-1:W]
    accm = np.zeros((Hp,Wp)); accs = np.zeros((Hp,Wp)); wacc = np.zeros((Hp,Wp))
    coords = [(i,j) for i in range(0,Hp-P+1,S) for j in range(0,Wp-P+1,S) if kp[i:i+P,j:j+P].sum() >= 400]
    with torch.no_grad():
        for b0 in range(0, len(coords), BW):
            cb = coords[b0:b0+BW]
            zb = np.stack([np.nan_to_num(zp[i:i+P,j:j+P]) for i,j in cb])
            kb = np.stack([kp[i:i+P,j:j+P] for i,j in cb])
            gb = np.stack([gp[i:i+P,j:j+P] for i,j in cb])
            t = lambda a: torch.tensor(a)[:,None].float().to(DEV)
            dt, kt, gt = t(zb), t(kb), t(gb)
            dn, gn, mu0, sd0 = normalize(dt, kt, gt)
            ridx = torch.zeros(len(cb), device=DEV, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu, lv = net(torch.cat([dn*kt, kt, gn], 1), ridx)
            mu = mu.float()
            if RESIDUAL: mu = mu + gn.float()
            est = (mu*sd0 + mu0)[:,0].cpu().numpy()
            sig = (torch.exp(0.5*lv.float())*sd0)[:,0].cpu().numpy()
            for q,(i,j) in enumerate(cb):
                accm[i:i+P,j:j+P] += est[q]*win; accs[i:i+P,j:j+P] += sig[q]*win; wacc[i:i+P,j:j+P] += win
    est = np.where(wacc>0, accm/np.maximum(wacc,1e-6), np.nan)[:H,:W]
    sig = np.where(wacc>0, accs/np.maximum(wacc,1e-6), np.nan)[:H,:W]
    dist_px, idx = ndi.distance_transform_edt(~pre, return_indices=True)
    res_m = abs(x1-x0)/W * math.cos(math.radians(float(lats.mean())))
    val = post & np.isfinite(est) & (dist_px <= MAXPX)
    if val.sum() < 50: continue
    nn = zin[idx[0], idx[1]]
    em = est[val]-z[val]; en = nn[val]-z[val]; eg = G[val]-z[val]
    rows.append(np.stack([em, en, eg, sig[val], dist_px[val]*res_m/1000, z[val]], 1).astype("float32"))
    blocks[os.path.basename(f)[:-4]] = dict(n=int(val.sum()), n_pre=int(pre.sum()), n_post=int(post.sum()),
        mae_model=float(np.abs(em).mean()), mae_nn=float(np.abs(en).mean()), mae_grav=float(np.abs(eg).mean()),
        bias=float(em.mean()), med_depth=float(np.median(z[val])))
    print(f"[{n+1}/{len(files)}] {os.path.basename(f)[:-4]} pre={pre.sum():7d} post={post.sum():7d} val={val.sum():7d} "
          f"model {np.abs(em).mean():5.1f} nn {np.abs(en).mean():5.1f} grav {np.abs(eg).mean():5.1f} bias {em.mean():+5.1f} depth~{np.median(z[val]):.0f}", flush=True)

A = np.concatenate(rows); np.save(f"temporal_cells_{TAG}.npy", A)
em, en, eg, sg, dk, dep = A.T
def st(sel): return dict(n=int(sel.sum()), mae_model=float(np.abs(em[sel]).mean()), mae_nn=float(np.abs(en[sel]).mean()),
                         mae_grav=float(np.abs(eg[sel]).mean()), bias=float(em[sel].mean()), mean_sigma=float(sg[sel].mean()),
                         frac_within_1sigma=float((np.abs(em[sel]) <= sg[sel]).mean()))
out = dict(ckpt=CKPT, step=int(ck["step"]), n_cells=int(len(A)), n_blocks=len(blocks), overall=st(np.ones(len(A), bool)),
           by_distance=[dict(km=[a, min(b, 99)], **st((dk>=a)&(dk<b))) for a, b in zip(BINS[:-1], BINS[1:]) if ((dk>=a)&(dk<b)).sum() >= 50],
           by_depth=[dict(m=[a, b], **st((-dep>=a)&(-dep<b))) for a, b in [(0,20),(20,50),(50,100),(100,200),(200,400),(400,9999)] if ((-dep>=a)&(-dep<b)).sum() >= 50],
           blocks=blocks)
json.dump(out, open(f"temporal_validation_{TAG}.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n_cells", "n_blocks", "overall")}, indent=1))
for r in out["by_distance"]: print("dist", r)
for r in out["by_depth"]: print("depth", r)
print("TEMPORAL_DONE")
