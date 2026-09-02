#!/usr/bin/env python3
"""Hazard head: predict the SHALLOWEST depth within 500 m of each 100 m cell (the
quantity a keel meets) from a masked 100 m depth field + gravity. Ground truth from
NONNA-10/100 pairs (build_hazard.py). Predicted everywhere — the sub-grid extreme is
unknown even where a 100 m sounding exists. Env: HZ_SIZE, HZ_STEPS, HZ_BATCH, HZ_CKPT,
HZ_EXCLUDE="lon,lat;lon,lat" (groups containing these points are held out: grounding
sites for the hindcast + geographic holdouts)."""
import glob, math, os, threading, queue, time, numpy as np, torch
from v5_data import GravityPrior, lat_of_y, lon_of_x, HOLDOUT_BBOXES
from v5_model import V5, normalize
DEV = "cuda"; P = 256
SIZE = os.environ.get("HZ_SIZE", "tiny"); STEPS = int(os.environ.get("HZ_STEPS", "3000"))
BATCH = int(os.environ.get("HZ_BATCH", "16")); CKPT = os.environ.get("HZ_CKPT", f"hazard_{SIZE}.pt")
EXCL = [tuple(map(float, p.split(","))) for p in os.environ.get("HZ_EXCLUDE", "").split(";") if p]
MARGIN = 0.4   # deg lon/lat around an excluded point
torch.manual_seed(0); rng = np.random.default_rng(0)
torch.zeros(8, device=DEV)
grav = GravityPrior()
train, held = [], []
for f in sorted(glob.glob("tiles_hz/*.npz")):
    d = np.load(f); bb = d["bbox3857"]
    lo0, lo1 = lon_of_x(min(bb[0], bb[2])), lon_of_x(max(bb[0], bb[2]))
    la0, la1 = lat_of_y(min(bb[1], bb[3])), lat_of_y(max(bb[1], bb[3]))
    H, W = d["z"].shape
    lons = np.linspace(lo0, lo1, W); lats = np.linspace(la1, la0, H)
    G = grav.sample(np.tile(lons, (H, 1)), np.tile(lats[:, None], (1, W))).astype(np.float32)
    rec = dict(z=d["z"], s=d["zshoal"], g=G, name=os.path.basename(f))
    ex = any(lo0-MARGIN <= x <= lo1+MARGIN and la0-MARGIN <= y <= la1+MARGIN for x, y in EXCL)
    ex |= any(not (lo1 < a or lo0 > c or la1 < b or la0 > dd) for a, b, c, dd in HOLDOUT_BBOXES)
    (held if ex else train).append(rec)
print(f"hazard corpus: {len(train)} train groups, {len(held)} held-out (grounding sites + geo holdouts)", flush=True)
print("held:", [r["name"] for r in held], flush=True)

def rand_mask():
    r = rng.random(); m = np.ones((P, P), np.float32)
    if r < 0.4:      # blobs (low-freq noise threshold) ~ real survey gaps
        n = rng.standard_normal((P//16, P//16))
        n = np.kron(n, np.ones((16, 16))); m[n > rng.uniform(-0.8, 0.8)] = 0
    elif r < 0.7:
        for _ in range(rng.integers(2, 6)):
            h, w = rng.integers(P//6, P//2, 2); i, j = rng.integers(P-h), rng.integers(P-w); m[i:i+h, j:j+w] = 0
    else:
        ang = rng.random()*math.pi; yy, xx = np.mgrid[0:P, 0:P]
        c = (xx-P/2)*math.cos(ang)+(yy-P/2)*math.sin(ang); m[c > rng.uniform(-P/6, P/6)] = 0
    return m

def sample(pool):
    for _ in range(200):
        r = pool[rng.integers(len(pool))]; H, W = r["z"].shape
        i, j = rng.integers(H-P), rng.integers(W-P)
        s = r["s"][i:i+P, j:j+P]
        if np.isfinite(s).mean() < 0.35: continue
        z = r["z"][i:i+P, j:j+P]
        return np.nan_to_num(z), np.isfinite(z).astype(np.float32), r["g"][i:i+P, j:j+P], np.nan_to_num(s), np.isfinite(s).astype(np.float32)
    return None

def batch(pool, B):
    out = [[] for _ in range(6)]
    while len(out[0]) < B:
        x = sample(pool)
        if x is None: continue
        for k, v in enumerate(x): out[k].append(v)
        out[5].append(rand_mask())
    t = lambda a: torch.tensor(np.stack(a))[:, None].float().to(DEV)
    return [t(a) for a in out]

Q = queue.Queue(maxsize=6)
def producer():
    while True: Q.put(batch(train, BATCH))
for _ in range(3): threading.Thread(target=producer, daemon=True).start()

net = V5(SIZE).to(DEV).to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
step0 = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=DEV); net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step0 = ck["step"]; print("resumed", step0)
t0 = time.time()
for step in range(step0+1, STEPS+1):
    z, k, g, s, sk, m = Q.get()
    mvis = k*m
    dn, gn, mu0, sd0 = normalize(z, mvis, g)
    x = torch.cat([dn*mvis, mvis, gn], 1).to(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mu, lv = net(x, torch.zeros(len(z), device=DEV, dtype=torch.long))
    mu = mu.float(); lv = lv.float()
    y = (s - mu0)/sd0
    w = sk*(1 + (1-m))                     # hidden cells weighted 2x
    nll = (0.5*(lv + (y-mu)**2*torch.exp(-lv))*w).sum()/(w.sum()+1)
    opt.zero_grad(); nll.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    if step % 100 == 0 or step == 1:
        mae = (((mu-y).abs()*sd0)*sk).sum()/(sk.sum()+1)
        print(f"[{step:5d}/{STEPS}] nll {nll.item():+.3f} shoal-MAE {mae.item():6.2f} m ({(time.time()-t0)/max(1,step-step0):.2f}s/it)", flush=True)
    if step % 500 == 0 or step == STEPS:
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "size": SIZE}, CKPT)
# ---- eval on held-out groups: shoal MAE, and the naive baselines (mean field as shoal; gravity)
net.eval(); res = dict(model=[], naive=[], grav=[], hid_model=[], hid_naive=[], cov=[], cov2=[])
with torch.no_grad():
    for _ in range(60):
        z, k, g, s, sk, m = batch(held, 4); mvis = k*m
        dn, gn, mu0, sd0 = normalize(z, mvis, g)
        mu, lv = net(torch.cat([dn*mvis, mvis, gn], 1), torch.zeros(len(z), device=DEV, dtype=torch.long))
        est = mu.float()*sd0 + mu0; sig = torch.exp(0.5*lv.float())*sd0
        vis = (sk*mvis) > 0; hid = (sk*(1-m)*k) > 0
        res["model"].append((est-s).abs()[vis].mean().item()); res["naive"].append((z-s).abs()[vis].mean().item()); res["grav"].append((g-s).abs()[vis].mean().item())
        if hid.sum() > 100:
            res["hid_model"].append((est-s).abs()[hid].mean().item()); res["hid_naive"].append((z-s).abs()[hid].mean().item())
        res["cov"].append(((est-s).abs() <= sig)[sk > 0].float().mean().item()); res["cov2"].append(((est-s).abs() <= 2*sig)[sk > 0].float().mean().item())
print("\n===== HAZARD held-out (grounding-site + geographic groups) =====")
print(f"shoal MAE where a 100 m sounding EXISTS: model {np.mean(res['model']):.2f} m | '100 m sounding is the shoal' {np.mean(res['naive']):.2f} m | gravity {np.mean(res['grav']):.2f} m")
print(f"shoal MAE where NO sounding (hidden):     model {np.mean(res['hid_model']):.2f} m | nearest-field {np.mean(res['hid_naive']):.2f} m")
print(f"σ coverage 1σ {np.mean(res['cov'])*100:.0f}%  2σ {np.mean(res['cov2'])*100:.0f}%")
print("HAZARD_DONE")
