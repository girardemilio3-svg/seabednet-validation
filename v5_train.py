#!/usr/bin/env python3
"""v5 training: masked-completion NLL over the multi-resolution corpus.

Physics-anchored normalization (gravity-channel stats), resolution-conditioned,
bf16, cosine LR, checkpoint/resume. Eval: geographic holdouts, MAE in metres,
with the gravity prior itself as the always-reported physics baseline.
Env: V5_SIZE=tiny|small|base  V5_STEPS  V5_BATCH
"""
import math, os, time, numpy as np, torch, torch.nn.functional as F
from v5_data import Corpus
from v5_model import V5, normalize

DEV = "cuda"
SIZE = os.environ.get("V5_SIZE", "tiny")
STEPS = int(os.environ.get("V5_STEPS", "4000"))
BATCH = int(os.environ.get("V5_BATCH", "16"))
P = 256
CKPT = os.environ.get("V5_CKPT", f"v5_{SIZE}.pt")
torch.manual_seed(0)
rng = np.random.default_rng(0)
torch.zeros(8, device=DEV)   # claim CUDA context BEFORE corpus preload (unified pool)
corpus = Corpus(P=P)
_POST_CORPUS_HOOKS = True

# ------------------------------------------------------------------ masks
# Real-gap masks are harvested ONCE into a bank (rejection sampling per-sample
# was the input bottleneck: data-starve 86% at step 100 of first tiny run).
_GAPS = []
def _build_gap_bank(n=600):
    t0 = __import__("time").time()
    while len(_GAPS) < n:
        s = corpus.sample(min_valid=0.15)
        if s and 0.2 < s["known"].mean() < 0.85:
            _GAPS.append(s["known"].copy())
    print(f"gap-mask bank: {len(_GAPS)} patterns ({__import__('time').time()-t0:.0f}s)", flush=True)

def gap_mask():
    return _GAPS[rng.integers(len(_GAPS))].copy()

def block_mask():
    m = np.ones((P, P), np.float32)
    for _ in range(rng.integers(2, 6)):
        h, w = rng.integers(P//6, P//2, 2)
        i, j = rng.integers(P-h), rng.integers(P-w)
        m[i:i+h, j:j+w] = 0
    return m

def half_mask():
    m = np.ones((P, P), np.float32); ang = rng.random()*math.pi
    yy, xx = np.mgrid[0:P, 0:P]
    c = (xx-P/2)*math.cos(ang)+(yy-P/2)*math.sin(ang)
    m[c > rng.uniform(-P/6, P/6)] = 0
    return m

def make_mask():
    r = rng.random()
    return gap_mask() if r < 0.45 else (block_mask() if r < 0.75 else half_mask())

# ------------------------------------------------------------------ batches
def get_batch(B=BATCH, holdout=False, center_mask=False):
    ds, ks, gs, ms, ridx = [], [], [], [], []
    while len(ds) < B:
        want10 = rng.random() < 0.3 and corpus.n10 > 20
        s = corpus.sample(min_valid=0.9, want_res=10.0 if want10 else 100.0,
                          holdout=holdout)
        if s is None:
            s = corpus.sample(min_valid=0.9, holdout=holdout)
        if s is None: continue
        if center_mask:
            m = np.ones((P, P), np.float32); h = int(P*0.6); i0 = (P-h)//2
            m[i0:i0+h, i0:i0+h] = 0
        else:
            m = make_mask()
        ds.append(s["depth"]); ks.append(s["known"]); gs.append(s["gravity"])
        ms.append(m); ridx.append(1 if s["res_m"] == 10.0 else 0)
    t = lambda a: torch.tensor(np.stack(a))[:, None].float().to(DEV)
    return (t(ds), t(ks), t(gs), t(ms),
            torch.tensor(ridx, device=DEV, dtype=torch.long))

# ------------------------------------------------------------------ setup
_build_gap_bank()
net = V5(SIZE).to(DEV).to(memory_format=torch.channels_last)
COMPILE = os.environ.get("V5_COMPILE", "0") == "1"
run_net = torch.compile(net, mode="max-autotune") if COMPILE else net
print(f"v5-{SIZE}: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params, "
      f"corpus {len(corpus.entries)} files ({corpus.n10} @10m), "
      f"channels_last on, compile={'ON' if COMPILE else 'off'}")
opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
step0 = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=DEV)
    net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"]); step0 = ck["step"]
    print(f"resumed @ {step0}")

# ---------------------------------------------------------- memory ceiling
# Absolute cap doctrine: the box NEVER goes down again. CUDA allocator capped;
# a guard thread checkpoints + exits if system MemAvailable dips under floor.
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("V5_CUDA_FRAC", "0.80")))
_MIN_AVAIL_GB = float(os.environ.get("V5_MIN_AVAIL_GB", "10"))
def _memguard():
    import threading as _th, time as _t, os as _os
    while True:
        avail = None
        with open("/proc/meminfo") as fh:
            for ln in fh:
                if ln.startswith("MemAvailable"):
                    avail = int(ln.split()[1]) / 1e6
                    break
        if avail is not None and avail < _MIN_AVAIL_GB:
            print(f"MEMGUARD: MemAvailable {avail:.1f} GB < {_MIN_AVAIL_GB} GB floor "
                  f"— checkpointing and exiting cleanly", flush=True)
            try:
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step0, "size": SIZE},
                           CKPT + ".memguard")
            except Exception: pass
            for w in globals().get("_workers", []):    # kill children FIRST —
                try: w.terminate()                     # os._exit skips cleanup
                except Exception: pass                 # (root cause of crash #2)
            _t.sleep(2)
            _os._exit(97)
        _t.sleep(3)
import threading as _thg
_thg.Thread(target=_memguard, daemon=True).start()

# ------------------------------------------------------------------ train
import multiprocessing as _mp
_ctx = _mp.get_context("fork")
NWORK = int(os.environ.get("V5_WORKERS", "4"))
BQ = _ctx.Queue(maxsize=8)
def _produce_proc(seed):
    # numpy-only in children — corpus + gap bank shared copy-on-write via fork
    import numpy as _np
    global rng
    rng = _np.random.default_rng(seed)
    corpus.rng = _np.random.default_rng(seed + 1)
    while True:
        ds, ks, gs, ms, ridx = [], [], [], [], []
        while len(ds) < BATCH:
            want10 = rng.random() < 0.3 and corpus.n10 > 20
            smp = corpus.sample(min_valid=0.9, want_res=10.0 if want10 else 100.0)
            if smp is None:
                smp = corpus.sample(min_valid=0.9)
            if smp is None: continue
            m = make_mask()
            ds.append(smp["depth"]); ks.append(smp["known"]); gs.append(smp["gravity"])
            ms.append(m); ridx.append(1 if smp["res_m"] == 10.0 else 0)
        BQ.put((_np.stack(ds), _np.stack(ks), _np.stack(gs), _np.stack(ms),
                _np.array(ridx, dtype=_np.int64)))
_workers = [_ctx.Process(target=_produce_proc, args=(1000+i,), daemon=True)
            for i in range(NWORK)]
for w in _workers: w.start()
print(f"{NWORK} producer processes forked (corpus shared CoW)", flush=True)
def _next_batch():
    a, b, c, dm, r = BQ.get()
    t = lambda arr: torch.tensor(arr)[:, None].float().to(DEV)
    return t(a), t(b), t(c), t(dm), torch.tensor(r, device=DEV, dtype=torch.long)
t0 = time.time(); data_wait = 0.0
ACCUM = int(os.environ.get("V5_ACCUM", "1"))
RESIDUAL = os.environ.get("V5_RESIDUAL", "0") == "1"   # micro-batches per optimizer step
for step in range(step0+1, STEPS+1):
    opt.zero_grad()
    for _acc in range(ACCUM):
        tw = time.time()
        d, k, g, m, ridx = _next_batch()
        data_wait += time.time() - tw
        mvis = k*m
        dn, gn, mu0, sd0 = normalize(d, mvis, g)
        x = torch.cat([dn*mvis, mvis, gn], 1).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, lv = run_net(x, ridx)
        mu = mu.float(); lv = lv.float()
        if RESIDUAL: mu = mu + gn.float()   # far-field defaults to gravity
        y = (d - mu0)/sd0
        hid = k*(1-m)
        nll = (0.5*(lv + (y-mu)**2*torch.exp(-lv))*hid).sum()/(hid.sum()+1)
        vis = (F.l1_loss(mu, y, reduction="none")*mvis).sum()/(mvis.sum()+1)
        loss = (nll + 0.3*vis) / ACCUM
        loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 100 == 0 or step == 1:
        hmae_m = (((mu-y).abs()*sd0)*hid).sum()/(hid.sum()+1)
        el = time.time()-t0
        print(f"[{step:6d}/{STEPS}] nll {nll.item():+.3f}  hidden-MAE {hmae_m.item():6.2f} m  "
              f"({el/max(1,step-step0):.2f}s/it, data-starve {100*data_wait/max(el,1e-9):.0f}%, "
              f"vram-need {torch.cuda.max_memory_allocated()/1e9:.1f}G "
              f"hoard {torch.cuda.memory_reserved()/1e9:.1f}G)",
              flush=True)
    if step % 1000 == 0 or step == STEPS:
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "size": SIZE}, CKPT)

# ------------------------------------------------------------------ eval
for w in _workers: w.terminate()
time.sleep(2)
net.eval()
maes, gmaes, sigs, errs = [], [], [], []
with torch.no_grad():
    for _ in range(24):
        d, k, g, m, ridx = get_batch(B=1, holdout=True, center_mask=True)
        mvis = k*m
        dn, gn, mu0, sd0 = normalize(d, mvis, g)
        mu, lv = net(torch.cat([dn*mvis, mvis, gn], 1), ridx)
        if RESIDUAL: mu = mu.float() + gn.float()
        est = mu.float()*sd0 + mu0
        sig = torch.exp(0.5*lv.float())*sd0
        hid = (k*(1-m)) > 0
        if hid.sum() < 300: continue
        maes.append((est-d).abs()[hid].mean().item())
        gmaes.append((g-d).abs()[hid].mean().item())
        sigs.append(sig[hid].cpu().numpy()); errs.append((est-d).abs()[hid].cpu().numpy())
s = np.concatenate(sigs); e = np.concatenate(errs)
print(f"\n===== v5-{SIZE} HELD-OUT (geographic) =====")
print(f"v5 MAE {np.mean(maes):.2f} m   gravity-prior baseline {np.mean(gmaes):.2f} m")
print(f"sigma corr {np.corrcoef(s, e)[0, 1]:.3f}")
print("V5_DONE")
