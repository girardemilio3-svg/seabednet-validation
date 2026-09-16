#!/usr/bin/env python3
"""ICESat-2 photon check, version 2: along-track bottom detection with continuity, a Poisson background test, and an afterpulse
guard. Rule fixed before the run:
  surface  = per 20 m along-track bin, median of photons within 1.5 m of the pass-wide surface mode
  bottom   = per 10 m bin, densest 0.5 m depth slice 2.5-40 m below the local surface (the 1.5-2.5 m band is the surface tail / ice),
             with a gap: every slice between 1.0 m below the surface and 1.0 m above the bottom must hold < thr/2 photons, counted a hit if count >= max(4, bg + 5*sqrt(bg))
             where bg is the mean count per 0.5 m slice in the 15-40 m band of the same pass (deep-water photon noise)
  segment  = >= 6 consecutive hit bins (60 m) whose depths change < 1.5 m bin to bin
  afterpulse guard: a segment with median depth 3.3-5.0 m AND depth std < 0.4 m is discarded (fixed-offset detector echo)
  pass verdict: bottom seen = at least one surviving segment; depth = median of its bin depths
  claim SUPPORTED iff >= 2 passes see a bottom shallower than 0.6 x chart depth, and the control false-positive rate < 5 %
usage: python3 is2_v2.py -> is2_v2.json"""
import glob, json, math, os, numpy as np
CL = json.load(open("is2_claims.json")); ctl = json.load(open("is2_controls.json")) if os.path.exists("is2_controls.json") else []
def passes(npz):
    a = np.load(npz, allow_pickle=True); h, lat, lon, trk = a["h"], a["lat"], a["lon"], a["trk"]
    for t in np.unique(trk):
        m = (trk == t) & np.isfinite(h)
        if m.sum() < 300: continue
        yield t, h[m], lat[m], lon[m]
def analyse(t, h, lat, lon):
    lo, hi = np.percentile(h, [5, 95]); cnt, edges = np.histogram(h, np.arange(lo, hi + 0.25, 0.25))
    if cnt.max() < 30: return None
    surf0 = edges[np.argmax(cnt)] + 0.125
    lat0 = lat.mean(); x = (lon - lon.mean())*111000*math.cos(math.radians(lat0)); y = (lat - lat0)*111000
    v = np.array([x, y]); c = np.cov(v); w, e = np.linalg.eigh(c); d = e[:, np.argmax(w)]; s = x*d[0] + y*d[1]; s -= s.min()
    nb = int(s.max()//10) + 1; hits = np.full(nb, np.nan)
    surf_bin = np.full(nb, np.nan)
    for k in range(nb):
        mm = (s >= k*10 - 5) & (s < k*10 + 15) & (np.abs(h - surf0) < 1.5)
        if mm.sum() >= 5: surf_bin[k] = np.median(h[mm])
    ok = np.isfinite(surf_bin)
    if ok.sum() < 3: return None
    surf_bin[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), surf_bin[ok])
    dep = np.full(len(h), np.nan); kb = np.clip((s//10).astype(int), 0, nb - 1); dep = surf_bin[kb] - h
    deep = dep[(dep >= 15) & (dep <= 40)]; nslice = 50; bg = len(deep)/max(nslice*nb, 1)     # mean photons per 0.5 m slice per 10 m bin
    thr = max(4.0, bg + 5*math.sqrt(max(bg, 1e-6)))
    for k in range(nb):
        mm = (kb == k) & (dep >= 1.0) & (dep <= 40)
        if mm.sum() < 4: continue
        c2, e2 = np.histogram(dep[mm], np.arange(1.0, 40.5, 0.5)); cand = c2.copy(); cand[e2[:-1] < 2.5] = 0; p = int(np.argmax(cand))
        gap = c2[(e2[:-1] >= 1.0) & (e2[:-1] < e2[p] - 1.0)]
        if cand[p] >= thr and (len(gap) == 0 or gap.max() < thr/2): hits[k] = e2[p] + 0.25
    segs = []; cur = []
    for k in range(nb):
        if np.isfinite(hits[k]) and (not cur or abs(hits[k] - hits[cur[-1]]) < 1.5): cur.append(k)
        else:
            if len(cur) >= 6: segs.append(cur)
            cur = [k] if np.isfinite(hits[k]) else []
    if len(cur) >= 6: segs.append(cur)
    out = []
    for sg in segs:
        dd = hits[sg]; med = float(np.median(dd)); sd = float(np.std(dd))
        after = 3.3 <= med <= 5.0 and sd < 0.4
        out.append(dict(len_m=len(sg)*10, depth_m=round(med, 1), std_m=round(sd, 2), afterpulse_like=bool(after)))
    return dict(track=str(t), n_photons=int(len(h)), bins=nb, bg_per_slice=round(bg, 3), thr=round(thr, 1), segments=out)
def site(npz):
    res = []
    for t, h, lat, lon in passes(npz):
        r = analyse(t, h, lat, lon)
        if r: res.append(r)
    good = [(r, [g for g in r["segments"] if not g["afterpulse_like"]]) for r in res]
    seen = [min(g["depth_m"] for g in gs) for r, gs in good if gs]
    return dict(passes=len(res), passes_with_bottom=len(seen), bottom_depths_m=sorted(seen), afterpulse_segments=sum(sum(1 for g in r["segments"] if g["afterpulse_like"]) for r in res), tracks=res)
R = dict(rule=__doc__.split("Rule fixed")[1][:900], claims=[], controls=[])
for i, c in enumerate(CL):
    f = f"is2_claims/claim_{i:02d}.npz"
    if not os.path.exists(f): R["claims"].append(dict(claim=i, region=c["region"], note="no photons")); continue
    s = site(f); chart = abs(c["chart_m"]); pred = abs(c["predicted_shoal_m"])
    shallow = [d for d in s["bottom_depths_m"] if d <= 0.6*chart]
    s.update(claim=i, region=c["region"], lat=c["lat"], lon=c["lon"], predicted_shoal_m=pred, chart_m=chart, passes_shallower_than_0p6chart=len(shallow), supported_pending_control=len(shallow) >= 2)
    s.pop("tracks"); R["claims"].append(s); print(f"claim {i:2d} {c['region'][:34]:34s} pred {pred:5.1f} chart {chart:6.1f} | passes {s['passes']:2d} bottom {s['passes_with_bottom']:2d} shallow-hits {len(shallow)} depths {s['bottom_depths_m'][:6]} afterpulse-discarded {s['afterpulse_segments']}", flush=True)
for i, c in enumerate(ctl):
    f = f"is2_controls/ctl_{i:02d}.npz"
    if not os.path.exists(f): continue
    s = site(f); s.update(site=c["site"], chart_m=c["chart_m"]); s.pop("tracks"); R["controls"].append(s)
    print(f"control {c['site']:22s} chart {c['chart_m']:4d} | passes {s['passes']:2d} bottom {s['passes_with_bottom']:2d} depths {s['bottom_depths_m'][:8]} afterpulse-discarded {s['afterpulse_segments']}", flush=True)
if R["controls"]:
    P = sum(s["passes"] for s in R["controls"]); B = sum(s["passes_with_bottom"] for s in R["controls"]); R["control_false_positive_rate"] = round(B/max(P, 1), 3)
    R["verdict"] = ("CONTROL FAILS (fp rate >= 5%): not usable as evidence" if B/max(P, 1) >= 0.05 else "control passes")
    print("control passes", P, "with bottom", B, "fp rate", R["control_false_positive_rate"], "|", R["verdict"])
sup = [s["claim"] for s in R["claims"] if s.get("supported_pending_control")]; R["claims_supported_pending_control"] = sup; print("claims with >=2 shallow passes:", sup)
json.dump(R, open("is2_v2.json", "w"), indent=1)
