#!/usr/bin/env python3
"""Score predictions on NONNA-Temporal-Churchill v1.
usage: python3 score.py predictions.npz   (arrays: block, row, col, pred [, sigma]) — same order/keys as targets_corridor.npz, or any subset matched on (block,row,col)."""
import sys, numpy as np, json
T = np.load("targets_corridor.npz"); P = np.load(sys.argv[1])
key = lambda a: a["block"].astype(np.int64)*10**8 + a["row"].astype(np.int64)*10**4 + a["col"].astype(np.int64)
kt, kp = key(T), key(P); order = np.argsort(kt); pos = np.searchsorted(kt[order], kp); ok = (pos < len(kt)) & (kt[order][np.minimum(pos, len(kt)-1)] == kp)
ti = order[pos[ok]]; pred = P["pred"][ok].astype(float); truth = T["depth"][ti].astype(float); dist = T["dist_px"][ti].astype(float)*0.1
e = pred - truth; out = dict(n=int(ok.sum()), coverage=float(ok.sum()/len(kt)), mae=float(np.abs(e).mean()), bias=float(e.mean()), rmse=float(np.sqrt((e**2).mean())))
out["by_distance_km"] = {f"{a}-{b}": float(np.abs(e[(dist>=a)&(dist<b)]).mean()) for a, b in [(0,.5),(.5,1),(1,2),(2,4),(4,8)] if ((dist>=a)&(dist<b)).any()}
out["by_depth_m"] = {f"{a}-{b}": float(np.abs(e[(-truth>=a)&(-truth<b)]).mean()) for a, b in [(0,20),(20,50),(50,100),(100,200),(200,400),(400,9999)] if ((-truth>=a)&(-truth<b)).any()}
if "sigma" in P: s = P["sigma"][ok].astype(float); out["frac_within_1sigma"] = float((np.abs(e) <= s).mean()); out["frac_within_2sigma"] = float((np.abs(e) <= 2*s).mean())
print(json.dumps(out, indent=1))
