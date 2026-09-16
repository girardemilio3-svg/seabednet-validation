#!/usr/bin/env python3
"""Offline ensemble on the temporal benchmark from the aligned per-cell arrays ([err_model, err_nn, err_grav, sigma, dist_km, depth]).
usage: python3 ensemble_temporal.py TAG member1.npy member2.npy ...  -> temporal_validation_<TAG>.json (same schema as temporal_eval)"""
import sys, json, numpy as np
tag = sys.argv[1]; A = [np.load(f, mmap_mode="r") for f in sys.argv[2:]]
n = min(a.shape[0] for a in A); ref = np.asarray(A[0][:n, 5])
for a in A[1:]: assert a.shape[0] == n and np.allclose(np.asarray(a[:n, 5]), ref), "cells not aligned"
E = np.mean(np.stack([np.asarray(a[:n, 0]) for a in A]), 0); S = np.sqrt(np.mean(np.stack([np.asarray(a[:n, 3])**2 for a in A]), 0))
nn = np.asarray(A[0][:n, 1]); gr = np.asarray(A[0][:n, 2]); depth = np.asarray(A[0][:n, 5]); dist = np.asarray(A[0][:n, 4])
def stats(m):
    if m.sum() == 0: return dict(n=0, mae_model=None, mae_nn=None, mae_grav=None, frac_within_1sigma=None, bias=None)
    return dict(n=int(m.sum()), mae_model=float(np.mean(np.abs(E[m]))), mae_nn=float(np.mean(np.abs(nn[m]))), mae_grav=float(np.mean(np.abs(gr[m]))), frac_within_1sigma=float(np.mean(np.abs(E[m]) <= S[m])), bias=float(np.mean(E[m])))
out = dict(ckpt="+".join(sys.argv[2:]), n_cells=int(n), overall=stats(np.ones(n, bool)),
           by_depth=[dict(m=[lo, hi], **stats((-depth >= lo) & (-depth < hi))) for lo, hi in ((0, 20), (20, 50), (50, 200), (200, 100000))],
           by_distance=[dict(km=[lo, hi], **stats((dist >= lo) & (dist < hi))) for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 1000))])
json.dump(out, open(f"temporal_validation_{tag}.json", "w"), indent=1)
o = out["overall"]; print(tag, "MAE", round(o["mae_model"], 2), "nn", round(o["mae_nn"], 2), "1sig", round(o["frac_within_1sigma"], 3), "bias", round(o["bias"], 2), "| 0-20", round(out["by_depth"][0]["mae_model"], 2), "20-50", round(out["by_depth"][1]["mae_model"], 2))
