#!/usr/bin/env python3
"""Distance-blend of model and nearest sounding on the temporal benchmark, fitted on half the blocks and tested on the other half.
Blend: est = w(d)*model + (1-w(d))*nearest, w(d) piecewise-constant in distance bins, w fitted per bin by least squares on the fit
blocks. Cells arrays: [err_model, err_nn, err_grav, sigma, dist_km, depth]; err = est - truth (signed), so blended err is the same blend.
usage: python3 hybrid_eval.py TAG cells.npy [cells2.npy ...]  (members averaged first) -> temporal_validation_<TAG>.json"""
import sys, json, numpy as np
tag = sys.argv[1]; A = [np.load(f, mmap_mode="r") for f in sys.argv[2:]]; n = A[0].shape[0]
E = np.mean(np.stack([np.asarray(a[:n, 0]) for a in A]), 0); S = np.sqrt(np.mean(np.stack([np.asarray(a[:n, 3])**2 for a in A]), 0))
nn = np.asarray(A[0][:n, 1]); gr = np.asarray(A[0][:n, 2]); dist = np.asarray(A[0][:n, 4]); depth = np.asarray(A[0][:n, 5])
blocks = json.load(open("temporal_validation_temporal_ctlfull.json"))["blocks"]; bid = np.concatenate([np.full(v["n"], k) for k, v in enumerate(blocks.values())]); assert len(bid) == n
bins = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 1e9]; b = np.digitize(dist, bins) - 1
res = {}; Eh = E.copy()
for fold in (0, 1):
    fit = (bid % 2) == fold; test = ~fit; W = []
    for k in range(len(bins) - 1):
        m = fit & (b == k); x = nn[m] - E[m]   # err_blend = E + (1-w)*(nn - E) ; minimise over t=(1-w): t = -<E,x>/<x,x>
        t = float(np.clip(-np.dot(E[m], x)/max(np.dot(x, x), 1e-9), 0, 1)) if m.sum() > 1000 else 0.0; W.append(round(1 - t, 3))
        mt = test & (b == k); Eh[mt] = E[mt] + t*(nn[mt] - E[mt])
    res[f"w_fold{fold}"] = W
def stats(m): return dict(n=int(m.sum()), mae_model=float(np.mean(np.abs(Eh[m]))), mae_nn=float(np.mean(np.abs(nn[m]))), mae_grav=float(np.mean(np.abs(gr[m]))), frac_within_1sigma=float(np.mean(np.abs(Eh[m]) <= S[m])), bias=float(np.mean(Eh[m])))
out = dict(ckpt="hybrid:" + "+".join(sys.argv[2:]), n_cells=int(n), overall=stats(np.ones(n, bool)), blend_weights_model=res, bins_km=bins[:-1],
           by_depth=[dict(m=[lo, hi], **stats((-depth >= lo) & (-depth < hi))) for lo, hi in ((0, 20), (20, 50), (50, 200), (200, 100000))],
           by_distance=[dict(km=[lo, hi], **stats((dist >= lo) & (dist < hi))) for lo, hi in ((0, 0.5), (0.5, 1), (1, 3), (3, 6))], model_alone_mae=float(np.mean(np.abs(E))))
json.dump(out, open(f"temporal_validation_{tag}.json", "w"), indent=1)
o = out["overall"]; print(tag, "hybrid MAE", round(o["mae_model"], 2), "vs model alone", round(out["model_alone_mae"], 2), "| 0-20", round(out["by_depth"][0]["mae_model"], 2), "20-50", round(out["by_depth"][1]["mae_model"], 2), "| weights", res)
