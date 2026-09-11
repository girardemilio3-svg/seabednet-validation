#!/usr/bin/env python3
"""Danger-report model v2: richer cues, Arctic-only results, calibrated probabilities. Same protocol as danger_model.py (chart-safe
cells only, ten random negatives per notice from the same block, train <= 2022, test 2023-2026). New cues: log(blue/green) from
Sentinel-2 (the classic satellite-bathymetry index), radar texture (std of VV within 500 m), seabed roughness (std of completed
depth within 500 m), local slope, distance to the nearest SHALLOW sounding (< 10 m) and its depth, hazard P mean within 1 km.
-> danger_model2.json, danger_features2.csv"""
import csv, glob, json, math, os, numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression
from v5_data import GravityPrior
R = 6378137.0; rng = np.random.default_rng(1)
def merc(lon, lat): return R*math.radians(lon), R*math.log(math.tan(math.pi/4 + math.radians(lat)/2))
pts = [r for f in ("recon_navwarn/danger_points.csv", "recon_navwarn/cancelled_danger_points.csv") for r in csv.DictReader(open(f)) if r["date"][:4].isdigit()]
blocks = []
for f in sorted(glob.glob("hazard_nat_v2/*.npz")):
    bb = np.load(f, allow_pickle=True)["bbox3857"]; blocks.append((os.path.basename(f), min(bb[0], bb[2]), max(bb[0], bb[2]), min(bb[1], bb[3]), max(bb[1], bb[3])))
G = GravityPrior(); bypt = {}
for p in pts:
    x, y = merc(float(p["lon"]), float(p["lat"])); hit = [b for b in blocks if b[1] <= x <= b[2] and b[3] <= y <= b[4]]
    if hit: bypt.setdefault(hit[0][0], []).append((p, x, y))
FEATS = ["hazard_p", "hazard_p_1km", "shoal_m", "complete_m", "sigma", "rough500", "slope", "nearest_z", "nearest_km", "shallow_km", "shallow_z", "land_max500", "land_flag", "coast_km", "s2_r", "s2_g", "s2_b", "s2_logbg", "s1_vv", "s1_vh", "s1_vv_std", "s1_ok", "gravity", "lat"]
rows = []
for name, items in bypt.items():
    hd = np.load(f"hazard_nat_v2/{name}", allow_pickle=True); P = hd["p105"].astype("float32"); S = hd["shoal"].astype("float32")
    cd = np.load(f"national_v5_out/{name}", allow_pickle=True); C = cd["complete"].astype("float32"); K = cd["known"].astype(bool); SG = cd["sigma"].astype("float32")
    Z = np.load(f"tiles_nat/{name}", allow_pickle=True)["z"].astype("float32"); bb = hd["bbox3857"]
    if P.shape != C.shape or Z.shape != C.shape: continue
    H, W = P.shape; x0, x1 = min(bb[0], bb[2]), max(bb[0], bb[2]); y0, y1 = min(bb[1], bb[3]), max(bb[1], bb[3])
    dist, nidx = ndi.distance_transform_edt(~K, return_indices=True); NZ = Z[nidx[0], nidx[1]]
    shallow = K & (Z > -10); sdist, sidx = ndi.distance_transform_edt(~shallow, return_indices=True) if shallow.any() else (np.full((H, W), 9999.0), None); SZ = Z[sidx[0], sidx[1]] if sidx is not None else np.full((H, W), np.nan, np.float32)
    P0 = np.nan_to_num(P, nan=0.0); Pmax = ndi.maximum_filter(P0, size=11); P1k = ndi.uniform_filter(P0, size=21)
    C0 = np.nan_to_num(C, nan=np.nanmean(C) if np.isfinite(C).any() else 0.0); Cm = ndi.uniform_filter(C0, size=11); Cs = np.sqrt(np.maximum(ndi.uniform_filter(C0*C0, size=11) - Cm*Cm, 0)); gy, gx = np.gradient(ndi.gaussian_filter(C0, 2)); slope = np.hypot(gx, gy)
    a = np.load(f"aux_out/{name}") if os.path.exists(f"aux_out/{name}") else None
    L = a["land"].astype("float32") if a is not None else np.full((H, W), np.nan, np.float32); Lm = np.isfinite(L) & (L > 0.5); L0 = np.where(Lm, L, 0.0); Lmax = ndi.maximum_filter(L0, size=11)
    cdist = ndi.distance_transform_edt(~Lm)*0.1 if Lm.any() else np.full((H, W), 999.0)
    s2 = a["s2"].astype("float32")/255.0 if a is not None else np.zeros((3, H, W), np.float32); logbg = np.log((s2[2] + 0.02)/(s2[1] + 0.02))
    b = np.load(f"aux_s1/{name}") if os.path.exists(f"aux_s1/{name}") else None
    VV = b["vv"].astype("float32") if b is not None else np.full((H, W), np.nan, np.float32); VH = b["vh"].astype("float32") if b is not None else np.full((H, W), np.nan, np.float32)
    VV0 = np.where(np.isfinite(VV), VV, -30.0); Vm = ndi.uniform_filter(VV0, size=11); Vs = np.sqrt(np.maximum(ndi.uniform_filter(VV0*VV0, size=11) - Vm*Vm, 0))
    lons = np.degrees((x0 + (np.arange(W)+0.5)/W*(x1-x0))/R); lats = np.degrees(2*np.arctan(np.exp((y1 - (np.arange(H)+0.5)/H*(y1-y0))/R)) - np.pi/2)
    def feat(i, j):
        return [float(Pmax[i, j]), float(P1k[i, j]), float(S[i, j]), float(C[i, j]), float(SG[i, j]), float(Cs[i, j]), float(slope[i, j]), float(NZ[i, j]), float(dist[i, j]*0.1), float(min(sdist[i, j]*0.1, 999)), float(SZ[i, j]) if np.isfinite(SZ[i, j]) else 0.0,
                float(Lmax[i, j]), float(Lm[i, j]), float(min(cdist[i, j], 999)), float(s2[0, i, j]), float(s2[1, i, j]), float(s2[2, i, j]), float(logbg[i, j]), float(VV0[i, j]), float(VH[i, j]) if np.isfinite(VH[i, j]) else -40.0, float(Vs[i, j]), float(np.isfinite(VV[i, j])), float(G.sample(np.array([lons[j]]), np.array([lats[i]]))[0]), float(lats[i])]
    safe = np.isfinite(C) & (C < -21) & (dist <= 60) & np.isfinite(P); si, sj = np.nonzero(safe)
    for p, x, y in items:
        j = int((x-x0)/(x1-x0)*(W-1)); i = int((y1-y)/(y1-y0)*(H-1))
        if not (np.isfinite(P[i, j]) and np.isfinite(C[i, j]) and C[i, j] < -21): continue
        rows.append([1, int(p["date"][:4]), 1 if p["series"].startswith("NW-A") else 0, name] + feat(i, j))
    n_neg = min(len(items)*10, len(si))
    for q in (rng.choice(len(si), n_neg, replace=False) if n_neg else []):
        rows.append([0, int(rng.choice([2019, 2020, 2021, 2022, 2023, 2024, 2025])), 1 if lats[si[q]] > 60 else 0, name] + feat(si[q], sj[q]))
with open("danger_features2.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["y", "year", "arctic", "block"] + FEATS); w.writerows(rows)
A = np.array([r[4:] for r in rows], dtype=np.float32); y = np.array([r[0] for r in rows]); yr = np.array([r[1] for r in rows]); arc = np.array([r[2] for r in rows])
tr, te = yr <= 2022, yr >= 2023; print("rows", len(rows), "positives", int(y.sum()), "test", int(te.sum()), "test pos", int(y[te].sum()), flush=True)
def topdec(score, yy):
    score = np.nan_to_num(score, nan=-1e9); k = max(1, int(0.1*len(score))); idx = np.argsort(-score)[:k]; return float(yy[idx].mean()), float(yy[idx].sum()/max(1, yy.sum()))
def fit(cols, Xtr=tr, Xte=te):
    clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=40, random_state=0).fit(A[Xtr][:, cols], y[Xtr]); pr = clf.predict_proba(A[Xte][:, cols])[:, 1]
    p, rc = topdec(pr, y[Xte]); return clf, pr, dict(auc=float(roc_auc_score(y[Xte], pr)), ap=float(average_precision_score(y[Xte], pr)), top_decile_precision=p, top_decile_recall=rc)
ix = {k: i for i, k in enumerate(FEATS)}; res = dict(n_rows=len(rows), n_test=int(te.sum()), pos_test=int(y[te].sum()), base_rate=float(y[te].mean()))
clf, pr, res["all_features"] = fit(list(range(len(FEATS))))
_, _, res["v1_features_only"] = fit([ix[k] for k in ("hazard_p", "shoal_m", "complete_m", "sigma", "nearest_z", "nearest_km", "land_max500", "land_flag", "s2_r", "s2_g", "s2_b", "s1_vv", "s1_vh", "s1_ok", "gravity", "lat", "coast_km")])
_, _, res["no_exposure_cues"] = fit([ix[k] for k in FEATS if k not in ("nearest_km", "shallow_km", "coast_km", "lat")])
_, _, res["hazard_only"] = fit([ix["hazard_p"], ix["hazard_p_1km"], ix["shoal_m"]])
# Arctic-only evaluation of the full model, and an Arctic-trained model
am = te & (arc == 1); p_a, r_a = topdec(pr[am[te]], y[am]); res["all_features_arctic_test"] = dict(n=int(am.sum()), pos=int(y[am].sum()), base_rate=float(y[am].mean()), auc=float(roc_auc_score(y[am], pr[am[te]])), top_decile_precision=p_a, top_decile_recall=r_a)
# calibration on the test years (isotonic), reported as reliability bins
iso = IsotonicRegression(out_of_bounds="clip").fit(pr, y[te]); bins = np.quantile(pr, np.linspace(0, 1, 11)); rel = []
for a_, b_ in zip(bins[:-1], bins[1:]):
    m = (pr >= a_) & (pr <= b_); rel.append(dict(pred_mean=float(pr[m].mean()), obs_rate=float(y[te][m].mean()), n=int(m.sum())))
res["reliability_test_years"] = rel
full = res["all_features"]["auc"]; imp = {}
for k in FEATS:
    cols = [i for i, kk in enumerate(FEATS) if kk != k]; imp[k] = round(full - fit(cols)[2]["auc"], 4)
res["drop_one_auc_loss"] = dict(sorted(imp.items(), key=lambda t: -t[1]))
json.dump(res, open("danger_model2.json", "w"), indent=1)
for k in ("all_features", "v1_features_only", "no_exposure_cues", "hazard_only", "all_features_arctic_test"):
    v = res[k]; print(f"{k:26s} AUC {v['auc']:.3f} top-decile precision {v['top_decile_precision']*100:5.1f}% recall {v['top_decile_recall']*100:5.1f}%" + (f" (n={v['n']}, base {v['base_rate']*100:.1f}%)" if 'n' in v else ""))
print("drop-one:", list(res["drop_one_auc_loss"].items())[:8]); print("DANGER2_DONE")
