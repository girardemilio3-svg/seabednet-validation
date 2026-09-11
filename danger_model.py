#!/usr/bin/env python3
"""Danger-report model: learn where mariners end up reporting a danger, from the hazard field and every other cue, and test it on
years it never saw. Positives: Coast Guard danger positions (live + cancelled archive) with a date; negatives: random cells in the same
blocks that the chart calls safe (completed depth < -21 m) within 6 km of a sounding. Features per cell: hazard P (v2), predicted
shallowest depth, completed depth, model sigma, nearest published sounding depth and distance, coastal elevation within the cell and
its 500 m max, land flag, Sentinel-2 RGB, winter Sentinel-1 VV/VH, gravity prior, latitude. Split: train on notices dated <= 2022,
test on 2023-2026. Report AUC and precision at the top decile against single-cue baselines. -> danger_model.json, danger_features.csv"""
import csv, glob, json, math, os, numpy as np
from scipy import ndimage as ndi
from v5_data import GravityPrior
R = 6378137.0; rng = np.random.default_rng(0)
def merc(lon, lat): return R*math.radians(lon), R*math.log(math.tan(math.pi/4 + math.radians(lat)/2))
pts = [r for f in ("recon_navwarn/danger_points.csv", "recon_navwarn/cancelled_danger_points.csv") for r in csv.DictReader(open(f)) if r["date"][:4].isdigit()]
blocks = []
for f in sorted(glob.glob("hazard_nat_v2/*.npz")):
    bb = np.load(f, allow_pickle=True)["bbox3857"]; blocks.append((os.path.basename(f), min(bb[0], bb[2]), max(bb[0], bb[2]), min(bb[1], bb[3]), max(bb[1], bb[3])))
G = GravityPrior(); bypt = {}
for p in pts:
    x, y = merc(float(p["lon"]), float(p["lat"]))
    hit = [b for b in blocks if b[1] <= x <= b[2] and b[3] <= y <= b[4]]
    if hit: bypt.setdefault(hit[0][0], []).append((p, x, y))
print("danger points with a date:", len(pts), "in blocks:", sum(len(v) for v in bypt.values()), "blocks:", len(bypt), flush=True)
FEATS = ["hazard_p", "shoal_m", "complete_m", "sigma", "nearest_z", "nearest_km", "land_m", "land_max500", "land_flag", "s2_r", "s2_g", "s2_b", "s1_vv", "s1_vh", "s1_ok", "gravity", "lat", "coast_dist_px"]
rows = []
for name, items in bypt.items():
    hd = np.load(f"hazard_nat_v2/{name}", allow_pickle=True); P = hd["p105"].astype("float32"); S = hd["shoal"].astype("float32")
    cd = np.load(f"national_v5_out/{name}", allow_pickle=True); C = cd["complete"].astype("float32"); K = cd["known"].astype(bool); SG = cd["sigma"].astype("float32")
    Z = np.load(f"tiles_nat/{name}", allow_pickle=True)["z"].astype("float32"); bb = hd["bbox3857"]
    if P.shape != C.shape or Z.shape != C.shape: continue
    H, W = P.shape; x0, x1 = min(bb[0], bb[2]), max(bb[0], bb[2]); y0, y1 = min(bb[1], bb[3]), max(bb[1], bb[3])
    dist, nidx = ndi.distance_transform_edt(~K, return_indices=True); NZ = Z[nidx[0], nidx[1]]
    a = np.load(f"aux_out/{name}") if os.path.exists(f"aux_out/{name}") else None
    L = a["land"].astype("float32") if a is not None else np.full((H, W), np.nan, np.float32); Lm = np.isfinite(L) & (L > 0.5)
    L0 = np.where(Lm, L, 0.0); Lmax = ndi.maximum_filter(L0, size=11); cdist = ndi.distance_transform_edt(~Lm) if Lm.any() else np.full((H, W), 9999.0)
    s2 = a["s2"].astype("float32")/255.0 if a is not None else np.zeros((3, H, W), np.float32)
    b = np.load(f"aux_s1/{name}") if os.path.exists(f"aux_s1/{name}") else None
    VV = b["vv"].astype("float32") if b is not None else np.full((H, W), np.nan, np.float32); VH = b["vh"].astype("float32") if b is not None else np.full((H, W), np.nan, np.float32)
    lons = np.degrees((x0 + (np.arange(W)+0.5)/W*(x1-x0))/R); lats = np.degrees(2*np.arctan(np.exp((y1 - (np.arange(H)+0.5)/H*(y1-y0))/R)) - np.pi/2)
    def feat(i, j):
        near = P[max(0, i-5):i+6, max(0, j-5):j+6]; pm = float(np.nanmax(near)) if np.isfinite(near).any() else np.nan
        return [pm, float(S[i, j]), float(C[i, j]), float(SG[i, j]), float(NZ[i, j]), float(dist[i, j]*0.1), float(L0[i, j]), float(Lmax[i, j]), float(Lm[i, j]), float(s2[0, i, j]), float(s2[1, i, j]), float(s2[2, i, j]),
                float(VV[i, j]) if np.isfinite(VV[i, j]) else -30.0, float(VH[i, j]) if np.isfinite(VH[i, j]) else -40.0, float(np.isfinite(VV[i, j])), float(G.sample(np.array([lons[j]]), np.array([lats[i]]))[0]), float(lats[i]), float(min(cdist[i, j], 999))]
    safe = np.isfinite(C) & (C < -21) & (dist <= 60) & np.isfinite(P)
    si, sj = np.nonzero(safe)
    for p, x, y in items:
        j = int((x-x0)/(x1-x0)*(W-1)); i = int((y1-y)/(y1-y0)*(H-1))
        if not np.isfinite(P[i, j]): continue
        rows.append([1, int(p["date"][:4]), p["category"], name] + feat(i, j))
    n_neg = min(len(items)*10, len(si))
    if n_neg:
        for q in rng.choice(len(si), n_neg, replace=False):
            rows.append([0, int(rng.choice([2019, 2020, 2021, 2022, 2023, 2024, 2025])), "negative", name] + feat(si[q], sj[q]))
    print(name, len(items), flush=True)
with open("danger_features.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["y", "year", "category", "block"] + FEATS); w.writerows(rows)
print("rows", len(rows), "positives", sum(r[0] for r in rows), flush=True)
# ---- model: gradient boosting, year split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
A = np.array([r[4:] for r in rows], dtype=np.float32); y = np.array([r[0] for r in rows]); yr = np.array([r[1] for r in rows])
tr, te = yr <= 2022, yr >= 2023
clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0).fit(A[tr], y[tr])
pr = clf.predict_proba(A[te])[:, 1]
def topdec(score, yy):
    k = max(1, int(0.1*len(score))); idx = np.argsort(-score)[:k]; return float(yy[idx].mean()), float(yy[idx].sum()/max(1, yy.sum()))
res = dict(n_train=int(tr.sum()), n_test=int(te.sum()), pos_test=int(y[te].sum()), model=dict(auc=float(roc_auc_score(y[te], pr)), ap=float(average_precision_score(y[te], pr)), top_decile_precision=topdec(pr, y[te])[0], top_decile_recall=topdec(pr, y[te])[1]))
for nm, col, sign in [("hazard_p_only", 0, 1), ("nearest_sounding_only", 4, 1), ("nearest_distance_only", 5, -1), ("completed_depth_only", 2, 1)]:
    s = sign*A[te, col]; s = np.nan_to_num(s, nan=-1e9); res[nm] = dict(auc=float(roc_auc_score(y[te], s)), top_decile_precision=topdec(s, y[te])[0], top_decile_recall=topdec(s, y[te])[1])
imp = sorted(zip(FEATS, clf.feature_importances_ if hasattr(clf, "feature_importances_") else [0]*len(FEATS)), key=lambda t: -t[1])[:8] if hasattr(clf, "feature_importances_") else None
res["baseline_rate"] = float(y[te].mean()); json.dump(res, open("danger_model.json", "w"), indent=1)
print(json.dumps(res, indent=1)); print("DANGER_MODEL_DONE")
