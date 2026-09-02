#!/usr/bin/env python3
"""σ recalibration from the independent GMRT cells (shelf water 50-400 m).

Fits a monotone map σ_raw -> σ_cal such that |err| <= σ_cal for 68% of cells within each
σ bin (isotonic on the 68th percentile of |err| per raw-σ quantile bin), and a second
map for the 95th percentile. Also fits the depth gate: beyond which depth the gravity
prior beats the model. Writes sigma_calibration.json used by the atlas/planner.
"""
import json, numpy as np
from sklearn.isotonic import IsotonicRegression

A = np.load("indep_cells_corridor_out.npy"); em, en, eg, sg, dk, dep = A.T
sh = (-dep >= 50) & (-dep < 400)
ae = np.abs(em[sh]); s = sg[sh]
q = np.quantile(s, np.linspace(0, 1, 41))
xs, y68, y95, ns = [], [], [], []
for a, b in zip(q[:-1], q[1:]):
    t = (s >= a) & (s <= b)
    if t.sum() < 200: continue
    xs.append(float(np.median(s[t]))); y68.append(float(np.quantile(ae[t], .68))); y95.append(float(np.quantile(ae[t], .95))); ns.append(int(t.sum()))
iso68 = IsotonicRegression(out_of_bounds="clip").fit(xs, y68, sample_weight=ns)
iso95 = IsotonicRegression(out_of_bounds="clip").fit(xs, y95, sample_weight=ns)
cal68 = iso68.predict(s); cal95 = iso95.predict(s)
grid = np.linspace(0, 80, 161)
print(f"shelf cells {sh.sum()}  raw: |err|<=σ {np.mean(ae<=s)*100:.0f}%  <=2σ {np.mean(ae<=2*s)*100:.0f}%")
print(f"calibrated: |err|<=σ68 {np.mean(ae<=cal68)*100:.0f}%  <=σ95 {np.mean(ae<=cal95)*100:.0f}%  mean scale σ68/σraw {np.mean(cal68/np.maximum(s,.1)):.2f}")
# depth gate: MAE model vs gravity by depth
gate = []
for a, b in [(0,50),(50,100),(100,200),(200,300),(300,400),(400,600),(600,1000),(1000,2000),(2000,9999)]:
    t = (-dep >= a) & (-dep < b)
    if t.sum() < 100: continue
    gate.append(dict(depth=[a, b], n=int(t.sum()), mae_model=float(np.abs(em[t]).mean()), mae_grav=float(np.abs(eg[t]).mean())))
    print(f"depth {a:5d}-{b:<5d} n={t.sum():7d} model {np.abs(em[t]).mean():5.1f} grav {np.abs(eg[t]).mean():5.1f}")
json.dump(dict(source="indep_cells_corridor_out.npy (GMRT research-cruise multibeam, shelf 50-400 m)",
               n=int(sh.sum()), sigma_raw_grid=grid.tolist(), sigma68=iso68.predict(grid).tolist(),
               sigma95=iso95.predict(grid).tolist(), knots=dict(x=xs, p68=y68, p95=y95, n=ns),
               coverage_raw_1sigma=float(np.mean(ae<=s)), coverage_cal_68=float(np.mean(ae<=cal68)),
               coverage_cal_95=float(np.mean(ae<=cal95)), depth_gate=gate),
          open("sigma_calibration.json", "w"), indent=1)
# Thamesborg site: raw σ 48.3 m -> calibrated
print("Thamesborg raw σ 48.3 ->", f"σ68 {iso68.predict([48.3])[0]:.1f} m, σ95 {iso95.predict([48.3])[0]:.1f} m")
print("SIGMA_CAL_DONE")
