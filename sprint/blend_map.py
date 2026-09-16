#!/usr/bin/env python3
"""Post-process a national map: blend the model with the nearest sounding by distance, est = w(d)*model + (1-w(d))*nearest.
w(d) is the average of the two block-fold fits in temporal_validation_hybrid_ens3tta.json (fitted on half the corridor blocks,
tested on the other half: 12.68 -> 12.45 m). usage: python3 blend_map.py <in_dir> <out_dir>"""
import sys, glob, os, json, numpy as np
from scipy import ndimage as ndi
IN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
h = json.load(open("temporal_validation_hybrid_ens3tta.json")); W = np.mean([h["blend_weights_model"]["w_fold0"], h["blend_weights_model"]["w_fold1"]], 0); bins = h["bins_km"]
def w_of(dkm): return W[np.clip(np.digitize(dkm, bins) - 1, 0, len(W) - 1)]
n = 0
for f in sorted(glob.glob(f"{IN}/*.npz")):
    name = os.path.basename(f); a = np.load(f, allow_pickle=True); C = a["complete"].astype("float32"); K = a["known"].astype(bool)
    Z = np.load(f"tiles_nat/{name}", allow_pickle=True)["z"].astype("float32")
    if not K.any(): np.savez_compressed(f"{OUT}/{name}", **{k: a[k] for k in a.files}); n += 1; continue
    dist, idx = ndi.distance_transform_edt(~K, return_indices=True); NZ = Z[idx[0], idx[1]]; w = w_of(dist*0.1).astype("float32")
    fin = np.isfinite(C) & np.isfinite(NZ); B = C.copy(); B[fin] = w[fin]*C[fin] + (1 - w[fin])*NZ[fin]
    np.savez_compressed(f"{OUT}/{name}", complete=B.astype(a["complete"].dtype), sigma=a["sigma"], known=a["known"], bbox3857=a["bbox3857"], **({"n_members": a["n_members"]} if "n_members" in a.files else {}), blend="distance blend v1"); n += 1
print("BLEND_DONE", n, "blocks; weights", np.round(W, 3).tolist(), "bins_km", bins)
