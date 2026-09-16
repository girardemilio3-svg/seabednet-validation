#!/usr/bin/env python3
"""v6 national map = equal-weight ensemble of three full-data models: the published v5-small (national_v5_out), the same recipe
retrained from scratch (national_ctl_out) and the winter-radar variant (national_s1_out). complete = mean of the available
members; sigma = sqrt(mean of variances); known/bbox from the published run. -> national_v6_out/<block>.npz"""
import glob, os, numpy as np
MEM = os.environ.get("V6_MEMBERS", "national_v5_out,national_ctl_out,national_s1_out").split(","); OUT = os.environ.get("V6_OUT", "national_v6_out")
os.makedirs(OUT, exist_ok=True); n = 0; nm = {}
for f in sorted(glob.glob(MEM[0] + "/*.npz")):
    name = os.path.basename(f); a = np.load(f, allow_pickle=True)
    members = [a["complete"].astype("float32")]; sig = [a["sigma"].astype("float32")]
    for d in MEM[1:]:
        p = f"{d}/{name}"
        if os.path.exists(p):
            b = np.load(p, allow_pickle=True)
            if b["complete"].shape == members[0].shape: members.append(b["complete"].astype("float32")); sig.append(b["sigma"].astype("float32"))
    M = np.stack(members); Sg = np.stack(sig)
    comp = np.nanmean(M, 0); sg = np.sqrt(np.nanmean(Sg**2, 0)); nm[len(members)] = nm.get(len(members), 0) + 1
    np.savez_compressed(f"{OUT}/{name}", complete=comp.astype("float16"), sigma=sg.astype("float16"), known=a["known"], bbox3857=a["bbox3857"], n_members=np.int8(len(members))); n += 1
print("ENSEMBLE_DONE blocks", n, "members per block", nm)
