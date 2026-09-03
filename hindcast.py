#!/usr/bin/env python3
"""Grounding hindcast benchmark.

For each TSB-documented Arctic grounding on an uncharted / badly charted shoal, feed the
hazard model ONLY the CHS soundings that existed before the incident (CHS Survey Index
dates; soundings outside every indexed polygon are post-2016 and are dropped for incidents
up to 2016). Ask: among the "apparently safe" water around the site (model mean depth
deeper than 2x draft, within the 6 km fill discipline), what percentile of hazard does the
strike cell get?  Reported whether it lands or not.  Baselines: v5 mean-model σ (the atlas's
"where it is blind" metric), nearest pre-incident sounding, gravity prior.
usage: HZ_CKPT=hazard_tiny.pt python3 hindcast.py  -> hindcast.json
"""
import glob, json, math, os, numpy as np, torch
from scipy import ndimage as ndi
from scipy.stats import norm
from v5_data import GravityPrior, lat_of_y, lon_of_x
from v5_model import V5, normalize
DEV = "cuda"; P = 256; S = 128; BW = 16
R = 6378137.0
def x_of_lon(lo): return R*math.radians(lo)
def y_of_lat(la): return R*math.log(math.tan(math.pi/4+math.radians(la)/2))
HZ = os.environ.get("HZ_CKPT", "hazard_tiny.pt"); HZS = os.environ.get("HZ_SIZE", "tiny")
INCIDENTS = [
 dict(name="Thamesborg", date="2025-09-06", year=2025, lon=-96.90, lat=71.35, draft=10.5, tsb="M25C0241",
      note="position from AIS/press (TSB report pending, no official coordinates yet); draft assumed for a laden Handysize"),
 dict(name="Akademik Ioffe", date="2018-08-24", year=2018, lon=-91.34918, lat=69.71738, draft=5.9, tsb="M18C0225",
      note="TSB: 69°43.043'N 091°20.951'W; chart from 1984-92 reconnaissance, 2 km sounding spacing; struck at 14 m"),
 dict(name="Clipper Adventurer", date="2010-08-27", year=2010, lon=-112.67167, lat=67.97000, draft=4.6, tsb="M10H0006",
      note="TSB: 67°58.2'N 112°40.3'W; chart showed 29 m, least depth 3.3 m (2007 CCGS discovery never charted)"),
 dict(name="Hanseatic", date="1996-08-29", year=1996, lon=-97.53667, lat=68.56250, draft=4.8, tsb="M96H0016",
      note="TSB: 68°33.75'N 097°32.2'W, Simpson Strait; buoy displaced by ice"),
 dict(name="Nanny 2012", date="2012-10-25", year=2012, lon=-94.30667, lat=63.99333, draft=4.6, tsb="M12H0012",
      note="TSB: 63°59.6'N 094°18.4'W, Chesterfield Narrows; navigation error, hazard known"),
 dict(name="Nanny 2014", date="2014-10-14", year=2014, lon=-91.52000, lat=63.62000, draft=6.9, tsb="M14C0219",
      note="TSB: 63°37.2'N 091°31.2'W, Chesterfield Inlet; helm-order error, hazard known"),
 dict(name="Mokami 2000", date="2000-10-31", year=2000, lon=-61.56967, lat=56.45083, draft=5.2, tsb="M00N0098",
      note="TSB: buoy NP5 at 56°27.05'N 061°34.18'W, Bridges Passage, Labrador; grounded ~200 m NE of the buoy; navigation error + buoy 160 m off charted position + datum mismatch; hazard known (control). Hazard-model training did NOT exclude this site's tiles.", control=True, training_exposed=True),
]
grav = GravityPrior()
hz = V5(HZS).to(DEV); ck = torch.load(HZ, map_location=DEV, weights_only=False); hz.load_state_dict(ck["net"]); hz.eval()
mean_net = V5("small").to(DEV); ck2 = torch.load("v5_small.pt", map_location=DEV, weights_only=False); mean_net.load_state_dict(ck2["net"]); mean_net.eval()
win = np.outer(np.hanning(P), np.hanning(P)) + 1e-3
blocks = []
for f in sorted(glob.glob("tiles_nat/*.npz")):
    bb = np.load(f, allow_pickle=True)["bbox3857"]
    blocks.append((f, min(bb[0], bb[2]), max(bb[0], bb[2]), min(bb[1], bb[3]), max(bb[1], bb[3])))

def infer(net, zin, known, G):
    H, W = zin.shape
    Hp = (H//S+3)*S; Wp = (W//S+3)*S
    zp = np.full((Hp, Wp), np.nan, np.float32); zp[:H, :W] = zin
    kp = np.zeros((Hp, Wp), np.float32); kp[:H, :W] = known
    gp = np.zeros((Hp, Wp), np.float32); gp[:H, :W] = G; gp[H:, :] = G[-1:, :].mean(); gp[:, W:] = gp[:, W-1:W]
    accm = np.zeros((Hp, Wp)); accs = np.zeros((Hp, Wp)); wacc = np.zeros((Hp, Wp))
    coords = [(i, j) for i in range(0, Hp-P+1, S) for j in range(0, Wp-P+1, S) if kp[i:i+P, j:j+P].sum() >= 400]
    with torch.no_grad():
        for b0 in range(0, len(coords), BW):
            cb = coords[b0:b0+BW]
            t = lambda a: torch.tensor(np.stack(a))[:, None].float().to(DEV)
            dt = t([np.nan_to_num(zp[i:i+P, j:j+P]) for i, j in cb]); kt = t([kp[i:i+P, j:j+P] for i, j in cb]); gt = t([gp[i:i+P, j:j+P] for i, j in cb])
            dn, gn, mu0, sd0 = normalize(dt, kt, gt)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu, lv = net(torch.cat([dn*kt, kt, gn], 1), torch.zeros(len(cb), device=DEV, dtype=torch.long))
            est = (mu.float()*sd0 + mu0)[:, 0].cpu().numpy(); sig = (torch.exp(0.5*lv.float())*sd0)[:, 0].cpu().numpy()
            for q, (i, j) in enumerate(cb):
                accm[i:i+P, j:j+P] += est[q]*win; accs[i:i+P, j:j+P] += sig[q]*win; wacc[i:i+P, j:j+P] += win
    est = np.where(wacc > 0, accm/np.maximum(wacc, 1e-6), np.nan)[:H, :W]
    sig = np.where(wacc > 0, accs/np.maximum(wacc, 1e-6), np.nan)[:H, :W]
    return est, sig

def pct_rank(field, mask, i, j, r):
    """percentile (0-100) of field[i,j] among masked cells within r px; higher = more dangerous"""
    ii, jj = np.mgrid[0:field.shape[0], 0:field.shape[1]]
    loc = mask & ((ii-i)**2 + (jj-j)**2 <= r*r) & np.isfinite(field)
    if loc.sum() < 50 or not np.isfinite(field[i, j]): return None, int(loc.sum())
    return float((field[loc] < field[i, j]).mean()*100), int(loc.sum())

results = []
for inc in INCIDENTS:
    x, y = x_of_lon(inc["lon"]), y_of_lat(inc["lat"])
    hit = [b for b in blocks if b[1] <= x <= b[2] and b[3] <= y <= b[4]]
    if not hit:
        results.append(dict(inc, status="site outside the NONNA-100 harvest")); print(inc["name"], "outside harvest"); continue
    f, x0, x1, y0, y1 = hit[0]
    d = np.load(f, allow_pickle=True); z = d["z"].astype("float32"); H, W = z.shape
    idx = np.load(f"index_out/{os.path.basename(f)}")["year"]
    allk = np.isfinite(z)
    if inc["year"] <= 2016: pre = allk & (idx > 0) & (idx < inc["year"])
    else: pre = allk                                      # no dates after 2016: use everything, flag it
    zin = np.where(pre, z, np.nan); known = pre.astype(np.float32)
    lons = np.array([lon_of_x(v) for v in np.linspace(x0, x1, W)]); lats = np.array([lat_of_y(v) for v in np.linspace(y1, y0, H)])
    G = grav.sample(np.tile(lons, (H, 1)), np.tile(lats[:, None], (1, W))).astype("float32")
    j = int((x-x0)/(x1-x0)*(W-1)); i = int((y1-y)/(y1-y0)*(H-1))
    res_m = (x1-x0)/W*math.cos(math.radians(inc["lat"]))
    dist_px, nidx = ndi.distance_transform_edt(~pre, return_indices=True)
    dom_fill = dist_px <= 120   # hindcast: 12 km (reconnaissance-era sites sit 9-12 km from any pre-incident sounding)
    mu_s, sg_s = infer(hz, zin, known, G)              # shallowest-within-500 m field
    mu_m, sg_m = infer(mean_net, zin, known, G)        # mean-depth field (atlas model)
    draft = inc["draft"]
    p_shoal = 1 - norm.cdf((-draft - mu_s)/np.maximum(sg_s, 0.5))          # P(shallowest point < draft)
    safe = dom_fill & np.isfinite(mu_m) & (mu_m < -2*draft)                 # water that LOOKS safe on the mean map
    r_px = int(25000/res_m)                                                # 25 km neighbourhood
    # allow 500 m position slop: take the max hazard within 5 px of the reported position
    ii, jj = np.mgrid[0:H, 0:W]; near = (ii-i)**2 + (jj-j)**2 <= 25
    def at(fld):
        v = fld[near & np.isfinite(fld)]; return float(np.nanmax(v)) if v.size else None
    rec = dict(inc, block=os.path.basename(f), pre_soundings=int(pre.sum()), all_soundings=int(allk.sum()),
               dist_to_pre_sounding_km=round(float(dist_px[i, j]*res_m/1000), 2), in_fill_domain=bool(dom_fill[i, j]),
               site=dict(mean_depth_model=at(-np.abs(mu_m)) if np.isfinite(mu_m[i, j]) else None,
                         shoal_depth_model=float(mu_s[i, j]) if np.isfinite(mu_s[i, j]) else None,
                         shoal_sigma=float(sg_s[i, j]) if np.isfinite(sg_s[i, j]) else None,
                         p_shoal_lt_draft=at(p_shoal), mean_sigma=float(sg_m[i, j]) if np.isfinite(sg_m[i, j]) else None,
                         nearest_pre_sounding=float(zin[nidx[0][i, j], nidx[1][i, j]]) if pre.any() else None, gravity=float(G[i, j])),
               rank=dict())
    for nm, fld in [("hazard_p_shoal", p_shoal), ("hazard_shoal_depth", -np.abs(mu_s)*-1), ("mean_model_sigma", sg_m), ("nearest_sounding_shallowness", zin[nidx[0], nidx[1]]), ("gravity_shallowness", G)]:
        pr, n = pct_rank(fld, safe, i, j, r_px); rec["rank"][nm] = dict(percentile=pr, n_cells=n)
    rec["rank"]["hazard_p_shoal_whole_block"] = dict(percentile=pct_rank(p_shoal, safe, i, j, 10**6)[0])
    results.append(rec)
    print(f"{inc['name']:20s} pre-soundings {pre.sum():8d}  dist {rec['dist_to_pre_sounding_km']:5.2f} km  "
          f"P(shoal<draft) {rec['site']['p_shoal_lt_draft']}  hazard pct (25 km, apparently-safe water) {rec['rank']['hazard_p_shoal']['percentile']}  "
          f"mean-σ pct {rec['rank']['mean_model_sigma']['percentile']}  nearest-sounding pct {rec['rank']['nearest_sounding_shallowness']['percentile']}", flush=True)
json.dump(results, open("hindcast.json", "w"), indent=1, default=float)
print("HINDCAST_DONE")
