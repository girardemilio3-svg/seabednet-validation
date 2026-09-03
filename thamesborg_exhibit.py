#!/usr/bin/env python3
"""Thamesborg exhibit — Franklin Strait, Sept 2025 grounding.

Left: what the archive offered (surveyed soundings only — mostly blank).
Right: SeabedNet completion — inferred seabed with shallow-hazard emphasis + σ.
"""
import math, numpy as np
from scipy import ndimage as ndi
R = 6378137.0
def lat_of_y(y): return math.degrees(2*math.atan(math.exp(y/R))-math.pi/2)
def lon_of_x(x): return math.degrees(x/R)

d = np.load("corridor_out/b0027_0030.npz", allow_pickle=True)
c = d["complete"].astype("float32"); k = d["known"].astype(bool)
s = d["sigma"].astype("float32"); bb = d["bbox3857"]
H, W = c.shape
lo0, lo1 = lon_of_x(min(bb[0],bb[2])), lon_of_x(max(bb[0],bb[2]))
la0, la1 = lat_of_y(min(bb[1],bb[3])), lat_of_y(max(bb[1],bb[3]))
lons = np.linspace(lo0, lo1, W); lats = np.linspace(la1, la0, H)
fill = np.isfinite(c) & ~k & (ndi.distance_transform_edt(~k) <= 60)

# land context from topo grid
g = np.load("planetary/gravity_prior_canada.npz")
gz = g["z"]; glon = g["lon"]; glat = g["lat"]
gx = np.clip((lons-glon[0])/(glon[1]-glon[0])*(gz.shape[1]-1), 0, gz.shape[1]-1).astype(int)
gy = np.clip((lats-glat[0])/(glat[1]-glat[0])*(gz.shape[0]-1), 0, gz.shape[0]-1).astype(int)
land = gz[np.ix_(gy, gx)] > 0

TL = (-96.90, 71.35)   # western tip, Tasmania Islands (AIS-derived)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
depth_cm = LinearSegmentedColormap.from_list("nav", [
    (0.00, "#7a1010"), (0.06, "#c8401e"), (0.14, "#e8933c"),   # <25 m: hazard
    (0.30, "#e8d97c"), (0.55, "#4f93c9"), (1.00, "#0b2a55")])  # deep: blue
DMAX = 160.0
def depth_rgba(z, alpha_where):
    n = np.clip(-z/DMAX, 0, 1)
    r = depth_cm(n); r[...,3] = np.where(alpha_where, 1.0, 0.0)
    return r
def panel(ax, title):
    back = np.zeros((H, W, 4)); back[...,0:3] = (0.03,0.045,0.09); back[...,3] = 1
    ax.imshow(back, extent=[lo0,lo1,la0,la1], origin="upper", aspect=1/math.cos(math.radians(70.8)), zorder=0)
    lr = np.zeros((H, W, 4)); lr[...,0:3] = 0.14; lr[...,3] = np.where(land,1.0,0.0)
    ax.imshow(lr, extent=[lo0,lo1,la0,la1], origin="upper", aspect=1/math.cos(math.radians(70.8)), zorder=1)
    ax.scatter(*TL, s=340, marker="X", color="#ff4d4d", edgecolor="white", lw=1.4, zorder=9)
    ax.set_title(title, color="white", fontsize=13.5, weight="bold", loc="left", pad=8)
    ax.set_axis_off()

fig, axes = plt.subplots(1, 2, figsize=(21, 9.6), dpi=140)
fig.subplots_adjust(wspace=0.06, top=0.90)
fig.patch.set_facecolor("#04060c")
panel(axes[0], "SEPT 6 2025 — WHAT THE ARCHIVE OFFERED\nCHS NONNA soundings — Tasmania Islands, Franklin Strait")
axes[0].imshow(depth_rgba(c, k), extent=[lo0,lo1,la0,la1], origin="upper",
               aspect=1/math.cos(math.radians(70.8)), zorder=2)
axes[0].annotate("MV THAMESBORG strikes an\n“uncharted shoal” (TSB M25C0241)", TL, xytext=(14,-34),
                 textcoords="offset points", color="#ff9d9d", fontsize=11, weight="bold")
axes[0].text(0.03, 0.95, "55% of this water: no published sounding\nthe ship struck 1.7 km past the edge of the surveyed swath (court: CATZOC C)", transform=axes[0].transAxes,
             color="#8fa0c0", fontsize=11.5, style="italic", va="top")

panel(axes[1], "THE SAME STRAIT, COMPLETED\nSeabedNet v5 inferred seabed + uncertainty")
axes[1].imshow(depth_rgba(c, k | fill), extent=[lo0,lo1,la0,la1], origin="upper",
               aspect=1/math.cos(math.radians(70.8)), zorder=2)
hi = fill & (s > np.nanpercentile(np.where(fill, s, np.nan), 85))
hs = np.zeros((H, W, 4)); hs[...,0:3] = (1,1,1); hs[...,3] = np.where(hi, 0.18, 0.0)
axes[1].imshow(hs, extent=[lo0,lo1,la0,la1], origin="upper",
               aspect=1/math.cos(math.radians(70.8)), zorder=3)
shal = (k | fill) & (c > -25) & np.isfinite(c) & ~land
axes[1].contour(lons, lats, np.where(shal, 1.0, 0.0), levels=[0.5],
                colors="#ff8f5d", linewidths=0.7, zorder=4)
axes[1].text(0.03, 0.075, "red/orange: shallower than 25 m — grounding depth for an ice-class freighter",
             transform=axes[1].transAxes, color="#ffb08a", fontsize=11)
axes[1].text(0.03, 0.04, "white haze: highest-σ zones — verify-first water", transform=axes[1].transAxes,
             color="#c9d4ea", fontsize=11)
axes[1].annotate("98th percentile of σ for this strait\ncalibrated σ ≥ 14 m (68%), ≥ 20 m (95%)", TL, xytext=(18,28), textcoords="offset points",
                 color="#e6d3ff", fontsize=11, weight="bold")
fig.text(0.015, 0.015, "CHS NONNA-100 (OGL; published archive, not all CHS holdings) · SeabedNet v5-small inference within 6 km of soundings · "
         "grounding at Tasmania Islands western tip (AIS/press; TSB coords unpublished) · planning prior — NOT for navigation",
         color="#66779a", fontsize=9)
fig.savefig("thamesborg_exhibit.png", dpi=140, facecolor="#04060c", bbox_inches="tight")
print("THAMESBORG_DONE")
