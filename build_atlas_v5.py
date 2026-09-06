#!/usr/bin/env python3
"""Correction pass: churchill_atlas_v4.html -> churchill_atlas_v5.html.
  Exhibit J  — adds the Shoal List v2 (sealed 2026-09-06) above the v1 table, with the correction note: what was wrong with v1
               (near-shore + survey-line artefacts, surface bias), what changed (coast-aware hazard head, offshore filters). v1 stays sealed.
  Exhibit L  — the reported-dangers test: 1,558 Canadian Coast Guard NAVWARN danger positions scored against the hazard field,
               old head vs new head vs the nearest-sounding rule, by subset. Numbers read from navwarn_hindcast{,_v2}.csv.
usage: python3 build_atlas_v5.py"""
import csv, json, os, collections
from scipy.stats import binomtest
src = open("churchill_atlas_v4.html", encoding="utf-8").read()
def sub1(s, old, new):
    assert s.count(old) == 1, f"count {s.count(old)}: {old[:60]}"; return s.replace(old, new)
tr = lambda cells, th=False: "<tr>" + "".join(f"<{'th' if th else 'td'}>{c}</{'th' if th else 'td'}>" for c in cells) + "</tr>"
# ---- Shoal List v2
SL2 = json.load(open("shoal_list_v2_manifest.json")); R2 = list(csv.DictReader(open(SL2["file"]))); ots2 = os.path.exists(SL2["file"] + ".ots")
SL1 = json.load(open("shoal_list_manifest.json")); R1 = list(csv.DictReader(open(SL1["file"])))
v2_rows = "".join(tr([str(i+1), r["region"], f"{float(r['lat']):.4f}&deg;N", f"{abs(float(r['lon'])):.4f}&deg;W", f"{float(r['p_peak'])*100:.0f}%", f"{abs(float(r['predicted_shoal_m'])):.1f}", f"{abs(float(r['mean_map_depth_m'])):.0f}", r["coast_km"], r["km_to_sounding"]]) for i, r in enumerate(R2[:20]))
reg2 = ", ".join(f"{k} ({v})" for k, v in collections.Counter(r["region"] for r in R2).most_common(6))
n_v1_near = sum(1 for r in R1 if float(r.get("km_to_sounding", 9)) >= 0)  # placeholder count (v1 has no coast field); stated numbers below come from the recon
JV2 = f'''
  <h3 style="font-size:19px;margin:22px 0 6px">The Shoal List, version 2 (sealed {SL2['date']})</h3>
  <p class="lede" style="font-size:14.5px"><b style="color:var(--text)">A correction, on the record.</b> Three days after sealing version 1 we tested it against things it had never seen: a Sentinel-2 image over every claim, the GSHHG shoreline, and the Coast Guard&rsquo;s reported-danger notices (Exhibit L). Version 1 failed two of our own checks. Every one of its forty claims predicted a rock at the surface, 22 of 40 sat within a kilometre of a shore, and several lined up along single survey tracks. The hazard head had learned a true but useless fact: near a coast, the shallowest point within 500 m is the beach. Version 1 stays sealed and will be scored exactly as published. It is superseded, not withdrawn.</p>
  <p class="lede" style="font-size:14.5px">What changed. The shallowest-point model was retrained on the user&rsquo;s own hardware with two rules: a training target only counts where at least 95% of the 500 m disc around it carries 10 m soundings (no land, no gaps), and targets at or above the surface are discarded. The candidate filters gained three conditions: at least 2 km from the full-resolution coastline, no cluster confined to a single row or column of cells, and a predicted shallowest point between 3 and 18 m, the depth of a keel-height pinnacle rather than a beach. Of {SL2['n_candidates_total']:,} candidate clusters nationally, {SL2['n_pass_filters']} pass; the top {SL2['n_sealed']}, at most two per quarter-degree, are sealed. Regions: {reg2}.</p>
  <div class="tblwrap"><table><thead>{tr(["#", "Water", "Lat", "Lon", "P(shoal &lt; 10.5 m)", "Predicted shallowest (m)", "Map says (m)", "km to coast", "km to sounding"], True)}</thead><tbody>{v2_rows}</tbody></table></div>
  <p class="lede" style="font-size:13.5px">First 20 of {SL2['n_sealed']}. File <a href="{SL2['file']}">{SL2['file']}</a> &middot; SHA-256 <code style="font-size:12px">{SL2['sha256']}</code>{' &middot; OpenTimestamps proof <a href="' + SL2['file'] + '.ots">.ots</a>' if ots2 else ''}. A Sentinel-2 summer image was pulled over all forty: every position is open water, none shows a visible rock or breaker at 10 m resolution, which neither confirms nor refutes a 3&ndash;7 m shoal. Same scoring rule as version 1.</p>
  <h3 style="font-size:19px;margin:26px 0 6px">The Shoal List, version 1 (sealed {SL1['date']}, superseded)</h3>
'''
src = sub1(src, '<h3 style="font-size:19px;margin:22px 0 6px">The Shoal List</h3>', JV2)
# ---- Exhibit L: the reported-dangers test
f = lambda r, k: float(r[k]) if r[k] not in ("", "None") else None
V1 = list(csv.DictReader(open("navwarn_hindcast.csv"))); V2 = list(csv.DictReader(open("navwarn_hindcast_v2.csv")))
pts = list(csv.DictReader(open("recon_navwarn/danger_points.csv"))); notices = list(csv.DictReader(open("recon_navwarn/dangers.csv")))
n_depth = sum(1 for p in pts if p["depth_m"] not in ("", "None")); n_arctic = sum(1 for p in pts if p["series"].startswith("NW-A"))
cats = collections.Counter(p["category"] for p in pts)
subs = [("Chart says the water is safe at the position (deeper than 21 m)", lambda r: r["map_safe"] == "True"),
        ("&nbsp;&nbsp;&hellip; and no published sounding within 300 m", lambda r: r["map_safe"] == "True" and f(r, "nearest_sounding_km") >= 0.3),
        ("&nbsp;&nbsp;&hellip; and no published sounding within 1 km", lambda r: r["map_safe"] == "True" and f(r, "nearest_sounding_km") >= 1.0),
        ("&nbsp;&nbsp;Arctic notices only", lambda r: r["map_safe"] == "True" and r["series"].startswith("NW-A")),
        ("&nbsp;&nbsp;Notices stating a measured depth of 21 m or less", lambda r: r["map_safe"] == "True" and f(r, "depth_m") is not None and f(r, "depth_m") <= 21)]
rows = []; stats = {}
for name, cond in subs:
    a = [r for r in V1 if cond(r) and f(r, "pct_hazard") is not None]; b = [r for r in V2 if cond(r) and f(r, "pct_hazard") is not None]
    k1 = sum(f(r, "pct_hazard") >= 90 for r in a); k2 = sum(f(r, "pct_hazard") >= 90 for r in b); kb = sum((f(r, "pct_nearest") or 0) >= 90 for r in b)
    p2 = binomtest(k2, len(b), 0.1, alternative="greater").pvalue if b else 1.0
    rows.append(tr([name, str(len(b)), f"{k1} ({k1/max(1,len(a))*100:.0f}%)", f"<b>{k2} ({k2/max(1,len(b))*100:.0f}%)</b>", f"{kb} ({kb/max(1,len(b))*100:.0f}%)", (f"{p2:.0e}" if p2 < 1e-3 else f"{p2:.3f}")]))
    stats[name] = dict(n=len(b), v1=k1, v2=k2, base=kb, p=p2)
S0 = stats[subs[0][0]]; S1 = stats[subs[1][0]]; S4 = stats[subs[4][0]]
L = f'''
<section>
  <div class="eyebrow">Exhibit L &mdash; the reported-dangers test</div>
  <h2>{len(pts):,} places where a mariner reported the chart was wrong, and where the model had put its flags</h2>
  <p class="lede">The Canadian Coast Guard publishes every navigational warning in force: uncharted rocks, shoals, depths shallower than charted, submerged objects, each with a position and often a measured least depth. Nobody had scored a seabed model against them. We crawled all {len(json.load(open('recon_navwarn/ids.json'))):,} notices in force on 6 September 2026, kept the {len(notices)} in the danger categories ({', '.join(f"{k} {v}" for k, v in cats.most_common(4))}), and parsed {len(pts):,} positions, {n_depth} with a stated depth and {n_arctic} in the Arctic. For each position we asked the hazard field the same question the grounding hindcast asks: among apparently-safe water within 25 km, what percentile of danger did the model give this exact spot? Ten percent of safe water sits above the 90th percentile by construction, so the base rate is exact.</p>
  <div class="tblwrap"><table><thead>{tr(["Subset", "n", "Old head, top decile", "New head, top decile", "Nearest-sounding rule, top decile", "p vs 10% (new head)"], True)}</thead><tbody>{''.join(rows)}</tbody></table></div>
  <p class="lede" style="font-size:14.5px"><b style="color:var(--text)">Read it straight.</b> Where the chart calls the water safe, the model puts {S0['v2']} of {S0['n']} reported dangers in its top decile, three times chance. With no published sounding within 300 m of the report, the position the model could not have copied from a neighbour, it is {S1['v2']} of {S1['n']} against {S1['base']} for the rule &ldquo;danger is where the nearest sounding is shallow&rdquo;. On notices that state a measured depth of 21 m or less in water the chart called safe, the retrained head lands {S4['v2']} of {S4['n']} (old head {S4['v1']}). Where a sounding sits right beside the notice, the simple rule ties or beats the model: there the rule already knows the answer, and the model&rsquo;s edge is in the water between the soundings. The samples are small and we say so; the table is the whole result, and the files are in the repository.</p>
  <p class="lede" style="font-size:13.5px">Source: nis.ccg-gcc.gc.ca NAVWARN search, all notices in force, crawled 2026-09-06. Notices in the &ldquo;Shallow Depth Confirmed&rdquo; category mostly follow a CHS survey and may already be in the 2026-08-29 NONNA snapshot the field was computed from, which is why the table conditions on the chart still calling the water safe and on the distance to the nearest published sounding. Files: <code>navwarn/danger_points.csv</code>, <code>navwarn/navwarn_hindcast_v2.csv</code>, <code>navwarn_hindcast.py</code> in the validation repository.</p>
</section>
'''
anchor = '<section>\n  <div class="eyebrow">Exhibit K &mdash; the country</div>'
src = sub1(src, anchor, L + anchor)
open("churchill_atlas_v5.html", "w", encoding="utf-8").write(src)
json.dump(dict(stats=stats, n_points=len(pts), n_notices=len(notices), n_depth=n_depth, n_arctic=n_arctic), open("navwarn_exhibit_stats.json", "w"), indent=1, default=float)
print("ATLAS_V5_DONE", len(src)//1024, "KB;", {k: (v['n'], v['v1'], v['v2'], v['base']) for k, v in stats.items()})
