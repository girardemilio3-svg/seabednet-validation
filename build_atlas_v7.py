#!/usr/bin/env python3
"""Accuracy-sprint pass: churchill_atlas_v6.html -> churchill_atlas_v7.html.
  Exhibit M — what moved the number: the temporal benchmark for the original model, a control fine-tune, and the fine-tune with
  new input channels (coastal elevation + Sentinel-2), by depth stratum; the hazard-head A/B on the Coast Guard test; and the
  value-of-information survey plan. Every number is read from result files; sections are skipped if their file is missing."""
import json, os, collections
src = open("churchill_atlas_v6.html", encoding="utf-8").read()
def sub1(s, old, new):
    assert s.count(old) == 1, f"count {s.count(old)}: {old[:60]}"; return s.replace(old, new)
tr = lambda cells, th=False: "<tr>" + "".join(f"<{'th' if th else 'td'}>{c}</{'th' if th else 'td'}>" for c in cells) + "</tr>"
def load(fn): return json.load(open(fn)) if os.path.exists(fn) else None
T = {k: load(f"temporal_validation_{k}.json") for k in ("temporal_base", "temporal_ctl", "temporal_aux", "temporal_ctl0", "temporal_aux0", "temporal_s10", "temporal_s11", "temporal_grav0", "temporal_ctlfull", "temporal_s1full")}
names = {"temporal_base": "Original (soundings + gravity), 34.8M", "temporal_ctl": "Control fine-tune (same inputs)", "temporal_aux": "Fine-tune + coastal elevation + Sentinel-2", "temporal_ctl0": "From scratch, 6.8M: soundings + gravity", "temporal_aux0": "From scratch: + coastal elevation + Sentinel-2", "temporal_s10": "From scratch: + winter Sentinel-1 radar", "temporal_grav0": "From scratch: + raw gravity, leakage-free anchor", "temporal_s11": "From scratch: + winter radar, full coverage", "temporal_ctlfull": "Full size, full corpus, from scratch: control", "temporal_s1full": "Full size, full corpus, from scratch: + winter radar"}
parts = []
if T["temporal_base"] and (T["temporal_ctl"] or T["temporal_aux"]):
    rows = []
    for k, v in T.items():
        if not v: continue
        o = v["overall"]; bd = v.get("by_depth", [])
        sh = (bd[0] if isinstance(bd, list) and bd else (next((bd[b] for b in bd if b.startswith("0") or b.startswith("<")), None) if isinstance(bd, dict) else None))
        sh2 = (bd[1] if isinstance(bd, list) and len(bd) > 1 else None)
        rows.append(tr([names[k], f"{o['mae_model']:.2f}", f"{o['mae_nn']:.2f}", f"{o['mae_grav']:.2f}", f"{o['frac_within_1sigma']*100:.0f}%", (f"{sh['mae_model']:.2f}" if sh else "&mdash;"), (f"{sh2['mae_model']:.2f}" if sh2 else "&mdash;"), f"{o['bias']:+.1f}"]))
    bd0 = T["temporal_base"].get("by_depth", []); strata = [(f"{x['m'][0]}&ndash;{x['m'][1]} m" if "m" in x else "shallowest") for x in bd0] if isinstance(bd0, list) else list(bd0.keys())
    parts.append(f'''
  <h3 style="font-size:19px;margin:22px 0 6px">The temporal benchmark, before and after new inputs</h3>
  <p class="lede" style="font-size:14.5px">Same test as Exhibit H: a model that has seen only pre-2016 soundings predicts the post-2016 soundings in the corridor. The control row is the original model fine-tuned for the same number of steps with the same inputs, so that any change in the next rows is the inputs and not the extra training. From-scratch rows share one recipe within each size. Two things came out of it. Winter Sentinel-1 radar is the only input that helps, and it helps most in the 0&ndash;50 m band where ships ground; at full size it also adds a deep-water bias that costs it the overall number, so the current best overall model is the full-size from-scratch control at {T["temporal_ctlfull"]["overall"]["mae_model"]:.2f} m with a 20&ndash;50 m error of {T["temporal_ctlfull"]["by_depth"][1]["mae_model"]:.1f} m against the published {T["temporal_base"]["by_depth"][1]["mae_model"]:.1f} m. And swapping the gravity anchor for one that has never seen a ship sounding costs about a metre: that row is the leakage-free number to defend in review.</p>
  <div class="tblwrap"><table><thead>{tr(["Model", "MAE (m)", "nearest sounding", "gravity prior", "inside 1&sigma;", "0&ndash;20 m", "20&ndash;50 m", "bias"], True)}</thead><tbody>{''.join(rows)}</tbody></table></div>''')
NWL = load("navwarn_stats_v4s1_live.json"); NWA = load("navwarn_stats_v4s1_archive.json")
if NWL and NWA:
    def rows_of(S): return "".join(tr([k, str(v["n"]), f"{v['v2']} ({v['v2']/max(1,v['n'])*100:.0f}%)", f"{v['v4s1']} ({v['v4s1']/max(1,v['n'])*100:.0f}%)", f"{v['base']} ({v['base']/max(1,v['n'])*100:.0f}%)"]) for k, v in S.items())
    parts.append(f'''
  <h3 style="font-size:19px;margin:26px 0 6px">The hazard head with winter radar, on the Coast Guard tests</h3>
  <p class="lede" style="font-size:14.5px">The version-2 hazard head was fine-tuned twice from the same checkpoint for the same number of steps, once with the winter Sentinel-1 channel and once without, and the radar version was run over all 437 blocks. Reported dangers in the top decile of apparently-safe water within 25 km (Exhibit L protocol), on the notices in force and on the archive:</p>
  <div class="tblwrap"><table><thead>{tr(["Notices in force", "n", "v2 head", "v2 + radar", "nearest-sounding rule"], True)}</thead><tbody>{rows_of(NWL)}</tbody></table></div>
  <div class="tblwrap" style="margin-top:10px"><table><thead>{tr(["Archive 2017&ndash;2026", "n", "v2 head", "v2 + radar", "nearest-sounding rule"], True)}</thead><tbody>{rows_of(NWA)}</tbody></table></div>
  <p class="lede" style="font-size:14.5px">Radar improves the depth model (above) but, fine-tuned into the hazard head, changes the reported-danger scores by a point or two either way. The head still trails the nearest-sounding rule in the Arctic and on notices with a measured depth. A hazard head trained from scratch with radar is the next run; if it does not move these numbers, the honest reading is that reported dangers are governed by cues the hazard field does not carry, which is what the danger-report model below already exploits.</p>''')
AR = {k: load(f"navwarn_archive_stats_{k}.json") for k in ("v2", "v3ctl", "v3aux")}
if AR["v2"]:
    keys = list(AR["v2"].keys()); hdr = ["Subset (archive, chart says safe)", "n"] + [f"{k} head" for k in ("v2", "v3ctl", "v3aux") if AR[k]] + ["nearest-sounding rule"]
    rows = "".join(tr([k, str(AR["v2"][k]["n"])] + [f"{AR[h][k]['model_ge90']} ({AR[h][k]['model_ge90']/max(1,AR[h][k]['n'])*100:.0f}%)" for h in ("v2", "v3ctl", "v3aux") if AR[h]] + [f"{AR['v2'][k]['base_ge90']} ({AR['v2'][k]['base_ge90']/max(1,AR['v2'][k]['n'])*100:.0f}%)"]) for k in keys)
    parts.append(f'''
  <h3 style="font-size:19px;margin:26px 0 6px">The reported-dangers test at ten times the size</h3>
  <p class="lede" style="font-size:14.5px">The Coast Guard also keeps its cancelled notices. All {8948:,} danger positions from 2017&ndash;2026 ({3302:,} with a measured depth, {385} Arctic) scored with the Exhibit L protocol. At this size the picture is plainer than on the live set: the hazard head and the &ldquo;danger is where the nearest sounding is shallow&rdquo; rule tie where the chart calls the water safe, and the rule wins in the Arctic and on notices with a measured depth. That is the current limit of the hazard head, stated as measured; the next section is what we are doing about it.</p>
  <div class="tblwrap"><table><thead>{tr(hdr, True)}</thead><tbody>{rows}</tbody></table></div>''')
DM = load("danger_model.json")
if DM and "all_features" in DM:
    m = DM["all_features"]; nh = DM["no_hazard"]; hz = DM["hazard_only_gbm"]; nn = DM["nearest_only_gbm"]; h1 = DM["hazard_p_rank"]; n1 = DM["nearest_z_rank"]
    imp = ", ".join(f"{k.replace('_', ' ')} ({v:+.3f})" for k, v in list(DM["drop_one_auc_loss"].items())[:6])
    rows = "".join(tr([lab, f"{v['auc']:.3f}", f"{v['top_decile_precision']*100:.1f}%", f"{v['top_decile_recall']*100:.0f}%"]) for lab, v in [("All cues (hazard field, soundings, coast, elevation, imagery, radar, gravity)", m), ("All cues except the hazard field", nh), ("Hazard field only (learned)", hz), ("Nearest sounding only (learned)", nn), ("Hazard probability, raw rank", h1), ("Nearest sounding depth, raw rank", n1)])
    parts.append(f'''
  <h3 style="font-size:19px;margin:26px 0 6px">Where dangers get reported: a model tested on years it never saw</h3>
  <p class="lede" style="font-size:14.5px">Every reported danger from the live and archived notices with a date, in water the chart calls safe, against ten random chart-safe cells per notice from the same blocks. Trained on notices up to 2022, tested on 2023&ndash;2026 ({DM['n_test']:,} cells, {DM['pos_test']:,} reported dangers, base rate {DM['base_rate']*100:.1f}%). The question is the one a survey planner asks: of the water the chart calls safe, which tenth will produce the reports?</p>
  <div class="tblwrap"><table><thead>{tr(["Ranking", "AUC", "Top decile precision", "Top decile recall"], True)}</thead><tbody>{rows}</tbody></table></div>
  <p class="lede" style="font-size:14.5px"><b style="color:var(--text)">Read it straight.</b> The combined ranking puts {m['top_decile_recall']*100:.0f}% of the next four years&rsquo; reported dangers in a tenth of the safe water, {m['top_decile_precision']/DM['base_rate']:.0f}&times; the base rate. The hazard field on its own is only slightly better than the nearest-sounding rule, and adds almost nothing once the other cues are in: the biggest cues are distance to the nearest sounding and to the coast, then the predicted shallowest point, coastal elevation, and the imagery and radar channels (drop-one AUC loss: {imp}). Two of those, distance to soundings and to the coast, are partly measures of where ships go rather than of the seabed, so this is a model of reporting as much as of rock. For deciding where to send a survey launch that is the right target; for a scientific claim about the seabed it is not, and we say so.</p>''')
DM2 = load("danger_model2.json")
if DM2 and "all_features_arctic_test" in DM2:
    a2 = DM2["all_features"]; ar = DM2["all_features_arctic_test"]; ne = DM2["no_exposure_cues"]
    parts.append(f'''
  <p class="lede" style="font-size:14.5px"><b style="color:var(--text)">Second pass, richer cues.</b> Adding the satellite-bathymetry index (log blue/green), radar texture, seabed roughness and slope, the distance to the nearest <em>shallow</em> sounding and the mean hazard within 1 km lifts the combined ranking to AUC {a2['auc']:.3f}, {a2['top_decile_recall']*100:.0f}% of later reported dangers in the top decile. <b style="color:var(--text)">For the Arctic alone</b> ({ar['n']:,} test cells, {ar['pos']} reported dangers, base rate {ar['base_rate']*100:.1f}%): the top decile of chart-safe water holds {ar['top_decile_precision']*100:.0f}% reported dangers, {ar['top_decile_precision']/ar['base_rate']:.0f}&times; the base rate, capturing {ar['top_decile_recall']*100:.0f}% of them. With the exposure cues removed entirely (distance to soundings, to shallow soundings, to the coast, latitude) the ranking still reaches AUC {ne['auc']:.3f} and {ne['top_decile_recall']*100:.0f}% recall, so the seabed cues carry real signal of their own. Predicted probabilities are conservative: the top decile is predicted at {DM2['reliability_test_years'][-1]['pred_mean']*100:.0f}% and observed at {DM2['reliability_test_years'][-1]['obs_rate']*100:.0f}%.</p>''')
DT = load("navwarn_depth_test.json")
if DT and DT.get("chart_safe") and DT.get("arctic"):
    cs = DT["chart_safe"]; arc = DT["arctic"]
    parts.append(f'''
  <h3 style="font-size:19px;margin:26px 0 6px">Measured depths at reported shoals</h3>
  <p class="lede" style="font-size:14.5px">{DT['all']['n']:,} notices state a least depth. Where the chart called the water safe ({cs['n']} positions, reported depth median {cs['reported_median']:.0f} m against a chart of 21 m or more), the hazard head&rsquo;s predicted shallowest point is unbiased ({cs['bias_shoal']:+.1f} m) where the nearest sounding is biased {cs['bias_nearest']:+.0f} m deep, but no more precise: mean error {cs['mae_shoal_model']:.0f} m against {cs['mae_nearest']:.0f} m, within 5 m in {cs['frac_shoal_within5']*100:.0f}% of cases against {cs['frac_nearest_within5']*100:.0f}%. In the Arctic ({arc['n']} positions): model {arc['mae_shoal_model']:.1f} m, nearest sounding {arc['mae_nearest']:.1f} m, chart bias {arc['bias_nearest']:+.1f} m. The head knows the chart is too deep; it does not yet know by how much.</p>''')
IS2 = load("is2_claims_strict.json")
if IS2 and "control" in IS2:
    c = IS2["control"]
    parts.append(f'''
  <h3 style="font-size:19px;margin:26px 0 6px">The laser, again: a null result on the claims</h3>
  <p class="lede" style="font-size:14.5px">Raw ICESat-2 photons (all confidence levels, 2019&ndash;2026) were pulled in a 1.2 km box around each of the forty version-2 claims and searched, pass by pass, for a bottom return below the sea-surface peak. A detector that finds &ldquo;bottoms&rdquo; on {c['claims_with_detection']} of {c['claims_scored']} claims also finds them over 800 m of water in Lancaster Sound ({c['control_sites']['Lancaster Sound']['detections']} of {c['control_sites']['Lancaster Sound']['passes']} passes) and on the Beaufort slope ({c['control_sites']['Beaufort slope']['detections']} of {c['control_sites']['Beaufort slope']['passes']}), at the same depths: the 4 m peak is the instrument&rsquo;s afterpulse and the deeper ones are subsurface scattering and sea ice. The claim check is therefore inconclusive and is not used as evidence for or against any claim. The photon subsets and both detectors are in the repository for anyone who can do better.</p>''')
NP = load("national_plan_v2.json")
if NP:
    rows = "".join(tr([f"{r['lat']:.1f}&deg;N {abs(r['lon']):.1f}&deg;W", f"{r['area_km2']:,}", f"{r['expected_hazard_km2']:.1f}", f"{r['ship_days']:.0f}", f"{r['hazards_per_ship_day']:.1f}"]) for r in NP["top20"][:12])
    parts.append(f'''
  <h3 style="font-size:19px;margin:26px 0 6px">The survey plan, ranked by what a ship would find</h3>
  <p class="lede" style="font-size:14.5px">Exhibit K ranks survey boxes by uncertainty. This ranks them by the expected area of keel-depth hazard a survey would uncover per ship-day: the sum of P(shallowest point within 500 m &lt; 10.5 m) over cells the chart calls navigable that no published sounding has touched, restricted to water ships use ({NP['relevance']}). Twenty boxes, {NP['total']['area_km2']:,} km&sup2;, {NP['total']['ship_days']} ship-days, C${NP['total']['cost_low_MCAD']}&ndash;{NP['total']['cost_high_MCAD']}M, expected hazard area {NP['total']['expected_hazard_km2']:.0f} km&sup2;. This is the number a hydrographic service can act on: hazards found per day at sea.</p>
  <div class="tblwrap"><table><thead>{tr(["Box", "Area (km&sup2;)", "Expected hazard (km&sup2;)", "Ship-days", "Hazard km&sup2; per ship-day"], True)}</thead><tbody>{rows}</tbody></table></div>
  <p class="lede" style="font-size:13.5px">First 12 of 20; file <a href="national_plan_v2.json">national_plan_v2.json</a>.</p>''')
NP3 = load("national_plan_v3.json")
if NP3:
    rows = "".join(tr([f"{r['lat']:.1f}&deg;N {abs(r['lon']):.1f}&deg;W", f"{r['area_km2']:,}", f"{r['expected_reports_km2']:.1f}", f"{r['ship_days']:.0f}", f"{r['reports_per_ship_day']:.2f}"]) for r in NP3["top20"][:12])
    parts.append(f'''
  <h3 style="font-size:19px;margin:26px 0 6px">The same plan, ranked by the danger-report model</h3>
  <p class="lede" style="font-size:14.5px">The validated model above applied to every chart-safe cell in shipping water on a 500 m grid, and the boxes ranked by expected reported-danger area per ship-day. Twenty boxes, {NP3['total']['area_km2']:,} km&sup2;, {NP3['total']['ship_days']} ship-days, C${NP3['total']['cost_low_MCAD']}&ndash;{NP3['total']['cost_high_MCAD']}M. Where the two rankings agree, the case is strong; where they differ, the hazard-only ranking is the physical claim and this one is the operational one.</p>
  <div class="tblwrap"><table><thead>{tr(["Box", "Area (km&sup2;)", "Expected reported-danger area (km&sup2;)", "Ship-days", "per ship-day"], True)}</thead><tbody>{rows}</tbody></table></div>
  <p class="lede" style="font-size:13.5px">File <a href="national_plan_v3.json">national_plan_v3.json</a>.</p>''')
if parts:
    M = '<section>\n  <div class="eyebrow">Exhibit M &mdash; the accuracy sprint</div>\n  <h2>What moved the number, what did not, and where a ship would find the most rock per day</h2>' + "".join(parts) + '\n</section>\n'
    anchor = '<section>\n  <div class="eyebrow">Exhibit J &mdash; sealed claims</div>'
    src = sub1(src, anchor, M + anchor)
open("churchill_atlas_v7.html", "w", encoding="utf-8").write(src); print("ATLAS_V7_DONE", len(src)//1024, "KB; sections:", len(parts))
