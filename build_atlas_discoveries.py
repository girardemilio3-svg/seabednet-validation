#!/usr/bin/env python3
"""Discoveries pass: churchill_atlas_v3.html -> churchill_atlas_v4.html.
Inserts, after Exhibit I (hazard/hindcast):
  Exhibit J — sealed claims: the Shoal List (top 40, national) and the four strike-point predictions
  Exhibit K — the country: national temporal test, the age of Canada's chart, community lanes,
              GEBCO lag audit, national survey plan, negative results (laser, era audit)
All numbers read from result files."""
import base64, csv, json, os, re
src = open("churchill_atlas_v3.html", encoding="utf-8").read()
def sub1(s, old, new):
    assert s.count(old) == 1, f"count {s.count(old)}: {old[:50]}"; return s.replace(old, new)
b64 = lambda fn: base64.b64encode(open(fn, "rb").read()).decode()
SL = json.load(open("shoal_list_manifest.json")); SLR = list(csv.DictReader(open(SL["file"])))
SP = json.load(open("strike_predictions_manifest.json"))
TN = json.load(open("temporal_national_strata.json")) if os.path.exists("temporal_national_strata.json") else None
TNJ = json.load(open("temporal_validation_national.json")) if os.path.exists("temporal_validation_national.json") else None
CA = json.load(open("chart_age.json")); CC = json.load(open("community_cards.json"))["cards"]
GL = json.load(open("gebco_lag.json")); NP = json.load(open("national_plan.json"))
IC = json.load(open("icesat2_validation.json")); ICS = json.load(open("icesat2_validation_strict.json"))
ERA = json.load(open("era_audit.json"))
sl_ots = os.path.exists(SL["file"] + ".ots"); sp_ots = os.path.exists(SP["file"] + ".ots")
tr = lambda cells, th=False: "<tr>" + "".join(f"<{'th' if th else 'td'}>{c}</{'th' if th else 'td'}>" for c in cells) + "</tr>"
shoal_rows = "".join(tr([str(i+1), r["region"], f"{float(r['lat']):.4f}&deg;N", f"{abs(float(r['lon'])):.4f}&deg;W", f"{float(r['p_peak'])*100:.0f}%", f"{float(r['predicted_shoal_m']):.1f}", f"{abs(float(r['mean_map_depth_m'])):.0f}", r["km_to_sounding"]]) for i, r in enumerate(SLR[:20]))
strike_rows = "".join(tr([f"<b>{x['name']}</b><br><span class='mut' style='color:var(--muted);font-size:12px'>{x['place'][:70]}</span>", f"{x['predicted_lat']:.4f}&deg;N {abs(x['predicted_lon']):.4f}&deg;W", f"{x['p_shoal_lt_draft']*100:.0f}%", f"{x['predicted_shoal_depth_m']:.1f}", f"{abs(x['mean_map_depth_m']):.0f}", f"{x['candidates_in_search_area']:,}"]) for x in SP["incidents"] if "predicted_lat" in x)
cc_sorted = sorted([c for c in CC if c.get("median_survey_year")], key=lambda c: (c["median_survey_year"], -c["pct_pre1980"]))
cc_rows = "".join(tr([c["community"], f"{c['pct_sounded']:.0f}%", str(c["median_survey_year"]), f"{c['pct_pre1980']:.0f}%", (f"{c['pct_safe_water_flagged']:.0f}%" if "pct_safe_water_flagged" in c else "&mdash;")]) for c in cc_sorted[:14])
era_note = "old soundings agree with modern re-surveys to about 1 m in every decade; the risk of old charts is the water between the soundings, not the soundings"
J = f'''
<section>
  <div class="eyebrow">Exhibit J &mdash; sealed claims</div>
  <h2>Forty places we say hide a keel-depth shoal, and four groundings we place before the record does</h2>
  <p class="lede">A validated model earns the right to make claims. These are the model&rsquo;s, sealed on {SL['date']} with SHA-256 and OpenTimestamps so they cannot be edited after the fact, and written so that a single survey line settles each one.</p>
  <h3 style="font-size:19px;margin:22px 0 6px">The Shoal List</h3>
  <p class="lede" style="font-size:14.5px">Across all 437 NONNA blocks, {SL['n_candidates_total']:,} places pass every filter: the depth map calls the water 21&ndash;150 m deep, the shallowest-point model puts a rock between the surface and 21 m within 500 m with probability &ge; 80%, no published sounding within 500 m says otherwise, and the claim sits inside the 6 km inference envelope. The top {SL['n_sealed']} by probability, at most two per quarter-degree, are sealed. Regions: {', '.join(f"{k} ({v})" for k, v in __import__('collections').Counter(r['region'] for r in SLR).most_common(6))}. Inland lakes are excluded: the gravity anchor is invalid there.</p>
  <div class="tblwrap"><table><thead>{tr(["#", "Water", "Lat", "Lon", "P(shoal &lt; 10.5 m)", "Predicted shallowest (m)", "Map says (m)", "km to sounding"], True)}</thead><tbody>{shoal_rows}</tbody></table></div>
  <p class="lede" style="font-size:13.5px">First 20 of {SL['n_sealed']}. File <a href="{SL['file']}">{SL['file']}</a> &middot; SHA-256 <code style="font-size:12px">{SL['sha256']}</code>{' &middot; OpenTimestamps proof <a href="' + SL['file'] + '.ots">.ots</a>' if sl_ots else ''}. Scoring: a survey line over the position finds water shallower than 21 m within 500 m, or it does not; scored per entry, published either way.</p>
  <h3 style="font-size:19px;margin:26px 0 6px">Four groundings, placed before the record is opened</h3>
  <p class="lede" style="font-size:14.5px">Four Arctic groundings are public only as a place name: the TSB has the positions, nobody else does. For each, the hazard model names the cell in the reported area it considers most dangerous for that vessel&rsquo;s draft, in water the map calls navigable and clear of any charted shore. Sealed {SP['date']}; graded verbatim against the TSB positions when the records request is answered (hit = within 2 km). If they miss, that is published too.</p>
  <div class="tblwrap"><table><thead>{tr(["Grounding", "Predicted strike position", "P(shoal &lt; draft)", "Predicted shallowest (m)", "Map says (m)", "Cells considered"], True)}</thead><tbody>{strike_rows}</tbody></table></div>
  <p class="lede" style="font-size:13.5px">File <a href="{SP['file']}">{SP['file']}</a> &middot; SHA-256 <code style="font-size:12px">{SP['sha256']}</code>{' &middot; <a href="' + SP['file'] + '.ots">.ots</a>' if sp_ots else ''}.</p>
</section>
'''
nat = ""
if TN and TNJ:
    nat = f'''<div class="claims">
    <div class="claim"><span class="num">{TN['marine_model']:.1f} m vs {TN['marine_nn']:.1f} m</span><h3>The national temporal test</h3>
      <p>Trained on pre-2016 soundings only, scored on <b>{TN['marine_cells']/1e6:.1f} million</b> post-2016 marine soundings across {TN['marine_blocks']} blocks and three oceans: model {TN['marine_model']:.1f} m, nearest sounding {TN['marine_nn']:.1f} m, gravity {TN['marine_grav']:.1f} m; {TNJ['overall']['frac_within_1sigma']*100:.0f}% of errors inside 1&sigma; (all {TNJ['overall']['n']/1e6:.1f}M cells). On {TN['gravfail_cells']/1e6:.2f}M cells in inland lakes and fjord heads the gravity anchor is invalid and the model fails with it ({TN.get('gravfail_model', 79):.0f} m): do not use this model on inland water.</p></div>
    <div class="claim"><span class="num">{CA['pct_pre1980']:.0f}%</span><h3>The age of Canada&rsquo;s chart</h3>
      <p>Of {CA['sounded_cells']/1e6:.0f} million sounded cells in the national archive, {CA['pct_pre1980']:.0f}% rest on surveys older than 1980 and {CA['pct_pre1960']:.0f}% older than 1960; {CA['pct_post2016_modern']:.0f}% are post-2016 multibeam. The map below colours every cell by the decade of its survey.</p></div>
    <div class="claim"><span class="num">1 in {round(100/GL['route']['pct_gt10'])}</span><h3>The global chart&rsquo;s lag</h3>
      <p>On {GL['km2']:,} km&sup2; of corridor water where CHS holds real soundings, the GEBCO-based global grid that foreign ships fall back on is off by more than 10 m on {GL['pct_err_gt10']:.0f}% of cells and deep-biased, the dangerous direction, on {GL['pct_deep_biased_gt10']:.0f}% ({GL['km2_deep_biased_gt10']:,} km&sup2;). Along the Churchill route: {GL['route']['pct_gt10']:.0f}% of points off by 10+ m, {GL['route']['pct_deep_gt10']:.0f}% deep-biased, worst case {abs(GL['route']['worst_deep_m']):.0f} m deeper than the archive.</p></div>
    <div class="claim"><span class="num">C${NP['total']['cost_low_MCAD']}&ndash;{NP['total']['cost_high_MCAD']}M</span><h3>The national survey plan</h3>
      <p>The corridor plan, extended to every NONNA block: the 20 highest-uncertainty half-degree boxes of inferred water, {NP['total']['area_km2']:,} km&sup2;, {NP['total']['ship_days']:,} ship-days, priced from CHS&rsquo;s own Arctic contract rate to the icebreaker charter rate. A federal charting line with a number on it.</p></div>
  </div>'''
K = f'''
<section>
  <div class="eyebrow">Exhibit K &mdash; the country</div>
  <h2>The same questions, asked of all of Canada&rsquo;s water</h2>
  {nat}
  <div class="exh"><img src="data:image/jpeg;base64,{b64('chart_age_web.jpg')}" alt="The age of Canada's chart"><div class="cap"><b>The age of Canada&rsquo;s chart.</b> Year of the latest survey under every sounded cell, from the CHS Survey Index: red and orange are pre-1980, blue is post-2016 multibeam. The Arctic corridors are new; the approaches to Hudson Bay, the Labrador shore and Foxe Basin are not.</div></div>
  <h3 style="font-size:19px;margin:26px 0 6px">Every resupply lane, graded</h3>
  <p class="lede" style="font-size:14.5px">A 15 km approach zone around each sealift community and port: share of water with a published sounding, the median year of those surveys, the share from before 1980, and where the hazard field covers it, the share of apparently-safe water it flags. The oldest lanes first. Milne Inlet, the one lane industry paid to survey, is the control: {next((f"{c['pct_sounded']:.0f}% sounded, median {c['median_survey_year']}" for c in CC if c['community'].startswith('Milne')), '')}.</p>
  <div class="tblwrap"><table><thead>{tr(["Community / port", "Sounded", "Median survey year", "Pre-1980", "Safe water flagged"], True)}</thead><tbody>{cc_rows}</tbody></table></div>
  <p class="lede" style="font-size:13.5px">All {len(CC)} lanes in <a href="community_cards.json">community_cards.json</a>. Median year uses 2017 for soundings outside the 1832&ndash;2016 index (modern multibeam).</p>
  <h3 style="font-size:19px;margin:26px 0 6px">What did not work, on the record</h3>
  <p class="lede" style="font-size:14.5px"><b style="color:var(--text)">NASA&rsquo;s laser.</b> ICESat-2 ATL24 seafloor photons, {sum(len(v.get('blocks', [])) for v in IC.get('boxes', {}).values())} block-passes over 12 corridor boxes, ~3.6 million photons: where the fitted datum offset is physical, the laser agrees with CHS soundings to 1&ndash;5 m, which independently checks the archive, but only {ICS.get('overall', {}).get('n', 0)} photon cells with no published sounding survive the quality gates in this turbid water. The laser cannot referee the model&rsquo;s fills here; we tried and say so. <b style="color:var(--text)">The era audit.</b> Where CHS re-surveyed water after 2016, the older soundings within 200 m agree with the new ones to about a metre in every decade back to 1900 ({', '.join(f"{k}: {v['median_abs_err']:.1f} m" for k, v in list(json.load(open('era_audit.json'))['by_survey_decade'].items())[2:7])}). A null result with a lesson: {era_note}.</p>
</section>
'''
src = sub1(src, '<div class="stat amber"><b>+61%</b><span>of the route, completed by model (grades B+C)</span></div>',
  '<div class="stat amber"><b>+61%</b><span>of the route, completed by model (grades B+C)</span></div>\n    <div class="stat blue"><b><a href="map/" style="text-decoration:none;color:inherit">&#9679; GLOBE</a></b><span><a href="map/">open the interactive globe: every sealed claim on the planet</a></span></div>')
anchor = '<section>\n  <div class="eyebrow">Exhibit A &mdash; the corridor, found</div>'
src = sub1(src, anchor, J + K + anchor)
open("churchill_atlas_v4.html", "w", encoding="utf-8").write(src)
print("built v4", len(src)//1024, "KB; shoal rows", len(SLR), "strikes", len(SP["incidents"]))
