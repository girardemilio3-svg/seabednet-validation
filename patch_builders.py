#!/usr/bin/env python3
"""One-shot patch of build_atlas_v2.py, build_atlas_hazard.py and build_report.py for the
review items: stronger baselines + GEBCO reference, gravity-leakage statement, blind hindcast
and sensitivity with skill p-values, channel cost, survey-rate derivation and rate range,
external timestamp, label fixes. All new numbers are read from baselines.json, gebco_eval.json,
hindcast_blind.json, survey_rates.json (guarded: sections render only if the file exists)."""
import re
def sub1(p, old, new):
    s = open(p, encoding="utf-8").read(); assert s.count(old) == 1, f"{p}: count {s.count(old)} for {old[:60]!r}"; open(p, "w", encoding="utf-8").write(s.replace(old, new))

# ============================================================ build_atlas_v2.py
p = "build_atlas_v2.py"
sub1(p, 'F = json.load(open("forecast_manifest.json")); prof = json.load(open("route_profile_v2.json"))',
'''F = json.load(open("forecast_manifest.json")); prof = json.load(open("route_profile_v2.json"))
BL = json.load(open("baselines.json")) if os.path.exists("baselines.json") else None
GB = json.load(open("gebco_eval.json")) if os.path.exists("gebco_eval.json") else None
SR = json.load(open("survey_rates.json")) if os.path.exists("survey_rates.json") else None
cf = json.load(open("corridor_found.json"))["stats"]; plan_days = sum(x["ship_days"] for x in json.load(open("survey_plan.json"))); plan_area = sum(x["area_km2"] for x in json.load(open("survey_plan.json")))
OTS = os.path.exists("forecast_2026-09-02.csv.ots")''')
# --- header labels
sub1(p, """  '<div class="stat amber"><b>+61%</b><span>completed by SeabedNet</span></div>\\n'""",
        """  '<div class="stat amber"><b>+61%</b><span>of the route, completed by model (grades B+C)</span></div>\\n'""")
sub1(p, "f\"{(1-to['mae_model']/to['mae_grav'])*100:.0f}% better than the gravity-derived bathymetry every global chart falls back on.</p>\")",
        "f\"{(1-to['mae_model']/to['mae_grav'])*100:.0f}% better than the gravity-derived bathymetry every global chart falls back on. (The 17% and CHS&rsquo;s 15.8% are different measures: soundings under this route versus Arctic waters surveyed to modern standard; they are not cited as corroborating each other.)</p>\")")
# --- Exhibit G: baseline rows + leakage paragraph
sub1(p, '''def row(label, a, b, c, bold=False):
    l = f"<b>{label}</b>" if bold else label
    return f"<tr><td>{l}</td><td>{a}</td><td>{b}</td><td>{c}</td></tr>"''',
'''def row(label, a, b, c, bold=False, extra=""):
    l = f"<b>{label}</b>" if bold else label
    return f"<tr><td>{l}</td><td>{a}</td><td>{b}</td><td>{c}</td>{extra}</tr>"
def bl1(k):
    if not BL: return ""
    d = BL["test1"]["shelf_50_400"] if k == "all" else None
    return ""
BLROW1 = ""; BLROW2 = ""; BLNOTE = ""
if BL:
    b1 = BL["test1"]["shelf_50_400"]; b2 = BL["test2"]["all"]
    g1 = GB["test1"]["shelf_50_400"]["gebco"] if GB else None; g2 = GB["test2"]["all"]["gebco"] if GB else None
    BLROW1 = f"<tr><td><b>Shelf, all distances &mdash; stronger baselines</b></td><td><b>{fmt(b1['model'])}</b></td><td>gravity trend + natural-neighbour residual {fmt(b1['trend_natural'])} &middot; + inverse-distance residual {fmt(b1['trend_idw'])}</td><td>{fmt(b1['gravity'])}" + (f" &middot; GEBCO {fmt(g1)}*" if g1 else "") + "</td></tr>"
    BLROW2 = f"<tr><td><b>All distances &mdash; stronger baselines</b></td><td><b>{fmt(to['mae_model'])}</b></td><td>gravity trend + natural-neighbour residual {fmt(b2['trend_natural'])} &middot; + inverse-distance residual {fmt(b2['trend_idw'])}</td><td>{fmt(b2['gravity'])}" + (f" &middot; GEBCO {fmt(g2)}*" if g2 else "") + "</td></tr>"
    L = BL["leakage"]
    BLNOTE = (f"<p class=\\"lede\\" style=\\"font-size:14.5px\\"><b style=\\"color:var(--text)\\">Baselines a hydrographer would actually use.</b> Copying the nearest sounding is a floor, not a method. The classical gap fill is a gravity trend plus interpolated residuals; we report it two ways (Delaunay natural-neighbour and inverse-distance-squared on the 12 nearest residuals). "
               f"On Test 1 the model beats both ({fmt(b1['model'])} vs {fmt(b1['trend_natural'])} / {fmt(b1['trend_idw'])} m); on Test 2, {fmt(to['mae_model'])} vs {fmt(b2['trend_natural'])} / {fmt(b2['trend_idw'])} m. "
               + (f"*GEBCO (NCEI global mosaic, 15 arc-second) is shown as the chart-world reference, not a baseline: it ingests NONNA-100 through IBCAO v5 (the Arctic Seabed 2030 compilation) and the same NCEI cruise archive, so it has seen both test sets; on the 100 m grid it scores {fmt(g1)} m (Test 1) and {fmt(g2)} m (Test 2). " if g1 else "")
               + f"<b style=\\"color:var(--text)\\">Gravity leakage, measured.</b> SRTM15+ V2.7 (April 2025) inherits the cumulative NCEI multibeam archive of its point releases, which is where the Test 1 cruises live. Its error on the Test 1 cells is {fmt(L['gravity_mae_on_test1_shelf_cells'])} m against {fmt(L['gravity_mae_on_nonna_sounded_shelf_cells_same_blocks'])} m on {L['n_sounded_cells']:,} ordinary NONNA-sounded shelf cells in the same blocks: the prior has almost certainly seen the cruise data. That leakage favours the <em>baseline</em>, which the model still edges on Test 1; the model itself takes gravity as an input and inherits some of it. Test 2 is the clean comparison: NONNA shelf soundings SRTM15+ does not resolve, where gravity scores {fmt(to['mae_grav'])} m against the model&rsquo;s {fmt(to['mae_model'])} m.</p>")''')
sub1(p, "    <tbody><tr><td colspan=\"4\" style=\"color:var(--muted);font-family:var(--mono);font-size:11px;letter-spacing:.1em\">TEST 1 &middot; INDEPENDENT MULTIBEAM, SHELF 50&ndash;400 M</td></tr>{ind_rows}\n    <tr><td colspan=\"4\" style=\"color:var(--muted);font-family:var(--mono);font-size:11px;letter-spacing:.1em\">TEST 2 &middot; PRE-2016 MODEL vs POST-2016 CHS SOUNDINGS, CORRIDOR</td></tr>{tmp_rows}</tbody>\n  </table></div>",
        "    <tbody><tr><td colspan=\"4\" style=\"color:var(--muted);font-family:var(--mono);font-size:11px;letter-spacing:.1em\">TEST 1 &middot; INDEPENDENT MULTIBEAM, SHELF 50&ndash;400 M</td></tr>{ind_rows}{BLROW1}\n    <tr><td colspan=\"4\" style=\"color:var(--muted);font-family:var(--mono);font-size:11px;letter-spacing:.1em\">TEST 2 &middot; PRE-2016 MODEL vs POST-2016 CHS SOUNDINGS, CORRIDOR</td></tr>{tmp_rows}{BLROW2}</tbody>\n  </table></div>\n  {BLNOTE}")
# --- Exhibit A reframe + channel cost
sub1(p, "src = sub1(src, \"gives the strike site a &sigma; of 48&nbsp;m",
'''src = sub1(src, "<b style=\\"color:var(--text)\\">not one kilometre over unanswered water</b>. Where the straight route crosses red, the found channel goes around.</p>",
  f"<b style=\\"color:var(--text)\\">not one kilometre over unanswered water</b>. Where the straight route crosses red, the found channel goes around. Read it as a map of where the risk is, not as a sailing direction: the channel is {cf['length_km']-2327:,} km longer than the straight route, about {(cf['length_km']-2327)/1.852/12:.0f} hours at 12 knots, roughly {(cf['length_km']-2327)/1.852/12/24*25:.0f} t of fuel and one day of hire per voyage (a Handysize at ~25 t/day; ~US$25k). Nobody should pay that on every voyage; the point is that the water the straight route crosses has never been answered for, and that is what the survey plan below is priced to fix.</p>")
src = sub1(src, "gives the strike site a &sigma; of 48&nbsp;m''')
# --- Exhibit E: rate derivation + two rates
sub1(p, "# ---------- Exhibit H: forecast (before Exhibit F)",
'''# ---------- Exhibit E: rate derivation and a second day rate
if SR:
    lo_cost = plan_area*SR["cost_per_km2_cad"]; lo_src = SR["source"]
    src = sub1(src, "route, priced in ship-days at C$183,000/day &mdash; the Coast Guard&rsquo;s July 2026 polar-icebreaker charter (C$22M for ~120 days).</p>",
      f"route, priced in ship-days. Coverage assumption: 40 km&sup2;/day, from a multibeam swath of ~3&times; water depth (300&ndash;400 m at 100&ndash;130 m), 8 knots for 20 hours (~300 line-km/day, ~100 km&sup2; raw), 20% line overlap and a 50% weather-and-ice downtime factor across an Arctic season. Two public contracts bracket the cost. Lower bound: {lo_src}, which prices the plan&rsquo;s {plan_area:,.0f} km&sup2; at about C${lo_cost/1e6:.0f}M. Upper bound: the Coast Guard&rsquo;s July 2026 polar-icebreaker charter at C$183,000/day (C$22M for ~120 days), which the table uses. For scale, {SR['amundsen_context']}.</p>")
    src = sub1(src, "Total: <b style=\\"color:var(--text)\\">~219 ship-days, C$40.1M</b>",
      f"Total: <b style=\\"color:var(--text)\\">~{plan_days:.0f} ship-days for {plan_area:,.0f} km&sup2;: C${lo_cost/1e6:.0f}M at the CHS contract rate, C$40.1M at the icebreaker charter rate</b>")
# ---------- Exhibit H: forecast (before Exhibit F)''')
# --- Exhibit H: seal wording + OTS
sub1(p, "    <tr><td>Model</td><td>{F['model']}</td></tr>",
        "    <tr><td>Model</td><td>SeabedNet v5-small, the 34.8M-parameter completion model used throughout this page (trained on the full archive), with the depth gate and calibrated &sigma; of Exhibit G</td></tr>\n    {'<tr><td>External timestamp</td><td>OpenTimestamps proof <a href=\"forecast_2026-09-02.csv.ots\">forecast_2026-09-02.csv.ots</a>, submitted to three public calendars (Bitcoin-anchored; verify with <code>ots verify</code>), plus the hash in a GitHub commit in the validation repository. Neither clock is ours.</td></tr>' if OTS else ''}")
sub1(p, "<h2>{F['n']:,} predicted depths, sealed on {F['date']}</h2>", "<h2>{F['n']:,} predicted depths, sealed on {F['date']} &mdash; with a clock we do not control</h2>")

# ============================================================ build_atlas_hazard.py
p = "build_atlas_hazard.py"
sub1(p, 'hc = json.load(open("hindcast.json")); hz = json.load(open("hazard_corridor.json"))',
'''hc = json.load(open("hindcast.json")); hz = json.load(open("hazard_corridor.json"))
HB = json.load(open("hindcast_blind.json")) if os.path.exists("hindcast_blind.json") else None
def ordn(p):
    p=int(round(p)); return f"{p}{'th' if 10<=p%100<=20 else {1:'st',2:'nd',3:'rd'}.get(p%10,'th')}"
BLIND = ""
if HB:
    S = HB["sites"]; K = HB["skill"]; names = ["Thamesborg", "Akademik Ioffe", "Clipper Adventurer", "Hanseatic", "Nanny 2012", "Nanny 2014"]
    def cell(n, mask, key="win25km_safe2x"):
        r = S.get(n, {}).get(mask, {}); v = r.get("ranks", {}).get(key, {}).get("percentile"); return "&mdash;" if v is None else f"{v:.0f}"
    rows_b = "".join(f"<tr><td><b>{n}</b></td><td>{cell(n,'mask_0km')}</td><td>{cell(n,'mask_10km')}</td><td>{cell(n,'mask_25km')}</td><td>{cell(n,'mask_25km','win10km_safe2x')}</td><td>{cell(n,'mask_25km','win50km_safe2x')}</td><td>{cell(n,'mask_25km','win25km_safe1.5x')}</td><td>{cell(n,'mask_25km','win25km_safe3x')}</td></tr>" for n in names if n in S)
    k25 = K["mask_25km"]; k0 = K["mask_0km"]
    th = S["Thamesborg"]["mask_25km"]; th_pct = th["ranks"]["win25km_safe2x"]["percentile"]
    BLIND = f"""
  <h2 style="font-size:24px;margin-top:34px">The same hindcast, blind, and how much the denominator matters</h2>
  <p class="lede">The table above still lets the model see every published sounding around a site at inference time, and NONNA carries no dates after 2016, so a post-grounding survey could be in that input. Here every sounding within 10 km and within 25 km of each site is removed from the input as well (the hazard model was trained with every tile group within 0.4&deg; of the sites excluded from the start). Then the two choices that set the percentile &mdash; the comparison window and the definition of &ldquo;looked safe&rdquo; &mdash; are varied. Thamesborg with a 25 km hole in its input: <b style="color:var(--text)">{ordn(th_pct)} percentile</b>, nearest remaining sounding {th['dist_to_input_sounding_km']:.1f} km away.</p>
  <div class="tblwrap"><table>
    <thead><tr><th>Grounding</th><th>Input: all soundings</th><th>Input: 10 km hole</th><th>Input: 25 km hole</th><th>25 km hole, 10 km window</th><th>25 km hole, 50 km window</th><th>25 km hole, safe = 1.5&times; draft</th><th>25 km hole, safe = 3&times; draft</th></tr></thead>
    <tbody>{rows_b}</tbody></table></div>
  <p class="lede" style="font-size:14px"><b style="color:var(--text)">Skill, not recall.</b> By construction 10% of apparently-safe cells sit above the 90th percentile, so the base rate is known exactly. With the 25 km hole, {k25['hits_all'][0]} of {k25['hits_all'][1]} sites land above it (binomial p = {k25['p_binomial_all']:.4f}); on the four uncharted-shoal groundings alone, {k25['hits_uncharted4'][0]} of {k25['hits_uncharted4'][1]} (p = {k25['p_binomial_uncharted4']:.4f}). With all soundings visible: {k0['hits_all'][0]} of {k0['hits_all'][1]} (p = {k0['p_binomial_all']:.4f}). The precision side of the ledger is stated too: corridor-wide, {hz['frac_p_gt_5pct']*100:.0f}% of cells exceed P = 5%, and on the route {hz['km_flagged']:.0f} km of {hz['km_mean_map_safe']:.0f} apparently-safe km are flagged. Six incidents cannot bound a false-alarm rate; what the flags buy is a survey order, and the survey is what tests them.</p>"""''')
sub1(p, "  <p class=\"lede\" style=\"font-size:14px\">Thamesborg uses all published soundings because NONNA carries no dates after 2016;",
        "  {BLIND}\n  <p class=\"lede\" style=\"font-size:14px\">Thamesborg (table above) uses all published soundings because NONNA carries no dates after 2016;")
sub1(p, "def ordn(p):\n    p=int(round(p)); return f\"{p}{'th' if 10<=p%100<=20 else {1:'st',2:'nd',3:'rd'}.get(p%10,'th')}\"\nn_scored = len(landed) + len(missed)", "n_scored = len(landed) + len(missed)")

# ============================================================ build_report.py
p = "build_report.py"
sub1(p, 'plan = json.load(open("survey_plan.json"));',
'''BL = json.load(open("baselines.json")) if os.path.exists("baselines.json") else None
GB = json.load(open("gebco_eval.json")) if os.path.exists("gebco_eval.json") else None
HB = json.load(open("hindcast_blind.json")) if os.path.exists("hindcast_blind.json") else None
SR = json.load(open("survey_rates.json")) if os.path.exists("survey_rates.json") else None
OTS = os.path.exists("forecast_2026-09-02.csv.ots")
plan = json.load(open("survey_plan.json"));''')
sub1(p, 't1 += tr(["<b>all shelf 50–400 m</b>",',
'''if BL:
    b1 = BL["test1"]["shelf_50_400"]
    t1 += tr(["<i>trend + natural-neighbour residual</i>", "", "", f1(b1["trend_natural"]), ""]) + tr(["<i>trend + inverse-distance residual</i>", "", "", f1(b1["trend_idw"]), ""])
    if GB: t1 += tr(["<i>GEBCO reference (not independent, §4.5)</i>", "", "", f1(GB["test1"]["shelf_50_400"]["gebco"]), ""])
t1 += tr(["<b>all shelf 50–400 m</b>",''')
sub1(p, 't2 += tr(["<b>all</b>",',
'''if BL:
    b2 = BL["test2"]["all"]
    t2 += tr(["<i>trend + natural-neighbour residual</i>", "", "", f1(b2["trend_natural"]), "", "", ""]) + tr(["<i>trend + inverse-distance residual</i>", "", "", f1(b2["trend_idw"]), "", "", ""])
    if GB: t2 += tr(["<i>GEBCO reference (not independent, §4.5)</i>", "", "", f1(GB["test2"]["all"]["gebco"]), "", "", ""])
t2 += tr(["<b>all</b>",''')
sub1(p, "<h2>5. Results: hazard field and grounding hindcast</h2>",
'''<h3>4.5 Stronger baselines, GEBCO, and gravity leakage</h3>
{("<p>Nearest-sounding is a floor. The classical gap fill is a gravity trend plus interpolated residuals (sounding minus SRTM15+ at the input soundings), interpolated by Delaunay natural-neighbour-style linear interpolation or by inverse-distance-squared over the 12 nearest residuals. On Test 1 these score %s and %s m against the model's %s m; on Test 2, %s and %s m against %s m (Tables 1 and 3, italic rows). GEBCO (NCEI global mosaic, GEBCO 2024/25, 15 arc-second) is reported as the chart-world reference and scores %s m (Test 1) and %s m (Test 2) on the 100 m grid; it is not an independent baseline because it ingests NONNA-100 through IBCAO v5 (the Arctic Seabed 2030 compilation) and the same NCEI cruise archive. <b>Gravity leakage.</b> SRTM15+ V2.7 (released April 2025) inherits the cumulative NCEI multibeam archive of its point releases. Its error on the Test 1 cells is %s m against %s m on %s ordinary NONNA-sounded shelf cells in the same blocks; the prior has almost certainly seen the cruise data. This favours the gravity baseline, which the model still edges on Test 1, and the model, which takes gravity as an input, inherits part of it. Test 2 is therefore the clean comparison: those NONNA shelf soundings are below SRTM15+'s resolution, and there gravity scores %s m against the model's %s m.</p>" % (f1(BL["test1"]["shelf_50_400"]["trend_natural"]), f1(BL["test1"]["shelf_50_400"]["trend_idw"]), f1(BL["test1"]["shelf_50_400"]["model"]), f1(BL["test2"]["all"]["trend_natural"]), f1(BL["test2"]["all"]["trend_idw"]), f1(T["overall"]["mae_model"]), f1(GB["test1"]["shelf_50_400"]["gebco"]) if GB else "—", f1(GB["test2"]["all"]["gebco"]) if GB else "—", f1(BL["leakage"]["gravity_mae_on_test1_shelf_cells"]), f1(BL["leakage"]["gravity_mae_on_nonna_sounded_shelf_cells_same_blocks"]), f"{BL['leakage']['n_sounded_cells']:,}", f1(T["overall"]["mae_grav"]), f1(T["overall"]["mae_model"]))) if BL else ""}

<h2>5. Results: hazard field and grounding hindcast</h2>''')
sub1(p, '{fig("thamesborg_exhibit_web.jpg",',
'''{("<h3>5.1 Blind inputs and sensitivity</h3><p>Table 6 lets the model see every published sounding around a site at inference; NONNA carries no dates after 2016, so a post-incident survey could be among them. Table 6b removes every sounding within 10 km and within 25 km of each site from the input as well (the hazard model was trained with all tile groups within 0.4° of the sites excluded), and varies the two denominator choices: comparison window (10/25/50 km) and the apparently-safe filter (1.5×, 2×, 3× draft on the mean map). Skill: by construction 10%% of apparently-safe cells lie above the 90th percentile, so with the 25 km hole the %d-of-%d result has binomial p = %.4f (all six) and %d of %d, p = %.4f, on the four uncharted-shoal groundings; with all soundings visible, %d of %d (p = %.4f). Precision cannot be bounded from six incidents; corridor-wide %.0f%% of cells exceed P = 5%% and %.0f of %.0f apparently-safe route-km are flagged.</p><div class='tw'><table><thead>%s</thead><tbody>%s</tbody></table></div><div class='tcap'><b>Table 6b.</b> Hazard percentile of the strike site under input holes and denominator choices (window / safe-filter). 25 km hole with the 25 km window and 2× draft is the reference blind number.</div>" % (HB["skill"]["mask_25km"]["hits_all"][0], HB["skill"]["mask_25km"]["hits_all"][1], HB["skill"]["mask_25km"]["p_binomial_all"], HB["skill"]["mask_25km"]["hits_uncharted4"][0], HB["skill"]["mask_25km"]["hits_uncharted4"][1], HB["skill"]["mask_25km"]["p_binomial_uncharted4"], HB["skill"]["mask_0km"]["hits_all"][0], HB["skill"]["mask_0km"]["hits_all"][1], HB["skill"]["mask_0km"]["p_binomial_all"], hz["frac_p_gt_5pct"]*100, hz["km_flagged"], hz["km_mean_map_safe"], tr(["Grounding", "all soundings", "10 km hole", "25 km hole", "25 km hole · 10 km window", "25 km hole · 50 km window", "25 km hole · safe 1.5×", "25 km hole · safe 3×"], True), "".join(tr([f"<b>{n}</b>"] + [("—" if (v := HB["sites"][n].get(mk, {}).get("ranks", {}).get(key, {}).get("percentile")) is None else f"{v:.0f}") for mk, key in [("mask_0km","win25km_safe2x"),("mask_10km","win25km_safe2x"),("mask_25km","win25km_safe2x"),("mask_25km","win10km_safe2x"),("mask_25km","win50km_safe2x"),("mask_25km","win25km_safe1.5x"),("mask_25km","win25km_safe3x")]]) for n in ["Thamesborg","Akademik Ioffe","Clipper Adventurer","Hanseatic","Nanny 2012","Nanny 2014"] if n in HB["sites"]))) if HB else ""}
{fig("thamesborg_exhibit_web.jpg",''')
sub1(p, '<div><b>Least-risk channel.</b> Cost = distance × (shallow-water penalty + σ/8 + 13 × no-data); water under 20 m and land impassable. Result: {cf[\'length_km\']:,} km, shallowest {abs(cf[\'shallowest_on_path\']):.0f} m, mean σ {cf[\'mean_sigma\']} m, {cf[\'frac_nodata\']*100:.0f}% over no-data water.</div>',
        '<div><b>Least-risk channel.</b> Cost = distance × (shallow-water penalty + σ/8 + 13 × no-data); water under 20 m and land impassable. Result: {cf[\'length_km\']:,} km, shallowest {abs(cf[\'shallowest_on_path\']):.0f} m, mean σ {cf[\'mean_sigma\']} m, {cf[\'frac_nodata\']*100:.0f}% over no-data water. It is {cf[\'length_km\']-2327:,} km longer than the straight route: about {(cf[\'length_km\']-2327)/1.852/12:.0f} h at 12 kn, ~{(cf[\'length_km\']-2327)/1.852/12/24*25:.0f} t of fuel and a day of hire per voyage for a Handysize (~US$25k). It is a map of where the risk sits, not a sailing direction; the survey plan is what removes the detour.</div>')
sub1(p, '<div><b>Survey plan.</b> Ten highest-σ boxes within 35 km of the route priced at 40 km²/day (assumption) and C$183,000/day (CCG polar-icebreaker charter, July 2026: C$22M / ~120 days) [10].</div>',
        '<div><b>Survey plan.</b> Ten highest-σ boxes within 35 km of the route. Coverage: 40 km²/day, from a swath of ~3× depth (300–400 m at 100–130 m), 8 kn for 20 h (~300 line-km, ~100 km² raw), 20% overlap and a 50% weather-and-ice downtime factor; stated as an assumption. Day rate: the CCG polar-icebreaker charter of July 2026 (C$183,000/day; C$22M / ~120 days) [10] is the upper bound used in Table 7{"; the lower bound is the %s, which prices the same area at about C$%.0fM. %s" % (SR["source"], plan_area*SR["cost_per_km2_cad"]/1e6, SR["amundsen_context"]) if SR else ""}.</div>')
sub1(p, "Scoring rule: any later survey through these cells reports the mean absolute error and the fraction of soundings inside each band; we publish the score regardless of outcome.</p>",
        "Scoring rule: any later survey through these cells reports the mean absolute error and the fraction of soundings inside each band; we publish the score regardless of outcome. The depths were produced by the 34.8M-parameter v5-small completion model with the depth gate and calibrated σ of §4.3.{' The file is timestamped with OpenTimestamps (proof <code>forecast_2026-09-02.csv.ots</code>, three public calendars, Bitcoin-anchored) and its hash is in a GitHub commit of the validation repository, so the seal does not depend on our clock.' if OTS else ''}</p>")
sub1(p, "(vi) Survey-day rate and km²/day are assumptions from public contracts, not quotes.",
        "(vi) Survey-day rate and km²/day are assumptions from public contracts and stated arithmetic, not quotes. (ix) The gravity prior has very likely ingested the Test 1 cruises (§4.5).")
print("patched")
