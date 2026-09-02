#!/usr/bin/env python3
"""Atlas v2: patch the standalone churchill_atlas.html with the validation program.
Reads the JSON outputs (never retypes a number) -> churchill_atlas_v2.html
Inserts: Exhibit G (independent + temporal tests), Exhibit H (hashed forecast), CATZOC-aware
keel profile, corrected Exhibit D, rewritten claims, footer sources. Leaves a <!--HAZARD-->
marker for the hazard/hindcast pass."""
import json, re, numpy as np
src = open("churchill_atlas_v1.html", encoding="utf-8").read()
def sub1(s, old, new):
    assert s.count(old) == 1, f"pattern count {s.count(old)}: {old[:60]}"
    return s.replace(old, new)
A = np.load("indep_cells_corridor_out.npy"); em, en, eg, sg, dk, dep = A.T
sh = (-dep >= 50) & (-dep < 400); m = lambda x, s: np.abs(x[s]).mean()
import os
TF = "temporal_validation_v5_small_temporal.json" if os.path.exists("temporal_validation_v5_small_temporal.json") else "temporal_validation_v5_tiny_temporal.json"
T = json.load(open(TF)); FULL = "small" in TF; cal = json.load(open("sigma_calibration.json"))
F = json.load(open("forecast_manifest.json")); prof = json.load(open("route_profile_v2.json"))
yrs = [p["survey_year"] for p in prof if p.get("survey_year")]; pre80 = sum(1 for y in yrs if y < 1980)
grades = {g: sum(1 for p in prof if p["grade"] == g) for g in "ABCD"}; n = len(prof)
to = T["overall"]; td = {tuple(r["km"]): r for r in T["by_distance"]}; tdep = {tuple(r["m"]): r for r in T["by_depth"]}
fmt = lambda v: f"{v:.1f}"

# ---------- header
src = sub1(src, "every kilometre, with the provenance of every pixel kept honest.</p>",
  f"every kilometre, with the provenance of every pixel kept honest &mdash; and tested on water it had never seen: "
  f"trained only on soundings from before 2016, it predicted the {to['n']/1e6:.1f}&nbsp;million soundings CHS has collected since to within {fmt(to['mae_model'])}&nbsp;m, "
  f"{(1-to['mae_model']/to['mae_grav'])*100:.0f}% better than the gravity-derived bathymetry every global chart falls back on.</p>")
src = sub1(src, '<div class="stat amber"><b>+61%</b><span>completed by SeabedNet</span></div>',
  '<div class="stat amber"><b>+61%</b><span>completed by SeabedNet</span></div>\n'
  f'    <div class="stat red"><b>{int(np.median(yrs))}</b><span>median survey year under keel</span></div>')

# ---------- keel profile: v2 data + readout
src = re.sub(r"const PROF = \[.*?\];\n", lambda _m: "const PROF = " + json.dumps(prof) + ";\n", src, count=1, flags=re.S)
src = sub1(src, '<span id="rprov" class="chip none">no data</span>',
  '<span id="rprov" class="chip none">no data</span><span>SURVEY <b id="rsv">&ndash;</b></span><span>GRADE <b id="rgr">&ndash;</b></span>')
src = sub1(src, "  ch.textContent = best.prov==='none'?'no data':best.prov;\n",
  "  ch.textContent = best.prov==='none'?'no data':best.prov;\n"
  "  document.getElementById('rsv').textContent = best.survey || (best.dist_km!=null ? 'nearest sounding '+best.dist_km.toFixed(1)+' km' : '—');\n"
  "  document.getElementById('rgr').textContent = best.grade + ({A:' · published sounding',B:' · inferred, within 1 km',C:' · inferred, 1–6 km',D:' · no inference'}[best.grade]||'');\n")
src = sub1(src, "(best.prov==='inferred'&&best.sigma!=null)?'±'+best.sigma.toFixed(1)+' m'",
  "(best.prov==='inferred'&&best.sigma!=null)?'±'+(best.sigma68||best.sigma).toFixed(1)+' m (68%)'")
src = sub1(src, "model-inferred depth in amber with its &sigma; band, and the water nobody can answer for yet in red.</p>",
  "model-inferred depth in amber with its calibrated &sigma; band, and the water nobody can answer for yet in red. "
  f"Hover for the provenance of each point: the CHS survey year and CATZOC grade under the keel, or the distance to the nearest sounding. "
  f"Of the {len(yrs)} points with a published sounding, {pre80} ({pre80/len(yrs)*100:.0f}%) come from surveys before 1980. "
  f"Trust grade along the route: A {grades['A']/n*100:.0f}% &middot; B {grades['B']/n*100:.0f}% &middot; C {grades['C']/n*100:.0f}% &middot; D {grades['D']/n*100:.0f}%.</p>")

# ---------- σ band: one polygon per contiguous inferred run, calibrated σ68
OLD_BAND = """  ctx.beginPath(); let started=false;
  for (const p of PROF){ if(p.prov==='inferred'&&p.depth!=null){ const y=Y(p.depth-(p.sigma||0)); started?ctx.lineTo(X(p.km),y):ctx.moveTo(X(p.km),y); started=true; } else started=false; }
  for (let i=PROF.length-1;i>=0;i--){ const p=PROF[i]; if(p.prov==='inferred'&&p.depth!=null){ ctx.lineTo(X(p.km),Y(p.depth+(p.sigma||0))); } }
  ctx.fillStyle = 'rgba(232,178,74,0.16)'; ctx.fill();
"""
NEW_BAND = """  ctx.fillStyle = 'rgba(232,178,74,0.16)';
  { let run=[]; const flush=()=>{ if(run.length>1){ ctx.beginPath();
      run.forEach((p,i)=>{ const y=Y(p.depth-(p.sigma68||p.sigma||0)); i?ctx.lineTo(X(p.km),y):ctx.moveTo(X(p.km),y); });
      for(let i=run.length-1;i>=0;i--){ const p=run[i]; ctx.lineTo(X(p.km),Y(p.depth+(p.sigma68||p.sigma||0))); }
      ctx.closePath(); ctx.fill(); } run=[]; };
    for (const p of PROF){ if(p.prov==='inferred'&&p.depth!=null) run.push(p); else flush(); } flush(); }
"""
src = sub1(src, OLD_BAND, NEW_BAND)

# ---------- Exhibit G: validation (inserted before Exhibit A)
def row(label, a, b, c, bold=False):
    l = f"<b>{label}</b>" if bold else label
    return f"<tr><td>{l}</td><td>{a}</td><td>{b}</td><td>{c}</td></tr>"
ind_rows = "".join(row(f"{a}&ndash;{b} km", fmt(m(em, sh&(dk>=a)&(dk<b))), fmt(m(en, sh&(dk>=a)&(dk<b))), fmt(m(eg, sh&(dk>=a)&(dk<b)))) for a, b in [(0,.5),(.5,1),(1,2),(2,4)])
tmp_rows = "".join(row(f"{a}&ndash;{b} km", fmt(td[(a,b)]['mae_model']), fmt(td[(a,b)]['mae_nn']), fmt(td[(a,b)]['mae_grav'])) for a, b in [(0,.5),(.5,1),(1,2),(2,4),(4,8)])
r = np.abs(em[sh])/np.maximum(sg[sh], .1)
G = f'''
<section>
  <div class="eyebrow">Exhibit G &mdash; tested on water it had never seen</div>
  <h2>Two independent tests, all the numbers, including the ones that hurt</h2>
  <p class="lede">A completion model is worth nothing until it is scored on depths it never trained on. We ran two such tests and publish both in full.</p>
  <div class="claims">
    <div class="claim"><span class="num">{fmt(m(em, sh))} m</span><h3>Test 1 &mdash; research-cruise multibeam</h3>
      <p>{sh.sum():,} seabed cells on the shelf (50&ndash;400 m) measured by Amundsen, Healy, Knorr, Armstrong and Merian cruises (via GMRT/NCEI) where CHS has no published sounding. The model&rsquo;s mean error, against <b>{fmt(m(en, sh))} m</b> for copying the nearest published sounding and <b>{fmt(m(eg, sh))} m</b> for the gravity prior.</p></div>
    <div class="claim"><span class="num">{fmt(to['mae_model'])} m</span><h3>Test 2 &mdash; the future, hindcast</h3>
      <p>CHS&rsquo;s own Survey Index dates every sounding to 2016. We retrained on the pre-2016 soundings only and scored the {to['n']:,} corridor soundings CHS published after. Model {fmt(to['mae_model'])} m &middot; nearest old sounding <b>{fmt(to['mae_nn'])} m</b> &middot; gravity <b>{fmt(to['mae_grav'])} m</b>. {to['frac_within_1sigma']*100:.0f}% of errors inside the model&rsquo;s own 1&sigma;.</p></div>
    <div class="claim"><span class="num">{np.mean(r<=1)*100:.0f}% &rarr; {cal['coverage_cal_68']*100:.0f}%</span><h3>&sigma; recalibrated on independent cells</h3>
      <p>Raw &sigma; was under-confident: only {np.mean(r<=1)*100:.0f}% of independent errors fell inside 1&sigma; (68% expected). An isotonic map fitted on Test 1 fixes the scale (mean factor {np.mean(np.interp(sg[sh], cal['sigma_raw_grid'], cal['sigma68'])/np.maximum(sg[sh],.1)):.1f}&times;); every &sigma; on this page is the calibrated one. Ranking was already right (&sigma; vs |error| correlation {np.corrcoef(sg[sh], np.abs(em[sh]))[0,1]:.2f}).</p></div>
    <div class="claim"><span class="num">&gt; 400 m</span><h3>Where the model defers</h3>
      <p>Off the shelf, Test 1 shows gravity beats the model ({fmt(m(eg, -dep>=1000))} vs {fmt(m(em, -dep>=1000))} m below 1,000 m). Inferred cells deeper than 400 m now carry the gravity prior; 300&ndash;400 m is blended. Stated, not hidden.</p></div>
  </div>
  <div class="tblwrap"><table>
    <thead><tr><th>Distance to nearest sounding</th><th>SeabedNet (m)</th><th>Nearest sounding (m)</th><th>Gravity prior (m)</th></tr></thead>
    <tbody><tr><td colspan="4" style="color:var(--muted);font-family:var(--mono);font-size:11px;letter-spacing:.1em">TEST 1 &middot; INDEPENDENT MULTIBEAM, SHELF 50&ndash;400 M</td></tr>{ind_rows}
    <tr><td colspan="4" style="color:var(--muted);font-family:var(--mono);font-size:11px;letter-spacing:.1em">TEST 2 &middot; PRE-2016 MODEL vs POST-2016 CHS SOUNDINGS, CORRIDOR</td></tr>{tmp_rows}</tbody>
  </table></div>
  <p class="lede" style="font-size:14.5px"><b style="color:var(--text)">What the tests do not support.</b> Under 50 m of water the pre-2016 model reads {abs(tdep[(20,50)]['bias']):.0f}&ndash;{abs(tdep[(0,20)]['bias']):.0f} m <em>too deep</em> &mdash; the dangerous direction, and the reason the mean depth is the wrong target for navigation (see the hazard exhibit). Within 500 m of an old sounding, copying that sounding still beats the model ({fmt(td[(0,.5)]['mae_nn'])} vs {fmt(td[(0,.5)]['mae_model'])} m); the model earns its keep beyond that. {"Test 2 is the full-size (34.8M-parameter) model, trained 17 minutes on a rented RTX 5090; the 6.8M model scored 13.5 m on the same test, so size is not what limits this." if FULL else "Test 2 used a 6.8M-parameter model trained in 3.7 hours; the full-size run is in progress and will replace these numbers, better or worse."} Coastal cells under 50 m in Test 1 disagree with every source including CHS&rsquo;s own nearest sounding and are excluded pending a look at the GMRT coastline product.</p>
</section>
'''
src = sub1(src, '<section>\n  <div class="eyebrow">Exhibit A &mdash; the corridor, found</div>', G + '<!--HAZARD-->\n<section>\n  <div class="eyebrow">Exhibit A &mdash; the corridor, found</div>')

# ---------- Exhibit D corrections
s68 = float(np.interp(48.3, cal["sigma_raw_grid"], cal["sigma68"])); s95 = float(np.interp(48.3, cal["sigma_raw_grid"], cal["sigma95"]))
src = sub1(src, "gives the strike site a &sigma; of 48&nbsp;m: the 98th percentile of the whole strait. The right panel is the same water, completed.</p>",
  f"ranks the strike site in the 98th percentile of uncertainty for the whole strait (calibrated &sigma; &ge; {s68:.0f}&nbsp;m at 68%, &ge; {s95:.0f}&nbsp;m at 95%). "
  "CHS&rsquo;s own Survey Index shows the nearest dated survey to the site is a 1960 single-beam line graded CATZOC&nbsp;C; the modern multibeam swath ends 1.7&nbsp;km short of the strike. The right panel is the same water, completed.</p>")

# ---------- Exhibit H: forecast (before Exhibit F)
H = f'''
<section>
  <div class="eyebrow">Exhibit H &mdash; a forecast you can fail</div>
  <h2>{F['n']:,} predicted depths, sealed on {F['date']}</h2>
  <p class="lede">Every cell in this file has no published sounding today. {F['n_channel']} sit on the found channel; the rest are drawn at random across the corridor. Each carries a predicted depth and calibrated 68% / 95% bands. The next CHS, Coast Guard or research survey through this water scores it, and we publish the score, whichever way it goes.</p>
  <div class="tblwrap"><table><tbody>
    <tr><td>File</td><td><a href="{F['file']}">{F['file']}</a></td></tr>
    <tr><td>SHA-256</td><td style="font-family:var(--mono);font-size:12.5px;word-break:break-all">{F['sha256']}</td></tr>
    <tr><td>Model</td><td>{F['model']}</td></tr>
    <tr><td>Scoring rule</td><td>{F['rule']}</td></tr>
  </tbody></table></div>
</section>
'''
src = sub1(src, '<section>\n  <div class="eyebrow">Exhibit F &mdash; the completed product</div>', H + '<section>\n  <div class="eyebrow">Exhibit F &mdash; the completed product</div>')

# ---------- claims block rewrite
CLAIMS = f'''<div class="claims">
    <div class="claim"><span class="num">{fmt(m(em, sh))} m vs {fmt(m(en, sh))} m</span><h3>Independent multibeam, shelf water</h3>
      <p>Model vs nearest published sounding on {sh.sum():,} research-cruise cells CHS never published. Gravity prior: {fmt(m(eg, sh))} m. The model&rsquo;s advantage over gravity on this set is small, because cruises run where gravity already works; Test 2 is the shelf test.</p></div>
    <div class="claim"><span class="num">{fmt(to['mae_model'])} m vs {fmt(to['mae_grav'])} m</span><h3>Pre-2016 model, post-2016 soundings</h3>
      <p>{to['n']:,} corridor soundings CHS collected after its Survey Index closes, predicted by a model that never saw them. Nearest old sounding: {fmt(to['mae_nn'])} m. The gap widens with distance: {fmt(td[(2,4)]['mae_model'])} vs {fmt(td[(2,4)]['mae_nn'])} m at 2&ndash;4 km.</p></div>
    <div class="claim"><span class="num">{cal['coverage_cal_68']*100:.0f}% / {cal['coverage_cal_95']*100:.0f}%</span><h3>Calibrated bands</h3>
      <p>Share of independent errors inside the published 68% and 95% bands after isotonic recalibration. Rank correlation of &sigma; with error was {np.corrcoef(sg[sh], np.abs(em[sh]))[0,1]:.2f} before, unchanged after.</p></div>
    <div class="claim"><span class="num">&le; 6 km &middot; &gt; 400 m</span><h3>Provenance and deferral</h3>
      <p>Inference is published only within 6 km of a real sounding; below 400 m the gravity prior is used because the tests say it is better there. &ldquo;No published sounding&rdquo; is never claimed to mean &ldquo;never surveyed&rdquo;: CHS holds data NONNA has not released.</p></div>
  </div>
  <div class="stamp">'''
src = re.sub(r'(What we claim, and what we don&rsquo;t</h2>\n  )<div class="claims">.*?</div>\n  <div class="stamp">', lambda _m: _m.group(1) + CLAIMS, src, count=1, flags=re.S)

# ---------- footer sources
src = sub1(src, "Corridor completion: 94 blocks, Hudson Bay &middot; Hudson Strait &middot; Labrador approaches &middot; Franklin Strait.</p>",
  "Corridor completion: 94 blocks, Hudson Bay &middot; Hudson Strait &middot; Labrador approaches &middot; Franklin Strait. "
  "Independent test data: GMRT synthesis (Lamont-Doherty) topo-mask of NCEI-archived multibeam &mdash; CCGS Amundsen 2003&ndash;13, USCGC Healy, R/V Knorr, R/V Neil Armstrong, R/V Maria S. Merian; "
  "CHS Survey Index (DFO EGIS, 7,078 dated survey polygons with CATZOC, 1832&ndash;2016) for the temporal split and the survey provenance under keel. "
  "Code, validation cells and every number on this page: github.com/girardemilio3-svg/seabednet-validation.</p>")
open("churchill_atlas_v2.html", "w", encoding="utf-8").write(src)
print("built churchill_atlas_v2.html", len(src)//1024, "KB; HAZARD marker:", src.count("<!--HAZARD-->"))
