#!/usr/bin/env python3
"""Hazard pass: fill the <!--HAZARD--> marker in churchill_atlas_v2.html with the hazard
exhibit (shallowest-point model, corridor hazard map, grounding hindcast table), and
refresh the temporal numbers if the full-size run has finished. Data-driven: the verdict
sentences are generated from hindcast.json, landing or not. -> churchill_atlas_v3.html"""
import base64, json, os, re, numpy as np
src = open("churchill_atlas_v2.html", encoding="utf-8").read()
def sub1(s, old, new):
    assert s.count(old) == 1, f"count {s.count(old)}: {old[:50]}"; return s.replace(old, new)
hc = json.load(open("hindcast.json")); hz = json.load(open("hazard_corridor.json"))
HB = json.load(open("hindcast_blind.json")) if os.path.exists("hindcast_blind.json") else None
def ordn(p):
    p=int(round(p)); return f"{p}{'th' if 10<=p%100<=20 else {1:'st',2:'nd',3:'rd'}.get(p%10,'th')}"
BLIND = ""
if HB:
    S = HB["sites"]; K = HB["skill"]; names = ["Thamesborg", "Akademik Ioffe", "Clipper Adventurer", "Hanseatic", "Nanny 2012", "Nanny 2014"]
    def cell(n, mask, key="win25km_safe2x"):
        r = S.get(n, {}).get(mask, {}); v = r.get("ranks", {}).get(key, {}).get("percentile"); return "&mdash;" if v is None else f"{v:.0f}"
    rows_b = "".join(f"<tr><td><b>{n}</b></td><td>{cell(n,'mask_0km')}</td><td><b>{cell(n,'mask_10km')}</b></td><td>{cell(n,'mask_25km')}</td><td>{cell(n,'mask_10km','win10km_safe2x')}</td><td>{cell(n,'mask_10km','win50km_safe2x')}</td><td>{cell(n,'mask_10km','win25km_safe1.5x')}</td><td>{cell(n,'mask_10km','win25km_safe3x')}</td></tr>" for n in names if n in S)
    k25 = K["mask_10km"]; k0 = K["mask_0km"]
    th = S["Thamesborg"]["mask_10km"]; th_pct = th["ranks"]["win25km_safe2x"]["percentile"]
    BLIND = f"""
  <h2 style="font-size:24px;margin-top:34px">The same hindcast, blind, and how much the denominator matters</h2>
  <p class="lede">The table above still lets the model see every published sounding around a site at inference time, and NONNA carries no dates after 2016, so a post-grounding survey could be in that input. Here every sounding within 10 km of each site is removed from the input as well (the hazard model was trained with every tile group within 0.4&deg; of the sites excluded from the start), and a 25 km hole is shown too: at 25 km the site falls outside the model&rsquo;s inference envelope and it returns <em>no estimate</em>, which is the right answer, not a miss. Then the two choices that set the percentile &mdash; the comparison window and the definition of &ldquo;looked safe&rdquo; &mdash; are varied. Thamesborg with a 10 km hole in its input: <b style="color:var(--text)">{ordn(th_pct)} percentile</b>, nearest remaining sounding {th['dist_to_input_sounding_km']:.1f} km away, absolute P(shoal &lt; draft) {th['p_shoal_at_site']*100:.1f}%.</p>
  <div class="tblwrap"><table>
    <thead><tr><th>Grounding</th><th>Input: all soundings</th><th>Input: 10 km hole</th><th>Input: 25 km hole</th><th>10 km hole, 10 km window</th><th>10 km hole, 50 km window</th><th>10 km hole, safe = 1.5&times; draft</th><th>10 km hole, safe = 3&times; draft</th></tr></thead>
    <tbody>{rows_b}</tbody></table></div>
  <p class="lede" style="font-size:14px"><b style="color:var(--text)">Skill, not recall.</b> By construction 10% of apparently-safe cells sit above the 90th percentile, so the base rate is known exactly. With the 10 km hole, {k25['hits_all'][0]} of {k25['hits_all'][1]} sites land above it (binomial p = {k25['p_binomial_all']:.4f}); on the four uncharted-shoal groundings alone, {k25['hits_uncharted4'][0]} of {k25['hits_uncharted4'][1]} (p = {k25['p_binomial_uncharted4']:.4f}). With all soundings visible: {k0['hits_all'][0]} of {k0['hits_all'][1]} (p = {k0['p_binomial_all']:.4f}). The precision side of the ledger is stated too: corridor-wide, {hz['frac_p_gt_5pct']*100:.0f}% of cells exceed P = 5%, and on the route {hz['km_flagged']:.0f} km of {hz['km_mean_map_safe']:.0f} apparently-safe km are flagged. Six incidents cannot bound a false-alarm rate; what the flags buy is a survey order, and the survey is what tests them.</p>"""
log = open(os.environ.get("HZ_LOG", "hazard_small.log")).read()
m_ex = re.search(r"model ([\d.]+) m \| '100 m sounding is the shoal' ([\d.]+) m \| gravity ([\d.]+) m", log)
m_hid = re.search(r"hidden\):\s+model ([\d.]+) m \| nearest-field ([\d.]+) m", log)
m_cov = re.search(r"σ coverage 1σ (\d+)%\s+2σ (\d+)%", log)
img = base64.b64encode(open("hazard_corridor_web.jpg", "rb").read()).decode()
# ---- hindcast table
rows = []; landed = []; missed = []
for r in hc:
    if "rank" not in r: rows.append(f"<tr><td>{r['name']}</td><td colspan='6' style='color:var(--muted)'>{r.get('status','')}</td></tr>"); continue
    pr = r["rank"]["hazard_p_shoal"]["percentile"]; ms = r["rank"]["mean_model_sigma"]["percentile"]; nn = r["rank"]["nearest_sounding_shallowness"]["percentile"]
    p = r["site"]["p_shoal_lt_draft"]
    f = lambda v: "&mdash;" if v is None else f"{v:.0f}"
    hot = " class='hot'" if (pr is not None and pr >= 90) else ""
    rows.append(f"<tr><td><b>{r['name']}</b><br><span style='color:var(--muted);font-size:12px'>{r['date']} &middot; TSB {r['tsb']}</span></td>"
                f"<td>{r['pre_soundings']:,}</td><td>{r['dist_to_pre_sounding_km']:.1f}</td><td>{'&mdash;' if p is None else f'{p*100:.0f}%'}</td>"
                f"<td{hot}>{f(pr)}</td><td>{f(ms)}</td><td>{f(nn)}</td></tr>")
    if pr is None: continue
    (landed if pr >= 90 else missed).append((r["name"], pr))
n_scored = len(landed) + len(missed)
if n_scored == 0: verdict = "No site could be scored inside the fill domain; the hindcast is reported as not yet possible on published data."
else:
    verdict = (f"On {len(landed)} of {n_scored} scored groundings the hazard field placed the strike site in the top 10% of danger among water that looked safe on the mean map"
               + (f" ({', '.join(f'{n} {ordn(p)}' for n, p in landed)})" if landed else "") + "."
               + (f" It missed {', '.join(f'{n} ({ordn(p)})' for n, p in missed)}." if missed else " It missed none."))
gap = hz["median_gap_m"]
H = f'''
<section>
  <div class="eyebrow">Exhibit I &mdash; the hazard field</div>
  <h2>Ships do not ground on the mean depth. This predicts the shallowest point.</h2>
  <p class="lede">Every bathymetry model, including the one above, predicts the average depth of a 100 m cell. A keel meets the shallowest rock in it. Where CHS holds both 10 m and 100 m data, the shallowest point within 500 m sits a median <b style="color:var(--text)">{gap:.0f} m above</b> the 100 m mean on this route. So we trained a second, 34.8M-parameter model on {475} NONNA-10/100 tile pairs to predict that shallowest point and its uncertainty, and turned it into one number per cell: the probability that a 10.5 m draft touches.</p>
  <div class="claims">
    <div class="claim"><span class="num">{m_ex.group(1) if m_ex else '&mdash;'} m</span><h3>Shallowest-point error, held-out tiles</h3>
      <p>Mean error of the predicted shallowest depth within 500 m on tiles held out around the grounding sites, against <b>{m_ex.group(2) if m_ex else '&mdash;'} m</b> if you assume the archive&rsquo;s 100 m depth is the shallowest point, and {m_ex.group(3) if m_ex else '&mdash;'} m for gravity. Where no sounding exists at all: {m_hid.group(1) if m_hid else '&mdash;'} m vs {m_hid.group(2) if m_hid else '&mdash;'} m. Bands: {m_cov.group(1) if m_cov else '&mdash;'}% inside 1&sigma;, {m_cov.group(2) if m_cov else '&mdash;'}% inside 2&sigma;.</p></div>
    <div class="claim"><span class="num">{hz['km_flagged']:.0f} km</span><h3>Route the mean map calls safe, the hazard field does not</h3>
      <p>Of {hz['km_mean_map_safe']:.0f} km of route deeper than 21 m on the mean map, {hz['km_flagged']:.0f} km carry more than a 5% chance that the shallowest point within 500 m is above a 10.5 m keel. Corridor-wide: {hz['frac_p_gt_5pct']*100:.1f}% of cells exceed 5%, {hz['frac_p_gt_20pct']*100:.1f}% exceed 20%.</p></div>
  </div>
  <div class="exh"><img src="data:image/jpeg;base64,{img}" alt="Hazard field of the Churchill corridor">
  <div class="cap"><b>P(shallowest point within 500 m &lt; 10.5 m), every cell.</b> Black is safe, yellow is a coin toss. Blue: the route. Red rings: the {hz['km_flagged']:.0f} route-km the mean map would have called safe.</div></div>
  <h2 style="font-size:24px;margin-top:34px">The groundings that already happened, hindcast</h2>
  <p class="lede">For each TSB-documented Arctic grounding we gave the hazard model only the soundings that existed before the ship hit (dated by CHS&rsquo;s Survey Index) and asked what percentile of danger the strike site got among the water that looked safe on the mean map within 25 km. {verdict} Baselines: the mean model&rsquo;s own &sigma; (&ldquo;where it is blind&rdquo;) and the shallowness of the nearest old sounding.</p>
  <div class="tblwrap"><table>
    <thead><tr><th>Grounding</th><th>Pre-incident soundings in block</th><th>km to nearest</th><th>P(shoal &lt; draft) at site</th><th>Hazard percentile</th><th>Mean-&sigma; percentile</th><th>Nearest-sounding percentile</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
  {BLIND}
  <p class="lede" style="font-size:14px">Thamesborg (table above) uses all published soundings because NONNA carries no dates after 2016; its position is from AIS and press, not TSB. Akademik Ioffe and Clipper Adventurer sit 9&ndash;12 km from any pre-incident sounding, at the edge of what any completion should be trusted for; their scores are reported anyway. The two Nanny groundings were navigation errors in known narrows and are included as controls, not claims. Full per-site JSON in the validation repo.</p>
</section>
'''
src = sub1(src, "<!--HAZARD-->", H)
open("churchill_atlas_v3.html", "w", encoding="utf-8").write(src)
print("built v3", len(src)//1024, "KB;", verdict)
