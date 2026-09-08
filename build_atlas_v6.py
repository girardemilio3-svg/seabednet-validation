#!/usr/bin/env python3
"""Stills pass: churchill_atlas_v5.html -> churchill_atlas_v6.html. Adds 'The corridor, rendered': six fully-built 4K frames
of the globe (map/stills/*.jpg), one per tour stop, embedded at 1600 px and linked at 3840 px. Replaces the fly-through film."""
import base64, glob, os
src = open("churchill_atlas_v5.html", encoding="utf-8").read()
def sub1(s, old, new):
    assert s.count(old) == 1, f"count {s.count(old)}: {old[:60]}"; return s.replace(old, new)
b64 = lambda fn: base64.b64encode(open(fn, "rb").read()).decode()
caps = {"01_franklin_strait": "Franklin Strait, where the Thamesborg struck on 6 September 2025. The relief is the model-completed seabed, exaggerated 3.5&times;; yellow is the hazard field where the chart calls the water navigable.",
        "02_simpson_strait_claim1": "Sealed shoal claim #1, Simpson Strait: the chart says 22 m; the retrained model says the shallowest point within 500 m lies at 4.5 m, 3 km from the nearest shore.",
        "03_queen_maud_gulf": "Queen Maud Gulf: five of the forty version-2 claims sit in the shallow, sparsely sounded water every Northwest Passage transit crosses.",
        "04_chesterfield_inlet": "Chesterfield Inlet: two Nanny groundings and the Dorsch, with the model&rsquo;s sealed guess of where the Dorsch struck.",
        "05_deception_bay": "Deception Bay, Raglan&rsquo;s ore dock: median survey year 1960, and every square metre of its apparently-safe water flagged.",
        "06_age_of_chart": "The age of Canada&rsquo;s chart: 28% of the sounded seabed rests on surveys older than 1980."}
figs = ""
for fn in sorted(glob.glob("map/stills/*_web.jpg")):
    key = os.path.basename(fn)[:-8]; full = f"map/stills/{key}.jpg"
    figs += f'<figure style="margin:0 0 18px"><a href="{full}"><img src="data:image/jpeg;base64,{b64(fn)}" alt="{key}" style="width:100%;border-radius:6px;display:block"></a><figcaption class="cap" style="font-size:13px;color:var(--muted);margin-top:6px">{caps.get(key, key)} <a href="{full}">4K</a></figcaption></figure>\n'
R = f'''
<section>
  <div class="eyebrow">The corridor, rendered</div>
  <h2>Six places, fully built: the seabed the model completed, the claims on it, and the ships that found the rocks first</h2>
  <p class="lede">Frames from the <a href="map/">interactive globe</a>, rendered at 4K from the same data as every number on this page: 10 m relief where CHS holds it, model-completed at 100 m elsewhere, Sentinel-2 imagery on land, the sea surface drawn to the exact coastline. Click a frame for the full-resolution file.</p>
  {figs}
</section>
'''
anchor = '<section>\n  <div class="eyebrow">Exhibit J &mdash; sealed claims</div>'
src = sub1(src, anchor, R + anchor)
open("churchill_atlas_v6.html", "w", encoding="utf-8").write(src); print("ATLAS_V6_DONE", len(src)//1024, "KB")
