#!/usr/bin/env python3
"""Render the fly-through film from the live globe page with Playwright + ffmpeg.
Camera path: piecewise flights between the tour stops (smoothstep-eased, with a hold at each stop),
sampled at FPS; each frame waits for the map to be idle (tiles loaded) before capture.
Captions burned in from the page's own tour text via a small overlay injected into the DOM.
usage: python3 render_film.py [--url http://localhost:8765/map/] [--fps 24] [--w 1920 --h 1080] [--test]
-> film/frames/*.png, film/churchill_corridor.mp4"""
import argparse, json, math, os, subprocess, time
from playwright.sync_api import sync_playwright
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://localhost:8765/map/"); ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080); ap.add_argument("--test", action="store_true"); ap.add_argument("--exag", type=float, default=3.5)
A = ap.parse_args(); os.makedirs("film/frames", exist_ok=True)
FLY, HOLD = (2.0, 1.0) if A.test else (7.0, 5.0)      # seconds per flight / per hold
ORBIT = dict(c=[-80, 62], z=2.3, p=0, b=0, t="", s="")
def ease(u): return u*u*(3-2*u)
with sync_playwright() as pw:
    br = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = br.new_page(viewport={"width": A.w, "height": A.h}, device_scale_factor=1)
    pg.goto(A.url, wait_until="load"); pg.wait_for_function("window.map && map.loaded && map.loaded()", timeout=180000)
    pg.evaluate("""() => { document.querySelector('.panel').style.display='none'; document.querySelector('.legend').style.display='none';
        document.querySelectorAll('.maplibregl-ctrl-top-right').forEach(e => e.style.display='none');
        const c = document.createElement('div'); c.id='film-cap'; c.style.cssText='position:absolute;left:60px;bottom:60px;max-width:900px;background:rgba(6,10,18,.82);border:1px solid #1b2436;border-radius:8px;padding:16px 22px;font:20px/1.45 IBM Plex Sans,system-ui,sans-serif;color:#d8e0ee;z-index:9;display:none';
        document.body.appendChild(c); const t = document.createElement('div'); t.style.cssText='position:absolute;left:60px;top:44px;font:600 34px/1.2 Fraunces,Georgia,serif;color:#d8e0ee;text-shadow:0 2px 12px #000;z-index:9'; t.textContent='Churchill Corridor Atlas'; document.body.appendChild(t);
        const u = document.createElement('div'); u.style.cssText='position:absolute;left:60px;top:90px;font:13px/1.4 IBM Plex Mono,monospace;letter-spacing:.18em;color:#e8b24a;z-index:9'; u.textContent='SEABEDNET · SEALED CLAIMS ON THE SEABED · SEPTEMBER 2026'; document.body.appendChild(u);
        window.__stops = STOPS; window.__age = (on) => { map.setPaintProperty('age-img','raster-opacity', on ? 0.9 : 0); map.setPaintProperty('hazard-img','raster-opacity', on ? 0 : 0.85); }; }""")
    pg.evaluate(f"() => map.setTerrain({{source:'terrain', exaggeration:{A.exag}}})")
    stops = pg.evaluate("() => window.__stops.map(s => ({c:s.c, z:s.z, p:s.p, b:s.b, t:s.t, s:s.s, age: !!s.age}))")
    path = [ORBIT] + stops + [ORBIT]
    frames = 0
    def cam(a, b, u):
        e = ease(u); lerp = lambda x, y: x + (y - x)*e
        return dict(center=[lerp(a["c"][0], b["c"][0]), lerp(a["c"][1], b["c"][1])], zoom=lerp(a["z"], b["z"]), pitch=lerp(a["p"], b["p"]), bearing=lerp(a["b"], b["b"]))
    def shoot(view, cap):
        global frames
        pg.evaluate("(v) => map.jumpTo(v)", view)
        pg.evaluate("(c) => { const e = document.getElementById('film-cap'); if (c && c.t) { e.style.display='block'; e.innerHTML = '<div style=\"font:500 12px/1 IBM Plex Mono,monospace;letter-spacing:.14em;color:#e8b24a;margin-bottom:6px\">'+c.t+'</div>'+c.s; } else e.style.display='none'; }", cap)
        try: pg.wait_for_function("() => map.loaded() && !map.isMoving()", timeout=20000)
        except Exception: pass
        time.sleep(0.05); pg.screenshot(path=f"film/frames/f{frames:05d}.png"); frames += 1
    for i in range(len(path)-1):
        a, b = path[i], path[i+1]
        pg.evaluate("(on) => window.__age(on)", bool(b.get("age")))
        nf = int(FLY*A.fps)
        for k in range(nf): shoot(cam(a, b, k/nf), b if k > nf*0.55 else None)
        for k in range(int(HOLD*A.fps)): shoot(cam(b, b, 1.0), b)
        print(f"segment {i+1}/{len(path)-1}: {frames} frames", flush=True)
    br.close()
subprocess.run(["ffmpeg", "-y", "-r", str(A.fps), "-i", "film/frames/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart", "film/churchill_corridor.mp4"], check=True, capture_output=True)
print("FILM_DONE", frames, "frames"); os.system("ls -la film/churchill_corridor.mp4")
