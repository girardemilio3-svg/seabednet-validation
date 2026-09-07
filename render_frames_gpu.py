#!/usr/bin/env python3
"""Offline film render: Xvfb + headed Chromium on the NVIDIA GPU, one frame at a time, each frame waits for the map to be
fully idle (tiles, terrain mesh, tint) before the screenshot. No pop-in by construction. usage: python3 render_frames_gpu.py [--test]
-> film/frames/f%05d.png ; then encode with ffmpeg."""
import os, sys, time, math, subprocess, json
from playwright.sync_api import sync_playwright
TEST = "--test" in sys.argv; FPS = 30; W, H = 1920, 1080
FLY, HOLD = (2.0, 1.0) if TEST else (5.2, 9.0)   # same timing as the live tour (flyTo 5.2 s, hold 9 s)
os.makedirs("film/frames", exist_ok=True)
env = dict(os.environ, DISPLAY=":99", VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json")
xvfb = subprocess.Popen(["Xvfb", ":99", "-screen", "0", f"{W}x{H+200}x24", "-nolisten", "tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1.5)
args = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--ignore-gpu-blocklist", "--disable-gpu-sandbox", "--enable-gpu-rasterization", f"--window-size={W},{H+140}", "--window-position=0,0", "--hide-scrollbars"]
def ease(u): return u*u*(3-2*u)
try:
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=False, args=args, env=env, ignore_default_args=["--enable-automation"])
        ctx = br.new_context(viewport={"width": W, "height": H}); pg = ctx.new_page()
        pg.goto("http://localhost:8765/map/?film=1&static=1&norun=1", wait_until="load"); pg.wait_for_function("window.map && map.loaded && map.loaded()", timeout=180000)
        pg.evaluate("() => { document.body.style.cursor='none'; if (window.__cover) window.__cover.remove(); }")
        stops = pg.evaluate("() => STOPS.map(s => ({c:s.c, z:s.z, p:s.p, b:s.b, t:s.t, s:s.s, age: !!s.age}))")
        home = dict(c=[-80, 62], z=1.6, p=0, b=0, t="", s="", age=False); path = [home] + stops
        frames = 0; t0 = time.time()
        def cam(a, b, u):
            e = ease(u); L = lambda x, y: x + (y-x)*e
            # flyTo-like arc: zoom out in the middle of the flight
            zmid = min(a["z"], b["z"]) - 2.2*math.sin(math.pi*u) if a["z"] > 3 and b["z"] > 3 else L(a["z"], b["z"])
            return dict(center=[L(a["c"][0], b["c"][0]), L(a["c"][1], b["c"][1])], zoom=max(zmid, L(a["z"], b["z"]) - 2.2), pitch=L(a["p"], b["p"]), bearing=L(a["b"], b["b"]))
        def shoot(view, st, show_cap):
            global frames
            pg.evaluate("(v) => map.jumpTo(v)", view)
            pg.evaluate("(a) => { if (window.__showCap) window.__showCap(a.st, a.show); }", dict(st=st, show=show_cap))
            pg.wait_for_function("() => map.loaded() && !map.isMoving()", timeout=60000); pg.wait_for_timeout(60)
            pg.screenshot(path=f"film/frames/f{frames:05d}.png"); frames += 1
        for i in range(len(path)-1):
            a, b = path[i], path[i+1]
            pg.evaluate("(on) => { map.setPaintProperty('age-img','raster-opacity', on ? 0.9 : 0); map.setPaintProperty('hazard-img','raster-opacity', on ? 0 : ['interpolate',['linear'],['zoom'],7,0.8,9,0.45,11,0.22]); }", bool(b["age"]))
            nf = int(FLY*FPS)
            for k in range(nf): shoot(cam(a, b, k/nf), b, k > nf*0.6)
            for k in range(int(HOLD*FPS)): shoot(cam(b, b, 1.0), b, True)
            print(f"segment {i+1}/{len(path)-1}: {frames} frames, {(time.time()-t0)/frames:.2f} s/frame", flush=True)
            if TEST and i >= 1: break
        br.close()
finally:
    xvfb.terminate()
print("FRAMES_DONE", frames)
