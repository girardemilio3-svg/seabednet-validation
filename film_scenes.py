#!/usr/bin/env python3
"""Scene-based film: each tour stop is recorded only after the map is fully built (no loading can appear on screen),
as a slow camera drift with its caption; scenes are joined with dissolves, plus the opening globe and the end card.
Xvfb + headed Chromium on the NVIDIA GPU + ffmpeg x11grab. usage: python3 film_scenes.py -> film/churchill_corridor.mp4"""
import os, sys, time, subprocess, signal, json
from playwright.sync_api import sync_playwright
W, H, FPS = 1920, 1080, 30; HOLD = 9.0; XF = 1.0
env = dict(os.environ, DISPLAY=":99", VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json")
os.makedirs("film/scenes", exist_ok=True)
xvfb = subprocess.Popen(["Xvfb", ":99", "-screen", "0", f"{W}x{H+200}x24", "-nolisten", "tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1.5)
args = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--ignore-gpu-blocklist", "--disable-gpu-sandbox", "--enable-gpu-rasterization", f"--window-size={W},{H+140}", "--window-position=0,0", "--hide-scrollbars"]
def record(path, seconds, gx, gy, gw, gh):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", str(FPS), "-video_size", f"{gw}x{gh}", "-i", f":99.0+{gx},{gy}", "-t", str(seconds), "-vf", f"scale={W}:{H}:flags=lanczos", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", path], env=env)
    return p
scenes = []
try:
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=False, args=args, env=env, ignore_default_args=["--enable-automation"])
        pg = br.new_context(viewport={"width": W, "height": H}).new_page()
        pg.goto("http://localhost:8765/map/?film=1&static=1&localsat=1&norun=1", wait_until="load"); pg.wait_for_function("window.map && map.loaded && map.loaded()", timeout=180000)
        g = pg.evaluate("() => ({iw: innerWidth, ih: innerHeight, sx: screenX, sy: screenY, oh: outerHeight})")
        gx, gy, gw, gh = max(0, g["sx"]), max(0, g["sy"] + (g["oh"] - g["ih"])), min(W, g["iw"] - g["iw"] % 2), min(H, g["ih"] - g["ih"] % 2)
        pg.evaluate("() => { document.body.style.cursor='none'; map.getCanvas().style.cursor='none'; }"); pg.mouse.move(gw-1, gh-1)
        stops = pg.evaluate("() => STOPS.map(s => ({c:s.c, z:s.z, p:s.p, b:s.b, t:s.t, s:s.s, age: !!s.age}))")
        # scene 0: the globe, slow spin in from space (few tiles, loads fast)
        pg.evaluate("() => { window.__showCap(null, false); map.jumpTo({center:[-80, 62], zoom: 1.6, pitch: 0, bearing: 0}); }"); pg.wait_for_timeout(1500); pg.wait_for_function("() => map.loaded()", timeout=180000); pg.wait_for_timeout(1000)
        rec = record("film/scenes/s00.mp4", 6.0, gx, gy, gw, gh); time.sleep(0.4)
        pg.evaluate("() => map.easeTo({center:[-80, 62], zoom: 2.6, duration: 5200, easing: t => t*t*(3-2*t), essential: true})"); rec.wait(); scenes.append("film/scenes/s00.mp4")
        for i, st in enumerate(stops):
            pg.evaluate("(on) => { map.setPaintProperty('age-img','raster-opacity', on ? 0.9 : 0); map.setPaintProperty('hazard-img','raster-opacity', on ? 0 : ['interpolate',['linear'],['zoom'],7,0.8,9,0.45,11,0.22]); }", st["age"])
            pg.evaluate("(v) => { window.__showCap(null, false); map.jumpTo({center: v.c, zoom: v.z, pitch: v.p, bearing: v.b - 4}); }", st)
            t0 = time.time(); pg.wait_for_timeout(1500); pg.wait_for_function("() => map.loaded()", timeout=300000); pg.wait_for_timeout(2500); pg.wait_for_function("() => map.loaded()", timeout=300000)
            pg.evaluate("(v) => window.__showCap(v, true)", st); pg.wait_for_timeout(400)
            print(f"stop {i+1} built in {time.time()-t0:.1f} s", flush=True)
            rec = record(f"film/scenes/s{i+1:02d}.mp4", HOLD, gx, gy, gw, gh); time.sleep(0.4)
            drift = "map.easeTo({bearing: v.b + 4, zoom: v.z + 0.08, duration: %d, easing: t => t, essential: true})" % int((HOLD-0.5)*1000)
            pg.evaluate("(v) => " + drift, st); rec.wait(); scenes.append(f"film/scenes/s{i+1:02d}.mp4")
        br.close()
finally:
    xvfb.terminate()
# assemble: title/globe, dissolves, end card
n = len(scenes); inputs = []; 
for s in scenes: inputs += ["-i", s]
inputs += ["-loop", "1", "-t", "5", "-framerate", str(FPS), "-i", "film/endcard.png"]
durs = [6.0] + [HOLD]*(n-1) + [5.0]
fc = ""; prev = "[0:v]"; offset = 0.0
for k in range(1, n+1):
    offset += durs[k-1] - XF
    src = f"[{k}:v]" if k < n else f"[{n}:v]scale={W}:{H},setsar=1[ec];[ec]"
    if k == n: fc += f"[{n}:v]scale={W}:{H},setsar=1[ec];"; src = "[ec]"
    out = f"[x{k}]" if k < n else "[xf]"
    fc += f"{prev}{src}xfade=transition=fade:duration={XF}:offset={offset:.3f}{out};"; prev = out
fc += f"[xf]fade=t=in:st=0:d=1.0,fade=t=out:st={sum(durs)-XF*n-1.0:.2f}:d=1.0[v]"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + inputs + ["-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "film/churchill_corridor.mp4"], check=True)
print("SCENES_DONE", n); os.system("ffprobe -v error -show_entries format=duration -of csv=p=0 film/churchill_corridor.mp4")
