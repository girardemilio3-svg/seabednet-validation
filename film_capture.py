#!/usr/bin/env python3
"""Capture the globe's film-mode tour in real time: Xvfb virtual display + headed Chromium on the
NVIDIA GPU (ANGLE/Vulkan) + ffmpeg x11grab at 1080p30, crf 18. Stops when the page flags __tourDone.
usage: python3 film_capture.py [--url http://localhost:8765/map/?film=1&static=1] -> film/churchill_corridor.mp4"""
import argparse, os, subprocess, time, signal
from playwright.sync_api import sync_playwright
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://localhost:8765/map/?film=1&static=1"); ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080); ap.add_argument("--fps", type=int, default=30)
A = ap.parse_args(); os.makedirs("film", exist_ok=True)
env = dict(os.environ, DISPLAY=":99", VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json")
xvfb = subprocess.Popen(["Xvfb", ":99", "-screen", "0", f"{A.w}x{A.h+200}x24", "-nolisten", "tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1.5)
args = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--ignore-gpu-blocklist", "--disable-gpu-sandbox", "--enable-gpu-rasterization",
        f"--window-size={A.w},{A.h+140}", "--window-position=0,0", "--hide-scrollbars", "--autoplay-policy=no-user-gesture-required"]
rec = None
try:
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=False, args=args, env=env, ignore_default_args=["--enable-automation"])
        ctx = br.new_context(viewport={"width": A.w, "height": A.h}); pg = ctx.new_page(); pg.goto(A.url, wait_until="load")
        pg.wait_for_function("window.map && map.loaded && map.loaded()", timeout=180000)
        time.sleep(1.0)
        g = pg.evaluate("() => ({iw: innerWidth, ih: innerHeight, sx: screenX, sy: screenY, oh: outerHeight, ow: outerWidth})")
        print("window geometry", g, flush=True)
        gx, gy, gw, gh = max(0, g["sx"]), max(0, g["sy"] + (g["oh"] - g["ih"])), min(A.w, g["iw"] - (g["iw"] % 2)), min(A.h, g["ih"] - (g["ih"] % 2))
        print("grab", gx, gy, gw, gh, flush=True)
        pg.evaluate("() => { document.body.style.cursor='none'; map.getCanvas().style.cursor='none'; }"); pg.mouse.move(gw-1, gh-1)
        tw = time.time(); pg.wait_for_function("() => !!window.__warmDone", timeout=600000); print(f"tiles warmed in {time.time()-tw:.0f}s", flush=True)
        time.sleep(1.0)
        rec = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", str(A.fps), "-video_size", f"{gw}x{gh}", "-i", f":99.0+{gx},{gy}", "-vf", f"scale={A.w}:{A.h}:flags=lanczos",
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "film/churchill_corridor_raw.mp4"], env=env)
        time.sleep(1.0); pg.evaluate("() => { window.__go = true; }")
        t0 = time.time()
        while time.time() - t0 < 240:
            if pg.evaluate("() => !!window.__tourDone"): break
            time.sleep(0.5)
        print(f"tour done after {time.time()-t0:.0f}s", flush=True)
        rec.send_signal(signal.SIGINT); rec.wait(timeout=30); rec = None
        br.close()
finally:
    if rec: rec.send_signal(signal.SIGINT)
    xvfb.terminate()
# trim the first second (page settle) and re-encode with faststart
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1.0", "-i", "film/churchill_corridor_raw.mp4", "-loop", "1", "-t", "5", "-framerate", str(A.fps), "-i", "film/endcard.png",
    "-filter_complex", f"[0:v]fade=t=in:st=0:d=1.2,fade=t=out:st=84.5:d=1.2,setsar=1[a];[1:v]scale={A.w}:{A.h},fade=t=in:st=0:d=1,fade=t=out:st=4:d=1,setsar=1[b];[a][b]concat=n=2:v=1:a=0[v]",
    "-map", "[v]", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "film/churchill_corridor.mp4"], check=True)
print("FILM_DONE"); os.system("ls -la film/churchill_corridor.mp4; ffprobe -v error -show_entries format=duration -of csv=p=0 film/churchill_corridor.mp4")
