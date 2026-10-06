"""
A short video of the wind layer: the lines' light moving at the simulated
wind speed, with a slow camera move. For a post or a comment, if wanted.

    python serve.py                       (or any http server on viewer/, port 8765)
    .venv-heat/Scripts/python tools/clip_wind.py [seconds]

Headless Chrome as in tools/render_wind.py: every frame is rendered on
purpose (fixed time step, never real time), so the clip is smooth whatever
the machine. Encoded to H.264 MP4 with the ffmpeg that ships in the
imageio-ffmpeg package. Writes data/wind/figures/post/clip_<stamp>.mp4.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
from render_post import CHROME, Tab  # noqa: E402
from render_wind import CAMS, READY  # noqa: E402

OUT = os.path.join(HERE, "data", "wind", "figures", "post")
BASE = os.environ.get("VIEWER_URL", "http://127.0.0.1:8765/")
PORT = 9337
FPS = 30
SIZE = (1080, 1350)
PAGE = "shadow-twin.html?layer=wind&season=dec&t=12:30&anim=1&full=1"

FRAME = """
(() => {
  const k = %(k)d, n = %(n)d, a = %(a)s, b = %(b)s;
  const f = k / Math.max(1, n - 1), e = f * f * (3 - 2 * f);
  const L = (p, q) => p + (q - p) * e;
  viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(L(a.lon, b.lon), L(a.lat, b.lat), L(a.h, b.h)),
    orientation: { heading: Cesium.Math.toRadians(L(a.heading, b.heading)),
                   pitch: Cesium.Math.toRadians(L(a.pitch, b.pitch)), roll: 0 } });
  const rate = WIND_SPEEDUP * Math.max(0.5, windNow().speed) / WIND.spacing_k1_m;
  for (const p of windPrims.values()) { p.phase = k / %(fps)d * rate; p.speed = windNow().speed; }
  viewer.scene.render();
  return viewer.canvas.toDataURL('image/jpeg', 0.92);
})()
"""

SETTLE = """
(async () => {
  if (typeof sunRay !== 'undefined' && sunRay) sunRay.show = false;
  if (typeof sunDot !== 'undefined' && sunDot) sunDot.show = false;
  viewer.scene.preRender.removeEventListener(windTick);
  const cam = %(cam)s;
  viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(cam.lon, cam.lat, cam.h),
    orientation: { heading: Cesium.Math.toRadians(cam.heading), pitch: Cesium.Math.toRadians(cam.pitch), roll: 0 } });
  const t0 = performance.now();
  let calm = 0;
  while (performance.now() - t0 < 180000) {
    viewer.scene.render();
    calm = (viewer.scene.globe.tilesLoaded && viewer.dataSourceDisplay.ready) ? calm + 1 : 0;
    if (calm >= 8 && performance.now() - t0 > 8000) break;
    await new Promise(r => setTimeout(r, 250));
  }
  return true;
})()
"""


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    n = int(secs * FPS)
    a = dict(CAMS["upwindP"])
    b = dict(a, h=a["h"] * 0.86, heading=a["heading"] + 8.0, pitch=a["pitch"] + 2.0)
    os.makedirs(OUT, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    frames = tempfile.mkdtemp(prefix="wind-clip-")
    prof = tempfile.mkdtemp(prefix="wind-clip-chrome-")
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={prof}", "--no-first-run", "--no-default-browser-check",
                             "--disable-extensions", "--hide-scrollbars", "--mute-audio",
                             "--ignore-gpu-blocklist", "--enable-gpu-rasterization",
                             f"--window-size={SIZE[0]},{SIZE[1]}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        targets = None
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json") as r:
                    targets = [t for t in json.load(r) if t.get("type") == "page"]
                if targets:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        tab = Tab(targets[0]["webSocketDebuggerUrl"])
        tab.call("Page.enable")
        tab.call("Runtime.enable")
        tab.call("Emulation.setDeviceMetricsOverride", width=SIZE[0], height=SIZE[1], deviceScaleFactor=1,
                 mobile=False)
        tab.call("Page.navigate", url=f"{BASE}{PAGE}&r={int(time.time())}")
        time.sleep(3)
        tab.js(READY, timeout_s=600)
        tab.js(SETTLE % {"cam": json.dumps(a)}, timeout_s=600)
        for k in range(n):
            uri = tab.js(FRAME % {"k": k, "n": n, "a": json.dumps(a), "b": json.dumps(b), "fps": FPS},
                         timeout_s=120)
            with open(os.path.join(frames, f"f{k:05d}.jpg"), "wb") as f:
                f.write(base64.b64decode(uri.split(",", 1)[1]))
            if k % 30 == 0:
                print(f"  frame {k}/{n}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    import imageio_ffmpeg
    out = os.path.join(OUT, f"clip_{stamp}.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(frames, "f%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-preset", "slow", "-movflags", "+faststart", out], check=True)
    import shutil
    shutil.rmtree(frames, ignore_errors=True)
    shutil.rmtree(prof, ignore_errors=True)
    print(f"wrote {out}  {os.path.getsize(out) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
