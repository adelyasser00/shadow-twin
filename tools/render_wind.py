"""
Render still images of the wind layer for posts.

    python serve.py                       (or any http server on viewer/, port 8765)
    .venv-heat/Scripts/python tools/render_wind.py [shot ...]

Same headless Chrome as tools/render_post.py, never your own browser. A still
cannot move, so each shot is a long exposure: the animated light is stepped
through EXPOSURES phases and the frames are stacked, keeping the brightest
value of every pixel. Moving light becomes a streak whose length shows the
wind speed, like a night photograph of traffic. Images go to
data/wind/figures/post/ and older images are never overwritten: a new run
gets a new time stamp in its name.
"""

from __future__ import annotations

import base64
import io
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

OUT = os.path.join(HERE, "data", "wind", "figures", "post")
BASE = os.environ.get("VIEWER_URL", "http://127.0.0.1:8765/")
PORT = 9335
EXPOSURES = 10

CAMS = {
    # Behind 57th Street looking up the avenues, as the heat post.
    "hero": {"lat": 40.75735, "lon": -73.984154, "h": 900, "heading": 28.9, "pitch": -32},
    "heroP": {"lat": 40.75600, "lon": -73.98494, "h": 1050, "heading": 28.9, "pitch": -36},
    # From the north-west, over the park, looking downwind at the towers' windward faces.
    "upwind": {"lat": 40.7745, "lon": -73.9895, "h": 650, "heading": 140.0, "pitch": -24},
    "upwindP": {"lat": 40.7760, "lon": -73.9905, "h": 780, "heading": 140.0, "pitch": -28},
    # Side view along 57th Street, to see air thrown down the towers.
    "side": {"lat": 40.7685, "lon": -73.9700, "h": 260, "heading": 225.0, "pitch": -12},
    "top": {"lat": 40.75650, "lon": -73.98330, "h": 2300, "heading": 28.9, "pitch": -58},
}

SHOTS = [
    # name, page, query, size, camera
    ("wind_dec_upwind", "shadow-twin.html", "layer=wind&season=dec&t=12:30&anim=1", (1920, 1080), "upwind"),
    ("wind_dec_upwindP", "shadow-twin.html", "layer=wind&season=dec&t=12:30&anim=1", (1080, 1350), "upwindP"),
    ("wind_dec_hero", "shadow-twin.html", "layer=wind&season=dec&t=12:30&anim=1", (1920, 1080), "hero"),
    ("wind_dec_side", "shadow-twin.html", "layer=wind&season=dec&t=12:30&anim=1", (1920, 1080), "side"),
    ("wind_dec_top", "shadow-twin.html", "layer=wind&season=dec&t=12:30&anim=1", (1920, 1080), "top"),
    ("wind_dec_2017_upwind", "shadow-twin-2017.html", "layer=wind&season=dec&t=12:30&anim=1", (1920, 1080), "upwind"),
]

READY = """
(async () => {
  const t0 = performance.now();
  const until = (f, ms) => new Promise((ok) => {
    const tick = () => { let v = false; try { v = f(); } catch (e) {}
      (v || performance.now() - t0 > ms) ? ok(v) : setTimeout(tick, 250); };
    tick();
  });
  await until(() => typeof viewer !== 'undefined' && viewer && HEAT && WIND && activeLayer === 'wind', 180000);
  await until(() => viewer.dataSourceDisplay.ready, 240000);
  await until(() => [...windPrims.values()].some(p => p.show && p._cmd), 60000);
  return Math.round((performance.now() - t0) / 1000);
})()
"""

SHOOT = """
(async () => {
  const cam = %(cam)s;
  if (typeof sunRay !== 'undefined' && sunRay) sunRay.show = false;
  if (typeof sunDot !== 'undefined' && sunDot) sunDot.show = false;
  viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(cam.lon, cam.lat, cam.h),
    orientation: { heading: Cesium.Math.toRadians(cam.heading), pitch: Cesium.Math.toRadians(cam.pitch), roll: 0 } });
  const t0 = performance.now();
  await new Promise(r => setTimeout(r, 1500));
  let calm = 0;
  while (performance.now() - t0 < 180000) {
    calm = (viewer.scene.globe.tilesLoaded && viewer.dataSourceDisplay.ready) ? calm + 1 : 0;
    if (calm >= 8 && performance.now() - t0 > 8000) break;
    await new Promise(r => setTimeout(r, 250));
  }
  // Freeze the clock and step the light by hand.
  viewer.scene.preRender.removeEventListener(windTick);
  const shots = [];
  for (let k = 0; k < %(n)d; k++) {
    for (const p of windPrims.values()) { p.phase = k * %(step)f; p.speed = windNow().speed; }
    viewer.scene.render();
    shots.push(viewer.canvas.toDataURL('image/png'));
  }
  const ST = Cesium.SceneTransforms;
  const toWin = ST.worldToWindowCoordinates || ST.wgs84ToWindowCoordinates;
  const towers = (HEAT.towers || []).map(t => {
    const b = DATA.buildings.find(bb => bb.ring && pointInRing(t.lonlat, bb.ring));
    const hgt = b ? b.height : t.height_m;
    const p = toWin(viewer.scene, Cesium.Cartesian3.fromDegrees(t.lonlat[0], t.lonlat[1], hgt));
    return { lonlat: t.lonlat, height_m: hgt, x: p ? p.x : null, y: p ? p.y : null };
  });
  const w = windNow(), run = currentWindRun();
  return { shots, w: viewer.canvas.width, h: viewer.canvas.height, towers,
           wind: { dir: w.dir, speed: w.speed, run: run && run.name }, time: hstep().time };
})()
"""


def stack(shots):
    from PIL import Image, ImageChops
    out = None
    for s in shots:
        im = Image.open(io.BytesIO(base64.b64decode(s.split(",", 1)[1]))).convert("RGB")
        out = im if out is None else ImageChops.lighter(out, im)
    return out


def main():
    wanted = set(sys.argv[1:])
    os.makedirs(OUT, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    prof = tempfile.mkdtemp(prefix="render-wind-chrome-")
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={prof}", "--no-first-run", "--no-default-browser-check",
                             "--disable-extensions", "--hide-scrollbars", "--mute-audio",
                             "--ignore-gpu-blocklist", "--enable-gpu-rasterization",
                             "--window-size=1920,1080", "about:blank"],
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
        if not targets:
            raise SystemExit("headless Chrome did not start")
        tab = Tab(targets[0]["webSocketDebuggerUrl"])
        tab.call("Page.enable")
        tab.call("Runtime.enable")
        for name, page, query, (w, h), cam in SHOTS:
            if wanted and name not in wanted:
                continue
            tab.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)
            tab.call("Page.navigate", url=f"{BASE}{page}?{query}&r={int(time.time())}")
            time.sleep(3)
            secs = tab.js(READY, timeout_s=600)
            out = tab.js(SHOOT % {"cam": json.dumps(CAMS[cam]), "n": EXPOSURES, "step": 26.0 / EXPOSURES},
                         timeout_s=600)
            img = stack(out["shots"])
            path = os.path.join(OUT, f"{name}_{stamp}.png")
            img.save(path)
            single = os.path.join(OUT, f"{name}_{stamp}_frame.png")
            stack(out["shots"][:1]).save(single)
            with open(path.replace(".png", ".json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"camera": CAMS[cam], "page": page, "query": query, "towers": out["towers"],
                           "wind": out["wind"], "time": out["time"], "size": [out["w"], out["h"]]}, f, indent=1)
            print(f"{name}: {out['w']}x{out['h']}, ready in {secs} s, wind {out['wind']} -> {path}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        import shutil
        shutil.rmtree(prof, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
