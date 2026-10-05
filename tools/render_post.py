"""
Render still images of the viewer for posts, at full resolution.

    python serve.py                       (or any http server on viewer/, port 8765)
    .venv-heat/Scripts/python tools/render_post.py

Drives a headless Chrome (its own throwaway profile, never your browser) over
the DevTools protocol: opens each view as a deep link, waits until the map
tiles, the buildings and the heat columns are really there, sets the camera,
and saves the 3D canvas as a PNG in data/heat/figures/post/. No UI panels: the
titles and legends are added afterwards by tools/compose_post.py, so they stay
sharp and consistent.
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

import websocket

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "heat", "figures", "post")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE = os.environ.get("VIEWER_URL", "http://127.0.0.1:8765/")
PORT = 9333

# Camera positions, degrees and metres. "hero" stands behind 57th Street and
# looks up the avenues, so the towers are in front and the park beyond.
CAMS = {
    "hero": {"lat": 40.75735, "lon": -73.984154, "h": 900, "heading": 28.9, "pitch": -32},
    "heroP": {"lat": 40.75600, "lon": -73.98494, "h": 1050, "heading": 28.9, "pitch": -36},
    "top": {"lat": 40.75650, "lon": -73.98330, "h": 2300, "heading": 28.9, "pitch": -58},
    "topP": {"lat": 40.75640, "lon": -73.98330, "h": 1900, "heading": 28.9, "pitch": -54},
}

SHOTS = [
    # name, page, query, size, camera, hide the flat ground layer
    ("raw_dec_change_hero", "shadow-twin.html", "layer=change&season=dec", (1920, 1080), "hero", False),
    ("raw_dec_change_portrait", "shadow-twin.html", "layer=change&season=dec", (1080, 1350), "heroP", False),
    ("raw_dec_change_top", "shadow-twin.html", "layer=change&season=dec", (1920, 1080), "top", False),
    ("raw_dec_felt_today", "shadow-twin.html", "layer=felt&season=dec", (1920, 1080), "hero", True),
    ("raw_dec_felt_2017", "shadow-twin-2017.html", "layer=felt&season=dec", (1920, 1080), "hero", True),
    ("raw_jun_change_hero", "shadow-twin.html", "layer=change&season=jun", (1920, 1080), "hero", False),
    # Portrait set, 4:5, for a swipeable carousel.
    ("rawP_dec_change_top", "shadow-twin.html", "layer=change&season=dec", (1080, 1350), "topP", False),
    ("rawP_dec_felt_today", "shadow-twin.html", "layer=felt&season=dec", (1080, 1350), "heroP", True),
    ("rawP_dec_felt_2017", "shadow-twin-2017.html", "layer=felt&season=dec", (1080, 1350), "heroP", True),
    ("rawP_jun_change", "shadow-twin.html", "layer=change&season=jun", (1080, 1350), "heroP", False),
]


class Tab:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=600, suppress_origin=True)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        my = self.n
        self.ws.send(json.dumps({"id": my, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == my:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr, timeout_s=300):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True,
                      returnByValue=True, timeout=timeout_s * 1000)
        if "exceptionDetails" in r:
            raise RuntimeError(json.dumps(r["exceptionDetails"])[:800])
        return r.get("result", {}).get("value")


WAIT_READY = """
(async () => {
  const t0 = performance.now();
  const until = (f, ms) => new Promise((ok) => {
    const tick = () => (f() || performance.now() - t0 > ms) ? ok(f()) : setTimeout(tick, 250);
    tick();
  });
  await until(() => typeof viewer !== 'undefined' && viewer && typeof HEAT !== 'undefined' && HEAT, 120000);
  await until(() => viewer.dataSourceDisplay.ready, 240000);
  await until(() => [...colCache.values()].every(p => p.ready) && colCache.size > 0, 120000);
  return {secs: Math.round((performance.now() - t0) / 1000), layer: activeLayer,
          buildings: viewer.dataSourceDisplay.ready, cols: colCache.size};
})()
"""

SHOOT = """
(async () => {
  const cam = %(cam)s;
  if (typeof sunRay !== 'undefined' && sunRay) sunRay.show = false;
  if (typeof sunDot !== 'undefined' && sunDot) sunDot.show = false;
  const base = viewer.imageryLayers.get(0); base.saturation = 0.25; base.brightness = 0.8;
  // The page's own opacity, so a late redraw of the ground layer stays hidden too.
  if (%(hide_ground)s) { opacity = 0; if (overlayLayer) overlayLayer.alpha = 0; }
  viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(cam.lon, cam.lat, cam.h),
    orientation: { heading: Cesium.Math.toRadians(cam.heading), pitch: Cesium.Math.toRadians(cam.pitch), roll: 0 } });
  const t0 = performance.now();
  const until = (f, ms) => new Promise((ok) => {
    const tick = () => (f() || performance.now() - t0 > ms) ? ok(f()) : setTimeout(tick, 250);
    tick();
  });
  // Settle: the map tiles and the building geometry must stay ready for a
  // while, and never less than 8 s in all (fills can lag the outlines).
  await new Promise(r => setTimeout(r, 1500));
  let calm = 0;
  while (performance.now() - t0 < 180000) {
    calm = (viewer.scene.globe.tilesLoaded && viewer.dataSourceDisplay.ready) ? calm + 1 : 0;
    if (calm >= 8 && performance.now() - t0 > 8000) break;
    await new Promise(r => setTimeout(r, 250));
  }
  await new Promise(r => setTimeout(r, 1500));
  viewer.scene.render();
  // Where each grown tower's roof lands in the picture, for labels.
  const ST = Cesium.SceneTransforms;
  const toWin = ST.worldToWindowCoordinates || ST.wgs84ToWindowCoordinates;
  const towers = (HEAT.towers || []).map(t => {
    const b = DATA.buildings.find(bb => bb.ring && pointInRing(t.lonlat, bb.ring));
    const hgt = b ? b.height : t.height_m;
    const p = toWin(viewer.scene, Cesium.Cartesian3.fromDegrees(t.lonlat[0], t.lonlat[1], hgt));
    return { lonlat: t.lonlat, height_m: hgt, x: p ? p.x : null, y: p ? p.y : null };
  });
  return { png: viewer.canvas.toDataURL('image/png'), w: viewer.canvas.width, h: viewer.canvas.height,
           tiles: viewer.scene.globe.tilesLoaded, step: hstep().time, towers };
})()
"""


def main():
    wanted = set(sys.argv[1:])
    os.makedirs(OUT, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="render-post-chrome-")
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
        for name, page, query, (w, h), cam, hide in SHOTS:
            if wanted and name not in wanted:
                continue
            tab.call("Emulation.setDeviceMetricsOverride", width=w, height=h,
                     deviceScaleFactor=1, mobile=False)
            url = f"{BASE}{page}?{query}&r={int(time.time())}"
            tab.call("Page.navigate", url=url)
            time.sleep(3)
            info = tab.js(WAIT_READY, timeout_s=600)
            out = tab.js(SHOOT % {"cam": json.dumps(CAMS[cam]), "hide_ground": "true" if hide else "false"},
                         timeout_s=300)
            png = base64.b64decode(out["png"].split(",", 1)[1])
            path = os.path.join(OUT, name + ".png")
            with open(path, "wb") as f:
                f.write(png)
            with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"step": out["step"], "size": [out["w"], out["h"]], "camera": CAMS[cam],
                           "page": page, "query": query, "towers": out["towers"]}, f, indent=1)
            print(f"{name}: {out['w']}x{out['h']} at {out['step']} EST, ready in {info['secs']} s, "
                  f"buildings {info['buildings']}, tiles {out['tiles']} -> {path}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        import shutil
        shutil.rmtree(prof, ignore_errors=True)     # the throwaway profile made above
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
