"""
Raw renders for the wind post: Billionaires' Row from the windward side, with
only the street layer that cleared the noise bar and the lines that come down
Central Park Tower's face. Also the plain viewer screenshot (proof the tool
is real) from the same camera.

    python serve.py                       (or any http server on viewer/, port 8765)
    .venv-heat/Scripts/python tools/post_wind_data.py
    .venv-heat/Scripts/python tools/render_post_wind.py [shot ...]

Headless Chrome with a throwaway profile, as tools/render_post.py. The page
opens on the wind layer (December, 12:30, today); then the park colour layer
and every other wind line are removed, the basemap is muted as in the heat
post, viewer/wind/post_windier.png is laid on the ground and
viewer/wind/post_lines_today.bin drawn as plain white lines. No text: the
words go on in tools/compose_post_wind.py. Each render also saves where the
towers, their bases and the lines land on the picture.

Writes data/wind/figures/post/raw_<shot>_<stamp>.png and .json; never overwrites.
"""

from __future__ import annotations

import base64
import json
import math
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
PORT = 9344
TOWERS = {"Central Park Tower": (-73.98102, 40.766438, 472.4),
          "Steinway Tower": (-73.977562, 40.76496, 435.3),
          "53 W 53rd": (-73.978155, 40.761827, 320.0)}
AIM = (-73.9795, 40.7645)          # between the three towers


def nw_camera(dist_m, h_m, pitch, aim=AIM, bearing_from=315.0, fov=None):
    """A camera dist_m from the aim point towards bearing_from, looking back at it.

    fov: vertical field of view in degrees for a portrait picture (Cesium's
    default is 60); narrower is a longer lens, flatter and more like a photo."""
    b = math.radians(bearing_from)
    lat = aim[1] + dist_m * math.cos(b) / 111320.0
    lon = aim[0] + dist_m * math.sin(b) / (111320.0 * math.cos(math.radians(aim[1])))
    heading = (bearing_from + 180.0) % 360.0
    cam = {"lat": lat, "lon": lon, "h": h_m, "heading": heading, "pitch": pitch}
    if fov:
        cam["fov"] = fov
    return cam


CAMS = {
    "nwA": nw_camera(800, 330, -17),
    "nwB": nw_camera(650, 430, -27),
    "nwC": nw_camera(950, 280, -12),
    "nwD": nw_camera(700, 260, -10, bearing_from=305.0),
    "nwE": nw_camera(900, 600, -30),
    "nwF": nw_camera(1100, 720, -31, bearing_from=320.0),
    "nwG": nw_camera(850, 540, -30, bearing_from=300.0),
    "nwH": nw_camera(700, 500, -33, aim=(-73.98102, 40.766438)),
}
# Down 57th Street from the west-north-west: the street runs at 119 degrees, so
# the camera stands on the street's line behind Central Park Tower and looks
# along it, into the canyon where the street layer can be seen. The aim point is
# 57th Street's middle, about 50 m south-south-west of the tower's centre.
ST57 = (-73.98102 + 50 * math.sin(math.radians(209)) / (111320 * math.cos(math.radians(40.7664))),
        40.766438 + 50 * math.cos(math.radians(209)) / 111320)
CAMS.update({
    "s57I": nw_camera(700, 350, -24, aim=ST57, bearing_from=299.0),
    "s57J": nw_camera(950, 430, -24, aim=ST57, bearing_from=299.0),
    "s57K": nw_camera(800, 300, -18, aim=ST57, bearing_from=305.0),
    "s57L": nw_camera(1000, 560, -29, aim=ST57, bearing_from=305.0),
    "s57M": nw_camera(1000, 420, -11, aim=ST57, bearing_from=299.0, fov=40),
    "s57N": nw_camera(1300, 520, -11, aim=ST57, bearing_from=299.0, fov=32),
    "s57O": nw_camera(950, 380, -13, aim=ST57, bearing_from=303.0, fov=45),
    "s57P": nw_camera(1200, 600, -17, aim=ST57, bearing_from=301.0, fov=36),
    # The chosen view for B1 and B3.
    "hero": nw_camera(1300, 520, -12, aim=ST57, bearing_from=299.0, fov=30),
    "hero2": nw_camera(1300, 520, -9.5, aim=ST57, bearing_from=299.0, fov=31),
    "hero3": nw_camera(1250, 470, -9.0, aim=ST57, bearing_from=299.0, fov=33),
    "hero4": nw_camera(1300, 520, -9.5, aim=ST57, bearing_from=305.0, fov=31),
    "hero5": nw_camera(1300, 520, -9.5, aim=ST57, bearing_from=310.0, fov=31),
    "hero6": nw_camera(1300, 560, -10.5, aim=ST57, bearing_from=314.0, fov=31),
    "hero7": nw_camera(1300, 520, -10.8, aim=ST57, bearing_from=299.0, fov=32),
    "hero8": nw_camera(1350, 540, -11.2, aim=ST57, bearing_from=299.0, fov=32),
})
SHOTS = [
    # name, page, camera, size, kind ("post": post layers, no UI; "ui": the plain viewer)
    ("b1_nwA", "shadow-twin.html", "nwA", (1080, 1350), "post"),
    ("b1_nwB", "shadow-twin.html", "nwB", (1080, 1350), "post"),
    ("b1_nwC", "shadow-twin.html", "nwC", (1080, 1350), "post"),
    ("b1_nwD", "shadow-twin.html", "nwD", (1080, 1350), "post"),
    ("b1_nwE", "shadow-twin.html", "nwE", (1080, 1350), "post"),
    ("b1_nwF", "shadow-twin.html", "nwF", (1080, 1350), "post"),
    ("b1_nwG", "shadow-twin.html", "nwG", (1080, 1350), "post"),
    ("b1_nwH", "shadow-twin.html", "nwH", (1080, 1350), "post"),
    ("b1_s57I", "shadow-twin.html", "s57I", (1080, 1350), "post"),
    ("b1_s57J", "shadow-twin.html", "s57J", (1080, 1350), "post"),
    ("b1_s57K", "shadow-twin.html", "s57K", (1080, 1350), "post"),
    ("b1_s57L", "shadow-twin.html", "s57L", (1080, 1350), "post"),
    ("b1_s57M", "shadow-twin.html", "s57M", (1080, 1350), "post"),
    ("b1_s57N", "shadow-twin.html", "s57N", (1080, 1350), "post"),
    ("b1_s57O", "shadow-twin.html", "s57O", (1080, 1350), "post"),
    ("b1_s57P", "shadow-twin.html", "s57P", (1080, 1350), "post"),
    ("b1_hero", "shadow-twin.html", "hero", (1080, 1350), "post"),
    ("b1_hero2", "shadow-twin.html", "hero2", (1080, 1350), "post"),
    ("b1_hero3", "shadow-twin.html", "hero3", (1080, 1350), "post"),
    ("b1_hero4", "shadow-twin.html", "hero4", (1080, 1350), "post"),
    ("b1_hero5", "shadow-twin.html", "hero5", (1080, 1350), "post"),
    ("b1_hero6", "shadow-twin.html", "hero6", (1080, 1350), "post"),
    ("b1_hero7", "shadow-twin.html", "hero7", (1080, 1350), "post"),
    ("b1_hero8", "shadow-twin.html", "hero8", (1080, 1350), "post"),
    ("b3_viewer", "shadow-twin.html", "hero7", (1080, 1350), "ui"),
    # The same, without the legend and stats panels: they show first-round wind speeds.
    ("b3_viewer_clean", "shadow-twin.html", "hero7", (1080, 1350), "ui_clean"),
]
QUERY = "layer=wind&season=dec&t=12:30&anim=1&full=1"

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
  return Math.round((performance.now() - t0) / 1000);
})()
"""

POST_LAYERS = """
(async () => {
  // Amber only for buildings that really grew (tools/tower_heights.py).
  const grew = %(grew)s;
  if (grew) {
    HEAT.towers = HEAT.towers.filter(t => grew.some(g =>
      Math.abs(g[0] - t.lonlat[0]) < 1e-4 && Math.abs(g[1] - t.lonlat[1]) < 1e-4));
    recolourBuildings();
  }
  // Only the post's layers: no park colour, no other wind lines, a muted basemap.
  windAnim = false;
  for (const p of windPrims.values()) p.show = false;
  if (overlayLayer) { viewer.imageryLayers.remove(overlayLayer, true); overlayLayer = null; }
  _overlaySeq++;                      // any overlay still loading is dropped
  windBasemap(false);
  const base = viewer.imageryLayers.get(0); base.saturation = 0.25; base.brightness = 0.85;
  if (typeof sunRay !== 'undefined' && sunRay) sunRay.show = false;
  if (typeof sunDot !== 'undefined' && sunDot) sunDot.show = false;
  // A flat, muted sky instead of the default blue gradient.
  viewer.scene.skyAtmosphere.show = false;
  if (viewer.scene.skyBox) viewer.scene.skyBox.show = false;
  viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#dfe1e3');
  viewer.scene.fog.enabled = false;
  const meta = await (await fetch('wind/post_windier.json', {cache: 'no-store'})).json();
  const b = meta.bounds;
  const prov = await Cesium.SingleTileImageryProvider.fromUrl('wind/post_windier.png?r=' + Date.now(),
    { rectangle: Cesium.Rectangle.fromDegrees(b.west, b.south, b.east, b.north) });
  const lay = viewer.imageryLayers.addImageryProvider(prov);
  lay.alpha = 0.92;
  const lines = await fetchLines('wind/post_lines_today.bin?r=' + Date.now());
  const p = new WindLines(lines, 'air', WIND.origin, WIND.spacing_k1_m, { solid: true, width: 2.4, color: [0.11, 0.12, 0.14], rim: [1, 1, 1] });
  window._postLines = viewer.scene.primitives.add(p);
  window._postLineData = lines;
  return lines.n;
})()
"""

SHOOT = """
(async () => {
  const cam = %(cam)s;
  if (cam.fov) viewer.camera.frustum.fov = Cesium.Math.toRadians(cam.fov);
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
  viewer.scene.render();
  const ST = Cesium.SceneTransforms;
  const toWin = ST.worldToWindowCoordinates || ST.wgs84ToWindowCoordinates;
  const win = (c) => { const p = toWin(viewer.scene, c); return p ? [p.x, p.y] : null; };
  const towers = {};
  for (const [nm, t] of Object.entries(%(towers)s)) {
    towers[nm] = { roof: win(Cesium.Cartesian3.fromDegrees(t[0], t[1], t[2])),
                   base: win(Cesium.Cartesian3.fromDegrees(t[0], t[1], 0)) };
  }
  // Every 3rd vertex of each post line, on the picture.
  const lines = [];
  const L = window._postLineData, M = window._postLines ? window._postLines.modelMatrix : null;
  if (L && M) {
    let o = 0;
    for (let k = 0; k < L.n; k++) {
      const pts = [];
      for (let v = 0; v < L.lens[k]; v += 3) {
        const a = (o + v) * 3;
        const local = new Cesium.Cartesian3(L.pts[a] / 10, L.pts[a + 1] / 10, L.pts[a + 2] / 10);
        const w = Cesium.Matrix4.multiplyByPoint(M, local, new Cesium.Cartesian3());
        pts.push(win(w));
      }
      lines.push(pts);
      o += L.lens[k];
    }
  }
  // The purple patch at the tower's foot, on the picture.
  let pool = null;
  try {
    const pm = await (await fetch('wind/post_windier.json', {cache: 'no-store'})).json();
    if (pm.pool_lonlat) pool = win(Cesium.Cartesian3.fromDegrees(pm.pool_lonlat[0], pm.pool_lonlat[1], 1));
  } catch (e) {}
  // 57th Street in front of Central Park Tower, at street level.
  const st57 = win(Cesium.Cartesian3.fromDegrees(%(st_lon)f, %(st_lat)f, 1));
  // Which way north points on the picture: a short step north from the aim point.
  const a0 = win(Cesium.Cartesian3.fromDegrees(%(aim_lon)f, %(aim_lat)f, 0));
  const a1 = win(Cesium.Cartesian3.fromDegrees(%(aim_lon)f, %(aim_lat)f + 0.002, 0));
  // Render again right before the capture: any await above lets the browser
  // clear the WebGL drawing buffer.
  viewer.scene.render();
  return { png: viewer.canvas.toDataURL('image/png'), w: viewer.canvas.width, h: viewer.canvas.height,
           towers, lines, pool, st57, north: [a0, a1], tiles: viewer.scene.globe.tilesLoaded };
})()
"""


def grown_list():
    """[lon, lat] of the listed towers that really grew, or None if not checked yet."""
    p = os.path.join(HERE, "data", "wind", "tower_heights_2017.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return [c["lonlat"] for c in d.get("_checked_grown_list", []) if c["grew"]] or None


def main():
    wanted = set(sys.argv[1:])
    os.makedirs(OUT, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prof = tempfile.mkdtemp(prefix="render-post-wind-")
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={prof}", "--no-first-run", "--no-default-browser-check",
                             "--disable-extensions", "--hide-scrollbars", "--mute-audio",
                             "--ignore-gpu-blocklist", "--enable-gpu-rasterization",
                             "--window-size=1080,1350", "about:blank"],
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
        for name, page, cam, (w, h), kind in SHOTS:
            if wanted and name not in wanted:
                continue
            tab.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)
            tab.call("Page.navigate", url=f"{BASE}{page}?{QUERY}&r={int(time.time())}")
            time.sleep(3)
            secs = tab.js(READY, timeout_s=600)
            if kind == "post":
                n = tab.js(POST_LAYERS % {"grew": json.dumps(grown_list())}, timeout_s=300)
            else:
                n = None
                if kind == "ui_clean":
                    tab.js("(() => { for (const id of ['legend', 'stats']) document.getElementById(id).style.display = 'none'; return 1; })()")
            out = tab.js(SHOOT % {"cam": json.dumps(CAMS[cam]), "towers": json.dumps(TOWERS),
                                  "aim_lon": AIM[0], "aim_lat": AIM[1], "st_lon": ST57[0], "st_lat": ST57[1]},
                         timeout_s=600)
            path = os.path.join(OUT, f"raw_{name}_{stamp}.png")
            if kind in ("ui", "ui_clean"):
                shot = tab.call("Page.captureScreenshot", format="png")
                data = base64.b64decode(shot["data"])
            else:
                data = base64.b64decode(out["png"].split(",", 1)[1])
            with open(path, "wb") as f:
                f.write(data)
            out.pop("png")
            out.update({"camera": CAMS[cam], "page": page, "query": QUERY, "kind": kind, "post_lines": n})
            with open(path.replace(".png", ".json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(out, f, indent=1)
            print(f"{name}: {out['w']}x{out['h']}, ready in {secs} s, lines {n}, tiles {out['tiles']} -> {path}",
                  flush=True)
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
