"""
Renders for the single-image LinkedIn post, into data/heat/post/.

    python serve.py                    (viewer on http://127.0.0.1:8765)
    .venv-heat/Scripts/python tools/render_linkedin.py

Same headless Chrome as tools/render_post.py, plus three overlays drawn on the
ground: the whole of Central Park outlined (59th to 110th Street), 59th Street,
and a dashed line where the heat model's analysis area ends (1.5 km north of
59th Street). North of that line nothing was computed, so the picture must not
suggest "no change" there by itself; the line says where the data stops.

Also captures the viewer itself, panels included, for screenshot use.
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import timedelta

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
from render_post import CHROME, Tab, WAIT_READY  # noqa: E402

OUT = os.path.join(HERE, "data", "heat", "post")
BASE = os.environ.get("VIEWER_URL", "http://127.0.0.1:8765/")
PORT = 9335
WINDOW_M = 1500.0


def overlays():
    """Park outline, 59th Street and the end of the modelled area, lon/lat."""
    import numpy as np
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform, unary_union
    with open(os.path.join(HERE, "data", "heat", "inputs", "domain.json"), encoding="utf-8") as f:
        m = json.load(f)
    ax = m["avenue_axis"]
    ux, uy, a0 = ax["ux"], ax["uy"], ax["a0_ft"]
    vx, vy = uy, -ux
    with open(os.path.join(HERE, "data", "nyc", "park.geojson"), encoding="utf-8") as f:
        gj = json.load(f)
    park = unary_union([shape(ft["geometry"]) for ft in gj["features"]
                        if (ft["properties"].get("signname") or "").lower() == "central park"])
    if park.geom_type == "MultiPolygon":
        park = max(park.geoms, key=lambda p: p.area)
    to_ft = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True).transform
    to_ll = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True).transform
    pc = transform(to_ft, park)
    xs, ys = np.asarray(pc.convex_hull.exterior.xy)
    across = xs * vx + ys * vy
    w0, w1 = float(across.min()) - 300, float(across.max()) + 300     # ft past the park edges
    ft = 1 / 0.3048006

    def line(along_m):
        a = a0 + along_m * ft
        return [list(to_ll(a * ux + w * vx, a * uy + w * vy)) for w in (w0, w1)]

    outline = [list(p) for p in park.simplify(0.00003).exterior.coords]
    mid59 = [(line(0)[0][i] + line(0)[1][i]) / 2 for i in (0, 1)]
    return {"outline": outline, "line59": line(0), "limit": line(WINDOW_M), "mid59": mid59,
            "heading": 28.9 + 0.0}


def day_composite(season="dec"):
    """The change at its coldest hour, per spot, over the reported window.

    The headline number counts park that feels 5 C colder at some hour; one
    hour shows only part of it, because the shadows move. This is the same
    change, each spot at its own worst hour, so the map and the number match.
    """
    import numpy as np
    sys.path.insert(0, HERE)
    from pipeline import export
    from pipeline.heat_export import CELL_HA, change_palette, columns, heat_grid, png_change
    from pipeline.heat_inputs import load
    from pipeline.verify_heat import read_step, report_steps
    from pyproj import Transformer
    _, masks, meta = load()
    report, grid = masks["report"], heat_grid(meta, masks)
    idx, bounds = grid.northup_index()
    painter = export.Painter(idx, bounds)
    worst = None
    for ts in report_steps(season):
        du = read_step("today", season, "utci", ts) - read_step("2017", season, "utci", ts)
        worst = du if worst is None else np.fmin(worst, du)
    to_ll = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True).transform
    fsl = grid.frame_slice
    ha5 = float((worst[report] <= -5).sum() * CELL_HA)
    return {"png": png_change(painter, worst[fsl], masks["building"][fsl], change_palette()),
            "cols": columns(worst, report, grid, to_ll), "ha5": ha5,
            "steps": [f"{t - timedelta(minutes=30):%H:%M}" for t in report_steps(season)]}


USE_COMPOSITE = """
(() => {
  const c = %(c)s;
  const st = hstep();
  st.img.change = c.png;
  st.cols = c.cols;
  for (const [k, p] of colCache) { viewer.scene.primitives.remove(p); colCache.delete(k); }
  setHour(heatStep);
  return st.time;
})()
"""


def camera_behind(mid, back_m, h, heading, pitch):
    """A camera back_m behind a point, looking along heading."""
    lon, lat = mid
    t = math.radians(heading)
    dn, de = -back_m * math.cos(t), -back_m * math.sin(t)
    return {"lat": lat + dn / 111320.0, "lon": lon + de / (111320.0 * math.cos(math.radians(lat))),
            "h": h, "heading": heading, "pitch": pitch}


ADD_OVERLAYS = """
(() => {
  const o = %(ov)s;
  const C = Cesium.Color;
  const deg = a => Cesium.Cartesian3.fromDegreesArray(a.flat());
  viewer.entities.add({ polyline: { positions: deg(o.outline), width: 4, clampToGround: true,
    material: C.fromCssColorString('#f0f3f7').withAlpha(0.95) } });
  viewer.entities.add({ polyline: { positions: deg(o.line59), width: 4, clampToGround: true,
    material: C.fromCssColorString('#f5c451') } });
  if (%(limit)s) viewer.entities.add({ polyline: { positions: deg(o.limit), width: 3, clampToGround: true,
    material: new Cesium.PolylineDashMaterialProperty({ color: C.fromCssColorString('#f0f3f7'), dashLength: 18 }) } });
  return true;
})()
"""

SHOOT = """
(async () => {
  const cam = %(cam)s;
  if (typeof sunRay !== 'undefined' && sunRay) sunRay.show = false;
  if (typeof sunDot !== 'undefined' && sunDot) sunDot.show = false;
  const base = viewer.imageryLayers.get(0); base.saturation = %(sat)s; base.brightness = %(bri)s;
  if (%(hide_ground)s) { opacity = 0; if (overlayLayer) overlayLayer.alpha = 0; }
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
  await new Promise(r => setTimeout(r, 1500));
  viewer.scene.render();
  const ST = Cesium.SceneTransforms;
  const toWin = ST.worldToWindowCoordinates || ST.wgs84ToWindowCoordinates;
  const pt = (lon, lat, h) => { const p = toWin(viewer.scene, Cesium.Cartesian3.fromDegrees(lon, lat, h || 0));
                                return p ? [p.x, p.y] : null; };
  const o = %(ov)s;
  // Where each grown tower's roof lands in the picture, for labels.
  const towers = (HEAT.towers || []).map(t => {
    const b = DATA.buildings.find(bb => bb.ring && pointInRing(t.lonlat, bb.ring));
    const hgt = b ? b.height : t.height_m;
    const p = pt(t.lonlat[0], t.lonlat[1], hgt);
    return { lonlat: t.lonlat, height_m: hgt, x: p ? p[0] : null, y: p ? p[1] : null };
  });
  return { png: viewer.canvas.toDataURL('image/png'), w: viewer.canvas.width, h: viewer.canvas.height,
           step: hstep().time, line59: o.line59.map(p => pt(p[0], p[1])), limit: o.limit.map(p => pt(p[0], p[1])),
           north: [pt(o.mid59[0], o.mid59[1]), pt(o.mid59[0], o.mid59[1] + 0.003)], towers };
})()
"""


def launch():
    prof = tempfile.mkdtemp(prefix="render-li-chrome-")
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={prof}", "--no-first-run", "--no-default-browser-check",
                             "--disable-extensions", "--hide-scrollbars", "--mute-audio",
                             "--ignore-gpu-blocklist", "--window-size=1920,1080", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json") as r:
                t = [x for x in json.load(r) if x.get("type") == "page"]
            if t:
                return proc, prof, Tab(t[0]["webSocketDebuggerUrl"])
        except OSError:
            pass
        time.sleep(0.5)
    raise SystemExit("headless Chrome did not start")


def main():
    wanted = set(sys.argv[1:])
    os.makedirs(OUT, exist_ok=True)
    ov = overlays()
    whole = camera_behind(ov["mid59"], 820, 1500, 28.9, -39.5)
    south = camera_behind(ov["mid59"], 900, 1000, 28.9, -36)
    # The carousel's tower slide (render_post "topP"): looking up the avenues
    # from over Midtown, the towers in front and their fingers beyond.
    towers = {"lat": 40.75640, "lon": -73.98330, "h": 1900, "heading": 28.9, "pitch": -54}
    shots = [
        # name, page, query, size, camera, hide flat layer, overlays, (saturation, brightness), muted towers
        ("li_main_raw", "shadow-twin.html", "layer=change&season=dec", (1080, 1350), whole, False, True, (0.2, 0.62), True),
        ("li_towers_raw", "shadow-twin.html", "layer=change&season=dec", (1080, 1350), towers, False, True, (0.35, 0.8), False),
        ("li_felt_2017_raw", "shadow-twin-2017.html", "layer=felt&season=dec", (1080, 640), south, True, False, (0.2, 0.62), False),
        ("li_felt_today_raw", "shadow-twin.html", "layer=felt&season=dec", (1080, 640), south, True, False, (0.2, 0.62), False),
    ]
    comp = None
    if not wanted or wanted & {"li_main_raw", "li_towers_raw"}:
        comp = day_composite()
        print(f"day composite {comp['steps'][0]} to {comp['steps'][-1]}: "
              f"{comp['ha5']:.2f} ha at least 5 C colder", flush=True)
        with open(os.path.join(OUT, "li_main_composite.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"steps": comp["steps"], "ha5": comp["ha5"]}, f, indent=1)
    proc, prof, tab = launch()
    try:
        tab.call("Page.enable")
        tab.call("Runtime.enable")
        for name, page, query, (w, h), cam, hide, ovl, (sat, bri), mute in shots:
            if wanted and name not in wanted:
                continue
            tab.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)
            tab.call("Page.navigate", url=f"{BASE}{page}?{query}&r={int(time.time())}")
            time.sleep(3)
            info = tab.js(WAIT_READY, timeout_s=600)
            if ovl:
                tab.js(ADD_OVERLAYS % {"ov": json.dumps(ov), "limit": "true"})
            if mute:
                # Grown towers in a quieter amber, so the blue fingers lead the eye.
                tab.js("""(() => { const base = buildingColour;
                  window.buildingColour = b => (HEAT && isHeat() && towerFor(b))
                    ? Cesium.Color.fromCssColorString('#b8955a') : base(b);
                  recolourBuildings(); return true; })()""")
            if ovl:
                tab.js(USE_COMPOSITE % {"c": json.dumps({"png": comp["png"], "cols": comp["cols"]})})
            out = tab.js(SHOOT % {"cam": json.dumps(cam), "hide_ground": "true" if hide else "false",
                                  "ov": json.dumps(ov), "sat": sat, "bri": bri}, timeout_s=300)
            with open(os.path.join(OUT, name + ".png"), "wb") as f:
                f.write(base64.b64decode(out["png"].split(",", 1)[1]))
            meta = {k: out[k] for k in ("w", "h", "step", "line59", "limit", "north", "towers")}
            meta["camera"] = cam
            with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(meta, f, indent=1)
            print(f"{name}: {out['w']}x{out['h']} at {out['step']} EST (ready {info['secs']} s)", flush=True)
        # The viewer itself, panels and all, for screenshot use.
        for name, query in (("li_viewer_change", "layer=change&season=dec"),
                            ("li_viewer_felt", "layer=felt&season=dec")):
            if wanted and name not in wanted:
                continue
            tab.call("Emulation.setDeviceMetricsOverride", width=1920, height=1080, deviceScaleFactor=1, mobile=False)
            tab.call("Page.navigate", url=f"{BASE}shadow-twin.html?{query}&r={int(time.time())}")
            time.sleep(3)
            tab.js(WAIT_READY, timeout_s=600)
            tab.js("""(async () => { const t0 = performance.now(); let calm = 0;
              while (performance.now() - t0 < 120000) {
                calm = (viewer.scene.globe.tilesLoaded && viewer.dataSourceDisplay.ready) ? calm + 1 : 0;
                if (calm >= 8 && performance.now() - t0 > 8000) break;
                await new Promise(r => setTimeout(r, 250)); }
              await new Promise(r => setTimeout(r, 1500)); return true; })()""", timeout_s=300)
            shot = tab.call("Page.captureScreenshot", format="png")
            with open(os.path.join(OUT, name + ".png"), "wb") as f:
                f.write(base64.b64decode(shot["data"]))
            print(f"{name}: viewer screenshot 1920x1080", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(prof, ignore_errors=True)      # the throwaway profile made above
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
