"""
Check the wind layer in a headless Chrome and save screenshots.

    python serve.py                       (or any http server on viewer/, port 8765)
    .venv-heat/Scripts/python tools/wind_qa.py [name ...]

Opens each case as a deep link, waits for the map, the buildings and the wind
lines, collects every exception and console error, measures the frame rate
with the animation on, and saves a screenshot to data/wind/qa/. Never your own
browser: a throwaway profile, as in tools/render_post.py.
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
from render_post import CHROME, Tab  # noqa: E402

OUT = os.path.join(HERE, "data", "wind", "qa")
BASE = os.environ.get("VIEWER_URL", "http://127.0.0.1:8765/")
PORT = 9334

CASES = [
    # name, page, query, size, mobile
    ("wind_dec", "shadow-twin.html", "layer=wind&season=dec", (1600, 900), False),
    ("wind_dec_2017", "shadow-twin-2017.html", "layer=wind&season=dec", (1600, 900), False),
    ("wind_live", "shadow-twin.html", "layer=wind&season=live", (1600, 900), False),
    ("wind_jun", "shadow-twin.html", "layer=wind&season=jun", (1600, 900), False),
    ("felt_local", "shadow-twin.html", "layer=change&season=dec", (1600, 900), False),
    ("felt_station", "shadow-twin.html", "layer=change&season=dec&localwind=0", (1600, 900), False),
    ("phone_wind", "shadow-twin.html", "layer=wind&season=dec", (390, 844), True),
    ("phone_wind_anim", "shadow-twin.html", "layer=wind&season=dec&anim=1", (390, 844), True),
]

WAIT = """
(async () => {
  const t0 = performance.now();
  const until = (f, ms) => new Promise((ok) => {
    const tick = () => { let v = false; try { v = f(); } catch (e) {}
      (v || performance.now() - t0 > ms) ? ok(v) : setTimeout(tick, 250); };
    tick();
  });
  await until(() => typeof viewer !== 'undefined' && viewer && typeof HEAT !== 'undefined' && HEAT, 120000);
  await until(() => viewer.dataSourceDisplay.ready, 240000);
  const wantWind = new URLSearchParams(location.search).get('layer') === 'wind';
  if (wantWind) await until(() => WIND && activeLayer === 'wind', 60000);
  if (wantWind && windAnim) await until(() => [...windPrims.values()].some(p => p.show && p._cmd), 60000);
  let calm = 0;
  while (performance.now() - t0 < 120000) {
    calm = viewer.scene.globe.tilesLoaded ? calm + 1 : 0;
    if (calm >= 6) break;
    await new Promise(r => setTimeout(r, 250));
  }
  // Frame rate over three seconds with whatever is on.
  let frames = 0; const f0 = performance.now();
  const cb = () => frames++;
  viewer.scene.postRender.addEventListener(cb);
  await new Promise(r => setTimeout(r, 3000));
  viewer.scene.postRender.removeEventListener(cb);
  const fps = frames / ((performance.now() - f0) / 1000);
  viewer.scene.render();
  return {
    secs: Math.round((performance.now() - t0) / 1000), layer: activeLayer, fps: Math.round(fps),
    wind: !!WIND, windheat: !!WINDHEAT, anim: windAnim, live: windLiveMode,
    liveErr: windLiveErr, prims: [...windPrims.entries()].map(([u, p]) => [u.split('/').pop(), p.show, p.lines.n]),
    legend: document.getElementById('legtitle').textContent + ' | ' + document.getElementById('legunits').textContent,
    stats: document.getElementById('statstable').innerText,
    png: viewer.canvas.toDataURL('image/png'),
    full: null
  };
})()
"""


def main():
    wanted = set(sys.argv[1:])
    os.makedirs(OUT, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="wind-qa-chrome-")
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={prof}", "--no-first-run", "--no-default-browser-check",
                             "--disable-extensions", "--hide-scrollbars", "--mute-audio",
                             "--ignore-gpu-blocklist", "--enable-gpu-rasterization",
                             "--window-size=1600,900", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = {}
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
        tab.call("Log.enable")
        for name, page, query, (w, h), mobile in CASES:
            if wanted and name not in wanted:
                continue
            tab.call("Emulation.setDeviceMetricsOverride", width=w, height=h,
                     deviceScaleFactor=2 if mobile else 1, mobile=mobile)
            tab.call("Emulation.setTouchEmulationEnabled", enabled=mobile)
            errors = []
            tab.call("Page.navigate", url=f"{BASE}{page}?{query}&r={int(time.time())}")
            time.sleep(2)
            res = tab.js(WAIT, timeout_s=600)
            # Drain the events that arrived meanwhile: exceptions and console errors.
            tab.ws.settimeout(0.5)
            try:
                while True:
                    m = json.loads(tab.ws.recv())
                    meth = m.get("method", "")
                    if meth == "Runtime.exceptionThrown":
                        d = m["params"]["exceptionDetails"]
                        errors.append("exception: " + (d.get("exception", {}).get("description") or d.get("text", ""))[:300])
                    elif meth == "Runtime.consoleAPICalled" and m["params"]["type"] in ("error", "warning"):
                        errors.append(m["params"]["type"] + ": " + " ".join(
                            str(a.get("value", a.get("description", ""))) for a in m["params"]["args"])[:300])
                    elif meth == "Log.entryAdded" and m["params"]["entry"]["level"] in ("error",):
                        e = m["params"]["entry"]
                        errors.append("log: " + e.get("text", "")[:200] + " " + e.get("url", "")[-60:])
            except Exception:
                pass
            tab.ws.settimeout(600)
            shot = tab.call("Page.captureScreenshot", format="png")
            with open(os.path.join(OUT, name + ".png"), "wb") as f:
                f.write(base64.b64decode(shot["data"]))
            res.pop("png", None)
            res["errors"] = errors
            results[name] = res
            print(f"{name}: layer {res['layer']}, {res['fps']} fps, anim {res['anim']}, "
                  f"prims {res['prims']}, {len(errors)} errors", flush=True)
            for e in errors:
                print("   ", e)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        import shutil
        shutil.rmtree(prof, ignore_errors=True)
    with open(os.path.join(OUT, "qa.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
