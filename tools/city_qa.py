"""
Open a city's viewer pages in a headless Chrome, collect page errors and take
a screenshot of each: the QA pass from the New York layers, for any city.

    python serve.py            (or: python -m http.server 8765 --bind 127.0.0.1 from viewer/)
    TWIN_CITY=rotterdam .venv-heat/Scripts/python tools/city_qa.py
    TWIN_CITY=rotterdam .venv-heat/Scripts/python tools/city_qa.py "layer=felt&season=dec"

Screenshots go to <city data>/figures/qa_<page>_<time>.png, never over an
older one. Harmless errors, as in New York: the vendored Cesium 404 (the page
falls back to the CDN) and the favicon.
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
sys.path.insert(0, HERE)

from pipeline import city  # noqa: E402

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE = os.environ.get("VIEWER_URL", "http://127.0.0.1:8765/")
PORT = 9334
HARMLESS = ("vendor/Cesium", "favicon", "2016.html", "2017.html", "index.html")


class Tab:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=600, suppress_origin=True)
        self.n = 0
        self.events = []

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
            if "method" in msg:
                self.events.append(msg)

    def js(self, expr, timeout_s=300):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True,
                      returnByValue=True, timeout=timeout_s * 1000)
        if "exceptionDetails" in r:
            raise RuntimeError(json.dumps(r["exceptionDetails"])[:800])
        return r.get("result", {}).get("value")


def launch():
    prof = tempfile.mkdtemp(prefix="city-qa-chrome-")
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={prof}", "--no-first-run", "--no-default-browser-check",
                             "--disable-extensions", "--hide-scrollbars", "--mute-audio",
                             "--ignore-gpu-blocklist", "--window-size=1600,1000", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json") as r:
                t = [x for x in json.load(r) if x.get("type") == "page"]
            if t:
                return proc, Tab(t[0]["webSocketDebuggerUrl"])
        except OSError:
            pass
        time.sleep(0.5)
    raise SystemExit("headless Chrome did not start")


READY = """
(async () => {
  const t0 = performance.now();
  const until = (f, ms) => new Promise((ok) => {
    const tick = () => { let v = false; try { v = f(); } catch (e) {}
      (v || performance.now() - t0 > ms) ? ok(v) : setTimeout(tick, 250); };
    tick();
  });
  const ok = await until(() => typeof viewer !== 'undefined' && viewer && viewer.dataSourceDisplay.ready, 240000);
  let calm = 0;
  while (performance.now() - t0 < 240000) {
    calm = (viewer.scene.globe.tilesLoaded && viewer.dataSourceDisplay.ready) ? calm + 1 : 0;
    if (calm >= 8 && performance.now() - t0 > 8000) break;
    await new Promise(r => setTimeout(r, 250));
  }
  await new Promise(r => setTimeout(r, 1500));
  viewer.scene.render();
  return { ready: !!ok, secs: Math.round((performance.now() - t0) / 1000),
           layer: typeof activeLayer !== 'undefined' ? activeLayer : null,
           title: document.title, heat: typeof HEAT !== 'undefined' && !!HEAT,
           png: viewer.canvas.toDataURL('image/png') };
})()
"""


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    rel = os.path.relpath(city.viewer_dir(), os.path.join(HERE, "viewer")).replace(os.sep, "/")
    out_dir = os.path.join(city.data_dir(), "figures")
    os.makedirs(out_dir, exist_ok=True)
    proc, tab = launch()
    report = {}
    try:
        tab.call("Runtime.enable")
        tab.call("Log.enable")
        tab.call("Network.enable")
        tab.call("Emulation.setDeviceMetricsOverride", width=1600, height=1000, deviceScaleFactor=1, mobile=False)
        pages = ["shadow-twin.html", f"shadow-twin-{city.before()}.html"]
        for page in pages:
            tab.events.clear()
            url = f"{BASE}{rel}/{page}" + (f"?{query}" if query else "")
            tab.call("Page.navigate", url=url)
            r = tab.js(READY, timeout_s=300)
            errs = []
            for ev in tab.events:
                m, p = ev["method"], ev.get("params", {})
                if m == "Runtime.exceptionThrown":
                    errs.append(p["exceptionDetails"].get("exception", {}).get("description", str(p))[:300])
                elif m == "Log.entryAdded" and p["entry"]["level"] == "error":
                    errs.append(f"{p['entry'].get('text', '')[:200]} {p['entry'].get('url', '')}")
                elif m == "Runtime.consoleAPICalled" and p.get("type") == "error":
                    errs.append(" ".join(str(a.get("value", a.get("description", ""))) for a in p["args"])[:300])
            real = [e for e in errs if not any(h in e for h in HARMLESS)]
            stamp = time.strftime("%Y%m%d_%H%M%S")
            png = os.path.join(out_dir, f"qa_{page.replace('.html', '')}_{stamp}.png")
            with open(png, "wb") as f:
                f.write(base64.b64decode(r.pop("png").split(",", 1)[1]))
            report[page] = {**r, "errors": real, "harmless": len(errs) - len(real), "screenshot": png}
            print(f"{page}: ready {r['ready']} in {r['secs']} s, title '{r['title']}', heat {r['heat']}, "
                  f"{len(real)} errors, {len(errs) - len(real)} harmless; {png}")
            for e in real:
                print(f"    ERROR {e}")
    finally:
        proc.kill()
    with open(os.path.join(out_dir, f"qa_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return 0 if all(not v["errors"] for v in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
