"""
The exposures behind a wind still, kept one by one, so the still can be
stacked with the light brightening toward its head (tools/compose_post_wind2.py):
a streak then shows which way the air goes, and its length how fast.

    python serve.py                       (or any http server on viewer/, port 8765)
    .venv-heat/Scripts/python tools/render_wind_frames.py [camera] [exposures] [step] [2017]
    .venv-heat/Scripts/python tools/render_wind_frames.py --city rotterdam --dir 247.5 zhUpwindP 12 0.6 [2016]

Same headless Chrome and cameras as tools/render_wind.py, never your own
browser; "2017" renders the 2017 build (another city: its before year).
--city reads viewer/cities/<city>/ and writes under data/cities/<city>/figures/post/;
--dir shows that wind direction's run instead of the heat day's wind. Writes data/wind/figures/post/frames_<camera>[_2017]_<stamp>/ with
frame_00.png ... and meta.json (camera, tower screen positions, wind).
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
from render_wind import BASE, CAMS, OUT, READY, SHOOT  # noqa: E402

PORT = 9339
PAGE = "shadow-twin.html"
QUERY = "layer=wind&season=dec&t=12:30&anim=1&full=1"
SIZE = (1080, 1350)


# --dir: show this wind direction's run instead of the heat day's wind (the viewer picks the nearest run).
WIND_DIR = """
(async () => {
  const s0 = windNow().speed;
  windNow = () => ({ dir: %(dir)f, speed: s0, season: hseason().id, live: false });
  await updateWindLines();
  refresh();
  await new Promise(r => setTimeout(r, 3000));
  const run = currentWindRun();
  return run && run.name;
})()
"""


def flag(args, name, default=None):
    if name not in args:
        return default
    k = args.index(name)
    v = args[k + 1]
    del args[k:k + 2]
    return v


def main():
    args = sys.argv[1:]
    city = flag(args, "--city")            # e.g. rotterdam: viewer/cities/<city>/, its before year from the page names
    wdir = flag(args, "--dir")
    cam = args[0] if len(args) > 0 else "upwindP"
    n = int(args[1]) if len(args) > 1 else 12
    step = float(args[2]) if len(args) > 2 else 1.25
    before = args[3] if len(args) > 3 else None
    prefix = f"cities/{city}/" if city else ""
    page = prefix + (f"shadow-twin-{before}.html" if before else PAGE)
    out_dir = os.path.join(OUT if not city else os.path.join(HERE, "data", "cities", city, "figures", "post"),
                           f"frames_{cam}{'_' + before if before else ''}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir)
    prof = tempfile.mkdtemp(prefix="render-wind-frames-")
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
        tab.call("Emulation.setDeviceMetricsOverride", width=SIZE[0], height=SIZE[1], deviceScaleFactor=1,
                 mobile=False)
        tab.call("Page.navigate", url=f"{BASE}{page}?{QUERY}&r={int(time.time())}")
        time.sleep(3)
        secs = tab.js(READY, timeout_s=600)
        if wdir is not None:
            print(f"wind run in view: {tab.js(WIND_DIR % {'dir': float(wdir)}, timeout_s=300)}", flush=True)
        res = tab.js(SHOOT % {"cam": json.dumps(CAMS[cam]), "n": n, "step": step}, timeout_s=900)
        for k, s in enumerate(res["shots"]):
            with open(os.path.join(out_dir, f"frame_{k:02d}.png"), "wb") as f:
                f.write(base64.b64decode(s.split(",", 1)[1]))
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"camera": CAMS[cam], "page": page, "query": QUERY, "exposures": n, "step": step,
                       "towers": res["towers"], "wind": res["wind"], "time": res["time"],
                       "size": [res["w"], res["h"]]}, f, indent=1)
        print(f"{cam}: {n} frames, ready in {secs} s, wind {res['wind']} -> {out_dir}", flush=True)
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
