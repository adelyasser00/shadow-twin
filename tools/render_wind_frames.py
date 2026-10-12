"""
The exposures behind a wind still, kept one by one, so the still can be
stacked with the light brightening toward its head (tools/compose_post_wind2.py):
a streak then shows which way the air goes, and its length how fast.

    python serve.py                       (or any http server on viewer/, port 8765)
    .venv-heat/Scripts/python tools/render_wind_frames.py [camera] [exposures] [step]

Same headless Chrome and cameras as tools/render_wind.py, never your own
browser. Writes data/wind/figures/post/frames_<camera>_<stamp>/ with
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


def main():
    cam = sys.argv[1] if len(sys.argv) > 1 else "upwindP"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    step = float(sys.argv[3]) if len(sys.argv) > 3 else 1.25
    out_dir = os.path.join(OUT, f"frames_{cam}_{datetime.now():%Y%m%d_%H%M%S}")
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
        tab.call("Page.navigate", url=f"{BASE}{PAGE}?{QUERY}&r={int(time.time())}")
        time.sleep(3)
        secs = tab.js(READY, timeout_s=600)
        res = tab.js(SHOOT % {"cam": json.dumps(CAMS[cam]), "n": n, "step": step}, timeout_s=900)
        for k, s in enumerate(res["shots"]):
            with open(os.path.join(out_dir, f"frame_{k:02d}.png"), "wb") as f:
                f.write(base64.b64decode(s.split(",", 1)[1]))
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"camera": CAMS[cam], "page": PAGE, "query": QUERY, "exposures": n, "step": step,
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
