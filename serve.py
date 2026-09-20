"""
Serve the viewer over http.

    python serve.py

Opens http://localhost:8765/shadow-twin.html in your browser.

Why this is needed at all: CesiumJS fetches its own asset files at runtime,
including approximateTerrainHeights.json and the skybox textures. Browsers
block those requests on a file:// URL under the same-origin policy, and when
approximateTerrainHeights.json fails the renderer stops with an unhelpful
error. Any http origin fixes it, including this one.

Double-clicking shadow-twin.html works only when it falls back to the Cesium
CDN, because https requests from a file:// page are allowed. The moment you
vendor Cesium locally you need a server. That is not a bug in the vendoring,
it is how the same-origin policy works.
"""

from __future__ import annotations

import http.server
import os
import socketserver
import sys
import threading
import webbrowser

PORT = int(os.environ.get("PORT", "8765"))
PAGE = "shadow-twin.html"


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Only shout about failures. A successful asset fetch is noise.
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write(f"  {args[0]} -> {args[1]}\n")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "viewer")
    if not os.path.exists(os.path.join(root, PAGE)):
        raise SystemExit(
            f"{PAGE} is not in {root}.\n"
            "Run: python -m pipeline.build_viewer"
        )
    os.chdir(root)

    url = f"http://localhost:{PORT}/{PAGE}"
    socketserver.TCPServer.allow_reuse_address = True
    try:
        server = socketserver.TCPServer(("127.0.0.1", PORT), Handler)
    except OSError as e:
        raise SystemExit(
            f"Could not open port {PORT}: {e}\n"
            f"Something else is using it. Try:  set PORT=8766 && python serve.py"
        )

    print(f"serving {root}")
    print(f"open {url}")
    print("ctrl-c to stop\n")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
