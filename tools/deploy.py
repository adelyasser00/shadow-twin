"""
Build a folder ready to drag onto Cloudflare Pages.

    python tools/deploy.py

Makes deploy/ containing:
    index.html   the "today" build, so the bare URL works
    2017.html    the 2017 survey build, for the comparison
    preview.png  link-preview image, if you put one in viewer/

Then go to the Cloudflare dashboard, Workers & Pages, Create, Pages, Upload
assets, and drag the deploy folder in.

Limits worth knowing: 25 MiB per file and 1,000 files for a drag-and-drop
deployment. This folder is three files and about a megabyte, so neither is
close. Static requests are free and unlimited.

Do NOT include viewer/vendor/. It is 23 MB of Cesium in thousands of files, and
the published page falls back to the Cesium CDN anyway, which serves it faster
than a static host will.
"""

from __future__ import annotations

import os
import shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(HERE, "viewer")
OUT = os.path.join(HERE, "deploy")

WANT = [("shadow-twin-today.html", "index.html", True),
        ("shadow-twin-2017.html", "2017.html", False),
        ("preview.png", "preview.png", False)]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    missing = []
    for src, dst, required in WANT:
        p = os.path.join(VIEWER, src)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(OUT, dst))
            mb = os.path.getsize(p) / 1e6
            flag = "  OVER 25 MiB, will be rejected" if mb > 25 else ""
            print(f"  {dst:<14} {mb:6.2f} MB   from {src}{flag}")
        elif required:
            missing.append(src)
        else:
            print(f"  {dst:<14} skipped, no {src} in viewer/")

    if missing:
        raise SystemExit(
            "missing " + ", ".join(missing) + "\n"
            "Run the scenarios first and copy each build:\n"
            "  python -m pipeline.run_nyc ... --burn-footprints\n"
            "  python -m pipeline.build_viewer\n"
            "  copy viewer\\shadow-twin.html viewer\\shadow-twin-today.html")

    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"\n{len(os.listdir(OUT))} files, {total/1e6:.2f} MB total")
    print(f"drag this folder onto Cloudflare Pages:\n  {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
