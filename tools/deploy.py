"""
Build a folder ready to drag onto Cloudflare Pages.

    python tools/deploy.py

Makes deploy/ containing:
    index.html          the "today" build, so the bare URL works
    2017.html           the 2017 survey build, for the comparison
    frames-today/       images for the today build, if it was split
    frames-2017/        images for the 2017 build, if it was split
    heat/               heat layers (heat.json and images), if they were exported
    preview.png         link-preview image, if you put one in viewer/

The shadow-twin-*.html copies are included too, so the comparison finds its
other half whichever name it asks for.

Then go to the Cloudflare dashboard, Workers & Pages, your project, Create
deployment, and drag the deploy folder in. Refreshing does not update the
site; only a new deployment does.

Limits worth knowing: 25 MiB per file and 1,000 files for a drag-and-drop
deployment. A whole-park build is a couple of hundred files, well inside both.

Do NOT include viewer/vendor/. It is 23 MB of Cesium in thousands of files, and
the published page falls back to the Cesium CDN anyway.
"""

from __future__ import annotations

import os
import shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(HERE, "viewer")
OUT = os.path.join(HERE, "deploy")

WANT = [("shadow-twin-today.html", "index.html", True),
        ("shadow-twin-2017.html", "2017.html", False),
        ("shadow-twin-today.html", "shadow-twin-today.html", False),
        ("shadow-twin-2017.html", "shadow-twin-2017.html", False),
        ("preview.png", "preview.png", False)]
FOLDERS = ["frames-today", "frames-2017", "heat"]


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
            print(f"  {dst:<24} {mb:6.2f} MB   from {src}{flag}")
        elif required:
            missing.append(src)
        else:
            print(f"  {dst:<24} skipped, no {src} in viewer/")

    for folder in FOLDERS:
        p = os.path.join(VIEWER, folder)
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(OUT, folder))
            files = os.listdir(p)
            mb = sum(os.path.getsize(os.path.join(p, f)) for f in files) / 1e6
            print(f"  {folder + '/':<24} {mb:6.2f} MB   {len(files)} images")

    if missing:
        raise SystemExit(
            "missing " + ", ".join(missing) + "\n"
            "Run both scenarios first; build_viewer names each build for you:\n"
            "  python -m pipeline.run_nyc ... --burn-footprints\n"
            "  python -m pipeline.build_viewer")

    n_files = sum(len(fs) for _, _, fs in os.walk(OUT))
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs)
    print(f"\n{n_files} files, {total/1e6:.2f} MB total"
          + ("   OVER the 1,000 file limit" if n_files > 1000 else ""))
    print(f"drag this folder onto Cloudflare Pages:\n  {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
