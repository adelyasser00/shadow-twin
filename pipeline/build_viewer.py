"""
Inline the solver payload into the viewer template.

    python -m pipeline.build_viewer

Produces viewer/shadow-twin.html, one self-contained file with every raster
embedded as a PNG data URI. No server, no CORS, no build step. Double click it,
or drop it on Cloudflare Pages as is.

If viewer/vendor/Cesium exists next to it, the page uses that and never touches
the network. Otherwise it falls back to the CDN. See tools/vendor_cesium.md.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(HERE, "viewer")
OUT_NAME = "shadow-twin.html"


def main():
    with open(os.path.join(VIEWER, "template.html")) as f:
        html = f.read()
    with open(os.path.join(VIEWER, "data.json")) as f:
        raw = f.read()

    if "/*__DATA__*/" not in html:
        raise SystemExit("template is missing the /*__DATA__*/ placeholder")

    out = html.replace("/*__DATA__*/", raw.replace("</script>", "<\\/script>"))
    path = os.path.join(VIEWER, OUT_NAME)
    with open(path, "w") as f:
        f.write(out)

    d = json.loads(raw)
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"wrote {path}")
    print(f"  {mb:.2f} MB total")
    print(f"  mode        {d['mode']}")
    print(f"  site        {d['site']['name']}")
    print(f"  grid        {d['site']['grid'][0]} x {d['site']['grid'][1]} "
          f"at {d['site']['cellsize_m']} m")
    for day in d.get("days", []):
        print(f"  day         {day['label']:<14} {len(day['hours'])} hours, "
              f"median {day['stats']['median_sun_hours']:.1f} h of sun")
    print(f"  sky view    {'present' if d.get('svf') else 'skipped'}")
    print(f"  buildings   {len(d.get('buildings', []))}")

    vendor = os.path.join(VIEWER, "vendor", "Cesium", "Cesium.js")
    print(f"  offline     {'yes, vendored Cesium found' if os.path.exists(vendor) else 'no, will use the CDN (see tools/vendor_cesium.md)'}")

    if mb > 9:
        print("\n  warning: over 9 MB. Cut --span, raise --res, or drop a day.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
