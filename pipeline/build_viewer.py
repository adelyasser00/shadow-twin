"""
Put the solver payload into the viewer template.

    python -m pipeline.build_viewer

Writes viewer/shadow-twin.html, plus a copy named for the scenario in the data:
shadow-twin-2017.html for the survey as flown, shadow-twin-today.html for a
--burn-footprints run. tools/deploy.py and the comparison mode look for those
two names, so there is nothing to rename by hand.

Small runs, like the original 700 m square, come out as one self-contained
file with every image inlined, as before.

Big runs, like the whole of Central Park, would be a 20 MB page: 76 shade
images at a couple of thousand pixels a side. So the images go into their own
folder next to the page, frames-2017/ or frames-today/, and the page asks for
each one when it is shown. The first picture appears after one image instead of
all of them. Pass --inline to force a single file anyway, --split to force the
folder.

Either way the page must be served over http (python serve.py, or the
deployed site), because Cesium fetches its own files at runtime.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(HERE, "viewer")
OUT_NAME = "shadow-twin.html"
SPLIT_ABOVE_MB = 6.0
DATA_FILE_ABOVE_MB = 1.0         # model data bigger than this gets its own file


def scenario_tag(d):
    sc = (d.get("model") or {}).get("scenario") or {}
    if sc.get("type") == "today":
        return "today"
    if sc.get("type") == "height cap":
        return f"cap{int(round(sc.get('cap_m', 0)))}"
    return "2017"


def _images(d):
    """Every (holder, name) whose 'png' is an inline image."""
    for day in d.get("days", []):
        for h in day["hours"]:
            yield h, f"{day['id']}_{h['hour'].replace(':', '')}.png"
        yield day["sunhours"], f"sunhours_{day['id']}.png"
    if d.get("svf"):
        yield d["svf"], "svf.png"


def split_images(d, tag):
    """Write inline images to viewer/frames-<tag>/ and point the data at them."""
    folder = f"frames-{tag}"
    path = os.path.join(VIEWER, folder)
    if os.path.isdir(path):
        shutil.rmtree(path)      # stale frames from an older run must not survive
    os.makedirs(path)
    # Same file names every run, so tell caches when the content changed.
    ver = hashlib.sha1(d.get("generated_utc", "").encode()).hexdigest()[:8]
    n = size = 0
    for holder, name in _images(d):
        uri = holder["png"]
        if not uri.startswith("data:image/png;base64,"):
            continue
        raw = base64.b64decode(uri.split(",", 1)[1])
        with open(os.path.join(path, name), "wb") as f:
            f.write(raw)
        holder["png"] = f"{folder}/{name}?v={ver}"
        n += 1
        size += len(raw)
    return folder, n, size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(VIEWER, "data.json"))
    ap.add_argument("--name", default=None, help="override the scenario tag, e.g. today")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--inline", action="store_true", help="one self-contained file")
    g.add_argument("--split", action="store_true", help="images in their own folder")
    args = ap.parse_args()

    with open(os.path.join(VIEWER, "template.html"), encoding="utf-8") as f:
        html = f.read()
    with open(args.data, encoding="utf-8") as f:
        d = json.load(f)
    if "/*__DATA__*/" not in html:
        raise SystemExit("template is missing the /*__DATA__*/ placeholder")

    tag = args.name or scenario_tag(d)
    inline_mb = sum(len(h["png"]) for h, _ in _images(d)) / 1024 / 1024
    split = args.split or (not args.inline and inline_mb > SPLIT_ABOVE_MB)
    folder = None
    if split:
        folder, n_img, img_bytes = split_images(d, tag)

    raw = json.dumps(d, separators=(",", ":"))
    data_file = None
    if split or len(raw) > DATA_FILE_ABOVE_MB * 1024 * 1024:
        # A big build keeps its data next to the page too, so the page stays
        # small: link previews (LinkedIn reads at most 3 MiB) and a first paint
        # that is not waiting on the whole model.
        data_file = f"data-{tag}.json"
        with open(os.path.join(VIEWER, data_file), "w", encoding="utf-8", newline="\n") as f:
            f.write(raw)
        ver = hashlib.sha1(raw.encode()).hexdigest()[:8]
        raw = json.dumps(f"{data_file}?v={ver}")
    out = html.replace("/*__DATA__*/", raw.replace("</script>", "<\\/script>"))
    written = []
    for name in (OUT_NAME, f"shadow-twin-{tag}.html"):
        path = os.path.join(VIEWER, name)
        # Unix line endings always, even on Windows. The comparison mode reads the
        # other build's data line by line, and mixed endings are how it broke.
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
        written.append(path)

    mb = os.path.getsize(written[0]) / 1024 / 1024
    s = d["site"]
    print(f"wrote {written[0]}")
    print(f"  and {written[1]}")
    print(f"  page        {mb:.2f} MB")
    if data_file:
        mb_d = os.path.getsize(os.path.join(VIEWER, data_file)) / 1024 / 1024
        print(f"  data        {mb_d:.2f} MB in viewer/{data_file}, fetched by the page")
    if split:
        print(f"  images      {n_img} files, {img_bytes/1024/1024:.1f} MB in viewer/{folder}/, "
              "loaded as they are shown")
    else:
        print(f"  images      inlined, {inline_mb:.1f} MB")
    print(f"  scenario    {tag}")
    print(f"  mode        {d['mode']}")
    print(f"  site        {s['name']}")
    fr = s.get("frame")
    if fr:
        print(f"  frame       {fr['length_m']:,} m by {fr['width_m']:,} m, "
              f"bearing {fr['bearing_deg']} deg, buffer {fr['buffer_m']:,} m")
    print(f"  grid        {s['grid'][0]} x {s['grid'][1]} at {s['cellsize_m']} m")
    for day in d.get("days", []):
        st = day["stats"]
        line = (f"  day         {day['label']:<24} {len(day['hours'])} hours, street median "
                f"{st['median_sun_hours']:.1f} h")
        if "resource" in st:
            line += f", {st['resource']['name']} never in sun {st['resource']['frac_no_sun']*100:.1f}%"
        print(line)
    print(f"  sky view    {'present' if d.get('svf') else 'skipped'}")
    b = d.get("buildings", [])
    ctx = sum(1 for x in b if x.get("context"))
    print(f"  buildings   {len(b) - ctx} in the frame, {ctx} context buildings around it")

    vendor = os.path.join(VIEWER, "vendor", "Cesium", "Cesium.js")
    print(f"  offline     {'yes, vendored Cesium found' if os.path.exists(vendor) else 'no, will use the CDN (see tools/vendor_cesium.md)'}")
    if mb > 9:
        print("\n  warning: the page itself is over 9 MB. Try --split, or cut the frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
