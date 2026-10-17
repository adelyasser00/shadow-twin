"""
The Rotterdam wind post, in the New York wind post's look (compose_post_wind2.py):
the same streaks, gradients, labels, key and credit, on Zalmhaven and the
usual winter wind, south-west (247.5 deg).

    python -m http.server 8765 --bind 127.0.0.1        (from viewer/)
    .venv-heat/Scripts/python tools/render_wind_frames.py --city rotterdam --dir 247.5 zhUpwindL 12 0.6
    .venv-heat/Scripts/python tools/compose_post_rotterdam.py [frames_dir] [--headline "..."]

The layer bar is cropped from a headless shot of the Rotterdam viewer
(tools/city_qa.py), so it reads "Colder since 2016". 1080 x 1350 PNG with a
400 px copy for the phone check, under data/cities/rotterdam/figures/post/.
New file names every run; nothing is overwritten.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compose_post_wind2 import AMBER, H, M, W, font, ramp_colour, shade, stack, tag, wrap  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "data", "cities", "rotterdam", "figures", "post")
ZALMHAVEN = (4.480722, 51.910355)

HEADLINE = "How the Netherlands' tallest tower reshapes the winter wind"
SUBTEXT = "Zalmhaven, Rotterdam  ·  December south-west wind  ·  2016 vs 2023"
SOURCE = ("Data: AHN3 (2016) and AHN5 (2023) LiDAR, 3DBAG outlines. Map: OpenStreetMap. "
          "Wind model, 8 m cells.")


def key(img, x, y):
    """As New York's key, with Rotterdam's wind and before year."""
    d = ImageDraw.Draw(img)
    f = font("Barlow-Medium", 24)
    ax = x + 10
    d.line([(ax, y + 34), (ax, y + 6)], fill=(255, 255, 255), width=3)
    d.polygon([(ax - 8, y + 12), (ax + 8, y + 12), (ax, y)], fill=(255, 255, 255))
    first = "South-west wind, blowing away from you"
    d.text((x + 32, y + 4), first, font=f, fill=(255, 255, 255))
    sx = x + 32 + d.textlength(first, font=f) + 34
    d.rectangle((sx, y + 10, sx + 18, y + 28), fill=AMBER)
    d.text((sx + 28, y + 4), "taller since 2016", font=f, fill=(255, 255, 255))
    y2 = y + 50
    bw = 150
    for i in range(bw):
        d.line([(x + i, y2 + 9), (x + i, y2 + 17)], fill=ramp_colour(i / (bw - 1)))
    d.text((x + bw + 14, y2), "Streaks: longer and yellower means faster", font=f, fill=(225, 229, 234))


def tower_xy(meta, lonlat):
    t = min(meta["towers"], key=lambda t: abs(t["lonlat"][0] - lonlat[0]) + abs(t["lonlat"][1] - lonlat[1]))
    return t["x"], t["y"]


def compose(frames_dir, headline, labels, ui_bar=None):
    with open(os.path.join(frames_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    img = stack(frames_dir)
    shade(img, 0, 520, 242, 0)
    shade(img, 930, H, 0, 248)
    d = ImageDraw.Draw(img)
    d.text((M, 58), "ROTTERDAM DIGITAL TWIN   /   WIND LAYER", font=font("BarlowCondensed-SemiBold", 27), fill=AMBER)
    fh_ = font("Barlow-Bold", 62)
    y = 98
    for ln in wrap(d, headline, fh_, W - 2 * M):
        d.text((M, y), ln, font=fh_, fill=(255, 255, 255))
        y += 68
    fs = font("Barlow-Medium", 30)
    for ln in wrap(d, SUBTEXT, fs, W - 2 * M):
        d.text((M, y + 10), ln, font=fs, fill=(214, 219, 225))
        y += 38

    lab = font("Barlow-SemiBold", 29)
    zx, zy = tower_xy(meta, ZALMHAVEN)
    for text, xy, to, amber in labels(zx, zy):
        tag(img, xy, text, lab, fg=AMBER if amber else (255, 255, 255), to=to)

    key(img, M, H - 268)
    if ui_bar:
        shot, box = ui_bar
        img.alpha_composite(Image.open(shot).convert("RGBA").crop(box), (M - 4, H - 160))
    d = ImageDraw.Draw(img)
    d.text((M, H - 100), "Simulation and graphic: Adel Yasser Aglan", font=font("Barlow-SemiBold", 24),
           fill=(240, 242, 245))
    d.text((M, H - 68), SOURCE, font=font("Barlow-Regular", 21), fill=(170, 176, 184))
    return img.convert("RGB")


def labels(zx, zy):
    """(text, box corner, leader end or None, amber), placed for the zhUpwindL camera."""
    return [("Zalmhaven, 215 m", (zx - 380, zy - 90), (zx, zy + 40), True),
            ("Nieuwe Maas", (790, 650), None, False)]


# The Rotterdam viewer's own layer bar (headless shot at 1080 x 1350, legend and stats hidden).
UI_BAR = (os.path.join(FIG, "raw_viewer_ui_20261017.png"), (339, 1290, 741, 1326))


def main():
    args = sys.argv[1:]
    headline = HEADLINE
    if "--headline" in args:
        k = args.index("--headline")
        headline = args[k + 1]
        del args[k:k + 2]
    frames = args[0] if args else sorted(glob.glob(os.path.join(FIG, "frames_zhUpwindL_2026*")))[-1]
    img = compose(frames, headline, labels, UI_BAR)
    path = os.path.join(FIG, f"post_zh_{datetime.now():%Y%m%d_%H%M%S}.png")
    img.save(path, format="PNG", optimize=True)
    img.resize((400, 500), Image.LANCZOS).save(path.replace(".png", "_phone400.png"))
    mb = os.path.getsize(path) / 1e6
    print(f"wrote {path}  {mb:.2f} MB")
    if mb >= 5:
        raise SystemExit("over 5 MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
