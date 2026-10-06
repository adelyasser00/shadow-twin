"""
The single-image LinkedIn post for the wind layer, from tools/render_wind.py.

    .venv-heat/Scripts/python tools/compose_wind.py [render.png]

1080 x 1350 (4:5), sRGB PNG, under 5 MB, same rules as the heat post
(tools/compose_linkedin.py): every word inside the centred 880 x 1150 safe
zone, nothing smaller than 22 px, one number on the image, read from
data/wind/wind_numbers.csv, never typed in. Writes a new file every run, so
older images are never overwritten.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from datetime import datetime

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
from compose_linkedin import (GROWN, INK, MUTED, SAFE, W, H, font, pill, shade, wrap)  # noqa: E402

FIG = os.path.join(HERE, "data", "wind", "figures", "post")
NUMBERS = os.path.join(HERE, "data", "wind", "wind_numbers.csv")
WIND_BLUE = "#7fd3f0"

# Words for the image. The number comes from the CSV.
TITLE = "How the wind moves through Billionaires' Row"
SUB = ("A cold northwester crossing Central Park, simulated in 3D. "
       "Light moves at the wind's speed. Amber: towers that grew since 2017.")
NUMBER_KEY = None              # e.g. "dec_1230_wind_part_median_c"; None: no number box
NUMBER_FMT = "{:+.1f} °C"
NUMBER_LINES = ["", ""]
CALLOUTS = []                   # (text, (x, y) anchor in the render, side "l"|"r")
SOURCE = ["Time-mean wind, 8 m cells, lattice Boltzmann LES on a laptop GPU. 2017 LiDAR vs today.",
          "Wind from the Central Park typical year, clear 18 December. Map © OpenStreetMap."]
NAMES = [((-73.98102, 40.76644), "Central Park Tower"),
         ((-73.97756, 40.76496), "Steinway Tower"),
         ((-73.97815, 40.76183), "53 W 53rd")]


def numbers():
    rows = {}
    if os.path.exists(NUMBERS):
        with open(NUMBERS, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[r["key"]] = r
    return rows


def tower_name(lonlat):
    for (lo, la), nm in NAMES:
        if abs(lo - lonlat[0]) < 0.0005 and abs(la - lonlat[1]) < 0.0004:
            return nm
    return None


def wind_arrow(img, x, y, deg_screen, text):
    """A fat arrow pointing the way the wind blows on the picture, with a label."""
    d = ImageDraw.Draw(img)
    a = math.radians(deg_screen)
    ca, sa = math.cos(a), math.sin(a)
    L = 120
    x1, y1 = x + L * ca, y + L * sa
    d.line((x, y, x1, y1), fill=WIND_BLUE, width=10)
    px, py = -sa, ca
    d.polygon([(x1 + 26 * ca, y1 + 26 * sa), (x1 + 18 * px, y1 + 18 * py), (x1 - 18 * px, y1 - 18 * py)],
              fill=WIND_BLUE)
    return pill(img, (x, y - 44), text, font(24, True), fg=WIND_BLUE)


def compose(raw_path):
    meta_path = raw_path.replace(".png", ".json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    img = Image.open(raw_path).convert("RGBA")
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    size = 48
    while size > 40 and d.textlength(TITLE, font=font(size, True)) > 880:
        size -= 1
    f_t, f_s = font(size, True), font(25)
    lt = wrap(d, TITLE, f_t, 880)
    ls = wrap(d, SUB, f_s, 880)
    top = SAFE[1] - 10
    ys = top + len(lt) * 56 + 6
    card = (SAFE[0] - 22, top - 18, SAFE[2] + 22, ys + len(ls) * 32 + 14)
    img = shade(img, card, alpha=222, radius=18)
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lt):
        d.text((SAFE[0], top + i * 56), ln, font=f_t, fill=INK)
    for i, ln in enumerate(ls):
        d.text((SAFE[0], ys + i * 32), ln, font=f_s, fill=MUTED)
    rows = numbers()
    if NUMBER_KEY and NUMBER_KEY in rows:
        v = float(rows[NUMBER_KEY]["value"])
        nb = (SAFE[2] - 262, card[3] + 40, SAFE[2] + 14, card[3] + 180)
        img = shade(img, nb, alpha=222, radius=14)
        d = ImageDraw.Draw(img)
        d.text((SAFE[2], nb[1] + 2), NUMBER_FMT.format(v), font=font(62, True), fill=WIND_BLUE, anchor="ra")
        for i, s in enumerate(NUMBER_LINES):
            d.text((SAFE[2], nb[1] + 78 + 28 * i), s, font=font(23), fill=INK, anchor="ra")
    # Tower names on their roofs.
    for t in meta.get("towers", []):
        nm = tower_name(t["lonlat"])
        if not nm or t["x"] is None:
            continue
        x, y = int(t["x"] * W / meta["size"][0]), int(t["y"] * H / meta["size"][1])
        if not (SAFE[0] < x < SAFE[2] and SAFE[1] < y < SAFE[3]):
            continue
        img = pill(img, (x, y - 30), nm, font(23, True), fg=GROWN, anchor="mm")
        d = ImageDraw.Draw(img)
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=GROWN)
    for text, (x, y), side in CALLOUTS:
        img = pill(img, (x, y), text, font(23), anchor="lm" if side == "r" else "rm")
    f = font(22)
    y0 = SAFE[3] - len(SOURCE) * 30
    img = shade(img, (SAFE[0] - 12, y0 - 8, SAFE[2] + 12, SAFE[3] + 4), alpha=205, radius=10)
    d = ImageDraw.Draw(img)
    for i, s in enumerate(SOURCE):
        d.text((SAFE[0], y0 + i * 30), s, font=f, fill=MUTED)
    return img


def main():
    raws = sys.argv[1:] or sorted(glob.glob(os.path.join(FIG, "wind_dec_upwindP_*[0-9].png")))[-1:]
    if not raws:
        raise SystemExit("no render; run tools/render_wind.py first")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    for raw in raws:
        img = compose(raw)
        out = os.path.join(FIG, f"linkedin_wind_{stamp}.png")
        img.convert("RGB").save(out, format="PNG", optimize=True)
        mb = os.path.getsize(out) / 1e6
        im = Image.open(out)
        im.resize((400, int(400 * im.height / im.width)), Image.LANCZOS).save(out.replace(".png", "_phone400.png"))
        print(f"wrote {out}  {mb:.2f} MB")
        if mb >= 5:
            raise SystemExit("over the 5 MB limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
