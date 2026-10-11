"""
The wind post, in the series look of POST_BRIEF.md (written once, reused).

    .venv-heat/Scripts/python tools/compose_post_wind.py [raw_b1_hero_*.png]

Style, decided for every post:
- IBM Plex Sans for the headline, IBM Plex Sans Condensed for map labels,
  IBM Plex Mono for the source line (data/fonts/, SIL Open Font Licence).
  Sentence case.
- No cards, no pills, no blur, no glow. Text sits on the picture: dark ink on
  the flat sky, and a thin halo wherever it crosses the city, so it reads.
- Editorial map labels: text with a thin leader line to the exact thing it
  names. Left-aligned, one margin, asymmetric.
- Colour: neutrals, the series amber (towers that grew since 2017), and one
  data ramp (here ColorBrewer Purples, colour-blind safe, clear of the amber).
- A number goes on the picture only if its gates passed, next to what it
  measures. Two files are written: without the number (usable now) and with
  it (usable only if the AIJ wind tunnel check passes; see POST_NOTES.md).

1080 x 1350 PNG, sRGB, under 5 MB, key content inside the centred
880 x 1150, nothing under 20 px. A 400 px wide copy is saved next to each,
for the phone check. New file names every run; nothing is overwritten.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "data", "wind", "figures", "post")
FONTS = os.path.join(HERE, "data", "fonts")
W, H = 1080, 1350
SAFE = (100, 100, 980, 1250)
INK = (21, 24, 28)                 # dark text on the light sky
PAPER = (242, 243, 244)            # halo behind dark text
WHITE = (250, 250, 250)
SHADOW = (20, 22, 26)              # halo behind light text
MUTED = (70, 76, 84)
AMBER = (245, 196, 81)             # the series amber, as the viewer draws grown towers
LEADER = (21, 24, 28)


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, f"{name}.ttf"), size)


F = {
    "head": lambda s: font("IBMPlexSans-SemiBold", s),
    "sub": lambda s: font("IBMPlexSans-Regular", s),
    "lab_b": lambda s: font("IBMPlexSansCondensed-SemiBold", s),
    "lab": lambda s: font("IBMPlexSansCondensed-Regular", s),
    "mono": lambda s: font("IBMPlexMono-Regular", s),
}


def wrap(d, text, f, width):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def text(d, xy, s, f, fill=INK, halo=PAPER, width=3, anchor="la"):
    d.text(xy, s, font=f, fill=fill, stroke_width=width, stroke_fill=halo, anchor=anchor)


def label(d, x, y, lines, target, side="r", dark=True, max_w=None):
    """Lines of text at (x, y), left-aligned, with a thin leader to target.

    side: which edge of the text block the leader leaves from ("r", "rb", "l", "b", "t").
    max_w: a line wider than this is set smaller (never under 23 px) so it
    cannot run onto the thing it names.
    """
    fill, halo = (INK, PAPER) if dark else (WHITE, SHADOW)
    fb, fr = F["lab_b"](29), F["lab"](26)
    yy = y
    widths = []
    for i, s in enumerate(lines):
        f = fb if i == 0 else fr
        if max_w:
            size = 29 if i == 0 else 26
            while size > 23 and d.textlength(s, font=f) > max_w:
                size -= 1
                f = F["lab_b"](size) if i == 0 else F["lab"](size)
        text(d, (x, yy), s, f, fill=fill, halo=halo)
        widths.append(d.textlength(s, font=f))
        yy += 34 if i == 0 else 31
    bw, bh = max(widths), yy - y
    if target is None:
        return (x, y, x + bw, y + bh)
    ax = {"r": x + bw + 10, "rb": x + bw + 10, "l": x - 10, "b": x + 12, "t": x + 12}[side]
    ay = {"r": y + 16, "rb": y + bh - 14, "l": y + 16, "b": y + bh + 6, "t": y - 6}[side]
    tx, ty = target
    # A white casing under a dark line, so the leader reads on any ground.
    d.line((ax, ay, tx, ty), fill=PAPER, width=5)
    d.line((ax, ay, tx, ty), fill=LEADER, width=2)
    d.ellipse((tx - 5, ty - 5, tx + 5, ty + 5), fill=LEADER, outline=PAPER, width=2)
    return (x, y, x + bw, y + bh)


def north_arrow(d, x, y, vec, length=44):
    """A plain arrow along the picture's north, with N at its tip."""
    (x0, y0), (x1, y1) = vec
    a = math.atan2(y1 - y0, x1 - x0)
    ca, sa = math.cos(a), math.sin(a)
    tail, tip = (x - length / 2 * ca, y - length / 2 * sa), (x + length / 2 * ca, y + length / 2 * sa)
    d.line((tail, tip), fill=PAPER, width=6)
    d.line((tail, tip), fill=INK, width=3)
    px, py = -sa, ca
    head = [(tip[0] + 12 * ca, tip[1] + 12 * sa), (tip[0] + 8 * px, tip[1] + 8 * py), (tip[0] - 8 * px, tip[1] - 8 * py)]
    d.polygon(head, fill=INK, outline=PAPER)
    text(d, (tip[0] + 30 * ca, tip[1] + 30 * sa), "N", F["lab_b"](24), anchor="mm")


def arrowheads(d, lines, every=0.62):
    """A small arrowhead on each line, part way along, pointing downstream."""
    for pts in lines:
        pts = [p for p in pts if p]
        if len(pts) < 6:
            continue
        k = int(len(pts) * every)
        (x0, y0), (x1, y1) = pts[k - 1], pts[k + 1]
        a = math.atan2(y1 - y0, x1 - x0)
        ca, sa = math.cos(a), math.sin(a)
        x, y = pts[k]
        px, py = -sa, ca
        tri = [(x + 9 * ca, y + 9 * sa), (x - 6 * ca + 6 * px, y - 6 * sa + 6 * py),
               (x - 6 * ca - 6 * px, y - 6 * sa - 6 * py)]
        d.polygon(tri, fill=(28, 30, 35), outline=WHITE)


def legend(d, x, y, info, with_numbers):
    """Purple ramp, left-aligned, no box."""
    text(d, (x, y), "Street wind vs 2017", F["lab_b"](25), fill=WHITE, halo=SHADOW)
    sw, sh, gap = 62, 18, 4
    yy = y + 36
    cols = info["colours"]
    for i, c in enumerate(cols if not with_numbers else []):
        rgb = tuple(int(c[j:j + 2], 16) for j in (1, 3, 5))
        d.rectangle((x + i * (sw + gap), yy, x + i * (sw + gap) + sw, yy + sh), fill=rgb, outline=SHADOW)
    f = F["lab"](24)
    if with_numbers:
        # One row per class: swatch, then its range.
        e = info["edges_pct"]
        rows = [f"+{e[0]:.0f} to {e[1]:.0f}%", f"+{e[1]:.0f} to {e[2]:.0f}%", f"+{e[2]:.0f} to {e[3]:.0f}%",
                f"over +{e[3]:.0f}%"]
        img_rows = yy
        for i, (c, tk) in enumerate(zip(cols, rows)):
            rgb = tuple(int(c[j:j + 2], 16) for j in (1, 3, 5))
            ry = img_rows + i * 28
            d.rectangle((x, ry, x + 34, ry + 18), fill=rgb, outline=SHADOW)
            text(d, (x + 44, ry - 5), tk, f, fill=WHITE, halo=SHADOW)
        nxt = img_rows + 4 * 28 + 4
        text(d, (x, nxt), "Uncoloured: no change", f, fill=WHITE, halo=SHADOW)
        text(d, (x, nxt + 28), "beyond the model's noise", f, fill=WHITE, halo=SHADOW)
        return
    else:
        text(d, (x, yy + sh + 6), "stronger", f, fill=WHITE, halo=SHADOW)
        text(d, (x + 4 * (sw + gap) - gap, yy + sh + 6), "much stronger", f, fill=WHITE, halo=SHADOW, anchor="ra")
        nxt = yy + sh + 36
    text(d, (x, nxt), "Uncoloured: no change", f, fill=WHITE, halo=SHADOW)
    text(d, (x, nxt + 28), "beyond the model's noise", f, fill=WHITE, halo=SHADOW)


def compose(raw, with_number, headline, number=None):
    with open(raw.replace(".png", ".json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(os.path.join(HERE, "viewer", "wind", "post_windier.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    with open(os.path.join(HERE, "data", "wind", "tower_heights_2017.json"), encoding="utf-8") as fh:
        heights = json.load(fh)
    img = Image.open(raw).convert("RGB")
    assert img.size == (W, H), img.size
    d = ImageDraw.Draw(img)
    arrowheads(d, meta["lines"])

    # Headline and two lines of reading instructions, dark on the flat sky.
    fh_ = F["head"](50)
    lines = wrap(d, headline, fh_, 880)
    y = SAFE[1] - 50
    for ln in lines:
        text(d, (SAFE[0], y), ln, fh_)
        y += 58
    sub = ("A cold December northwester in 3D, 2017 vs today. Lines: air that comes down the "
           "tower's windward face. Purple: street wind clearly stronger than in 2017.")
    fs = F["sub"](24)
    for ln in wrap(d, sub, fs, 880):
        text(d, (SAFE[0], y + 4), ln, fs, fill=MUTED)
        y += 30

    # Labels, placed for the "hero7" camera of tools/render_post_wind.py: from
    # there Steinway Tower stands right behind Central Park Tower and shows as
    # a sliver along its east edge.
    T = meta["towers"]
    h = heights
    cpt = T["Central Park Tower"]
    st_roof = T["Steinway Tower"]["roof"]
    w53_roof = T["53 W 53rd"]["roof"]
    effect = (f"street at its foot: +{number:.0f}% windier" if with_number and number is not None
              else "street at its foot: much windier")
    hc, hs, hw = h["Central Park Tower"], h["Steinway Tower"], h["53 W 53rd"]
    label(d, SAFE[0], cpt["roof"][1] - 22, ["Central Park Tower",
                                           f"{hc['roof_2017_m']} m in 2017, {hc['today_m']} m today", effect],
          (cpt["roof"][0] - 50, cpt["roof"][1] + 12), side="r", max_w=cpt["roof"][0] - 60 - SAFE[0])
    sliver_x = cpt["roof"][0] + 64
    label(d, 640, st_roof[1] - 70, ["Steinway Tower, behind it",
                                   f"{hs['roof_2017_m']} m in 2017, {hs['today_m']} m today",
                                   "street: no clear change"], (sliver_x, st_roof[1] + 26), side="l")
    label(d, 590, w53_roof[1] - 112, ["53 W 53rd", f"{hw['roof_2017_m']} m in 2017, {hw['today_m']} m today",
                                      "street: a few corners windier"], (w53_roof[0] + 8, w53_roof[1] + 10), side="rb")
    # The lines and the street, named where they are.
    pts = [p for ln in meta["lines"] for p in ln if p]
    top = min(pts, key=lambda p: p[1])
    label(d, SAFE[0], top[1] - 150, ["Air from high up,", "thrown down the face"], (top[0] - 8, top[1] + 24),
          side="r", dark=False)
    canyon = (cpt["roof"][0] + 90, cpt["base"][1] - 177)       # the purple patch in 57th Street's canyon
    label(d, 620, canyon[1] - 110, ["57th Street", "wind clearly stronger"], canyon, side="b", dark=False)
    north_arrow(d, SAFE[2] - 30, 840, meta["north"])
    legend(d, SAFE[0], cpt["base"][1] - 262, info, with_number)
    src = ("Time-mean wind, 8 m cells, lattice Boltzmann LES on a laptop GPU. Clear December day, "
           "Central Park typical year. 2017 LiDAR vs today. Map © OpenStreetMap contributors. "
           "Model and map: Adel Yasser Aglan. First-round model.")
    fm = F["mono"](19)
    sl = wrap(d, src, fm, 880)
    y0 = SAFE[3] - len(sl) * 26 + 4
    for i, ln in enumerate(sl):
        text(d, (SAFE[0], y0 + i * 26), ln, fm, fill=WHITE, halo=SHADOW, width=2)
    return img


def main():
    raws = sys.argv[1:] or sorted(glob.glob(os.path.join(FIG, "raw_b1_hero7_*[0-9].png")))[-1:]
    if not raws:
        raise SystemExit("no raw render; run tools/render_post_wind.py b1_hero first")
    import csv
    number = None
    p = os.path.join(HERE, "data", "wind", "wind_story.csv")
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["key"] == "central_park_tower_street_change_pct":
                number = float(r["value"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    headline = "Central Park Tower pulls the wind down to the street"
    out = []
    for raw in raws:
        for tag, with_n in (("nonumber", False), ("number", True)):
            img = compose(raw, with_n, headline, number)
            path = os.path.join(FIG, f"post_b1_{tag}_{stamp}.png")
            img.save(path, format="PNG", optimize=True)
            img.resize((400, 500), Image.LANCZOS).save(path.replace(".png", "_phone400.png"))
            mb = os.path.getsize(path) / 1e6
            print(f"wrote {path}  {mb:.2f} MB")
            if mb >= 5:
                raise SystemExit("over 5 MB")
            out.append(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
