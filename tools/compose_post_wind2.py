"""
The wind post, second look: the wide view over Central Park with the wind
moving through it, readable at a glance (Adel, 12 Oct: the editorial look of
compose_post_wind.py read as AI-made and hid the wind).

    .venv-heat/Scripts/python tools/render_wind_frames.py upwindP 12 0.6
    .venv-heat/Scripts/python tools/compose_post_wind2.py [frames_dir]
    .venv-heat/Scripts/python tools/render_wind_frames.py upwindP 12 0.6 2017
    .venv-heat/Scripts/python tools/compose_post_wind2.py --comment     (2017 vs today, the first comment)

The exposures are stacked with the light brightening toward its head, so each
streak shows which way the air goes; every streak covers the same time, so
its length is the speed, and its colour is the speed too (the viewer's ramp).
Text sits on dark gradients, never as halos over buildings. Fonts: Barlow
(data/fonts/barlow, SIL Open Font Licence).

1080 x 1350 PNG with a 400 px wide copy for the phone check. New file names
every run; nothing is overwritten.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "data", "wind", "figures", "post")
FONT = os.path.join(HERE, "data", "fonts", "barlow")
UI_SHOT = os.path.join(FIG, "raw_b3_viewer_clean_20261012_001944.png")   # the viewer's own layer bar
UI_BAR = (306, 1288, 775, 1328)
W, H = 1080, 1350
M = 64                                  # one left margin for everything
AMBER = (245, 196, 81)                  # the series amber, towers that grew since 2017
INK = (14, 16, 19)
# The viewer's line colours by speed (WIND_FS ramp in viewer/template.html), calm to strong.
RAMP = [(0.0, (115, 168, 255)), (0.4, (199, 235, 255)), (0.75, (255, 255, 255)), (1.0, (255, 214, 107))]


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT, name + ".ttf"), size)


def stack(frames_dir, gamma=1.8, w0=0.3, gain=1.4):
    """Long exposure whose light brightens toward the latest frame: streaks with a head."""
    fr = [np.asarray(Image.open(p).convert("RGB"), np.float32)
          for p in sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))]
    f = np.stack(fr)
    base = f.min(axis=0)
    w = w0 + (1 - w0) * (np.arange(len(fr)) / max(1, len(fr) - 1)) ** gamma
    img = base + gain * ((f - base) * w[:, None, None, None]).max(axis=0)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def shade(img, y0, y1, a0, a1):
    """Darken rows y0..y1 from alpha a0 to a1: a photographic vignette, not a box."""
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for y in range(y0, y1):
        t = (y - y0) / max(1, y1 - y0 - 1)
        d.line([(0, y), (W, y)], fill=(*INK, int(a0 + (a1 - a0) * t)))
    img.alpha_composite(ov)


def tag(img, xy, text, f, fg=(255, 255, 255), to=None):
    """A solid dark label, readable on anything, with an optional thin leader to what it names."""
    d = ImageDraw.Draw(img)
    x, y = xy
    asc, desc = f.getmetrics()
    box = (x, y, x + d.textlength(text, font=f) + 24, y + asc + desc + 10)
    if to:
        start = (box[2], (box[1] + box[3]) / 2) if to[0] > box[2] else ((box[0] + box[2]) / 2, box[3])
        d.line([start, to], fill=(255, 255, 255, 230), width=2)
        d.ellipse((to[0] - 4, to[1] - 4, to[0] + 4, to[1] + 4), fill=(255, 255, 255, 255))
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).rounded_rectangle(box, radius=3, fill=(*INK, 225))
    img.alpha_composite(ov)
    ImageDraw.Draw(img).text((x + 12, y + 5), text, font=f, fill=fg)


def ramp_colour(t):
    for (t0, c0), (t1, c1) in zip(RAMP, RAMP[1:]):
        if t <= t1:
            s = (t - t0) / (t1 - t0)
            return tuple(int(a + (b - a) * s) for a, b in zip(c0, c1))
    return RAMP[-1][1]


def key(img, x, y):
    """How to read the wind: which way it goes, and what length and colour mean."""
    d = ImageDraw.Draw(img)
    f = font("Barlow-Medium", 24)
    # Direction: in this view the wind blows away from the camera, into the picture.
    ax = x + 10
    d.line([(ax, y + 34), (ax, y + 6)], fill=(255, 255, 255), width=3)
    d.polygon([(ax - 8, y + 12), (ax + 8, y + 12), (ax, y)], fill=(255, 255, 255))
    d.text((x + 32, y + 4), "Northwest wind, blowing away from you", font=f, fill=(255, 255, 255))
    # Speed: a streak drawn with the ramp, short and calm to long and strong.
    y2 = y + 50
    bw = 150
    for i in range(bw):
        d.line([(x + i, y2 + 9), (x + i, y2 + 17)], fill=ramp_colour(i / (bw - 1)))
    d.text((x + bw + 14, y2), "Streaks: longer and yellower means faster", font=f, fill=(225, 229, 234))


HEADLINE = "The new supertalls pull winter wind down to the street"


def compose(frames_dir, headline=HEADLINE):
    with open(os.path.join(frames_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    img = stack(frames_dir)
    shade(img, 0, 520, 242, 0)
    shade(img, 930, H, 0, 248)
    d = ImageDraw.Draw(img)

    # What it is and who made it, small; the observation, large; how, one line.
    d.text((M, 58), "MY CENTRAL PARK DIGITAL TWIN   /   WIND LAYER", font=font("BarlowCondensed-SemiBold", 27),
           fill=AMBER)
    fh_ = font("Barlow-Bold", 62)
    y = 98
    for ln in wrap(d, headline, fh_, W - 2 * M):
        d.text((M, y), ln, font=fh_, fill=(255, 255, 255))
        y += 68
    fs = font("Barlow-Medium", 30)
    for ln in wrap(d, "I simulated the same December wind over Manhattan, with the buildings of 2017 and of today.",
                   fs, W - 2 * M - 40):
        d.text((M, y + 10), ln, font=fs, fill=(214, 219, 225))
        y += 38

    # Two labels only.
    cpt = next(t for t in meta["towers"] if abs(t["height_m"] - 472.4) < 1.0)
    lab = font("Barlow-SemiBold", 29)
    tag(img, (640, 560), "Central Park Tower, 472 m", lab, fg=AMBER, to=(cpt["x"] - 30, cpt["y"] + 160))
    tag(img, (M + 120, 905), "Central Park", lab)

    # How to read it, then proof it is a working tool, then the credit.
    key(img, M, H - 268)
    ui = Image.open(UI_SHOT).convert("RGBA").crop(UI_BAR)
    img.alpha_composite(ui, (M - 4, H - 160))
    d = ImageDraw.Draw(img)
    d.text((M, H - 100), "Simulation and graphic: Adel Yasser Aglan", font=font("Barlow-SemiBold", 24),
           fill=(240, 242, 245))
    d.text((M, H - 68), "Data: NYC 2017 LiDAR and building footprints. Map: OpenStreetMap. "
                        "First-round wind model, 8 m cells.", font=font("Barlow-Regular", 21), fill=(170, 176, 184))
    return img.convert("RGB")


CPT_LONLAT = (-73.98102, 40.766438)


def cpt_xy(meta):
    t = min(meta["towers"], key=lambda t: abs(t["lonlat"][0] - CPT_LONLAT[0]) + abs(t["lonlat"][1] - CPT_LONLAT[1]))
    return t["x"], t["y"]


def compose_comment(frames_today, frames_2017):
    """2017 and today side by side, same camera, same wind: only the buildings changed."""
    img = Image.new("RGBA", (W, H), (*INK, 255))
    d = ImageDraw.Draw(img)
    d.text((M, 50), "MY CENTRAL PARK DIGITAL TWIN   /   2017 VS TODAY", font=font("BarlowCondensed-SemiBold", 27),
           fill=AMBER)
    fh_ = font("Barlow-Bold", 52)
    y = 86
    for ln in wrap(d, "Same wind. Only the buildings changed.", fh_, W - 2 * M):
        d.text((M, y), ln, font=fh_, fill=(255, 255, 255))
        y += 58
    fs = font("Barlow-Medium", 28)
    y += 6
    for ln in wrap(d, "Same December day. In 2017 the wind crossed this block high above the street. "
                      "Today it runs into Central Park Tower and wraps around and down the new towers.", fs, W - 2 * M):
        d.text((M, y), ln, font=fs, fill=(214, 219, 225))
        y += 36
    box = (560, 270, 1080, 1150)                  # around Billionaires' Row, from the tops to the street
    pw, gap, top = (W - 12) // 2, 12, y + 16
    scale = pw / (box[2] - box[0])
    for i, (fr, name) in enumerate(((frames_2017, "2017"), (frames_today, "Today"))):
        with open(os.path.join(fr, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        panel = stack(fr).crop(box)
        panel = panel.resize((pw, int(panel.height * scale)), Image.LANCZOS)
        x0 = i * (pw + gap)
        img.alpha_composite(panel, (x0, top))
        tag(img, (x0 + 20, top + 18), name, font("Barlow-Bold", 40))
        cx, cy = cpt_xy(meta)
        to = (x0 + (cx - box[0]) * scale, top + (cy - box[1]) * scale)
        tag(img, (x0 + 20, top + 360 if i == 0 else top + 470), "Central Park Tower", font("Barlow-SemiBold", 27),
            fg=AMBER, to=to)
    d = ImageDraw.Draw(img)
    ky = top + int((box[3] - box[1]) * scale) + 22
    for i in range(150):
        d.line([(M + i, ky + 10), (M + i, ky + 18)], fill=ramp_colour(i / 149))
    d.text((M + 164, ky), "Streaks: air moving with the wind, away from you. Longer and yellower means faster.",
           font=font("Barlow-Medium", 22), fill=(225, 229, 234))
    d.text((M, H - 68), "Simulation and graphic: Adel Yasser Aglan. Same weather both years; first-round wind "
                        "model, 8 m cells.", font=font("Barlow-Regular", 21), fill=(170, 176, 184))
    return img.convert("RGB")


def main():
    if "--comment" in sys.argv:
        today = sorted(glob.glob(os.path.join(FIG, "frames_upwindP_2026*")))[-1]
        y2017 = sorted(glob.glob(os.path.join(FIG, "frames_upwindP_2017_*")))[-1]
        img = compose_comment(today, y2017)
        path = os.path.join(FIG, f"post_v2_comment_{datetime.now():%Y%m%d_%H%M%S}.png")
        img.save(path, format="PNG", optimize=True)
        img.resize((400, 500), Image.LANCZOS).save(path.replace(".png", "_phone400.png"))
        print(f"wrote {path}  {os.path.getsize(path) / 1e6:.2f} MB")
        return 0
    args = sys.argv[1:]
    headline = HEADLINE
    if "--headline" in args:
        k = args.index("--headline")
        headline = args[k + 1]
        del args[k:k + 2]
    frames = args[0] if args else sorted(glob.glob(os.path.join(FIG, "frames_upwindP_2026*")))[-1]
    img = compose(frames, headline)
    path = os.path.join(FIG, f"post_v2_{datetime.now():%Y%m%d_%H%M%S}.png")
    img.save(path, format="PNG", optimize=True)
    img.resize((400, 500), Image.LANCZOS).save(path.replace(".png", "_phone400.png"))
    mb = os.path.getsize(path) / 1e6
    print(f"wrote {path}  {mb:.2f} MB")
    if mb >= 5:
        raise SystemExit("over 5 MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
