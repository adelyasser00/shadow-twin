"""
The single-image LinkedIn post, and the 2017 vs today image for the first
comment, from the renders of tools/render_linkedin.py.

    .venv-heat/Scripts/python tools/compose_linkedin.py

1080 x 1350 (4:5), sRGB PNG, under 5 MB. Every word and label sits inside the
centred 880 x 1150 safe zone; nothing is smaller than 22 px. One number on the
image, read from data/heat/heat_numbers.csv, never typed in.
"""

from __future__ import annotations

import csv
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POST = os.path.join(HERE, "data", "heat", "post")
FONTS = r"C:\Windows\Fonts"
W, H = 1080, 1350
SAFE = (100, 100, 980, 1250)            # x0, y0, x1, y1: the centred 880 x 1150
INK, MUTED, PANEL = (240, 243, 247), (176, 184, 194), (13, 16, 20)
TOWER = "#b8955a"                       # grown towers, muted on the main image
GROWN = "#f5c451"                       # grown towers as the viewer draws them

HEADLINES = {
    "A": "New towers made Central Park feel colder",
    "B": "Where the new towers chill Central Park",
    "C": "Billionaires' Row casts a cold shadow",
}
SOURCE = ["Clear December day, typical year. SOLWEIG (UMEP). 2017 LiDAR vs today.",
          "Each spot at its coldest hour, 09:30 to 14:30. Modelled up to the dashed line.",
          "Map © OpenStreetMap contributors."]


def font(size, bold=False):
    return ImageFont.truetype(os.path.join(FONTS, "segoeuib.ttf" if bold else "segoeui.ttf"), size)


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


def shade(img, box, alpha=200, radius=12):
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(over).rounded_rectangle(box, radius=radius, fill=PANEL + (alpha,))
    return Image.alpha_composite(img, over)


def top_fade(img, height, alpha=235):
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for y in range(height):
        a = int(alpha * min(1.0, 1.25 * (1 - y / height)))
        d.line((0, y, img.width, y), fill=PANEL + (a,))
    return Image.alpha_composite(img, over)


def pill(img, xy, text, f, fg=INK, anchor="lm"):
    d = ImageDraw.Draw(img)
    x, y = xy
    tw = d.textlength(text, font=f)
    h = int(f.size * 1.45)
    x0 = x if anchor == "lm" else (x - tw - 24 if anchor == "rm" else x - tw / 2 - 12)
    x0 = min(max(SAFE[0], x0), SAFE[2] - tw - 24)
    box = (int(x0), int(y - h / 2), int(x0 + tw + 24), int(y + h / 2))
    img = shade(img, box, alpha=215, radius=10)
    ImageDraw.Draw(img).text((box[0] + 12, y), text, font=f, fill=fg, anchor="lm")
    return img


def numbers():
    rows = {}
    with open(os.path.join(HERE, "data", "heat", "heat_numbers.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["key"]] = r
    st = rows["dec_ha_any_hour_dutci_le_-5"]
    if st["status"] != "publishable":
        raise SystemExit("the number is not marked publishable; no numbers on the image")
    return {"ha5": float(st["value"]), "area": float(rows["dec_report_area"]["value"])}


def legend(img, x, y, heat):
    """A short bar, 2 to 14 C colder, four ticks."""
    bins = [e for e in heat["legend"]["change"] if e["max"] <= -1.9][::-1]       # 2-4 ... 12-14
    d = ImageDraw.Draw(img)
    f_t, f = font(24, True), font(22)
    bw, bh = 240, 22
    img = shade(img, (x - 16, y - 14, x + bw + 16, y + 132), alpha=210)
    d = ImageDraw.Draw(img)
    d.text((x, y), "Colder than 2017, °C", font=f_t, fill=INK)
    yb = y + 40
    step = bw / len(bins)
    for i, e in enumerate(bins):
        d.rectangle((x + i * step, yb, x + (i + 1) * step - 2, yb + bh), fill=e["color"])
    for v in (2, 6, 10, 14):
        tx = x + (v - 2) / 12 * bw
        d.line((tx, yb + bh, tx, yb + bh + 6), fill=INK, width=2)
        d.text((tx, yb + bh + 8), f"{v}", font=f, fill=INK, anchor="ma")
    d.rectangle((x, yb + 70, x + 26, yb + 88), fill=TOWER)
    d.text((x + 36, yb + 79), "grew since 2017", font=f, fill=INK, anchor="lm")
    return img


def north_arrow(img, cx, cy, vec):
    """Shaft and head along the projected north of the render."""
    (x0, y0), (x1, y1) = vec
    ang = math.atan2(y1 - y0, x1 - x0)
    ca, sa = math.cos(ang), math.sin(ang)
    img = shade(img, (cx - 46, cy - 46, cx + 46, cy + 46), alpha=210, radius=46)
    d = ImageDraw.Draw(img)
    tail, tip = (cx - 30 * ca, cy - 30 * sa), (cx + 2 * ca, cy + 2 * sa)
    d.line((tail, tip), fill=INK, width=4)
    head = (cx + 18 * ca, cy + 18 * sa)
    px, py = -sa, ca
    d.polygon([head, (tip[0] + 10 * px, tip[1] + 10 * py), (tip[0] - 10 * px, tip[1] - 10 * py)], fill=INK)
    d.text((cx + 32 * ca, cy + 32 * sa), "N", font=font(22, True), fill=INK, anchor="mm")
    return img


def source(img):
    f = font(22)
    y = SAFE[3] - len(SOURCE) * 30
    img = shade(img, (SAFE[0] - 12, y - 8, SAFE[2] + 12, SAFE[3] + 4), alpha=205, radius=10)
    d = ImageDraw.Draw(img)
    for i, s in enumerate(SOURCE):
        d.text((SAFE[0], y + i * 30), s, font=f, fill=MUTED)
    return img


def main_image(key, n, heat, meta):
    img = Image.open(os.path.join(POST, "li_main_raw.png")).convert("RGBA")
    img = top_fade(img, 250)
    d = ImageDraw.Draw(img)
    # Headline on the left, the one number on the right.
    f_h = font(46, True)
    lines = wrap(d, HEADLINES[key], f_h, 560)
    assert len(lines) <= 2, lines
    for i, ln in enumerate(lines):
        d.text((SAFE[0], SAFE[1] + i * 56), ln, font=f_h, fill=INK)
    d.text((SAFE[2], SAFE[1] - 8), f"{n['ha5']:.1f} ha", font=font(66, True), fill="#86b6ef", anchor="ra")
    d.text((SAFE[2], SAFE[1] + 74), "of the park feels 5 °C", font=font(24), fill=INK, anchor="ra")
    d.text((SAFE[2], SAFE[1] + 104), "colder, for an hour or more", font=font(24), fill=INK, anchor="ra")
    # Map labels from the render's own projection.
    l59, lim = meta["line59"], meta["limit"]
    img = pill(img, (SAFE[2], l59[1][1] - 30), "59th St", font(24, True), fg="#f5c451", anchor="rm")
    img = pill(img, (lim[1][0] + 14, lim[1][1]), "model stops here", font(22))
    img = pill(img, (W / 2, 266), "Central Park, 59th to 110th St", font(24, True), anchor="mm")
    img = north_arrow(img, SAFE[2] - 46, 306, meta["north"])
    img = legend(img, SAFE[2] - 256, 380, heat)
    img = source(img)
    return img


def pair_image(n, heat):
    # Each render cropped to the modelled park and the towers' tops.
    a, b = (Image.open(os.path.join(POST, f)).convert("RGBA").crop((0, 100, W, 580))
            for f in ("li_felt_2017_raw.png", "li_felt_today_raw.png"))
    y_a, y_b = 172, 172 + 480 + 8
    img = Image.new("RGBA", (W, H), PANEL + (255,))
    img.paste(a, (0, y_a))
    img.paste(b, (0, y_b))
    d = ImageDraw.Draw(img)
    d.text((SAFE[0], SAFE[1] - 4), "How the park feels at 14:30", font=font(44, True), fill=INK)
    img = pill(img, (SAFE[0], y_a + 36), "2017", font(30, True))
    img = pill(img, (SAFE[0], y_b + 36), "Today", font(30, True))
    # Felt-temperature key: the December range only.
    felt = [e for e in heat["legend"]["felt"] if e["max"] > -27 and e["min"] < 9]
    bw = 300
    x, y = SAFE[2] - bw - 6, y_a + 22
    img = shade(img, (x - 22, y - 12, x + bw + 22, y + 132), alpha=220)
    d = ImageDraw.Draw(img)
    d.text((x, y), "Feels like, °C", font=font(24, True), fill=INK)
    step = bw / len(felt)
    for i, e in enumerate(felt):
        d.rectangle((x + i * step, y + 38, x + (i + 1) * step - 2, y + 60), fill=e["color"])
    for i, e in enumerate(felt + [None]):
        v = felt[-1]["max"] if e is None else e["min"]
        if v in (-27, -13, 0, 9):
            d.text((x + i * step, y + 64), f"{v}".replace("-", "−"), font=font(22), fill=INK, anchor="ma")
    d.rectangle((x, y + 100, x + 26, y + 118), fill=GROWN)
    d.text((x + 36, y + 109), "grew since 2017", font=font(22), fill=INK, anchor="lm")
    lines = ["Same weather both years: only the buildings differ.",
             "Clear December day, typical year. SOLWEIG (UMEP). 2017 LiDAR vs today.",
             "Map © OpenStreetMap contributors."]
    d = ImageDraw.Draw(img)
    for i, s_ in enumerate(lines):
        d.text((SAFE[0], SAFE[3] - 3 * 30 + i * 30), s_, font=font(22), fill=MUTED)
    return img


def save(img, name):
    p = os.path.join(POST, name)
    img.convert("RGB").save(p, format="PNG", optimize=True)
    mb = os.path.getsize(p) / 1e6
    print(f"wrote {p}  {mb:.2f} MB")
    if mb >= 5:
        raise SystemExit(f"{name} is {mb:.1f} MB, over the 5 MB limit")
    # Phone check: what it looks like at 400 px wide.
    im = Image.open(p)
    im.resize((400, int(400 * im.height / im.width)), Image.LANCZOS).save(p.replace(".png", "_phone400.png"))


def main():
    n = numbers()
    with open(os.path.join(HERE, "viewer", "heat", "heat.json"), encoding="utf-8") as f:
        heat = json.load(f)
    with open(os.path.join(POST, "li_main_raw.json"), encoding="utf-8") as f:
        meta = json.load(f)
    for k in HEADLINES:
        save(main_image(k, n, heat, meta), f"linkedin_main_{k}.png")
    save(pair_image(n, heat), "linkedin_2017_vs_today.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
