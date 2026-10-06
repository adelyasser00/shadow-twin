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


# The tower view: the carousel's slide 2 look (title card, named towers with
# leader lines) on the whole-day change, as the first image of the post.
HERO_TITLES = {
    "A": "Three towers do almost all of it",
    "B": "Where the new towers chill Central Park",
    "C": "New towers made Central Park feel colder",
}
HERO_SUB = ("Blue: park that now feels 5 °C or more colder on a clear December day. "
            "Amber: towers that grew since 2017.")
NAMES = [((-73.98102, 40.76644), "Central Park Tower"),
         ((-73.97756, 40.76496), "Steinway Tower, 111 W 57th"),
         ((-73.97815, 40.76183), "53 W 53rd")]


def tower_name(lonlat):
    for (lo, la), nm in NAMES:
        if abs(lo - lonlat[0]) < 0.0005 and abs(la - lonlat[1]) < 0.0004:
            return nm
    return None


def label_towers(img, meta, towers, keep_out):
    """Name, height then and now, and park made 5 C colder, for each named tower.

    Each label goes where it hides the least of the picture's story: the blue
    fingers and the amber towers, counted in the render itself.
    """
    import numpy as np
    a = np.asarray(Image.open(os.path.join(POST, "li_towers_raw.png")).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    story = ((b > r + 30) & (b > g + 10)).astype(int) + 4 * ((r > b + 70) & (g > b + 30))
    f_b, f = font(25, True), font(23)
    by_pos = {tuple(t["lonlat"]): t for t in towers}
    placed = list(keep_out)
    # No label over any named tower's body, roof to street.
    placed += [(int(t["x"]) - 26, int(t["y"]) - 14, int(t["x"]) + 26, int(t["y"]) + 190)
               for t in meta["towers"] if t["x"] is not None and tower_name(t["lonlat"])]
    d = ImageDraw.Draw(img)
    for t in sorted(meta["towers"], key=lambda t: -by_pos.get(tuple(t["lonlat"]), {}).get("dec_ha_any_hour_5", 0)):
        info, name = by_pos.get(tuple(t["lonlat"])), tower_name(t["lonlat"])
        if not info or not name or t["x"] is None:
            continue
        x, y = int(t["x"]), int(t["y"])
        lines = [name, f"{info['lidar_p99_m']:.0f} m → {info['height_m']:.0f} m",
                 f"{info['dec_ha_any_hour_5']:.1f} ha of park 5 °C+ colder"]
        bw = int(max(d.textlength(lines[0], font=f_b), max(d.textlength(s_, font=f) for s_ in lines[1:])) + 28)
        bh = 98
        cands = [(x - bw // 2, y - bh - 28), (x - bw - 44, y - bh // 2), (x + 44, y - bh // 2),
                 (x - bw // 2, y - bh - 150), (x - bw - 44, y + 30), (x + 44, y + 30), (x - bw // 2, y + 70),
                 (x - bw // 2, y + 210), (x + 44, y + 120), (x - bw - 44, y + 120)]
        box, best = None, None
        for cx, cy in cands:
            cx = int(min(max(SAFE[0] - 14, cx), SAFE[2] + 14 - bw))
            bx = (cx, int(cy), cx + bw, int(cy) + bh)
            if bx[1] < SAFE[1] or bx[3] > SAFE[3]:
                continue
            if not all(bx[2] < q[0] - 8 or bx[0] > q[2] + 8 or bx[3] < q[1] - 8 or bx[1] > q[3] + 8
                       for q in placed):
                continue
            cost = int(story[bx[1]:bx[3], bx[0]:bx[2]].sum()) + 3 * math.hypot(bx[0] + bw / 2 - x, bx[1] + bh / 2 - y)
            if best is None or cost < best:
                box, best = bx, cost
        if box is None:
            print("no room for", name)
            continue
        img = shade(img, box, alpha=218, radius=10)
        d = ImageDraw.Draw(img)
        lx = min(max(x, box[0]), box[2])
        ly = box[3] if y > box[3] else (box[1] if y < box[1] else (box[1] + box[3]) // 2)
        d.line((lx, ly, x, y), fill=GROWN, width=3)
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=GROWN)
        d.text((box[0] + 14, box[1] + 8), lines[0], font=f_b, fill=GROWN)
        d.text((box[0] + 14, box[1] + 41), lines[1], font=f, fill=INK)
        d.text((box[0] + 14, box[1] + 67), lines[2], font=f, fill=INK)
        placed.append(box)
    return img


def hero_image(key, n, heat, meta, towers):
    img = Image.open(os.path.join(POST, "li_towers_raw.png")).convert("RGBA")
    d = ImageDraw.Draw(img)
    size = 48                                   # one line if it fits at 40 px or more
    while size > 40 and d.textlength(HERO_TITLES[key], font=font(size, True)) > 880:
        size -= 1
    f_t, f_s = font(size, True), font(25)
    lt = wrap(d, HERO_TITLES[key], f_t, 880)
    ls = wrap(d, HERO_SUB.replace("5 °C", "5 °C"), f_s, 880)
    top = SAFE[1] - 10
    ys = top + len(lt) * 56 + 6
    card = (SAFE[0] - 22, top - 18, SAFE[2] + 22, ys + len(ls) * 32 + 14)
    img = shade(img, card, alpha=222, radius=18)
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lt):
        d.text((SAFE[0], top + i * 56), ln, font=f_t, fill=INK)
    for i, ln in enumerate(ls):
        d.text((SAFE[0], ys + i * 32), ln, font=f_s, fill=MUTED)
    # The one number, on the east side where it covers only buildings.
    nb_ = (SAFE[2] - 262, card[3] + 92, SAFE[2] + 14, card[3] + 232)
    img = shade(img, nb_, alpha=222, radius=14)
    d = ImageDraw.Draw(img)
    d.text((SAFE[2], nb_[1] + 2), f"{n['ha5']:.1f} ha", font=font(62, True), fill="#86b6ef", anchor="ra")
    d.text((SAFE[2], nb_[1] + 78), "feels 5 °C+ colder", font=font(23), fill=INK, anchor="ra")
    d.text((SAFE[2], nb_[1] + 106), "for an hour or more", font=font(23), fill=INK, anchor="ra")
    lim = meta["limit"]
    img = pill(img, (lim[1][0] + 14, lim[1][1]), "model stops here", font(22))
    img = north_arrow(img, SAFE[2] - 46, nb_[3] + 66, meta["north"])
    img = source(img)
    keep_out = [card, nb_, (SAFE[0] - 12, SAFE[3] - len(SOURCE) * 30 - 8, SAFE[2] + 12, SAFE[3] + 4),
                (SAFE[2] - 92, nb_[3] + 20, SAFE[2] + 2, nb_[3] + 112)]
    img = label_towers(img, meta, towers, keep_out)
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


def link_preview():
    """viewer/preview.png, the 1200 x 627 card LinkedIn shows for the site link.

    The tower view cropped to the towers and their fingers, no text: the card
    already prints the page title under the image.
    """
    raw = Image.open(os.path.join(POST, "li_towers_raw.png")).convert("RGB")
    img = raw.crop((0, 250, 1080, 815)).resize((1200, 627), Image.LANCZOS)
    p = os.path.join(HERE, "viewer", "preview.png")
    img.save(p, optimize=True)
    print(f"wrote {p}  {os.path.getsize(p) / 1e6:.2f} MB")


def main():
    n = numbers()
    with open(os.path.join(HERE, "viewer", "heat", "heat.json"), encoding="utf-8") as f:
        heat = json.load(f)
    with open(os.path.join(POST, "li_main_raw.json"), encoding="utf-8") as f:
        meta = json.load(f)
    for k in HEADLINES:
        save(main_image(k, n, heat, meta), f"linkedin_main_{k}.png")
    with open(os.path.join(POST, "li_towers_raw.json"), encoding="utf-8") as f:
        tmeta = json.load(f)
    with open(os.path.join(HERE, "data", "heat", "compare", "towers.json"), encoding="utf-8") as f:
        towers = json.load(f)
    for k in HERO_TITLES:
        save(hero_image(k, n, heat, tmeta, towers), f"linkedin_towers_{k}.png")
    save(pair_image(n, heat), "linkedin_2017_vs_today.png")
    link_preview()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
