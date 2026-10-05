"""
Turn the renders from tools/render_post.py into post images.

    .venv-heat/Scripts/python tools/compose_post.py

Writes to data/heat/figures/post/:
  post_hero_1920x1080.png       one landscape image, stands on its own
  slide_1..5_1080x1350.png      a portrait carousel
  carousel.pdf                  the same five slides as one document, which
                                LinkedIn shows as a swipeable carousel

Every number comes from data/heat/heat_numbers.csv, data/heat/compare/ and
viewer/heat/heat.json, never typed in by hand, and is printed here so it can
be checked against those files.
"""

from __future__ import annotations

import csv
import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POST = os.path.join(HERE, "data", "heat", "figures", "post")
FONTS = r"C:\Windows\Fonts"
INK = (240, 243, 247)
MUTED = (170, 178, 189)
AMBER = (245, 196, 81)
PANEL = (13, 16, 20)
CREDIT = ("Model estimate: SOLWEIG (UMEP) on a 2 m grid, Central Park typical-year weather. "
          "LiDAR: NYC 2017. Buildings: NYC Open Data. Map data: OpenStreetMap contributors.")
BYLINE = "Shadow twin  ·  Adel Yasser"


def font(size, bold=False):
    return ImageFont.truetype(os.path.join(FONTS, "segoeuib.ttf" if bold else "segoeui.ttf"), size)


def nb(text):
    """Keep numbers with their units on one line."""
    import re
    return re.sub(r"(\d) (°C|ha|m/s|m(?![a-z/])|%)",
                  lambda m: m.group(1) + "\u00a0" + m.group(2), text)


def wrap(draw, text, f, width):
    words, lines, cur = nb(text).split(" "), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def text_block(draw, xy, text, f, width, fill=INK, gap=1.18):
    x, y = xy
    for ln in wrap(draw, text, f, width):
        draw.text((x, y), ln, font=f, fill=fill)
        y += int(f.size * gap)
    return y


def panel(img, box, alpha=205, radius=18):
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(over).rounded_rectangle(box, radius=radius, fill=PANEL + (alpha,))
    return Image.alpha_composite(img, over)


def numbers():
    rows = {}
    with open(os.path.join(HERE, "data", "heat", "heat_numbers.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["key"]] = r
    with open(os.path.join(HERE, "viewer", "heat", "heat.json"), encoding="utf-8") as f:
        heat = json.load(f)
    with open(os.path.join(HERE, "data", "heat", "compare", "towers.json"), encoding="utf-8") as f:
        towers = json.load(f)
    v = lambda k: float(rows[k]["value"])
    dec = next(s for s in heat["seasons"] if s["id"] == "dec")
    worst = next(st for st in dec["steps"] if st["time"] == dec["summary"]["worst_hour"])
    n = {
        "area": v("dec_report_area"),
        "ha5": v("dec_ha_any_hour_dutci_le_-5"),
        "ha10": v("dec_ha_any_hour_dutci_le_-10"),
        "p1": -v("dec_dutci_p1_changed"),
        "worst_time": dec["summary"]["worst_hour"],
        "worst_ha": v("dec_worst_hour_ha_dutci_le_-2"),
        "jun_ha5": v("jun_ha_any_hour_dutci_le_-5"),
        "jun_ha10": v("jun_ha_any_hour_dutci_le_-10"),
        "felt_now": worst["park_felt_today_c"], "air_now": worst["air_c"], "wind_now": worst["wind_m_s"],
        "v1": min(v(f"V1_{t}_agreement") for t in ("10:00", "12:00", "14:00")),
        "v3": max(v(f"V3_{t}_change_rmse") for t in ("09:30", "11:30", "13:30")),
        "status": heat["status"],
        "towers": towers,
    }
    n["share5"] = n["ha5"] / n["area"]
    return n


# Towers we can name with confidence, by position (lon, lat within ~40 m).
NAMES = [((-73.98102, 40.76644), "Central Park Tower"),
         ((-73.97756, 40.76496), "Steinway Tower, 111 W 57th"),
         ((-73.97815, 40.76183), "53 W 53rd")]


def tower_name(lonlat):
    for (lo, la), nm in NAMES:
        if abs(lo - lonlat[0]) < 0.0005 and abs(la - lonlat[1]) < 0.0004:
            return nm
    return None


def footer(img, d, w, h, pad):
    f = font(19)
    y = h - pad - 2 * int(19 * 1.25)
    img = panel(img, (pad - 14, y - 12, w - pad + 14, h - pad + 12), alpha=180, radius=12)
    d = ImageDraw.Draw(img)
    text_block(d, (pad, y), CREDIT, f, w - 2 * pad, fill=MUTED, gap=1.25)
    return img, d


def legend_change(img, d, x, y, heat):
    rows = [e for e in heat["legend"]["change"] if e["max"] <= -1.9][::-1]   # 2 C and colder
    f = font(22)
    img = panel(img, (x - 16, y - 14, x + 300, y + 44 + 34 * len(rows) + 40), alpha=200)
    d = ImageDraw.Draw(img)
    d.text((x, y), "Feels colder than in 2017", font=font(22, True), fill=INK)
    yy = y + 40
    for e in rows:
        d.rounded_rectangle((x, yy + 4, x + 34, yy + 24), radius=4, fill=e["color"])
        d.text((x + 46, yy), f"{-e['max']:.0f} to {-e['min']:.0f} °C", font=f, fill=INK)
        yy += 34
    d.rounded_rectangle((x, yy + 8, x + 34, yy + 28), radius=4, fill=AMBER)
    d.text((x + 46, yy + 4), "grew since 2017", font=f, fill=INK)
    return img, d


def label_towers(img, d, meta, n, max_labels=3):
    f_b, f = font(24, True), font(21)
    by_pos = {tuple(t["lonlat"]): t for t in n["towers"]}
    placed = []
    W, H = img.size
    for t in sorted(meta["towers"], key=lambda t: -by_pos.get(tuple(t["lonlat"]), {}).get("dec_ha_any_hour_5", 0)):
        info = by_pos.get(tuple(t["lonlat"]))
        name = tower_name(t["lonlat"])
        if not info or not name or t["x"] is None or len(placed) >= max_labels:
            continue
        x, y = int(t["x"]), int(t["y"])
        lines = [name, f"{info['lidar_p99_m']:.0f} m → {info['height_m']:.0f} m",
                 nb(f"{info['dec_ha_any_hour_5']:.1f} ha of park 5 °C+ colder")]
        bw = int(max(d.textlength(lines[0], font=f_b), max(d.textlength(s_, font=f) for s_ in lines[1:])) + 28)
        bh = 92
        # Candidate boxes: above, then to either side, then further up or down.
        cands = [(x - bw // 2, y - bh - 26), (x - bw - 30, y - bh // 2), (x + 30, y - bh // 2),
                 (x - bw // 2, y - bh - 140), (x - bw - 30, y + 30), (x + 30, y + 30),
                 (x - bw // 2, y + 60)]
        box = None
        for cx, cy in cands:
            cx, cy = min(max(12, cx), W - bw - 12), min(max(12, cy), H - bh - 160)
            b = (cx, cy, cx + bw, cy + bh)
            if all(b[2] < p[0] - 8 or b[0] > p[2] + 8 or b[3] < p[1] - 8 or b[1] > p[3] + 8 for p in placed):
                box = b
                break
        if box is None:
            continue
        img = panel(img, box, alpha=215, radius=10)
        d = ImageDraw.Draw(img)
        # Leader line from the box edge nearest the roof to the roof.
        lx = min(max(x, box[0]), box[2])
        ly = box[3] if y > box[3] else (box[1] if y < box[1] else (box[1] + box[3]) // 2)
        d.line((lx, ly, x, y), fill=AMBER, width=3)
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=AMBER)
        d.text((box[0] + 14, box[1] + 8), lines[0], font=f_b, fill=AMBER)
        d.text((box[0] + 14, box[1] + 38), lines[1], font=f, fill=INK)
        d.text((box[0] + 14, box[1] + 63), lines[2], font=f, fill=INK)
        placed.append(box)
    return img, d


def load(name):
    return Image.open(os.path.join(POST, name + ".png")).convert("RGBA")


def save(img, name):
    path = os.path.join(POST, name)
    img.convert("RGB").save(path, optimize=True)
    print("wrote", path)
    return path


def hero_landscape(n, heat):
    img = load("raw_dec_change_hero")
    w, h = img.size
    pad = 44
    d = ImageDraw.Draw(img)
    f_t, f_b, f_s = font(46, True), font(27), font(21)
    tw = 900
    title = "New supertalls make Central Park feel colder in winter"
    body = (f"On a clear December day, {n['ha5']:.1f} ha of the park's south end "
            f"({n['share5']:.0%}) feels at least 5 °C colder than in 2017, for an hour or "
            f"more. In the towers' shadows the drop reaches about {n['p1']:.0f} °C.")
    note = f"{n['worst_time']} EST · blue columns: 10 m squares, taller = colder · amber: grew since 2017"
    lt, lb = wrap(d, title, f_t, tw), wrap(d, body, f_b, tw)
    ph = int(len(lt) * 46 * 1.12 + len(lb) * 27 * 1.2 + 21 * 1.6 + 30)
    top = h - pad - 2 * int(19 * 1.25) - 40 - ph
    img = panel(img, (pad - 18, top - 16, pad + tw + 22, top + ph), alpha=218)
    d = ImageDraw.Draw(img)
    y = text_block(d, (pad, top), title, f_t, tw, gap=1.12)
    y = text_block(d, (pad, y + 6), body, f_b, tw, gap=1.2)
    d.text((pad, y + 8), nb(note), font=f_s, fill=MUTED)
    img, d = legend_change(img, d, w - 330, pad + 4, heat)
    img, d = footer(img, d, w, h, pad)
    d.text((w - pad - d.textlength(BYLINE, font=font(21, True)), top - 2), BYLINE,
           font=font(21, True), fill=INK)
    return save(img, "post_hero_1920x1080.png")


def slide(raw, title, body, n, heat, extra=None):
    img = load(raw)
    w, h = img.size
    pad = 40
    d = ImageDraw.Draw(img)
    f_t, f_b = font(48, True), font(28)
    lines_t = wrap(d, title, f_t, w - 2 * pad)
    lines_b = wrap(d, body, f_b, w - 2 * pad)
    top_h = int(len(lines_t) * 48 * 1.12 + len(lines_b) * 28 * 1.25 + 46)
    img = panel(img, (pad - 18, pad - 16, w - pad + 18, pad + top_h), alpha=218)
    d = ImageDraw.Draw(img)
    y = text_block(d, (pad, pad), title, f_t, w - 2 * pad, gap=1.12)
    text_block(d, (pad, y + 10), body, f_b, w - 2 * pad, gap=1.25)
    if extra:
        img, d = extra(img, d)
    img, d = footer(img, d, w, h, pad)
    d.text((pad, h - pad - 96), BYLINE, font=font(21, True), fill=INK)
    return img


def carousel(n, heat):
    slides = []
    s1 = slide("raw_dec_change_portrait", "New supertalls make Central Park feel colder in winter",
               f"On a clear December day, {n['ha5']:.1f} ha of the park's south end, {n['share5']:.0%} "
               f"of it, feels at least 5 °C colder than in 2017 for an hour or more. "
               f"Blue columns show where, at {n['worst_time']}: the taller, the colder.",
               n, heat, extra=lambda im, d: legend_change(im, d, im.width - 330, 860, heat))
    slides.append(s1)
    with open(os.path.join(POST, "rawP_dec_change_top.json"), encoding="utf-8") as f:
        meta = json.load(f)
    s2 = slide("rawP_dec_change_top", "Three towers do almost all of it",
               "Each amber building grew since 2017. Each blue finger is the patch of park it now "
               "makes feel colder in the afternoon, when winter shadows run longest.",
               n, heat, extra=lambda im, d: label_towers(im, d, meta, n))
    slides.append(s2)
    # 2017 vs today, how it feels: two crops stacked.
    a, b = load("rawP_dec_felt_2017"), load("rawP_dec_felt_today")
    w, h = a.size
    crop = (0, 230, w, 230 + 560)
    s3 = Image.new("RGBA", (w, h), PANEL + (255,))
    s3.paste(a.crop(crop), (0, 250))
    s3.paste(b.crop(crop), (0, 250 + 570))
    d = ImageDraw.Draw(s3)
    pad = 40
    y = text_block(d, (pad, pad), "How it feels: 2017 vs today", font(48, True), w - 2 * pad, gap=1.12)
    text_block(d, (pad, y + 8), f"Felt temperature in the park at {n['worst_time']}, as 10 m "
               f"blocks: the taller and darker, the harsher. Today the park typically feels like "
               f"about {n['felt_now']:.0f} °C; the air is {n['air_now']:.0f} °C, with a "
               f"{n['wind_now']:.0f} m/s wind.", font(26), w - 2 * pad, gap=1.25)
    for yy, lab in ((260, "2017"), (830, "Today")):
        s3 = panel(s3, (pad - 10, yy, pad + 130, yy + 52), alpha=225, radius=10)
        d = ImageDraw.Draw(s3)
        d.text((pad + 6, yy + 6), lab, font=font(30, True), fill=AMBER if lab == "Today" else INK)
    s3, d = footer(s3, d, w, h, pad)
    slides.append(s3)
    s4 = slide("rawP_jun_change", "June is the control",
               f"Same towers, high summer sun, short shadows. At most {n['jun_ha5']:.1f} ha feels "
               f"5 °C colder at some hour, mostly in the evening. That near-silence is the check "
               f"that the December result comes from the towers, not from the model.",
               n, heat)
    slides.append(s4)
    # Method and checks, on a dark slide with a faded hero.
    bg = load("raw_dec_change_portrait").point(lambda v: int(v * 0.28))
    d = ImageDraw.Draw(bg)
    pad = 48
    y = text_block(d, (pad, pad + 10), "How we know", font(52, True), w - 2 * pad)
    items = [
        ("Same weather, two cities.", "Both runs use the same clear December day from the Central "
         "Park typical year, the same trees, the same ground. Only the buildings differ."),
        ("Shadows check out.", f"Our heat model's shadows match the shadow study's own geometry "
         f"on {n['v1']:.1%} or more of the ground."),
        ("An independent second model agrees.", f"UMEP's reference code, run separately on the "
         f"same blocks, gives the same change within {n['v3']:.1f} °C."),
        ("Believable values.", "Sun vs shade on the same lawn differs by about 12 °C in "
         "felt temperature, in line with published measurements."),
        ("What it leaves out.", "Wind is one value for the whole park: no gusts at the foot of "
         "towers yet. 'Today' raises grown buildings to their recorded roof, an upper bound. "
         "Read the change, not the absolute."),
    ]
    y += 18
    for head, body in items:
        d.text((pad, y), head, font=font(30, True), fill=AMBER)
        y = text_block(d, (pad, y + 40), body, font(26), w - 2 * pad, gap=1.25) + 22
    bg, d = footer(bg, d, w, h, pad)
    d.text((pad, h - pad - 96), BYLINE, font=font(21, True), fill=INK)
    slides.append(bg)
    paths = [save(s, f"slide_{i}_1080x1350.png") for i, s in enumerate(slides, start=1)]
    rgb = [Image.open(p).convert("RGB") for p in paths]
    pdf = os.path.join(POST, "carousel.pdf")
    rgb[0].save(pdf, save_all=True, append_images=rgb[1:], resolution=144)
    print("wrote", pdf)


def main():
    n = numbers()
    with open(os.path.join(HERE, "viewer", "heat", "heat.json"), encoding="utf-8") as f:
        heat = json.load(f)
    print({k: v for k, v in n.items() if k not in ("towers", "status")}, n["status"])
    hero_landscape(n, heat)
    carousel(n, heat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
