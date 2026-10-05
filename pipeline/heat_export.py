"""
Heat layers for the viewer: how cold or hot it feels, how much that changed
since 2017, and 3D columns where it changed.

    .venv-heat/Scripts/python -m pipeline.heat_export

Reads the SOLWEIG runs in data/heat/runs/ and writes viewer/heat/heat.json plus
the images it names, next to the viewer pages. The page fetches it on demand;
a build without it simply shows no heat layers.

What is shown
-------------
  Feels like       UTCI, the "feels like" temperature of the Universal
                   Thermal Climate Index, at body height, every hour inside
                   the CEQR window. Never mean radiant temperature: that is a
                   physics input, it reaches 60 to 80 C in summer sun, and it
                   reads as fake to anyone who sees it.
  Colder since     UTCI today minus UTCI in 2017, same hour, same weather.
  2017             Only the buildings differ.
  3D columns       the same change, averaged over 10 m squares of the park,
                   standing as columns whose height is proportional to how
                   much colder it feels there (6 m per degree).
  3D heatmap       with Feels like, every 10 m square of the park stands as a
                   column coloured by its felt temperature. Height is thermal
                   stress (degrees outside the UTCI "no thermal stress" band,
                   9 to 26 C) above the mildest 2% of the park that day, 2 m
                   per degree: the harshest spots stand tallest.

Colour follows fixed breaks, never stretched to the data. Felt temperature is
diverging around the UTCI "no thermal stress" band (9 to 26 C, grey): blues
below it, reds above it, darker the further out. The change is diverging
around zero: blue colder, red warmer, no change transparent.
"""

from __future__ import annotations

import base64
import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from . import export
from .heat_inputs import HEAT_DIR, INPUTS_DIR, US_FT

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "viewer", "heat")
CELL_HA = 4.0 / 1e4
SECTION = 5                      # cells per side of a 3D column, 10 m
COLUMN_MIN = 1.0                 # degrees of change before a column stands
COLUMN_M_PER_C = 6.0
FELT_M_PER_C = 2.0               # felt heatmap: metres per degree of thermal stress
DISPLAY_PAD = 150                # cells around the window that are drawn, 300 m

# UTCI assessment scale (Blazejczyk et al. 2013).
UTCI_CATS = [(-99, -40, "extreme cold stress"), (-40, -27, "very strong cold stress"),
             (-27, -13, "strong cold stress"), (-13, 0, "moderate cold stress"),
             (0, 9, "slight cold stress"), (9, 26, "no thermal stress"),
             (26, 32, "moderate heat stress"), (32, 38, "strong heat stress"),
             (38, 46, "very strong heat stress"), (46, 99, "extreme heat stress")]
# Few enough steps per arm that neighbours stay visibly different (dataviz
# validator: adjacent lightness gap >= 0.06), with the category edges kept.
FELT_BREAKS = [-40, -27, -20, -13, -9, -5, 0, 9, 26, 29, 32, 35, 38, 42, 46, 60]
CHANGE_BREAKS = [-14, -12, -10, -8, -6, -4, -2, -0.5, 0.5, 2, 4, 6, 8, 10, 12, 14]

# Diverging poles and midpoint: the blue sequential ramp (700 to 250) and a red
# arm of matching lightness range; light ends clear 2:1 on the light basemap.
BLUE_DARK, BLUE_LIGHT = "#0d366b", "#86b6ef"
RED_LIGHT, RED_DARK = "#ef8f88", "#6b1414"
NEUTRAL = "#e9e7e1"


# ------------------------------------------------------------ colour helpers
def _srgb_to_lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin_to_srgb(c):
    c = max(0.0, min(1.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _to_oklab(hexc):
    r, g, b = (_srgb_to_lin(int(hexc[i:i + 2], 16) / 255) for i in (1, 3, 5))
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _from_oklab(L, a, b):
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return "#" + "".join(f"{round(_lin_to_srgb(v) * 255):02x}" for v in (r, g, bb))


def ramp(stops, n):
    """n colours evenly along a piecewise OKLab path through the stops."""
    labs = [_to_oklab(c) for c in stops]
    out = []
    for i in range(n):
        t = i / max(1, n - 1) * (len(labs) - 1)
        k = min(int(t), len(labs) - 2)
        f = t - k
        out.append(_from_oklab(*(labs[k][j] * (1 - f) + labs[k + 1][j] * f for j in range(3))))
    return out


def felt_palette():
    n_cold = FELT_BREAKS.index(9)                     # bins below 9 C
    n_hot = len(FELT_BREAKS) - 1 - FELT_BREAKS.index(26)
    cold = ramp([BLUE_DARK, BLUE_LIGHT], n_cold)       # coldest first
    hot = ramp([RED_LIGHT, RED_DARK], n_hot)
    return cold + [NEUTRAL] + hot


def change_palette():
    n = CHANGE_BREAKS.index(-0.5)
    cold = ramp([BLUE_DARK, BLUE_LIGHT], n)
    hot = ramp([RED_LIGHT, RED_DARK], len(CHANGE_BREAKS) - 1 - CHANGE_BREAKS.index(0.5))
    return cold + [None] + hot                         # None: no change, transparent


def category(v):
    for lo, hi, name in UTCI_CATS:
        if lo <= v < hi:
            return name
    return ""


# ------------------------------------------------------------------- grid
def heat_grid(meta, masks):
    """The heat domain as a pipeline Grid, framed on the window plus a margin."""
    from .grid import Grid
    g = meta["grid"]
    rr, cc = np.nonzero(masks["window"])
    r0 = max(0, int(rr.min()) - DISPLAY_PAD)
    r1 = min(g["rows"], int(rr.max()) + DISPLAY_PAD)
    c0 = max(0, int(cc.min()) - DISPLAY_PAD)
    c1 = min(g["cols"], int(cc.max()) + DISPLAY_PAD)
    return Grid(crs="EPSG:2263", unit_m=US_FT, res_m=g["res_m"], bearing_deg=-g["conv_deg"],
                conv_deg=g["conv_deg"], x0=g["x0_ft"], y0=g["y0_ft"], rows=g["rows"],
                cols=g["cols"], frame=(r0, r1, c0, c1),
                centre_lat=meta["site"]["lat"], centre_lon=meta["site"]["lon"])


# Image names repeat every export, so each link carries a version stamp and a
# browser never shows a cached image from an older run.
VERSION = datetime.now().strftime("%Y%m%d%H%M%S")


def write_png(uri, name):
    with open(os.path.join(OUT_DIR, name), "wb") as f:
        f.write(base64.b64decode(uri.split(",", 1)[1]))
    return f"heat/{name}?v={VERSION}"


def png_change(painter, values, mask, colours):
    """Palette PNG with the 'no change' bin transparent."""
    idx = painter.bin_index(values, CHANGE_BREAKS, mask)
    hole = CHANGE_BREAKS.index(-0.5) + 1               # palette index of the no-change bin
    idx[idx == hole] = 0
    cols = [c or "#000000" for c in colours]
    alphas = [0 if c is None else 225 for c in colours]
    return export.encode_indexed(idx, cols, alphas)


def section_means(arr, report, grid):
    """Mean of arr over the reported cells of each 10 m square in the frame."""
    r0, r1, c0, c1 = grid.frame
    H, W = (r1 - r0) // SECTION, (c1 - c0) // SECTION
    d = arr[r0:r0 + H * SECTION, c0:c0 + W * SECTION].reshape(H, SECTION, W, SECTION)
    m = report[r0:r0 + H * SECTION, c0:c0 + W * SECTION].reshape(H, SECTION, W, SECTION)
    n = m.sum(axis=(1, 3))
    s = np.where(m, d, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n >= (SECTION * SECTION) // 2, s / np.maximum(n, 1), np.nan)
    return mean


def section_lonlat(ii, jj, grid, to_ll):
    r0, _, c0, _ = grid.frame
    x, y = grid.rowcol_to_crs(r0 + ii * SECTION + SECTION / 2.0, c0 + jj * SECTION + SECTION / 2.0)
    lon, lat = to_ll(x, y)
    return np.atleast_1d(lon), np.atleast_1d(lat)


def columns(delta, report, grid, to_ll):
    """[lon, lat, change] for each 10 m square of the park that changed."""
    mean = np.nan_to_num(section_means(delta, report, grid), nan=0.0)
    ii, jj = np.nonzero(np.abs(mean) >= COLUMN_MIN)
    if not ii.size:
        return []
    lon, lat = section_lonlat(ii, jj, grid, to_ll)
    return [[round(float(lo), 6), round(float(la), 6), round(float(v), 1)]
            for lo, la, v in zip(lon, lat, mean[ii, jj])]


def gate_status():
    from .verify_heat import GATES_JSON
    if not os.path.exists(GATES_JSON):
        return {"validated": False, "gates": {}}
    with open(GATES_JSON, encoding="utf-8") as f:
        g = json.load(f)
    gates = {k: bool(v["pass"]) for k, v in g.items()
             if isinstance(v, dict) and v.get("pass") is not None}
    ok = all(gates.get(k) for k in ("V1", "V2", "V3"))
    return {"validated": ok, "gates": gates}


def main():
    from .heat_inputs import load
    from .run_heat import days, weather_list
    from .solar import sun_position
    from .verify_heat import read_step, report_steps, runs_dir
    from pyproj import Transformer

    layers, masks, meta = load()
    report = masks["report"]
    building = masks["building"]
    grid = heat_grid(meta, masks)
    fsl = grid.frame_slice
    idx, bounds = grid.northup_index()
    painter = export.Painter(idx, bounds)
    to_ll = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True).transform
    lat, lon = meta["site"]["lat"], meta["site"]["lon"]
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):                  # stale images must not survive
        if f.endswith(".png") or f == "heat.json":
            os.remove(os.path.join(OUT_DIR, f))

    # The felt-temperature 3D heatmap stands on every 10 m square of the park.
    # Positions once; per hour, the mean felt temperature in half degrees.
    probe = section_means(np.zeros(report.shape, dtype=np.float32), report, grid)
    sec_i, sec_j = np.nonzero(np.isfinite(probe))
    slon, slat = section_lonlat(sec_i, sec_j, grid, to_ll)
    sections = [[round(float(a), 6), round(float(b), 6)] for a, b in zip(slon, slat)]
    print(f"{len(sections):,} park squares of {SECTION * 2} m for the 3D heatmap", flush=True)

    def felt_ints(arr):
        m = section_means(arr, report, grid)[sec_i, sec_j]
        return [int(round(float(v) * 2)) for v in np.nan_to_num(m, nan=0.0)]

    felt_cols = felt_palette()
    chg_cols = change_palette()
    bmask = building[fsl]
    d = days()
    seasons = []
    for season, label, short, control in (
            ("dec", "Clear December day", "Dec", False),
            ("jun", "Clear June day (control)", "Jun", True)):
        sd = d["seasons"][season]
        wl = {w.datetime: w for w in weather_list(season)}
        steps = []
        for ts in report_steps(season):
            st = ts - timedelta(minutes=30)
            hhmm = st.strftime("%H%M")
            p = sun_position((st + timedelta(hours=5)).replace(tzinfo=timezone.utc), lat, lon)
            u17 = read_step("2017", season, "utci", ts)
            unow = read_step("today", season, "utci", ts)
            du = unow - u17
            img = {
                "today": write_png(painter.png(unow[fsl], FELT_BREAKS, felt_cols, alpha=215,
                                               mask=bmask), f"{season}_{hhmm}_today.png"),
                "y2017": write_png(painter.png(u17[fsl], FELT_BREAKS, felt_cols, alpha=215,
                                               mask=bmask), f"{season}_{hhmm}_2017.png"),
                "change": write_png(png_change(painter, du[fsl], bmask, chg_cols),
                                    f"{season}_{hhmm}_change.png"),
            }
            r = du[report]
            w = wl[ts]
            steps.append({
                "time": st.strftime("%H:%M"), "sun": {"alt": round(p.altitude, 2), "az": round(p.azimuth, 2)},
                "img": img,
                "air_c": w.ta, "wind_m_s": w.ws,
                "park_felt_today_c": round(float(np.nanmedian(unow[report])), 1),
                "park_felt_2017_c": round(float(np.nanmedian(u17[report])), 1),
                "ha_colder_2": round(float((r <= -2).sum() * CELL_HA), 2),
                "ha_colder_5": round(float((r <= -5).sum() * CELL_HA), 2),
                "biggest_drop_c": round(float(-np.nanmin(r)), 1),
                "cols": columns(du, report, grid, to_ll),
                "felt2": {"today": felt_ints(unow), "y2017": felt_ints(u17)},
            })
            print(f"{season} {st:%H:%M}: {steps[-1]['ha_colder_2']:.1f} ha at least 2 C colder, "
                  f"{len(steps[-1]['cols'])} columns", flush=True)
        summ_path = os.path.join(HEAT_DIR, "compare", season, "summary.json")
        summary = {}
        if os.path.exists(summ_path):
            with open(summ_path, encoding="utf-8") as f:
                s = json.load(f)
            summary = {k: s[k] for k in ("report_ha", "ha_any_hour_le_-2", "ha_any_hour_le_-5",
                                         "ha_any_hour_le_-10", "ha_hours_le_-2", "worst_hour",
                                         "worst_hour_ha_le_minus2", "bands") if k in s}
        # Heatmap baseline: the mildest ground of the day (2nd percentile of
        # thermal stress over the park and the reported hours) stands flat.
        allu = np.concatenate([np.asarray(st["felt2"]["today"] + st["felt2"]["y2017"]) / 2.0
                               for st in steps])
        stress = np.where(allu < 9, 9 - allu, np.where(allu > 26, allu - 26, 0.0))
        base = round(float(np.percentile(stress, 2)), 1)
        seasons.append({"id": season, "label": label, "short": short, "control": control,
                        "felt_base_stress": base,
                        "date": sd["chosen"]["date"], "steps": steps, "summary": summary,
                        "air_c": sd["ta_range_c"]})

    towers = []
    att = os.path.join(HEAT_DIR, "compare", "towers.json")
    if os.path.exists(att):
        with open(att, encoding="utf-8") as f:
            towers = json.load(f)

    felt_legend = []
    for i, c in enumerate(felt_cols):
        lo, hi = FELT_BREAKS[i], FELT_BREAKS[i + 1]
        felt_legend.append({"min": lo, "max": hi, "color": c, "label": category((lo + hi) / 2)})
    chg_legend = [{"min": CHANGE_BREAKS[i], "max": CHANGE_BREAKS[i + 1], "color": c}
                  for i, c in enumerate(chg_cols) if c is not None]
    w_, s_, e_, n_ = bounds
    bundle = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "runs": runs_dir(),
        "status": gate_status(),
        "bounds": {"west": w_, "south": s_, "east": e_, "north": n_},
        "area_ha": round(float(report.sum() * CELL_HA), 1),
        "seasons": seasons,
        "legend": {"felt": felt_legend, "change": chg_legend,
                   "categories": [{"min": lo, "max": hi, "label": n} for lo, hi, n in UTCI_CATS]},
        "columns": {"section_m": SECTION * 2, "m_per_c": COLUMN_M_PER_C, "min_c": COLUMN_MIN,
                    "felt_m_per_c": FELT_M_PER_C, "neutral": [9, 26]},
        "sections": sections,
        "towers": towers,
        "weather": {"station": d["header"]["city"], "wmo": d["header"]["wmo"], "epw": d["epw"]},
    }
    with open(os.path.join(OUT_DIR, "heat.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(bundle, f, separators=(",", ":"))
    size = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR))
    print(f"wrote {OUT_DIR}: {len(os.listdir(OUT_DIR))} files, {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
