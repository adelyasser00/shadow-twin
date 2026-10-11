"""
The comment image for the wind post: a vertical section through Central Park
Tower along the wind, 2017 above, today below, the diagram wind engineers
draw.

    .venv-heat/Scripts/python tools/post_wind_data.py
    .venv-heat/Scripts/python tools/post_wind_section.py

The plane runs along the wind (from the north-west, left to right) through
the middle of today's tower; both years are cut on the same plane of the same
box. Colour: wind speed in the plane as a share of the wind 300 m up, one
blue ramp, same scale in both panels. Thin lines follow the wind's components
in this plane. Bold lines start 30 to 70 m upwind of the windward face, 60 to
100 m up, in both years. Buildings are grey, the tower amber (the series
colour for "grew since 2017"); today's panel outlines, dashed, the tower as
the 2017 model had it in this same cut (its 8 m columns, so the core shows
above the 124 m working floor).

Plain, like a figure in a paper: white ground, IBM Plex, no boxes, no text
under 20 px. 1080 x 1350 PNG to data/wind/figures/post/post_b2_section_<stamp>.png,
never overwritten.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "data", "wind", "figures", "post")
FONTS = os.path.join(HERE, "data", "fonts")
AMBER = "#f5c451"
INK = "#15181c"
MUTED = "#4a5058"
BUILDING = "#8b9099"
X_RANGE = (-480.0, 370.0)
Z_TOP = 500.0
TOWER_COLS = 9                 # 8 m columns of the tower in the cut, from its windward face
SEEDS = [(sx, sz) for sx in (-70.0, -50.0, -30.0) for sz in (60.0, 80.0, 100.0)]
NL = "\n"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib import patheffects as pe
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    for f in os.listdir(FONTS):
        if f.endswith(".ttf"):
            fm.fontManager.addfont(os.path.join(FONTS, f))
    halo = [pe.withStroke(linewidth=4, foreground="white")]
    plt.rcParams.update({"font.family": "IBM Plex Sans", "font.size": 15, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.linewidth": 0.8})
    z = np.load(os.path.join(HERE, "data", "wind", "post", "section.npz"))
    with open(os.path.join(HERE, "data", "wind", "tower_heights_2017.json"), encoding="utf-8") as f:
        heights = json.load(f)
    res, u_ref = float(z["res"]), float(z["u_ref"])
    h17 = heights["Central Park Tower"]["roof_2017_m"]
    hnow = heights["Central Park Tower"]["today_m"]
    cmap = ListedColormap(matplotlib.colormaps["Blues"](np.linspace(0.08, 0.95, 256)))
    lab = dict(fontsize=15, color=INK, family="IBM Plex Sans Condensed", path_effects=halo, zorder=7)
    lead = dict(arrowstyle="-", color=INK, linewidth=1.0)

    fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor="white")
    fig.text(0.093, 0.966, "How the wind meets Central Park Tower", fontsize=30, weight="semibold", color=INK)
    fig.text(0.093, 0.917, "A vertical cut along the December northwester, through the tower." + NL +
             "Same wind, same cut, 2017 and today.", fontsize=16, color=MUTED, linespacing=1.35)
    panels = [("2017", f"2017: the tower at {h17} m", 0.575), ("today", f"Today: the tower at {hnow} m", 0.232)]
    for sc, title, bottom in panels:
        ax = fig.add_axes([0.093, bottom, 0.815, 0.30])
        ux, uz, solid = z[f"{sc}_ux"].astype(float), z[f"{sc}_uz"].astype(float), z[f"{sc}_solid"]
        k0 = int(z[f"{sc}_ground_k"].min())
        nz, nx = ux.shape
        x = float(z[f"{sc}_x0_m"]) + (np.arange(nx) + 0.5) * res
        zz = (np.arange(nz) - k0 + 0.5) * res
        sp = np.where(solid, np.nan, np.hypot(ux, uz) / u_ref)
        ax.imshow(sp, origin="lower", cmap=cmap, vmin=0, vmax=1.1, interpolation="bilinear",
                  extent=(x[0] - res / 2, x[-1] + res / 2, zz[0] - res / 2, zz[-1] + res / 2), aspect="equal")
        face = int(round((0.0 - float(z[f"{sc}_x0_m"])) / res))
        for i in range(nx):
            col = solid[:, i]
            top = int(np.nonzero(col)[0].max()) if col.any() else -1
            if top < k0:
                continue
            is_tower = face <= i < face + TOWER_COLS and (top - k0 + 1) * res > 40
            ax.add_patch(Rectangle((x[i] - res / 2, 0), res, (top - k0 + 1) * res, linewidth=0,
                                   facecolor=AMBER if is_tower else BUILDING, zorder=3))
        ax.add_patch(Rectangle((X_RANGE[0], -res), X_RANGE[1] - X_RANGE[0], res, facecolor=BUILDING,
                               linewidth=0, zorder=3))
        U = np.ma.masked_where(solid, ux)
        V = np.ma.masked_where(solid, uz)
        XX, ZZ = np.meshgrid(x, zz)
        ax.streamplot(XX, ZZ, U, V, density=1.4, color="#2b3138", linewidth=0.55, arrowsize=0.7,
                      broken_streamlines=True, zorder=4)
        ax.streamplot(XX, ZZ, U, V, start_points=np.array(SEEDS), color=INK, linewidth=2.2, arrowsize=1.4,
                      integration_direction="forward", density=8, zorder=5)
        if sc == "today":
            # The 2017 model's own columns in this cut, as a dashed step outline.
            s17 = z["2017_solid"]
            g17 = int(z["2017_ground_k"].min())
            xs, ys = [], []
            for i in range(face, face + TOWER_COLS):
                col = s17[:, i]
                top = (int(np.nonzero(col)[0].max()) - g17 + 1) * res if col.any() else 0.0
                xs += [x[i] - res / 2, x[i] + res / 2]
                ys += [top, top]
            xs = [xs[0]] + xs + [xs[-1]]
            ys = [0.0] + ys + [0.0]
            ax.plot(xs, ys, linestyle=(0, (5, 4)), color=INK, linewidth=1.6, zorder=6)
            ax.annotate("the tower in 2017," + NL + "as the model had it", xy=(xs[-2], max(ys) * 0.8),
                        xytext=(x[face] + 130, max(ys) + 60), arrowprops=lead, **lab)
            # No height in the words: wind numbers stay off images until the
            # AIJ check passes (POST_BRIEF.md, section 4).
            ax.annotate("air from high on the windward face" + NL + "comes down to the street",
                        xy=(-35.0, 45.0), xytext=(-460.0, 290.0), arrowprops=lead, **lab)
        else:
            ax.annotate("the same air stays above the street," + NL + "turning in a slow eddy",
                        xy=(-40.0, 85.0), xytext=(-460.0, 290.0), arrowprops=lead, **lab)
        ax.text(X_RANGE[0] + 8, Z_TOP - 38, "wind from the north-west  →", **lab)
        ax.set_title(title, loc="left", fontsize=19, weight="semibold", color=INK, pad=8)
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(-res, Z_TOP)
        ax.set_ylabel("height above the street, m")
        if sc == "today":
            ax.set_xlabel("distance along the wind from today's windward face, m")
        else:
            ax.set_xticklabels([])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    cax = fig.add_axes([0.093, 0.140, 0.30, 0.011])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1.1))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=14.5)
    fig.text(0.42, 0.146, "wind speed in this plane," + NL + "as a share of the wind 300 m up", fontsize=14.5,
             color=MUTED, va="center", linespacing=1.3)
    fig.text(0.093, 0.012, "Time-mean wind, 8 m cells, lattice Boltzmann LES on a laptop" + NL +
             "GPU. Clear December day, Central Park typical year. 2017 LiDAR" + NL +
             "vs today. Lines follow the wind's components in this plane." + NL +
             "Model and figure: Adel Yasser Aglan. First-round model.", fontsize=14.5, color=MUTED,
             family="IBM Plex Mono", linespacing=1.35)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, f"post_b2_section_{datetime.now():%Y%m%d_%H%M%S}.png")
    fig.savefig(out, dpi=100, facecolor="white")
    from PIL import Image
    im = Image.open(out)
    print(out, im.size, f"{os.path.getsize(out) / 1e6:.2f} MB")
    im.convert("RGB").resize((400, 500), Image.LANCZOS).save(out.replace(".png", "_phone400.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
