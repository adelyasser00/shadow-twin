"""
Diagnostic figures for the wind runs: how the wind grows with height above
Sheep Meadow, and the street-level wind ratio, for every set of runs kept.

    .venv-heat/Scripts/python tools/wind_diagnostics.py [run name]

Writes data/wind/figures/diag_<run>.png. For internal checks, not for posts.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pipeline.wind_export import RunBox, heat_points, sheep_meadow  # noqa: E402

SETS = [("runs_noslip", "no-slip walls, dense trees"),
        ("runs_lad03", "slip walls, dense trees"),
        ("runs", "slip walls, park trees (final)")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    name = sys.argv[1] if len(sys.argv) > 1 else "today_dec_315.0_8m"
    x, y, layers, masks, meta = heat_points()
    cx, cy, _, _ = sheep_meadow(layers, masks, meta, x, y)
    fig, axes = plt.subplots(1, 1 + len(SETS), figsize=(5 + 5 * len(SETS), 6))
    ax = axes[0]
    for sub, label in SETS:
        d = os.path.join(HERE, "data", "wind", sub, name)
        if not os.path.exists(os.path.join(d, "run.json")):
            continue
        with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
            m = json.load(f)
        z = np.load(os.path.join(d, "field.npz"))
        b = RunBox(m)
        i, j = b.from_map(cx, cy)
        i, j = int(round(float(i))), int(round(float(j)))
        sp = z["speed"].astype(np.float32)
        kf = int(z["kfirst"][j, i])
        prof = sp[kf:, j - 1:j + 2, i - 1:i + 2].mean(axis=(1, 2))
        h = (np.arange(len(prof)) + 0.5) * b.res
        u300 = np.interp(300.0, h, prof)
        ax.plot(prof / u300, h, label=label)
    hh = np.linspace(2, 500, 100)
    for a in (0.2, 0.45):
        ax.plot((hh / 300.0) ** a, hh, "k:", lw=0.8)
    ax.set_ylim(0, 500)
    ax.set_xlim(0, 1.4)
    ax.set_xlabel("wind speed / wind at 300 m")
    ax.set_ylabel("height above the lawn, m")
    ax.set_title("Above Sheep Meadow (dotted: city power laws 0.2, 0.45)")
    ax.legend(fontsize=8)
    for k, (sub, label) in enumerate(SETS):
        a2 = axes[1 + k]
        p = os.path.join(HERE, "data", "wind", sub, name, "k_heatgrid.npy")
        if os.path.exists(p):
            km = np.load(p)
            im = a2.imshow(km[::2, ::2], vmin=0, vmax=2.5, cmap="magma")
            plt.colorbar(im, ax=a2, fraction=0.04, label="street wind / open lawn")
        a2.set_title(label)
        a2.axis("off")
    out = os.path.join(HERE, "data", "wind", "figures")
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, f"diag_{name}.png")
    fig.tight_layout()
    fig.savefig(fn, dpi=90)
    print("wrote", fn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
