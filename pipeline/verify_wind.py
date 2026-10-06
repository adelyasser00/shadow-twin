"""
Checks for the wind layer. Thresholds were fixed before any result was seen.

    .venv-heat/Scripts/python -m pipeline.verify_wind

  W0  felt temperature recomputed with the station wind everywhere
      reproduces SOLWEIG's saved UTCI: worst hourly median difference under
      0.05 C (run by pipeline/wind_heat.py, read back here).
  W1  convergence: the two halves of the averaging window, street level:
      median |difference| at most 10% of the median street speed, every run.
  W2  grid: the hero direction at 6 m and at 8 m on the same smaller box.
      Over the park, correlation of the street wind ratio at least 0.8 and
      park median ratio within 15%.
  W3  open-lawn profile: above Sheep Meadow the wind grows with height up to
      200 m, and wind at 12 m over wind at 300 m is between 0.2 and 0.6, what
      city power laws with exponents from 0.2 to 0.45 give.
  W4  trees: the median wind ratio under park canopy is below the one on
      open park lawn.
  W5  plausibility: the 99th percentile of the street wind ratio over the
      frame is at most 3.0. Published speed-ups at the foot of tall buildings
      reach about 2 to 3 times the open-ground wind.

Results go to data/wind/verify/gates.json.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .wind_domain import WIND_DIR
from .wind_export import (RunBox, heat_points, lawn_reference, load_run, open_lawn, resample,
                          runs_available, sheep_meadow)

VERIFY_DIR = os.path.join(WIND_DIR, "verify")
GATES_JSON = os.path.join(VERIFY_DIR, "gates.json")


def main():
    from .heat_export import heat_grid
    runs = runs_available()
    x, y, layers, masks, meta = heat_points()
    cx, cy, _, _ = sheep_meadow(layers, masks, meta, x, y)
    grid = heat_grid(meta, masks)
    r0, r1, c0, c1 = grid.frame
    report = masks["report"]
    gates = {}

    # W0
    p = os.path.join(os.path.dirname(WIND_DIR), "..", "viewer", "wind", "windheat.json")
    p = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "viewer", "wind", "windheat.json"))
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            g = json.load(f).get("gate_w0", {})
        gates["W0"] = {"worst_median_abs_diff_c": g.get("W0_median_abs_diff_c"), "pass": g.get("pass")}

    # W1
    w1 = {}
    for m in runs:
        c = m.get("convergence_street_speed", {})
        v = c.get("median_abs_diff_over_median_speed")
        if v is not None:
            w1[m["_name"]] = v
    gates["W1"] = {"runs": w1, "worst": max(w1.values()) if w1 else None,
                   "pass": bool(w1) and max(w1.values()) <= 0.10}

    # W2
    small = {m["res_m"]: m for m in runs if m["_name"].endswith("_small")}
    if 6.0 in small and 8.0 in small:
        k = {}
        for res, m in small.items():
            run = load_run(m)
            ref = lawn_reference(run, x, y, open_lawn(layers, masks))
            k[res] = resample(run["street_speed"], run["ok"], run["box"], x[report], y[report]) / ref
        ok = np.isfinite(k[6.0]) & np.isfinite(k[8.0])
        corr = float(np.corrcoef(k[6.0][ok], k[8.0][ok])[0, 1])
        med6, med8 = float(np.median(k[6.0][ok])), float(np.median(k[8.0][ok]))
        gates["W2"] = {"correlation": round(corr, 3), "park_median_6m": round(med6, 3),
                       "park_median_8m": round(med8, 3), "rel_diff": round(abs(med6 - med8) / med6, 3),
                       "pass": corr >= 0.8 and abs(med6 - med8) / med6 <= 0.15}

    # W3, W4, W5 on the hero runs
    for m in runs:
        if m["_name"] not in ("today_dec_315.0_8m", "2017_dec_315.0_8m"):
            continue
        run = load_run(m)
        b = run["box"]
        i, j = b.from_map(cx, cy)
        i, j = int(round(float(i))), int(round(float(j)))
        sp = run["speed"].astype(np.float32)
        kf = int(run["kfirst"][j, i])
        prof = [float(sp[k, j - 1:j + 2, i - 1:i + 2].mean()) for k in range(kf, sp.shape[0])]
        h = (np.arange(len(prof)) + 0.5) * b.res
        u12 = float(np.interp(12.0, h, prof))
        u300 = float(np.interp(300.0, h, prof))
        mono = all(np.diff(np.array(prof)[h <= 200]) > -0.02 * max(prof))
        name = m["scenario"]
        gates[f"W3_{name}"] = {"u12_over_u300": round(u12 / u300, 3), "grows_to_200m": bool(mono),
                               "profile_m": [round(v / u300, 3) for v in prof[:40]],
                               "pass": bool(mono) and 0.2 <= u12 / u300 <= 0.6}
        kmap = np.load(os.path.join(m["_dir"], "k_heatgrid.npy"))
        canopy = layers["cdsm"] >= 2.0
        lawn = report & ~canopy
        under = report & canopy
        kl, ku = float(np.nanmedian(kmap[lawn])), float(np.nanmedian(kmap[under]))
        gates[f"W4_{name}"] = {"median_open_lawn": round(kl, 3), "median_under_canopy": round(ku, 3),
                               "pass": ku < kl}
        fr = kmap[r0:r1, c0:c1]
        fr = fr[np.isfinite(fr)]
        p99 = float(np.percentile(fr, 99))
        gates[f"W5_{name}"] = {"p99_ratio": round(p99, 3), "max_ratio": round(float(fr.max()), 3),
                               "pass": p99 <= 3.0}
    os.makedirs(VERIFY_DIR, exist_ok=True)
    with open(GATES_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(gates, f, indent=1)
    for k, v in gates.items():
        short = {a: b for a, b in v.items() if a not in ("runs", "profile_m")}
        print(f"{k}: {'PASS' if v.get('pass') else 'FAIL'}  {short}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
