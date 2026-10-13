"""
The wind numbers for a city, with New York's noise rule.

    TWIN_CITY=rotterdam .venv-heat/Scripts/python tools/city_wind_story.py

Reads the runs' street-level ratio maps on the heat grid (k_heatgrid.npy,
written by pipeline.wind_export), smooths each over 20 m as New York does,
and takes the change today / before - 1 for every direction run in both
years.

Noise bar (tools/post_wind_data.py for New York): open ground more than
QUIET_M from every cell that changed between the years, split into lawns and
streets; the bar is three times the larger of the two standard deviations of
the change there. If too little ground is that far from any change (a city
that grew everywhere), the distance halves until there is at least MIN_HA,
and the notes say so.

Per tower that grew (70 m or more): the median change on open ground 20 to
80 m from its footprint, New York's "street wind at the foot of Central Park
Tower". 8 m cells put one to three cells across that ring, so this is a
pattern, not a pedestrian comfort number (the wind tunnel check failed at
8 m; see the README).

Writes <wind dir>/wind_story.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pipeline import city  # noqa: E402

QUIET_M = 500.0
MIN_HA = 5.0
SIGMAS = 3.0
SMOOTH_CELLS = 10
TOWER_MIN_M = 70.0
# New York's measured bar. A city's own measurement can come out near zero
# when its quiet ground lies upwind of every change: two deterministic runs
# are identical there, so the spread is not noise at all. The bar used is the
# larger of the city's own and New York's.
NYC_BAR = 0.35
RING_M = (20.0, 80.0)


def smooth(a, ok):
    from scipy.ndimage import gaussian_filter
    num = gaussian_filter(np.where(ok, a, 0.0), SMOOTH_CELLS)
    den = gaussian_filter(ok.astype(np.float32), SMOOTH_CELLS)
    return np.where(ok & (den > 0.3), num / np.maximum(den, 1e-6), np.nan)


def main():
    from scipy import ndimage
    from pipeline.heat_inputs import load
    from pipeline.wind_export import runs_available
    from pyproj import Transformer
    layers, masks, meta = load()
    before = city.before()
    runs = [m for m in runs_available() if m["res_m"] == 8.0 and "small" not in m["_name"]]
    by = {(m["scenario"], m["season"], m["theta_from_deg"]): m for m in runs}
    open_ground = ~masks["building"] & ~masks["water"]
    d_changed = ndimage.distance_transform_edt(~masks["changed"]) * 2.0
    quiet_m = QUIET_M
    while quiet_m > 50 and (open_ground & (d_changed > quiet_m)).sum() * 4 / 1e4 < MIN_HA:
        quiet_m /= 2
    far = d_changed > quiet_m
    quiet_lawn = masks["park"] & open_ground & far
    quiet_streets = open_ground & ~masks["park"] & far
    report = masks["report"]
    t = meta["_transform"]
    to_m = Transformer.from_crs("EPSG:4326", meta["grid"]["crs_build"], always_xy=True).transform
    lab = masks["labels"]
    out = {"city": city.city_id(), "quiet_m": quiet_m,
           "quiet_ha": {"lawns": round(float(quiet_lawn.sum() * 4 / 1e4), 1),
                        "streets": round(float(quiet_streets.sum() * 4 / 1e4), 1)},
           "directions": []}
    print(f"quiet ground, over {quiet_m:.0f} m from any change: lawns {out['quiet_ha']['lawns']} ha, "
          f"streets {out['quiet_ha']['streets']} ha")
    for (sc, se, th), m in sorted(by.items()):
        if sc != "today" or (before, se, th) not in by:
            continue
        k = {}
        for s, mm in (("today", m), (before, by[(before, se, th)])):
            a = np.load(os.path.join(mm["_dir"], "k_heatgrid.npy"))
            k[s] = smooth(a, np.isfinite(a) & ~masks["building"])
        with np.errstate(invalid="ignore", divide="ignore"):
            change = k["today"] / np.maximum(k[before], 0.02) - 1.0
        sd_l = float(np.nanstd(change[quiet_lawn])) if quiet_lawn.any() else 0.0
        sd_s = float(np.nanstd(change[quiet_streets])) if quiet_streets.any() else 0.0
        bar_measured = SIGMAS * max(sd_l, sd_s)
        bar = max(bar_measured, NYC_BAR)
        rep = change[report]
        rec = {"season": se, "dir": th, "noise_lawns_pct": round(sd_l * 100, 1),
               "noise_streets_pct": round(sd_s * 100, 1), "bar_measured_pct": round(bar_measured * 100, 1),
               "bar_pct": round(bar * 100, 1),
               "report_ha_windier_above_bar": round(float((rep > bar).sum() * 4 / 1e4), 2),
               "report_ha_calmer_below_bar": round(float((rep < -bar).sum() * 4 / 1e4), 2),
               "report_median_pct": round(float(np.nanmedian(rep)) * 100, 1),
               "towers": []}
        print(f"\n{se} {th:.1f} deg: noise lawns {rec['noise_lawns_pct']}%, streets "
              f"{rec['noise_streets_pct']}%, measured bar {rec['bar_measured_pct']}%, bar used "
              f"{rec['bar_pct']}%; open ground of the window: "
              f"{rec['report_ha_windier_above_bar']} ha windier above the bar, "
              f"{rec['report_ha_calmer_below_bar']} ha calmer, median {rec['report_median_pct']}%")
        for b in meta["grown_buildings"]:
            if b["height_m"] < TOWER_MIN_M:
                continue
            x, y = to_m(*b["lonlat"])
            r, c = int((y - t.f) / t.e), int((x - t.c) / t.a)
            i = int(lab[r, c])
            if not i:
                continue
            fp = lab == i
            d = ndimage.distance_transform_edt(~fp) * 2.0
            ring = (d >= RING_M[0]) & (d <= RING_M[1]) & open_ground
            v = change[ring]
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            med = float(np.median(v))
            rec["towers"].append({"height_m": b["height_m"], "year": b["year"], "lonlat": b["lonlat"],
                                  "ring_ha": round(float(ring.sum() * 4 / 1e4), 2),
                                  "median_pct": round(med * 100, 1),
                                  "p90_pct": round(float(np.percentile(v, 90)) * 100, 1),
                                  "above_bar": bool(abs(med) > bar)})
            tw = rec["towers"][-1]
            print(f"  {b['height_m']:6.1f} m, built {b['year']}, {b['lonlat'][1]:.5f}, {b['lonlat'][0]:.5f}: "
                  f"20-80 m from it median {tw['median_pct']:+.1f}%, windiest tenth {tw['p90_pct']:+.1f}% "
                  f"({'above' if tw['above_bar'] else 'inside'} the noise bar)")
        out["directions"].append(rec)
    path = os.path.join(city.wind_dir(), "wind_story.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
