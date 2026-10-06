"""
The numbers a wind post could use, each with its plain meaning.

    .venv-heat/Scripts/python tools/wind_story.py [direction]

For the hero direction (default 315, the December day): for each tower that
grew since 2017, the street-level wind on open ground within 20 to 80 m of
its footprint, 2017 against today, and the same for the park's south edge.
All as ratios to the park's open lawns, and in m/s for the December day's
wind. Writes data/wind/wind_story.csv and prints it.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pipeline.run_wind import RUNS_DIR, run_name  # noqa: E402
from pipeline.wind_export import heat_points  # noqa: E402

NAMES = {(-73.98102, 40.76644): "Central Park Tower",
         (-73.97756, 40.76496): "Steinway Tower",
         (-73.97815, 40.76183): "53 W 53rd"}


def main():
    from pyproj import Transformer
    from scipy import ndimage
    d = float(sys.argv[1]) if len(sys.argv) > 1 else 315.0
    x, y, layers, masks, meta = heat_points()
    k = {sc: np.load(os.path.join(RUNS_DIR, run_name(sc, "dec", d, 8.0), "k_heatgrid.npy"))
         for sc in ("2017", "today")}
    with open(os.path.join(HERE, "data", "heat", "compare", "towers.json"), encoding="utf-8") as f:
        towers = json.load(f)
    with open(os.path.join(HERE, "viewer", "wind", "windheat.json"), encoding="utf-8") as f:
        wh = json.load(f)
    dec = next(s for s in wh["seasons"] if s["id"] == "dec")
    ws = float(np.median([st["wind_m_s"] for st in dec["steps"]]))
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32118", always_xy=True).transform
    bld = masks["building"]
    lab, _ = ndimage.label(bld)
    open_ground = ~bld & ~masks["water"]
    rows = []

    def add(key, value, unit, meaning):
        rows.append({"key": key, "value": round(float(value), 3), "unit": unit, "meaning": meaning})

    for t in towers:
        nm = next((v for (lo, la), v in NAMES.items()
                   if abs(lo - t["lonlat"][0]) < 5e-4 and abs(la - t["lonlat"][1]) < 4e-4), None)
        if not nm:
            continue
        tx, ty = tf(*t["lonlat"])
        r = int(round((y[0, 0] - ty) / 2.0))
        c = int(round((tx - x[0, 0]) / 2.0))
        fp = lab == lab[r, c] if lab[r, c] else None
        if fp is None:
            continue
        dist = ndimage.distance_transform_edt(~fp) * 2.0
        ring = open_ground & (dist >= 20) & (dist <= 80)
        a, b = np.nanmedian(k["2017"][ring]), np.nanmedian(k["today"][ring])
        a90, b90 = np.nanpercentile(k["2017"][ring], 90), np.nanpercentile(k["today"][ring], 90)
        key = nm.lower().replace(" ", "_")
        add(f"{key}_street_ratio_2017", a, "x lawn", f"street wind 20-80 m from {nm}, 2017, vs the open lawns")
        add(f"{key}_street_ratio_today", b, "x lawn", f"same, today")
        add(f"{key}_street_change_pct", (b / a - 1) * 100, "%", "change since 2017 of that median")
        add(f"{key}_street_p90_2017_m_s", a90 * ws, "m/s", f"windiest tenth around {nm}, 2017, Dec day wind")
        add(f"{key}_street_p90_today_m_s", b90 * ws, "m/s", "same, today")
    park = masks["report"]
    for band, (lo, hi) in {"south300": (0, 300), "south_all": (0, 1500)}.items():
        m = park & (masks["dist59"] >= lo) & (masks["dist59"] < hi)
        a, b = np.nanmedian(k["2017"][m]), np.nanmedian(k["today"][m])
        add(f"park_{band}_ratio_2017", a, "x lawn", f"park street wind, {lo}-{hi} m north of 59th, 2017")
        add(f"park_{band}_ratio_today", b, "x lawn", "same, today")
        add(f"park_{band}_change_pct", (b / a - 1) * 100, "%", "change since 2017")
    add("dec_day_station_wind_m_s", ws, "m/s", "median station wind over the reported hours")
    add("dec_ha_5c_colder_local_wind", dec["ha_any_hour_le_-5"], "ha",
        "park 5 C+ colder than 2017 for an hour or more, wind at each spot")
    add("dec_ha_5c_colder_station_wind", dec["ha_any_hour_le_-5_uniform"], "ha",
        "the same with one wind for the park (the published heat number's method)")
    out = os.path.join(HERE, "data", "wind", "wind_story.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "value", "unit", "meaning"])
        w.writeheader()
        w.writerows(rows)
    for r_ in rows:
        print(f"{r_['key']:42s} {r_['value']:>9} {r_['unit']:7s} {r_['meaning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
