"""
How tall each grown tower was in the 2017 LiDAR, with one stated definition.

    .venv-heat/Scripts/python tools/tower_heights.py

Definition: the 2017 height of a tower is the highest level that covers at
least a tenth of its footprint (the cells whose surface changed between 2017
and today): the highest height h whose +-5 m band holds 10% of those cells,
reported as the median of that band. Thin things on top, like a construction
crane's mast and jib, cover far less than a tenth and are ignored.

Why not the 99th percentile the heat post used: in May 2017 Central Park Tower
and 53 West 53rd Street had tower cranes above their working floors, and the
99th percentile lands on the crane (172 and 173 m), not on the building (124
and 144 m). The highest return of all is reported too, so nothing is hidden.

It also checks every building the heat layer lists as grown since 2017
(data/heat/compare/towers.json): it grew only if today's height is more than
GREW_M above the highest 2017 LiDAR return within 10 m of its changed cells.
The grown-building rule can raise a few edge cells of a building that was
already there (432 Park Avenue, finished in 2015, is one); those are listed
as not grown, and only grown ones are drawn amber in the posts.

Writes data/wind/tower_heights_2017.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

TOWERS = [((-73.98102, 40.766438), "Central Park Tower"),
          ((-73.977562, 40.76496), "Steinway Tower"),
          ((-73.978155, 40.761827), "53 W 53rd")]
SHARE = 0.10
BAND_M = 5.0
GREW_M = 15.0


def roof_level(v):
    """Highest level whose +-BAND_M band holds SHARE of the heights v."""
    for h in np.arange(v.max(), 0, -0.5):
        w = (v >= h - BAND_M) & (v <= h + BAND_M)
        if w.mean() >= SHARE:
            return float(np.median(v[w]))
    return float(np.median(v))


def main():
    from pyproj import Transformer
    from scipy import ndimage
    from pipeline.heat_inputs import load
    layers, masks, meta = load()
    h17 = layers["dsm_2017"] - layers["dem"]
    hnow = layers["dsm_today"] - layers["dem"]
    lab, _ = ndimage.label(masks["changed"], structure=np.ones((3, 3)))
    t = meta["_transform"]
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32118", always_xy=True).transform
    with open(os.path.join(HERE, "data", "heat", "compare", "towers.json"), encoding="utf-8") as f:
        published = json.load(f)
    out = {}
    for (lo, la), name in TOWERS:
        x, y = tf(lo, la)
        r, c = int((y - t.f) / t.e), int((x - t.c) / t.a)
        cells = lab == lab[r, c]
        v = h17[cells]
        pub = next((p["lidar_p99_m"] for p in published
                    if abs(p["lonlat"][0] - lo) < 5e-4 and abs(p["lonlat"][1] - la) < 4e-4), None)
        out[name] = {"roof_2017_m": round(roof_level(v)), "highest_2017_m": round(float(v.max())),
                     "published_p99_m": pub, "today_m": round(float(hnow[cells].max())),
                     "footprint_m2": int(cells.sum() * 4)}
        print(f"{name:20s} 2017 roof {out[name]['roof_2017_m']:4d} m (highest return "
              f"{out[name]['highest_2017_m']} m, published p99 {pub} m), today {out[name]['today_m']} m")
    checked = []
    for p in published:
        x, y = tf(*p["lonlat"])
        r, c = int((y - t.f) / t.e), int((x - t.c) / t.a)
        comp = lab == lab[r, c]
        around = ndimage.binary_dilation(comp, iterations=5)
        top17 = float(h17[around].max())
        now = float(hnow[comp].max())
        checked.append({"lonlat": p["lonlat"], "today_m": round(now), "highest_2017_within_10m": round(top17),
                        "changed_cells": int(comp.sum()), "dec_ha_any_hour_5": p["dec_ha_any_hour_5"],
                        "grew": bool(now - top17 > GREW_M)})
        print(f"  listed {p['lonlat']}: today {now:.0f} m, 2017 highest nearby {top17:.0f} m -> "
              f"{'grew' if checked[-1]['grew'] else 'NOT grown'}")
    out["_checked_grown_list"] = checked
    with open(os.path.join(HERE, "data", "wind", "tower_heights_2017.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
