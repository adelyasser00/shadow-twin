"""
Put a 2017 run and a today run side by side, cell by cell.

    python tools/compare_runs.py data/runs/cp_2017.npz data/runs/cp_today.npz
    python tools/compare_runs.py data/runs/cp_2017.npz data/runs/cp_today.npz --csv data/runs/profile.csv

Both files come from run_nyc --save-rasters, on the same frame. The viewer
shows two pictures; this answers the question the pictures raise: how much of
the park lost its sun, and where.

  never in sun     share and hectares of the park with no direct sun at all
                   inside the CEQR window, in each run
  newly dark       hectares that had some sun in 2017 and have none today
  lost 1 h or more hectares that lost at least an hour of direct sun
  mean change      average change in sun hours per park cell

Then the same, band by band, north from the park's south edge, which is where
the towers are. If the change is the towers, it should fade with distance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return z, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", help="the 2017 run's .npz")
    ap.add_argument("after", help="the today run's .npz")
    ap.add_argument("--step", type=float, default=500.0, help="band width, metres")
    ap.add_argument("--csv", default=None, help="also write the bands to a CSV")
    args = ap.parse_args()

    a, ma = load(args.before)
    b, mb = load(args.after)
    if ma["frame"] != mb["frame"] or ma["cellsize"] != mb["cellsize"]:
        sys.exit("these two runs are on different frames; run both with the same arguments")
    if "resource" not in a.files:
        sys.exit("no resource in these runs; pass --resource to run_nyc")
    park = a["resource"] & b["resource"]
    cell_ha = ma["cellsize"] ** 2 / 1e4
    name = ma.get("resource_name", "the park")
    days = [k[4:] for k in a.files if k.startswith("sun_") and k in b.files]
    rows_m = np.asarray(ma["profile_rows"], dtype=np.float64)

    print(f"{name}: {park.sum() * cell_ha:,.1f} ha on the same frame, "
          f"{ma['scenario']} vs {mb['scenario']}\n")
    print(f"  {'day':<6} {'never in sun':>22}   {'newly dark':>10}   {'lost 1 h+':>9}   {'mean change':>11}")
    out_rows = []
    for d in days:
        sa, sb = a[f"sun_{d}"][park], b[f"sun_{d}"][park]
        na, nb = sa < 0.25, sb < 0.25
        newly = (~na & nb).sum() * cell_ha
        lost1 = ((sa - sb) >= 1.0).sum() * cell_ha
        print(f"  {d:<6} {na.mean()*100:5.1f}% -> {nb.mean()*100:5.1f}%  "
              f"({na.sum()*cell_ha:5.1f} -> {nb.sum()*cell_ha:5.1f} ha)   "
              f"{newly:7.1f} ha   {lost1:6.1f} ha   {(sb - sa).mean():+8.2f} h")
        if "published" in a.files and "published" in b.files:
            pa = a[f"sun_{d}"][a["published"]] < 0.25
            pb = b[f"sun_{d}"][b["published"]] < 0.25
            print(f"         on the published 700 m frame's ground: "
                  f"{pa.mean()*100:5.1f}% -> {pb.mean()*100:5.1f}%")
        for k in range(int(np.ceil(rows_m.max() / args.step))):
            band = (rows_m >= k * args.step) & (rows_m < (k + 1) * args.step)
            cells = park & band[:, None]
            if cells.sum() * cell_ha < 0.5:
                continue
            ba, bb = a[f"sun_{d}"][cells], b[f"sun_{d}"][cells]
            out_rows.append({"day": d, "from_m": int(k * args.step),
                             "to_m": int((k + 1) * args.step),
                             "area_ha": round(float(cells.sum() * cell_ha), 2),
                             "never_before": round(float((ba < 0.25).mean()), 4),
                             "never_after": round(float((bb < 0.25).mean()), 4),
                             "lost_1h_ha": round(float(((ba - bb) >= 1.0).sum() * cell_ha), 2),
                             "mean_change_h": round(float((bb - ba).mean()), 3)})

    first = days[0]
    print(f"\n  {first}, by distance north of the park's south edge:")
    print(f"  {'band, m':<13} {'never in sun':>18}   {'lost 1 h+':>9}   {'mean change':>11}")
    for r in (r for r in out_rows if r["day"] == first):
        print(f"  {r['from_m']:>5}-{r['to_m']:<6} {r['never_before']*100:6.1f}% -> "
              f"{r['never_after']*100:5.1f}%   {r['lost_1h_ha']:6.1f} ha   {r['mean_change_h']:+8.2f} h")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\n  wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
