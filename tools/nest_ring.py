"""
The street ring around the tower (20 to 80 m from its footprint, as
tools/wind_nest.py scores it), read from 2 m nest runs by several
statistics, because the median alone proved jumpy (NL_NOTES D22, D23).

    TWIN_CITY=rotterdam .venv-heat/Scripts/python -I tools/nest_ring.py table <box> [<box> ...]
        the change before -> today by median, mean, quartiles, p90
    TWIN_CITY=rotterdam .venv-heat/Scripts/python -I tools/nest_ring.py twins <box A> <box B>
        how far each statistic moves between two equally valid runs
        (another gust seed, another GPU)
    TWIN_CITY=rotterdam .venv-heat/Scripts/python -I tools/nest_ring.py noise <box A> <box B>
        D25: two seeds of a box whose runs saved their halves: do the halves
        gauge the seed gap, and how well is each change known

A box is a run folder name without its scenario, e.g.
dec_247.5_2m_fw_w600_avg6_sides_pzh; a path prefix "cloud:" reads
nest_cloud/cloud_copy/ instead of the nest folder. Runs that saved their
two half-run street layers also get each statistic's own noise: half the
gap between the change computed on each half (the two halves are
independent samples of the same flow; the full run is their average).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATS = {"median": np.median, "mean": np.mean,
         "p25": lambda v: np.percentile(v, 25), "p75": lambda v: np.percentile(v, 75),
         "p90": lambda v: np.percentile(v, 90)}


def folder(N, box, scenario):
    if box.startswith("cloud:"):
        return os.path.join(os.path.dirname(N.OUT), "nest_cloud", "cloud_copy", f"{scenario}_{box[6:]}")
    return os.path.join(N.OUT, f"{scenario}_{box}")


def ring_values(N, d, x, y, ring):
    """Ring cells' street speed for the full run and, when saved, each half."""
    from pipeline.wind_export import RunBox, resample
    with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
        m = json.load(f)
    z = np.load(os.path.join(d, "street.npz"))
    ok = (~z["roof"]) & np.isfinite(z["street_speed"])
    out = {"full": resample(z["street_speed"], ok, RunBox(m), x[ring], y[ring])}
    for h in ("street_half1", "street_half2"):
        if h in z.files:
            out[h] = resample(z[h], ok, RunBox(m), x[ring], y[ring])
    return out


def changes(N, box, ring_xy):
    """Change before -> today (%) by each statistic, and its noise from the halves when both years have them."""
    a = ring_values(N, folder(N, box, N.BEFORE), *ring_xy)
    b = ring_values(N, folder(N, box, "today"), *ring_xy)
    ok = np.isfinite(a["full"]) & np.isfinite(b["full"])
    res = {}
    for name, f in STATS.items():
        r = {"change": (f(b["full"][ok]) / f(a["full"][ok]) - 1) * 100}
        if "street_half1" in a and "street_half1" in b:
            c = [(f(b[h][ok]) / f(a[h][ok]) - 1) * 100 for h in ("street_half1", "street_half2")]
            r["noise"] = abs(c[0] - c[1]) / 2
        res[name] = r
    res["_n"] = int(ok.sum())
    return res


def table(args):
    import tools.wind_nest as N
    ring_xy = N.tower_ring()
    print(f"change {N.BEFORE} -> today on the street ring around {N.TOWER} (noise from the halves, when saved)")
    print(f"{'box':52s} " + " ".join(f"{s:>14s}" for s in STATS))
    for box in args.boxes:
        r = changes(N, box, ring_xy)
        cells = [f"{r[s]['change']:+6.1f}" + (f" +-{r[s]['noise']:4.1f}" if "noise" in r[s] else "      ")
                 for s in STATS]
        print(f"{box:52s} " + " ".join(f"{c:>14s}" for c in cells))


def twins(args):
    import tools.wind_nest as N
    ring_xy = N.tower_ring()
    ra, rb = changes(N, args.a, ring_xy), changes(N, args.b, ring_xy)
    out = {"a": args.a, "b": args.b, "stats": {}}
    print(f"{args.a}\n  vs {args.b}")
    for s in STATS:
        d = rb[s]["change"] - ra[s]["change"]
        out["stats"][s] = {"a": round(ra[s]["change"], 2), "b": round(rb[s]["change"], 2), "moved": round(d, 2)}
        print(f"  {s:6s} {ra[s]['change']:+6.1f}% vs {rb[s]['change']:+6.1f}%: moved {d:+5.1f} points")
    if args.save:
        with open(args.save, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)


def noise_check(args):
    """D25: do the halves gauge the noise? And is the mean change known to 5 points?"""
    import tools.wind_nest as N
    ring_xy = N.tower_ring()
    ra, rb = changes(N, args.a, ring_xy), changes(N, args.b, ring_xy)
    out = {"a": args.a, "b": args.b, "stats": {}}
    for s in STATS:
        if "noise" not in ra[s] or "noise" not in rb[s]:
            raise SystemExit(f"{s}: a run without saved halves")
        gap = rb[s]["change"] - ra[s]["change"]
        halves = float(np.sqrt((ra[s]["noise"] ** 2 + rb[s]["noise"] ** 2) / 2))
        noise = max(halves, abs(gap) / 2)
        out["stats"][s] = {"a": round(ra[s]["change"], 2), "b": round(rb[s]["change"], 2), "seed_gap": round(gap, 2),
                           "halves_sigma": round(halves, 2), "halves_gauge_ok": abs(gap) <= 2.5 * halves,
                           "noise": round(noise, 2), "both_seeds": round((ra[s]["change"] + rb[s]["change"]) / 2, 2)}
        print(f"  {s:6s} {ra[s]['change']:+6.1f}% / {rb[s]['change']:+6.1f}%: seed gap {gap:+5.1f}, halves sigma "
              f"{halves:4.1f} ({'gauge ok' if abs(gap) <= 2.5 * halves else 'halves miss it'}); "
              f"both seeds {out['stats'][s]['both_seeds']:+.1f}% +/- {noise:.1f}")
    m = out["stats"]["mean"]
    out["mean_within_5_points"] = m["noise"] <= 5.0
    if not out["mean_within_5_points"]:
        out["averaging_factor_for_5_points"] = round((m["noise"] / 5.0) ** 2, 2)
    print(f"  mean change {m['both_seeds']:+.1f}% +/- {m['noise']:.1f}: "
          + ("within 5 points" if out["mean_within_5_points"] else
             f"needs about {out['averaging_factor_for_5_points']}x the averaging for 5 points"))
    if args.save:
        with open(args.save, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    t = sub.add_parser("table")
    t.add_argument("boxes", nargs="+")
    w = sub.add_parser("twins")
    w.add_argument("a")
    w.add_argument("b")
    w.add_argument("--save", help="write the comparison as JSON here")
    n = sub.add_parser("noise")
    n.add_argument("a")
    n.add_argument("b")
    n.add_argument("--save", help="write the verdict as JSON here")
    args = ap.parse_args()
    {"table": table, "twins": twins, "noise": noise_check}[args.what](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
