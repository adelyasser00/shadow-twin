"""
What ground do your LiDAR tiles cover, and does it cover the run you want?

    python -m pipeline.tile_info --laz data/nyc
    python -m pipeline.tile_info --laz data/nyc --preset central-park --resource data/nyc/park.geojson
    python -m pipeline.tile_info --laz data/nyc --lat 40.7668 --lon -73.9790 --span 700
    python -m pipeline.tile_info --laz data/nyc --scan

Checks the exact grid run_nyc will build from the same arguments, frame and
buffer both, and names the tiles you are missing.

How coverage is worked out, and what the old version got wrong
--------------------------------------------------------------
NYC's 2017 tiles are 2500 ft squares on the state plane grid (EPSG:2263),
named by their lower-left corner in thousands of feet: 990217 starts at
x 990,000 ft, y 217,500 ft. Each file's header carries its own bounding box.

The first version of this tool merged every header into one big rectangle and
tested the target against that. With an L-shaped set of tiles, the rectangle
includes the missing corner, so it reported ground as covered that was not.
That is what happened with the first 700 m run. It was put down to the tiles
being rotated. It was the union.

This version tests each tile's own box, cell by cell, and names each missing
tile from the naming scheme so you can search for it on the downloader.

--scan goes further and reads the points themselves, then reports what share
of each tile's box actually has returns. It settles the rotation question for
good: an axis-aligned tile fills its box apart from water, a rotated one leaves
four empty corners. It reads every point, so it takes about as long as a run.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

TILE_FT = 2500.0


def tile_name(x, y, size=TILE_FT):
    """NYC 2017 tile name for a point in EPSG:2263 feet."""
    xt = math.floor(x / size) * size
    yt = math.floor(y / size) * size
    return f"{int(xt // 1000):03d}{int(yt // 1000):03d}"


def _wgs(crs, xmin, ymin, xmax, ymax):
    from pyproj import Transformer
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
    return min(lon), min(lat), max(lon), max(lat)


def _pattern(tiles):
    """Print the tiles you have as a small map, north up."""
    xs = sorted({math.floor(t["xmin"] / TILE_FT + 0.5) for t in tiles})
    ys = sorted({math.floor(t["ymin"] / TILE_FT + 0.5) for t in tiles})
    have = {(math.floor(t["xmin"] / TILE_FT + 0.5), math.floor(t["ymin"] / TILE_FT + 0.5))
            for t in tiles}
    xs = list(range(xs[0] - 1, xs[-1] + 2))
    ys = list(range(ys[0] - 1, ys[-1] + 2))
    print("\nyour tiles, north up (# have, . missing):")
    print("         " + "".join(f"{int(x * TILE_FT // 1000):>5}" for x in xs))
    for y in reversed(ys):
        row = "".join(f"{'#' if (x, y) in have else '.':>5}" for x in xs)
        print(f"  {int(y * TILE_FT // 1000):>5}  {row}")


def _scan(tiles, log=print, block_ft=65.0):
    """Real coverage per tile, from the points, on a coarse block grid."""
    from . import laz
    out = {}
    for t in tiles:
        nx = int(math.ceil((t["xmax"] - t["xmin"]) / block_ft))
        ny = int(math.ceil((t["ymax"] - t["ymin"]) / block_ft))
        occ = np.zeros((ny, nx), dtype=bool)
        with laz._open(t["path"]) as f:
            for pts in f.chunk_iterator(4_000_000):
                c = ((np.asarray(pts.x) - t["xmin"]) / block_ft).astype(np.int64).clip(0, nx - 1)
                r = ((t["ymax"] - np.asarray(pts.y)) / block_ft).astype(np.int64).clip(0, ny - 1)
                occ[r, c] = True
        k = max(1, ny // 8)
        corners = [occ[:k, :k].mean(), occ[:k, -k:].mean(), occ[-k:, :k].mean(), occ[-k:, -k:].mean()]
        out[t["name"]] = {"fill": float(occ.mean()), "corners": corners, "occ": occ}
        log(f"  {t['name']:<10} returns in {occ.mean()*100:5.1f}% of its box, corners "
            + " ".join(f"{c*100:3.0f}%" for c in corners))
    return out


def main():
    from .run_nyc import PRESETS
    from . import grid as G, laz

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laz", required=True, help="folder of .laz/.las, or one file")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--resource", default=None)
    ap.add_argument("--resource-where", default=None)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--span", type=float, default=700.0)
    ap.add_argument("--length", type=float, default=None)
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--bearing", type=float, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--buffer", type=float, default=None)
    ap.add_argument("--res", type=float, default=2.0)
    ap.add_argument("--scan", action="store_true",
                    help="read the points and report real coverage per tile (slow)")
    args = ap.parse_args()

    info = laz.probe(args.laz)
    crs, unit = info["crs"], info["unit_m"]
    tiles = info["tiles"]

    print(f"CRS {crs.to_string()}, 1 unit = {unit:.6f} m\n")
    print(f"{'tile':<10}{'points':>14}   {'x range, ft':<19}  {'y range, ft':<19}  lat range")
    print("-" * 92)
    named_ok = 0
    for t in tiles:
        w, s, e, n = _wgs(crs, t["xmin"], t["ymin"], t["xmax"], t["ymax"])
        guess = tile_name(0.5 * (t["xmin"] + t["xmax"]), 0.5 * (t["ymin"] + t["ymax"]))
        named_ok += guess == t["name"]
        flag = "" if guess == t["name"] else f"   (box says {guess})"
        print(f"{t['name']:<10}{t['points']:>14,}   {t['xmin']:>8.0f}-{t['xmax']:<8.0f}   "
              f"{t['ymin']:>8.0f}-{t['ymax']:<8.0f}   {s:.4f}-{n:.4f}{flag}")
    print("-" * 92)
    print(f"{'total':<10}{sum(t['points'] for t in tiles):>14,} points in {len(tiles)} tiles")
    scheme = named_ok == len(tiles)
    if scheme:
        print("every file name matches its box on the 2500 ft grid, so missing tiles "
              "can be named")
        for t in tiles:
            w_ = t["xmax"] - t["xmin"]
            h_ = t["ymax"] - t["ymin"]
            if w_ > TILE_FT * 1.02 or h_ > TILE_FT * 1.02:
                print(f"  note: {t['name']} box is {w_:.0f} x {h_:.0f} ft, bigger than a tile")
    else:
        print(f"only {named_ok} of {len(tiles)} file names match the 2500 ft scheme; "
              "missing tiles will be shown as coordinates")
    _pattern(tiles)

    scan = None
    if args.scan:
        print("\nscanning points for real coverage (reads every point)")
        scan = _scan(tiles)
        low = [k for k, v in scan.items() if min(v["corners"]) < 0.5]
        if low:
            print(f"  tiles with an empty corner: {', '.join(low)}. If their other corners "
                  "are full, look for water or a survey edge there before blaming rotation.")
        else:
            print("  every tile has returns in all four corners: the tiles are squares on "
                  "the state plane grid, not rotated.")

    # ---- the run to check ---------------------------------------------
    preset = PRESETS.get(args.preset, {})
    bearing = args.bearing if args.bearing is not None else preset.get("bearing", 0.0)
    margin = args.margin if args.margin is not None else preset.get("margin", 250.0)
    buffer_m = args.buffer if args.buffer is not None else preset.get("buffer", 0.0)
    fit = preset.get("frame_from_resource", False)
    geoms = None
    if fit:
        if not args.resource:
            raise SystemExit("--preset central-park needs --resource (the park polygon)")
        from . import resource as R
        geoms, names = R.load(args.resource, args.resource_where)
        R.check_fit(geoms, names, args.resource_where)
    elif args.lat is None or args.lon is None:
        print("\nPass --preset central-park --resource <park.geojson>, or --lat/--lon, "
              "to check a run.")
        return 0

    g = G.build(crs, unit, args.res, lat=args.lat, lon=args.lon, span_m=args.span,
                length_m=args.length, width_m=args.width, bearing_deg=bearing,
                buffer_m=buffer_m, margin_m=margin, fit_geoms=geoms)
    L, Wd = g.frame_size_m()
    print(f"\nrun: frame {L:,.0f} m by {Wd:,.0f} m, long axis {g.bearing_deg:.1f} deg, "
          f"centred {g.centre_lat:.5f}, {g.centre_lon:.5f}")
    print(f"     buffer {g.buffer_m:,.0f} m, so the domain is "
          f"{g.rows * g.res_m / 1000:.1f} by {g.cols * g.res_m / 1000:.1f} km")
    corners = g.corners_lonlat()
    print("     frame corners: " + "  ".join(f"{la:.5f},{lo:.5f}" for lo, la in corners))

    # Coverage on a coarse copy of the same grid, 10 m cells.
    step = max(1, int(round(10.0 / g.res_m)))
    rr, cc = np.mgrid[0:g.rows:step, 0:g.cols:step]
    x, y = g.rowcol_to_crs(rr + 0.5, cc + 0.5)
    covered = np.zeros(x.shape, dtype=bool)
    for t in tiles:
        covered |= (x >= t["xmin"]) & (x < t["xmax"]) & (y >= t["ymin"]) & (y < t["ymax"])
    if scan:
        real = np.zeros(x.shape, dtype=bool)
        for t in tiles:
            occ = scan[t["name"]]["occ"]
            ny, nx = occ.shape
            inb = (x >= t["xmin"]) & (x < t["xmax"]) & (y >= t["ymin"]) & (y < t["ymax"])
            c = ((x[inb] - t["xmin"]) / 65.0).astype(int).clip(0, nx - 1)
            r = ((t["ymax"] - y[inb]) / 65.0).astype(int).clip(0, ny - 1)
            real[inb] |= occ[r, c]
        from scipy import ndimage
        covered = ndimage.binary_fill_holes(real) & covered
    r0, r1, c0, c1 = g.frame
    in_frame = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    in_buf = ~in_frame

    fcov = covered[in_frame].mean()
    print(f"\n  frame covered by LiDAR:  {fcov*100:5.1f}%")
    missing = {}
    for xi, yi in zip(x[in_frame & ~covered], y[in_frame & ~covered]):
        k = tile_name(xi, yi) if scheme else f"x {xi:.0f} ft, y {yi:.0f} ft"
        missing[k] = missing.get(k, 0) + 1
    if missing:
        tot = in_frame.sum()
        print("  missing for the frame (download these):")
        ranked = sorted(missing.items(), key=lambda kv: -kv[1])
        for k, v in ranked[:12]:
            print(f"    {k}   {v / tot * 100:5.1f}% of the frame")
        if len(ranked) > 12:
            print(f"    ... and {len(ranked) - 12} more. A list this long means the "
                  "frame is not where you think; check it before downloading anything.")
        print("  Without them those cells are blanked in every image and left out of "
              "every number.")
    else:
        print("  the frame is fully covered. Go.")

    if buffer_m > 0:
        bcov = covered[in_buf].mean()
        print(f"\n  buffer covered by LiDAR: {bcov*100:5.1f}%. The rest casts as footprint "
              "prisms (no trees, flat roofs).")
        # The buffer that matters most is the side the winter sun shines from:
        # grid down, toward Midtown, within 1.5 km of the frame.
        near = in_buf & (rr >= r1) & (rr < r1 + int(1500 / g.res_m)) & \
            (cc >= c0 - int(600 / g.res_m)) & (cc < c1 + int(600 / g.res_m))
        south = {}
        for xi, yi in zip(x[near & ~covered], y[near & ~covered]):
            k = tile_name(xi, yi) if scheme else f"x {xi:.0f}, y {yi:.0f}"
            south[k] = south.get(k, 0) + 1
        if south:
            best = sorted(south.items(), key=lambda kv: -kv[1])[:6]
            print("  tiles that would make the south buffer measured instead of footprint "
                  "prisms, most useful first (optional):")
            print("    " + "  ".join(k for k, _ in best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
