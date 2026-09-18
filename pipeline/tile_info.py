"""
Inspect LiDAR tiles without reading a single point.

    python -m pipeline.tile_info --laz data/nyc
    python -m pipeline.tile_info --laz data/nyc --lat 40.7655 --lon -73.9800 --span 1600

LAS and LAZ headers carry the bounding box, so this is instant even for a 400 MB
tile. It tells you what ground you actually have, whether your target square is
covered, and which direction is missing. No more guessing tile numbers off a map
screenshot.
"""

from __future__ import annotations

import argparse
import glob
import os


def _to_wgs84(crs, xmin, ymin, xmax, ymax):
    """
    Corner box in lat/lon. Uses rasterio if it is installed, otherwise pyproj,
    which laspy already pulls in. tile_info is the first thing anyone runs, so
    it should not be the thing that demands the heaviest dependency.
    """
    try:
        from rasterio.warp import transform_bounds
        return transform_bounds(crs, "EPSG:4326", xmin, ymin, xmax, ymax,
                                densify_pts=21)
    except ImportError:
        from pyproj import Transformer
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs = [xmin, xmax, xmin, xmax]
        ys = [ymin, ymin, ymax, ymax]
        lon, lat = tr.transform(xs, ys)
        return min(lon), min(lat), max(lon), max(lat)


def _bounds_wgs84(path):
    import laspy

    with laspy.open(path) as f:
        h = f.header
        crs = h.parse_crs()
        n = h.point_count
        xmin, ymin = h.mins[0], h.mins[1]
        xmax, ymax = h.maxs[0], h.maxs[1]
        zmin, zmax = h.mins[2], h.maxs[2]
    if crs is None:
        return None
    w, s, e, nth = _to_wgs84(crs, xmin, ymin, xmax, ymax)
    return {
        "name": os.path.basename(path), "points": n, "crs": crs.to_string(),
        "west": w, "south": s, "east": e, "north": nth,
        "zmin": zmin, "zmax": zmax,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--laz", required=True, help="folder of .laz/.las, or one file")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--span", type=float, default=1600.0, help="metres a side")
    args = ap.parse_args()

    if os.path.isdir(args.laz):
        files = sorted(glob.glob(os.path.join(args.laz, "*.la[sz]")))
    else:
        files = [args.laz]
    if not files:
        raise SystemExit(f"no .laz or .las files in {args.laz}")

    infos = []
    print(f"{'tile':<18}{'points':>14}   lat range            lon range")
    print("-" * 78)
    for f in files:
        i = _bounds_wgs84(f)
        if i is None:
            print(f"{os.path.basename(f):<18}  no CRS in header")
            continue
        infos.append(i)
        print(f"{i['name']:<18}{i['points']:>14,}   "
              f"{i['south']:.4f} to {i['north']:.4f}   "
              f"{i['west']:.4f} to {i['east']:.4f}")

    if not infos:
        return 1

    W = min(i["west"] for i in infos); E = max(i["east"] for i in infos)
    S = min(i["south"] for i in infos); N = max(i["north"] for i in infos)
    print("-" * 78)
    print(f"{'ALL TILES':<18}{sum(i['points'] for i in infos):>14,}   "
          f"{S:.4f} to {N:.4f}   {W:.4f} to {E:.4f}")
    print(f"\ncentre of what you have: {(S+N)/2:.4f}, {(W+E)/2:.4f}")
    print(f"height range across tiles: {min(i['zmin'] for i in infos):.0f} to "
          f"{max(i['zmax'] for i in infos):.0f} (in the file's own units)")

    if args.lat is None or args.lon is None:
        print("\nPass --lat and --lon to check whether a target square is covered.")
        return 0

    import math
    half = args.span / 2.0
    dlat = half / 111_320.0
    dlon = half / (111_320.0 * math.cos(math.radians(args.lat)))
    tw, te = args.lon - dlon, args.lon + dlon
    ts, tn = args.lat - dlat, args.lat + dlat

    print(f"\ntarget square: {args.span:.0f} m centred on {args.lat}, {args.lon}")
    print(f"  needs  {ts:.4f} to {tn:.4f}   {tw:.4f} to {te:.4f}")

    missing = []
    if tw < W: missing.append(f"WEST  (short by {(W - tw) * 111320 * math.cos(math.radians(args.lat)):.0f} m)")
    if te > E: missing.append(f"EAST  (short by {(te - E) * 111320 * math.cos(math.radians(args.lat)):.0f} m)")
    if ts < S: missing.append(f"SOUTH (short by {(S - ts) * 111320:.0f} m)")
    if tn > N: missing.append(f"NORTH (short by {(tn - N) * 111320:.0f} m)")

    inside = any(i["west"] <= args.lon <= i["east"] and i["south"] <= args.lat <= i["north"]
                 for i in infos)
    print(f"\n  centre point is {'INSIDE' if inside else 'NOT INSIDE'} your tiles")
    if not inside:
        print("  -> the run will fail. You have the wrong tiles for this coordinate.")
    if missing:
        print("  tiles needed in these directions:")
        for m in missing:
            print(f"    {m}")
        print("\n  Either download the neighbours in those directions, or shrink")
        print(f"  --span. The largest square fully inside what you have is about")
        largest = min((E - W) * 111320 * math.cos(math.radians(args.lat)), (N - S) * 111320)
        print(f"  {largest:.0f} m, centred on {(S+N)/2:.4f}, {(W+E)/2:.4f}.")
    else:
        print("  fully covered. Go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
