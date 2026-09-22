"""
Buildings from NYC's official Building Footprints dataset.

The LiDAR point cloud stays the shadow-casting surface: it is measured, and it
includes trees, which really do cast shadows. But the tiles are unclassified,
so deriving *buildings* from it means guessing, and the guesses produced trees
drawn as towers and slivers stacked into translucent slabs.

The city already publishes every building outline with its roof height and
ground elevation, surveyed and maintained by NYC OTI. One polygon per building,
no trees. This module rasterises those onto the solver grid so the facade
solver and the 3D view both use real buildings.

Download: NYC Open Data, "Building Footprints" (id 5zhs-2jue), export GeoJSON.
Heights are in feet. Field names differ between exports, so both spellings
are accepted.
"""

from __future__ import annotations

import json

import numpy as np

FT = 0.3048006096012192
HEIGHT_KEYS = ("height_roof", "heightroof", "HEIGHTROOF", "Height Roof")
GROUND_KEYS = ("ground_elevation", "groundelev", "GROUNDELEV")


def _num(props, keys):
    for k in keys:
        v = props.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def load(path, bounds, crs, transform, shape, log=print):
    """
    Read footprints inside `bounds` (WGS84 w,s,e,n) and rasterise them onto
    the solver grid. Returns (buildings, labels, n, heights_m, built).
    """
    from pyproj import Transformer
    from rasterio import features
    from shapely.geometry import shape as shp, mapping
    from shapely.ops import transform as shp_tf

    w, s, e, n = bounds
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"] if "features" in gj else gj
    log(f"  {len(feats):,} footprints in file")

    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    buildings, shapes = [], []
    lab = 0
    skipped_h = 0
    for ft in feats:
        g = ft.get("geometry")
        if not g:
            continue
        poly = shp(g)
        if poly.is_empty:
            continue
        minx, miny, maxx, maxy = poly.bounds
        if maxx < w or minx > e or maxy < s or miny > n:
            continue
        h_ft = _num(ft.get("properties", {}), HEIGHT_KEYS)
        if h_ft is None or h_ft <= 0:
            skipped_h += 1
            continue
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
        lab += 1
        h_m = h_ft * FT
        ring = [[round(x, 6), round(y, 6)] for x, y in poly.exterior.coords]
        buildings.append({"id": lab, "ring": ring, "height": round(h_m, 1)})
        shapes.append((shp_tf(to_grid, poly), lab))

    log(f"  {lab:,} buildings inside the frame"
        + (f", {skipped_h} skipped for missing roof height" if skipped_h else ""))
    if not lab:
        raise SystemExit("No footprints fell inside the frame. Wrong file, or "
                         "the export does not cover this part of Manhattan.")

    labels = features.rasterize(
        shapes, out_shape=shape, transform=transform, fill=0,
        dtype="int32", all_touched=False,
    )
    h_lookup = np.zeros(lab + 1, dtype=np.float32)
    for b in buildings:
        h_lookup[b["id"]] = b["height"]
    heights = h_lookup[labels]
    built = labels > 0
    log(f"  footprints cover {built.mean()*100:.1f}% of the grid, "
        f"tallest {h_lookup.max():.0f} m ({h_lookup.max()/FT:.0f} ft)")
    return buildings, labels, lab, heights, built
