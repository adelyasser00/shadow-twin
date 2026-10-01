"""
The sunlight-sensitive resource CEQR cares about: here, the park.

A parks file can hold one park or every park in the city. Rasterising the
whole file would quietly fold a playground or Theodore Roosevelt Park into the
"Central Park" figures once the frame grows, so every feature is listed by
name at load time, and --resource-where keeps only the ones you mean:

    --resource-where "Central Park"            any text field containing it
    --resource-where "signname=Central Park"   one field, exact match

Matching ignores case.
"""

from __future__ import annotations

import json

import numpy as np

NAME_KEYS = ("signname", "name", "NAME", "park_name", "name311", "SIGNNAME",
             "propname", "eapply")


def _name(props):
    for k in NAME_KEYS:
        v = props.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for v in props.values():
        if isinstance(v, str) and 3 <= len(v) <= 60:
            return v
    return "(unnamed)"


def _match(props, where):
    if not where:
        return True
    if "=" in where:
        k, v = where.split("=", 1)
        pv = props.get(k.strip())
        return isinstance(pv, str) and pv.strip().lower() == v.strip().lower()
    w = where.lower()
    return any(isinstance(v, str) and w in v.lower() for v in props.values())


def load(path, where=None, log=print):
    """Shapely geometries in lon/lat, and the names that were kept."""
    from shapely.geometry import shape as shp

    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"] if "features" in gj else [gj]
    kept, names, dropped = [], [], []
    for ft in feats:
        g = ft.get("geometry")
        if not g:
            continue
        props = ft.get("properties") or {}
        nm = _name(props)
        if _match(props, where):
            geom = shp(g)
            if not geom.is_valid:
                geom = geom.buffer(0)
            kept.append(geom)
            names.append(nm)
        else:
            dropped.append(nm)
    log(f"  resource file: {len(feats)} feature(s), kept {len(kept)}"
        + (f" matching {where!r}" if where else ""))
    for nm in names[:8]:
        log(f"    kept     {nm}")
    if len(names) > 8:
        log(f"    ... and {len(names) - 8} more")
    if not kept:
        raise SystemExit(f"No resource features matched {where!r}. "
                         f"Names in the file include: {', '.join(dropped[:10])}")
    if len(kept) > 1 and not where:
        log("  warning: more than one feature and no --resource-where. Every one "
            "of them counts as the resource. Check the names above.")
    return kept, names


def check_fit(geoms, names, where):
    """
    Refuse to fit a frame around a file that is really a whole city of parks.

    NYC's parks export holds about 2,000 properties. Fitting a frame around
    all of them asks for a grid across five boroughs, which is nonsense and
    runs out of memory before it says so.
    """
    distinct = sorted(set(names))
    if len(distinct) > 1 and not where:
        raise SystemExit(
            f"The resource file holds {len(geoms)} features with {len(distinct)} "
            "different names, and the frame is fitted around the resource.\n"
            "Keep only the park you mean, for Central Park:\n"
            '    --resource-where "signname=Central Park"')
    from shapely.ops import unary_union
    w, s, e, n = unary_union(geoms).bounds
    if (e - w) > 0.1 or (n - s) > 0.1:
        raise SystemExit(
            f"The kept resource spans {(e - w):.2f} by {(n - s):.2f} degrees, over "
            "10 km. That is not one park. Tighten --resource-where; kept names: "
            + ", ".join(distinct[:8]))


def rasterise(geoms_lonlat, grid):
    """Resource mask on the domain grid."""
    from rasterio import features
    from shapely.ops import transform as shp_tf
    tr = grid.to_crs()
    shapes = [(shp_tf(tr, g), 1) for g in geoms_lonlat]
    return features.rasterize(shapes, out_shape=(grid.rows, grid.cols),
                              transform=grid.transform, fill=0,
                              dtype="uint8").astype(bool)


def window_mask(grid, lat, lon, span_m, edge_m):
    """
    Cells of the frame inside an earlier north-up square run, less its edge
    margin. Used to put a new run's numbers next to a published one on exactly
    the same ground.
    """
    from pyproj import Transformer
    x, y = grid.cell_centres_crs()
    cx, cy = Transformer.from_crs("EPSG:4326", grid.crs,
                                  always_xy=True).transform(lon, lat)
    half = (span_m / 2.0 - edge_m) / grid.unit_m
    return (np.abs(x - cx) <= half) & (np.abs(y - cy) <= half)
