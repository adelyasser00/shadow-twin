"""
Buildings from NYC's official Building Footprints dataset.

The LiDAR point cloud stays the shadow-casting surface: it is measured, and it
includes trees, which really do cast shadows. But the tiles are unclassified,
so deriving *buildings* from it means guessing, and the guesses produced trees
drawn as towers and slivers stacked into translucent slabs.

The city already publishes every building outline with its roof height and
construction year, surveyed and maintained by NYC OTI. One polygon per
building, no trees. This module rasterises those onto the solver grid so the
facade solver and the 3D view both use real buildings.

Footprints do three jobs:

  Outlines. Which cells belong to which building, for the walls and the 3D view.

  Heights for "today". Current recorded roof heights, which is how towers
  finished after the 2017 survey get into the today run.

  Casters beyond the survey. Where the buffer runs past the last LiDAR tile,
  each footprint stands as a prism at its roof height so it still casts shadow
  into the frame. For the 2017 run, only buildings with a construction year
  before 2017 stand there, so a tower that did not exist yet casts nothing.

Download: NYC Open Data, "Building Footprints" (id 5zhs-2jue), export GeoJSON.
Heights are in feet. Field names differ between exports, so several spellings
are accepted. The full city export is about a gigabyte; crop it once with

    python -m pipeline.nyc_footprints crop data/nyc/footprints.geojson data/nyc/footprints_crop.geojson --laz data/nyc

and point the runner at the crop. Every run after that starts in seconds.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

FT = 0.3048006096012192
HEIGHT_KEYS = ("height_roof", "heightroof", "HEIGHTROOF", "Height Roof")
GROUND_KEYS = ("ground_elevation", "groundelev", "GROUNDELEV")
YEAR_KEYS = ("construction_year", "cnstrct_yr", "CNSTRCT_YR", "Construction Year")
KEEP_KEYS = HEIGHT_KEYS + GROUND_KEYS + YEAR_KEYS + (
    "bin", "BIN", "base_bbl", "doitt_id", "name", "feat_code")

# LiDAR capture was May 2017. A building with that construction year or later
# did not exist, or was not finished, when the survey flew.
SURVEY_YEAR = 2017


def _num(props, keys):
    for k in keys:
        v = props.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _read_geojson(path, log=print):
    size = os.path.getsize(path) / 1e6
    if size > 200:
        log(f"  reading {size:,.0f} MB of GeoJSON. This is slow and memory hungry; "
            "crop it once with `python -m pipeline.nyc_footprints crop`.")
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    return gj["features"] if "features" in gj else gj


def read(path, bbox_lonlat=None, log=print):
    """
    Footprints as plain records, clipped to a lon/lat box.

    Each record: poly (shapely, lon/lat, largest part), height_m (or None),
    year (int or None).
    """
    from shapely.geometry import shape as shp

    feats = _read_geojson(path, log)
    log(f"  {len(feats):,} footprints in file")
    if bbox_lonlat:
        w, s, e, n = bbox_lonlat
    out, no_height, have_year = [], 0, 0
    for ft in feats:
        g = ft.get("geometry")
        if not g:
            continue
        try:
            poly = shp(g)
        except Exception:
            continue
        if poly.is_empty:
            continue
        if bbox_lonlat:
            minx, miny, maxx, maxy = poly.bounds
            if maxx < w or minx > e or maxy < s or miny > n:
                continue
        props = ft.get("properties", {}) or {}
        h_ft = _num(props, HEIGHT_KEYS)
        year = _num(props, YEAR_KEYS)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.is_empty:
                continue
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda p: p.area)
            if poly.geom_type != "Polygon":
                continue
        if h_ft is None or h_ft <= 0:
            no_height += 1
        if year and year > 1000:
            have_year += 1
        out.append({"poly": poly,
                    "height_m": (h_ft * FT) if h_ft and h_ft > 0 else None,
                    "year": int(year) if year and year > 1000 else None})
    log(f"  {len(out):,} footprints inside the area"
        + (f", {no_height:,} without a roof height" if no_height else "")
        + f", {have_year:,} with a construction year")
    if not out:
        raise SystemExit("No footprints fell inside the area. Wrong file, or "
                         "the export does not cover this part of Manhattan.")
    return out


def rasterise(recs, grid, log=print):
    """
    Burn every footprint onto the domain grid. Returns labels (int32, 0 is no
    building, label n is recs[n-1]) and the footprint polygons in CRS units.
    """
    from rasterio import features
    from shapely.ops import transform as shp_tf

    to_crs = grid.to_crs()
    polys = [shp_tf(to_crs, r["poly"]) for r in recs]
    shapes = [(p, i + 1) for i, p in enumerate(polys)]
    labels = features.rasterize(shapes, out_shape=(grid.rows, grid.cols),
                                transform=grid.transform, fill=0,
                                dtype="int32", all_touched=False)
    log(f"  footprints cover {(labels > 0).mean()*100:.1f}% of the domain, "
        f"{(labels[grid.frame_slice] > 0).mean()*100:.1f}% of the frame")
    return labels, polys


def label_medians(values, labels, n):
    """Median of `values` inside each label 1..n, NaN where a label has no cells."""
    m = labels > 0
    lab = labels[m]
    val = values[m]
    order = np.lexsort((val, lab))
    lab, val = lab[order], val[order]
    counts = np.bincount(lab, minlength=n + 1)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out = np.full(n + 1, np.nan, dtype=np.float32)
    idx = np.nonzero(counts > 0)[0]
    lo = starts[idx] + (counts[idx] - 1) // 2
    hi = starts[idx] + counts[idx] // 2
    out[idx] = 0.5 * (val[lo] + val[hi])
    return out


def label_quantile(values, labels, n, q):
    """The q-quantile of `values` inside each label 1..n, NaN where empty."""
    m = labels > 0
    lab = labels[m]
    val = values[m]
    order = np.lexsort((val, lab))
    lab, val = lab[order], val[order]
    counts = np.bincount(lab, minlength=n + 1)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out = np.full(n + 1, np.nan, dtype=np.float32)
    idx = np.nonzero(counts > 0)[0]
    k = starts[idx] + np.floor(q * (counts[idx] - 1)).astype(np.int64)
    out[idx] = val[k]
    return out


def label_fraction(mask, labels, n):
    """Share of each label's cells where mask is True."""
    tot = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float64)
    hit = np.bincount(labels.ravel(), weights=mask.ravel().astype(np.float64),
                      minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, hit / np.maximum(tot, 1), 0.0)


def scenario_heights(recs, labels, lidar_heights, nodata, today: bool, log=print):
    """
    One height per footprint, for the scenario being run.

    today   the recorded roof height. Falls back to the LiDAR if the record has
            none.
    2017    what the survey measured: the median LiDAR height inside the
            footprint. A footprint that sits mostly outside the survey keeps
            its recorded height only if it was built before 2017, and drops
            to zero otherwise, because it was not there.

    Returns (heights indexed by label, description). Label 0 is always 0.
    """
    n = len(recs)
    fp = np.array([np.nan] + [r["height_m"] if r["height_m"] else np.nan for r in recs],
                  dtype=np.float32)
    year = np.array([0] + [r["year"] or 0 for r in recs], dtype=np.int32)
    lid = label_medians(lidar_heights, labels, n)
    outside = label_fraction(nodata, labels, n) > 0.5

    if today:
        h = np.where(np.isfinite(fp), fp, lid)
        src = "recorded roof heights, NYC Building Footprints"
    else:
        late = (year >= SURVEY_YEAR) & outside
        h = np.where(outside, np.where(late, 0.0, fp), lid)
        if late.any():
            log(f"  {int(late.sum())} footprints beyond the survey were built in "
                f"{SURVEY_YEAR} or later and stand at zero for the 2017 run")
        src = ("median LiDAR height inside each footprint, from the 2017 survey; "
               "recorded heights only beyond the survey")
    h = np.nan_to_num(h, nan=0.0).astype(np.float32)
    h[0] = 0.0
    h[h < 0] = 0.0
    return h, src


def buildings_for_viewer(recs, polys_crs, heights, grid, context_min_m=100.0,
                         context_reach_m=1500.0, context_max=250,
                         ring_m=200.0, ring_max=6000,
                         min_height_m=3.0, log=print):
    """
    The building list the browser draws.

    Frame buildings: every footprint whose interior point is inside the frame
    and that stood at least `min_height_m` in this scenario. These carry the
    wall scores.

    Context, drawn plain and never scored:
      the ring    every building within `ring_m` of the frame, so the city
                  carries on past the frame edge instead of stopping dead
      towers      anything at least `context_min_m` tall within
                  `context_reach_m`: the ones whose shadows reach in, which
                  would otherwise walk in from nowhere
    """
    frame = grid.frame_polygon_crs()
    near = frame.buffer(context_reach_m / grid.unit_m)
    ring = frame.buffer(ring_m / grid.unit_m) if ring_m > 0 else None
    out_frame, out_ring, out_tow = [], [], []
    for i, (r, p) in enumerate(zip(recs, polys_crs), start=1):
        h = float(heights[i])
        if h < min_height_m:
            continue
        ring_xy = [[round(x, 6), round(y, 6)] for x, y in r["poly"].exterior.coords]
        if frame.contains(p.representative_point()):
            out_frame.append({"id": i, "ring": ring_xy, "height": round(h, 1)})
        elif ring is not None and ring.intersects(p):
            out_ring.append({"id": i, "ring": ring_xy, "height": round(h, 1),
                             "context": True})
        elif h >= context_min_m and near.intersects(p):
            out_tow.append({"id": i, "ring": ring_xy, "height": round(h, 1),
                            "context": True})
    out_tow.sort(key=lambda b: -b["height"])
    out_tow = out_tow[:context_max]
    out_ring = out_ring[:ring_max]
    log(f"  {len(out_frame):,} buildings in the frame, {len(out_ring):,} in the "
        f"{ring_m:.0f} m ring around it, {len(out_tow)} towers of "
        f"{context_min_m:.0f} m or more out to {context_reach_m/1000:.1f} km")
    return out_frame, out_ring + out_tow


# ------------------------------------------------------------------- crop
def crop(src, dst, bbox, log=print):
    """Write only the footprints inside a lon/lat box, with the fields we use."""
    from shapely.geometry import shape as shp
    feats = _read_geojson(src, log)
    w, s, e, n = bbox
    keep = []
    for ft in feats:
        g = ft.get("geometry")
        if not g:
            continue
        try:
            minx, miny, maxx, maxy = shp(g).bounds
        except Exception:
            continue
        if maxx < w or minx > e or maxy < s or miny > n:
            continue
        props = {k: v for k, v in (ft.get("properties") or {}).items() if k in KEEP_KEYS}
        keep.append({"type": "Feature", "properties": props, "geometry": g})
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"type": "FeatureCollection", "features": keep}, f,
                  separators=(",", ":"))
    log(f"kept {len(keep):,} of {len(feats):,} footprints, "
        f"{os.path.getsize(dst)/1e6:.1f} MB written to {dst}")


def _bbox_from_tiles(laz_dir, pad_m):
    from pyproj import Transformer
    from . import laz
    info = laz.probe(laz_dir)
    u = info["unit_m"]
    xmin = min(t["xmin"] for t in info["tiles"]) - pad_m / u
    ymin = min(t["ymin"] for t in info["tiles"]) - pad_m / u
    xmax = max(t["xmax"] for t in info["tiles"]) + pad_m / u
    ymax = max(t["ymax"] for t in info["tiles"]) + pad_m / u
    tr = Transformer.from_crs(info["crs"], "EPSG:4326", always_xy=True)
    lon, lat = tr.transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
    return min(lon), min(lat), max(lon), max(lat)


def main():
    ap = argparse.ArgumentParser(description="Footprint utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crop", help="keep only the footprints near your tiles")
    c.add_argument("src")
    c.add_argument("dst")
    c.add_argument("--laz", help="crop to these tiles plus --pad metres")
    c.add_argument("--pad", type=float, default=3000.0,
                   help="metres beyond the tiles to keep, default 3000")
    c.add_argument("--bbox", help="west,south,east,north in degrees, instead of --laz")
    args = ap.parse_args()
    if args.bbox:
        bbox = tuple(float(v) for v in args.bbox.split(","))
    elif args.laz:
        bbox = _bbox_from_tiles(args.laz, args.pad)
    else:
        raise SystemExit("give --laz (tiles folder) or --bbox")
    print(f"cropping to {bbox[1]:.4f} to {bbox[3]:.4f} N, "
          f"{bbox[0]:.4f} to {bbox[2]:.4f} E")
    crop(args.src, args.dst, bbox)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
