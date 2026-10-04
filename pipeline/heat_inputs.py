"""
Inputs for the heat layer: ground, buildings, trees and land cover on a
north-up 2 m grid, for 2017 and for today.

    .venv-heat/Scripts/python -m pipeline.heat_inputs --laz data/nyc --footprints data/nyc/footprints_crop.geojson --resource data/nyc/park.geojson

Everything is built from the same rasters and functions as the shadow study
(laz.rasterise, the footprint rasteriser, the 2017 and "grown" today heights),
so the buildings are the shadow study's buildings. The one new idea is moving
trees out of the surface and into a canopy layer, because SOLWEIG treats canopy
as partly transparent and buildings as solid.

  DEM    LiDAR ground, gaps filled from the nearest measured ground, exactly
         as the shadow pipeline fills it.
  DSM    the DEM everywhere, plus the shadow study's surface on footprint
         cells: real roofs inside the survey, prisms beyond it. Today adds
         the "grown" buildings, same rule as run_nyc --burn-mode grown.
  CDSM   LiDAR surface minus ground on cells that are neither building nor
         water, 0 below 2 m, capped at 40 m, and 0 within 4 m of any
         footprint. LiDAR roofs spill about one cell past their walls; read
         as canopy, every building edge would grow a 50 m tree.
  LC     buildings 2, water 7, the rest of the park 5 (grass), everything
         else 1 (asphalt). Same in both years.

Grid
----
SOLWEIG assumes raster rows run north to south, so the grid is north-up on the
EPSG:2263 axes, 2 m cells. The GeoTIFFs are written in EPSG:32118, which is the
same Lambert projection with the same origin in metres instead of US survey
feet, so SOLWEIG reads a 2.0 m pixel. Written in feet it would read 6.56 m
pixels and make every distance 3.28 times too long.

Domain
------
Data-driven, because a caster missing from both years fakes a change: a tower
that shades a cell in both years, left out, lets a grown tower take the credit.

  window   Central Park cells within 1.5 km of 59th Street, measured along the
           avenues exactly like the shadow study's bands
  casters  every footprint within 500 m of the window, plus every footprint
           of 100 m or more whose shadow can reach the window at some point
           of the December CEQR window (checked every 5 minutes, with 15
           minutes either side and 10% on the shadow length)
  margins  300 m west, north and east, where sky view is overestimated and
           nothing is reported. The south side needs none: that is where the
           casters are, and it is never cut.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAT_DIR = os.path.join(HERE, "data", "heat")
INPUTS_DIR = os.path.join(HEAT_DIR, "inputs")

RES_M = 2.0
AVENUE_BEARING = 28.9          # Manhattan's avenues, degrees east of true north
WINDOW_M = 1500.0              # analysis window, metres north of 59th Street
NEAR_M = 500.0                 # every footprint this close to the window casts
TALL_M = 100.0                 # far casters considered from this height
SOUTH_REACH_M = 3000.0         # how far south of 59th Street to look for them
MARGIN_M = 300.0               # west, north and east margins, never reported
SOUTH_PAD_M = 20.0             # keeps the southernmost caster off the edge
CANOPY_MIN_M = 2.0
CANOPY_MAX_M = 40.0
EDGE_SKIP_M = 4.0              # same rule as facade.EDGE_SKIP_M
GROWN_M = 6.0                  # same rule as run_nyc.GROWN_M
EST = -5.0

# SOLWEIG land cover codes.
LC_PAVED, LC_BUILDING, LC_GRASS, LC_WATER = 1, 2, 5, 7

GEOTIFF_CRS = "EPSG:32118"     # NAD83 / New York Long Island, metres
US_FT = 1200.0 / 3937.0


# ---------------------------------------------------------------- geometry
def avenue_axis(crs, lon, lat):
    """Unit vector in CRS units pointing up the avenues, and the convergence."""
    from .grid import convergence_deg
    conv = convergence_deg(crs, lon, lat)
    t = math.radians(AVENUE_BEARING + conv)
    return (math.sin(t), math.cos(t)), conv


def ceqr_dec_suns(lat, lon, step_min=5, pad_min=15):
    """Sun positions through the December CEQR window, with a pad either side."""
    from .run_nyc import CEQR_DAYS
    from .solar import sun_position
    spec = next(d for d in CEQR_DAYS if d["id"] == "dec")
    y, m, d = (int(x) for x in spec["date"].split("-"))
    h0, m0 = (int(x) for x in spec["window"][0].split(":"))
    h1, m1 = (int(x) for x in spec["window"][1].split(":"))
    t, end = h0 * 60 + m0 - pad_min, h1 * 60 + m1 + pad_min
    out = []
    while t <= end:
        local = datetime(y, m, d) + timedelta(minutes=t)
        p = sun_position((local - timedelta(hours=EST)).replace(tzinfo=timezone.utc), lat, lon)
        if p.altitude > 0:
            out.append(p)
        t += step_min
    return out


def shadow_reaches(poly, h_m, suns, target, unit_m, conv, stretch=1.1):
    """Can a prism of this footprint and height shade `target` for any sun?"""
    from shapely.affinity import translate
    hull = poly.convex_hull
    for p in suns:
        L = stretch * h_m / math.tan(math.radians(p.altitude)) / unit_m
        b = math.radians((p.azimuth + 180.0 + conv) % 360.0)
        sh = hull.union(translate(hull, L * math.sin(b), L * math.cos(b))).convex_hull
        if sh.intersects(target):
            return True
    return False


def make_grid(crs, unit_m, bounds_crs, window_bounds_crs, centre_lonlat):
    """A north-up Grid covering bounds_crs, 2 m cells, frame on the window."""
    from .grid import Grid, convergence_deg
    lon, lat = centre_lonlat
    conv = convergence_deg(crs, lon, lat)
    res_u = RES_M / unit_m
    xmin, ymin, xmax, ymax = bounds_crs
    cols = int(math.ceil((xmax - xmin) / res_u))
    rows = int(math.ceil((ymax - ymin) / res_u))
    g = Grid(crs=crs, unit_m=unit_m, res_m=RES_M, bearing_deg=-conv, conv_deg=conv,
             x0=xmin, y0=ymax, rows=rows, cols=cols, frame=(0, rows, 0, cols),
             centre_lat=lat, centre_lon=lon)
    wx0, wy0, wx1, wy1 = window_bounds_crs
    r0, c0 = g.crs_to_rowcol(wx0, wy1)
    r1, c1 = g.crs_to_rowcol(wx1, wy0)
    g.frame = (max(0, int(math.floor(r0))), min(rows, int(math.ceil(r1))),
               max(0, int(math.floor(c0))), min(cols, int(math.ceil(c1))))
    return g


def metric_transform(grid):
    """The grid's transform in EPSG:32118 metres. Checked, not assumed."""
    from pyproj import Transformer
    from rasterio.transform import Affine
    tf = grid.transform
    if abs(tf.b) > 1e-9 or abs(tf.d) > 1e-9 or tf.a <= 0 or tf.e >= 0:
        raise SystemExit("heat grid is not north-up")
    tr = Transformer.from_crs("EPSG:2263", GEOTIFF_CRS, always_xy=True)
    for x, y in ((tf.c, tf.f), (tf.c + 10_000, tf.f - 10_000)):
        mx, my = tr.transform(x, y)
        if abs(mx - x * US_FT) > 1e-3 or abs(my - y * US_FT) > 1e-3:
            raise SystemExit("EPSG:2263 and EPSG:32118 are not a pure unit change here")
    return Affine(tf.a * US_FT, 0.0, tf.c * US_FT, 0.0, tf.e * US_FT, tf.f * US_FT)


def flag_suspects(grown_list, footprints_path, grid, labels, top17, log=print):
    """
    Mark grown buildings that are probably footprint artefacts.

    NYC Building Footprints can split one building into several polygons (by
    BIN, or several BINs on one tax lot), each carrying the building's highest
    roof. The grown rule then raises a low wing to the tower's height. A grown
    polygon is a suspect when a neighbouring polygon on the same BIN or lot,
    or touching it with the same recorded roof, already reaches that roof in
    the 2017 LiDAR (within GROWN_M). Nothing is changed here; this is a list
    for a person to check. A construction year before 2017 is noted too, but
    is weak evidence on its own: a tower added onto an old base keeps the
    base's year.
    """
    from shapely import STRtree
    from shapely.geometry import Point, shape
    with open(footprints_path, encoding="utf-8") as f:
        feats = json.load(f)["features"]
    w, s_, e, n_ = grid.bounds_lonlat((0, grid.rows, 0, grid.cols), pad_deg=0.002)
    geoms, props = [], []
    for ft in feats:
        g = ft.get("geometry")
        if not g:
            continue
        try:
            shp = shape(g)
        except Exception:
            continue
        if shp.is_empty:
            continue
        x0, y0, x1, y1 = shp.bounds
        if x1 < w or x0 > e or y1 < s_ or y0 > n_:
            continue
        geoms.append(shp)
        props.append(ft.get("properties") or {})
    tree = STRtree(geoms)
    to_crs = grid.to_crs()

    def roof_m(pr):
        try:
            return float(pr.get("height_roof")) * 0.3048006096012192
        except (TypeError, ValueError):
            return None

    for b in grown_list:
        b["bin"] = b["bbl"] = b["sibling_lidar_p99_m"] = None
        reasons, notes = [], []
        pt = Point(*b["lonlat"])
        hits = [i for i in tree.query(pt) if geoms[i].contains(pt)]
        if hits:
            me = hits[0]
            b["bin"], b["bbl"] = props[me].get("bin"), props[me].get("base_bbl")
            best = None
            for j in tree.query(geoms[me].buffer(0.0001)):     # about 10 m
                if j == me:
                    continue
                pj = props[j]
                same = ((b["bin"] and pj.get("bin") == b["bin"])
                        or (b["bbl"] and pj.get("base_bbl") == b["bbl"])
                        or (roof_m(pj) is not None and abs(roof_m(pj) - b["height_m"]) < 1.0
                            and geoms[j].distance(geoms[me]) < 0.00006))
                if not same:
                    continue
                q = geoms[j].representative_point()
                r, c = grid.crs_to_rowcol(*to_crs(q.x, q.y))
                r, c = int(r), int(c)
                if 0 <= r < grid.rows and 0 <= c < grid.cols and labels[r, c] > 0:
                    v = top17[labels[r, c]]
                    if np.isfinite(v):
                        best = float(v) if best is None else max(best, float(v))
            b["sibling_lidar_p99_m"] = None if best is None else round(best, 1)
            if best is not None and best >= b["height_m"] - GROWN_M:
                reasons.append(f"a sibling footprint (same BIN, lot or roof) already stands "
                               f"{best:.0f} m in the 2017 LiDAR")
        if b["year"] and b["year"] < 2017:
            notes.append(f"construction year {b['year']}")
        b["suspect"] = "; ".join(reasons)
        b["note"] = "; ".join(notes)


# ------------------------------------------------------------------ build
def build(args, log=print):
    from pyproj import Transformer
    from shapely.geometry import Polygon, box
    from shapely.ops import transform as shp_tf, unary_union
    from . import laz
    from . import nyc_footprints as NF
    from . import resource as R

    info = laz.probe(args.laz)
    crs, unit_m = info["crs"], info["unit_m"]
    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform

    # ---- the window, from the park and the avenue axis -------------------
    res_geoms, res_names = R.load(args.resource, args.resource_where, log=log)
    R.check_fit(res_geoms, res_names, args.resource_where)
    park_crs = unary_union([shp_tf(to_crs, g) for g in res_geoms])
    c = park_crs.centroid
    (ux, uy), conv = avenue_axis(crs, *to_ll(c.x, c.y))
    vx, vy = uy, -ux                                   # across the avenues, eastward
    xs, ys = np.asarray(park_crs.convex_hull.exterior.xy)
    along = xs * ux + ys * uy
    a0 = float(along.min())                            # 59th Street, CRS units
    across = xs * vx + ys * vy
    w0, w1 = float(across.min()) - 10, float(across.max()) + 10
    a1 = a0 + WINDOW_M / unit_m

    def strip(a_lo, a_hi):
        pts = [(a * ux + w * vx, a * uy + w * vy) for a, w in
               ((a_lo, w0), (a_hi, w0), (a_hi, w1), (a_lo, w1))]
        return Polygon(pts)
    window_poly = park_crs.intersection(strip(a0, a1))
    log(f"window: Central Park within {WINDOW_M:.0f} m of 59th Street, "
        f"{window_poly.area * unit_m**2 / 1e4:.1f} ha (polygon)")
    wll = to_ll(window_poly.centroid.x, window_poly.centroid.y)

    # ---- casters ---------------------------------------------------------
    near_poly = window_poly.buffer(NEAR_M / unit_m)
    pad_deg = (SOUTH_REACH_M + 1000.0) / 111_000.0
    wb = shp_tf(to_ll, window_poly).bounds
    recs_all = NF.read(args.footprints, (wb[0] - pad_deg, wb[1] - pad_deg,
                                         wb[2] + pad_deg, wb[3] + pad_deg), log=log)
    polys_all = [shp_tf(to_crs, r["poly"]) for r in recs_all]
    suns = ceqr_dec_suns(wll[1], wll[0])
    log(f"casters: December CEQR window, {len(suns)} sun positions from "
        f"{min(p.altitude for p in suns):.1f} deg up")
    far, near_n, tall_checked, tall_unknown = [], 0, 0, 0
    for r, p in zip(recs_all, polys_all):
        if p.intersects(near_poly):
            near_n += 1
            continue
        h = r["height_m"]
        if h is None:
            tall_unknown += 1
            continue
        if h < TALL_M:
            continue
        cen = p.centroid
        a = cen.x * ux + cen.y * uy
        if (a0 - a) * unit_m > SOUTH_REACH_M or (a - a1) * unit_m > SOUTH_REACH_M:
            continue
        tall_checked += 1
        if shadow_reaches(p, h, suns, window_poly, unit_m, conv):
            far.append((r, p))
    south_far = sum(1 for r, p in far if (p.centroid.x * ux + p.centroid.y * uy) < a0)
    log(f"  {near_n:,} footprints within {NEAR_M:.0f} m of the window")
    log(f"  {tall_checked} footprints of {TALL_M:.0f} m or more beyond that, "
        f"{len(far)} of them can shade the window in December "
        f"({south_far} south of 59th Street)")
    for r, p in sorted(far, key=lambda rp: -rp[0]["height_m"])[:12]:
        lo, la = to_ll(p.centroid.x, p.centroid.y)
        d = ((p.centroid.x * ux + p.centroid.y * uy) - a0) * unit_m
        log(f"    {r['height_m']:6.1f} m  {la:.5f}, {lo:.5f}  {d:+7.0f} m along the avenues")

    # ---- domain ----------------------------------------------------------
    xmin, ymin, xmax, ymax = near_poly.bounds
    for _, p in far:
        bx0, by0, bx1, by1 = p.bounds
        xmin, ymin, xmax, ymax = min(xmin, bx0), min(ymin, by0), max(xmax, bx1), max(ymax, by1)
    m_u = MARGIN_M / unit_m
    north_extra = (args.north_pad_m / unit_m) if args.north_pad_m is not None else None
    ymax_dom = ymax + m_u
    if north_extra is not None:
        # Cut the north side only: keep the window plus this much.
        ymax_dom = min(ymax_dom, window_poly.bounds[3] + north_extra)
        log(f"  north extent cut to {args.north_pad_m:.0f} m past the window")
    bounds = (xmin - m_u, ymin - SOUTH_PAD_M / unit_m, xmax + m_u, ymax_dom)
    grid = make_grid(crs, unit_m, bounds, window_poly.bounds, wll)
    L, W = grid.rows * RES_M, grid.cols * RES_M
    log(f"domain {W/1000:.2f} km east-west by {L/1000:.2f} km north-south, "
        f"{grid.rows:,} x {grid.cols:,} cells at {RES_M} m, north-up")

    # ---- the shadow study's rasters on this grid -------------------------
    cache = os.path.join(HEAT_DIR, "cache", "lidar")
    prep = laz.rasterise(args.laz, grid, log=log, cache_dir=cache, derive_buildings=False)
    ground = prep["ground"].astype(np.float32)
    lidar = prep["surface"].astype(np.float32)
    nodata, water = prep["nodata"], prep["water"]

    recs = NF.read(args.footprints, grid.bounds_lonlat((0, grid.rows, 0, grid.cols),
                                                       pad_deg=0.002), log=log)
    labels, polys = NF.rasterise(recs, grid, log=log)
    n = len(recs)
    h17, src17 = NF.scenario_heights(recs, labels, prep["heights"], nodata, today=False, log=log)
    hnow, srcnow = NF.scenario_heights(recs, labels, prep["heights"], nodata, today=True, log=log)
    fp = labels > 0

    # Today, as run_nyc builds it with --burn-footprints --burn-mode grown.
    stand = nodata & fp
    surfnow = np.where(stand, ground + hnow[labels], lidar).astype(np.float32)
    target = ground + hnow[labels]
    raise_all = fp & (target > surfnow + 3.0)
    top17 = NF.label_quantile(prep["heights"], labels, n, 0.99)
    fp_h = np.array([0.0] + [r["height_m"] or 0.0 for r in recs], dtype=np.float32)
    grown = ~np.isfinite(top17) | (fp_h > np.nan_to_num(top17, nan=0.0) + GROWN_M)
    grown[0] = False
    outside = NF.label_fraction(nodata, labels, n) > 0.5
    grown &= ~outside
    grown &= np.bincount(labels.ravel(), minlength=n + 1) > 0
    raise_ = raise_all & grown[labels]
    surfnow = np.where(raise_, target, surfnow).astype(np.float32)
    log(f"today: {int(grown[1:].sum())} buildings stand over {GROWN_M:.0f} m above what "
        f"the 2017 survey measured in them, {int(raise_.sum()):,} cells raised")

    # 2017. Inside the survey this is run_nyc's 2017 surface exactly. On the
    # prisms beyond it, run_nyc gives a building that is partly outside the
    # survey its LiDAR median in 2017 (pulled down by the empty cells) and its
    # recorded roof today, so an unchanged building would differ between the
    # years. Here only grown buildings and buildings new since 2017 differ;
    # every other prism stands at today's height in both years.
    year = np.array([0] + [r["year"] or 0 for r in recs], dtype=np.int32)
    late_outside = outside & (year >= NF.SURVEY_YEAR) & (hnow > h17)
    late_outside[0] = False
    differs = grown | late_outside
    h17_heat = np.where(differs, h17, hnow).astype(np.float32)
    surf17 = np.where(stand, ground + h17_heat[labels], lidar).astype(np.float32)
    dev = stand & (h17_heat[labels] != h17[labels])
    dev_labels = np.unique(labels[dev])
    log(f"2017: {int(dev.sum()):,} prism cells in {dev_labels.size} unchanged buildings beyond "
        f"the survey stand at the same height as today, not run_nyc's 2017 LiDAR median")

    # ---- heat layers -----------------------------------------------------
    dem = ground
    dsm17 = np.where(fp, surf17, dem).astype(np.float32)
    dsmnow = np.where(fp, surfnow, dem).astype(np.float32)

    from scipy import ndimage
    d_fp = ndimage.distance_transform_edt(~fp) * RES_M
    edge = (~fp) & (d_fp <= EDGE_SKIP_M)
    raw = np.where(fp | water, 0.0, lidar - ground).astype(np.float32)
    canopy = (raw >= CANOPY_MIN_M)
    removed_edge = canopy & edge
    cdsm = np.where(canopy & ~edge, np.minimum(raw, CANOPY_MAX_M), 0.0).astype(np.float32)
    capped = int((canopy & ~edge & (raw > CANOPY_MAX_M)).sum())
    veg = cdsm > 0
    pct = np.percentile(cdsm[veg], [5, 25, 50, 75, 95, 99, 100]) if veg.any() else [0] * 7
    log(f"canopy: {int(veg.sum()):,} cells ({veg.mean()*100:.1f}% of the domain); "
        f"the {EDGE_SKIP_M:.0f} m edge rule removed {int(removed_edge.sum()):,} cells; "
        f"{capped:,} capped at {CANOPY_MAX_M:.0f} m")
    log("  canopy height percentiles 5/25/50/75/95/99/max: "
        + " / ".join(f"{v:.1f}" for v in pct) + " m")

    park = R.rasterise(res_geoms, grid)
    lc = np.full((grid.rows, grid.cols), LC_PAVED, dtype=np.uint8)
    lc[park & ~water & ~fp] = LC_GRASS
    lc[water & ~fp] = LC_WATER
    lc[fp] = LC_BUILDING

    # ---- masks -----------------------------------------------------------
    cx, cy = grid.cell_centres_crs((0, grid.rows, 0, grid.cols))
    dist59 = ((cx * ux + cy * uy) - a0) * unit_m
    window = park & (dist59 >= 0) & (dist59 < WINDOW_M) & ~nodata
    rr, cc = np.mgrid[0:grid.rows, 0:grid.cols]
    m_cells = int(round(MARGIN_M / RES_M))
    inner = (cc >= m_cells) & (cc < grid.cols - m_cells) & (rr >= m_cells)
    report = window & inner & ~water & ~fp
    log(f"window {window.sum() * RES_M**2 / 1e4:.1f} ha on the grid; reported "
        f"{report.sum() * RES_M**2 / 1e4:.1f} ha (no water, no buildings, "
        f"{MARGIN_M:.0f} m from the west, north and east edges); "
        f"{int((window & ~inner).sum())} window cells fall in the margin")

    changed = dsmnow != dsm17
    explained = raise_ | (stand & differs[labels])
    meta = {
        "grid": {"crs_build": "EPSG:2263", "crs_geotiff": GEOTIFF_CRS, "res_m": RES_M,
                 "rows": grid.rows, "cols": grid.cols,
                 "x0_ft": grid.x0, "y0_ft": grid.y0, "conv_deg": grid.conv_deg,
                 "transform_m": list(metric_transform(grid))[:6]},
        "site": {"lat": wll[1], "lon": wll[0], "utc_offset": EST},
        "avenue_axis": {"bearing_deg": AVENUE_BEARING, "ux": ux, "uy": uy,
                        "a0_ft": a0, "conv_deg": conv},
        "window_m": WINDOW_M, "near_m": NEAR_M, "margin_m": MARGIN_M,
        "casters_far": [{"height_m": round(r["height_m"], 1), "year": r["year"],
                         "lonlat": [round(v, 6) for v in to_ll(p.centroid.x, p.centroid.y)],
                         "along_m": round(((p.centroid.x * ux + p.centroid.y * uy) - a0) * unit_m)}
                        for r, p in sorted(far, key=lambda rp: -rp[0]["height_m"])],
        "footprints_near": near_n, "tall_without_height": tall_unknown,
        "lidar": {"tiles_used": prep.get("tiles_used"), "ground_source": prep["ground_source"],
                  "source": prep.get("source_desc"),
                  "nodata_frac": float(nodata.mean()), "water_frac": float(water.mean())},
        "heights": {"2017": src17, "today": srcnow},
        "scenario": {"buildings_grown": int(grown[1:].sum()), "cells_raised": int(raise_.sum()),
                     "prism_cells_held_equal": int(dev.sum()), "prism_buildings_held_equal": int(dev_labels.size),
                     "outside_new_buildings": int(late_outside.sum()),
                     "cells_changed": int(changed.sum()),
                     "cells_changed_in_window_bbox": int(changed[grid.frame_slice].sum())},
        "canopy": {"cells": int(veg.sum()), "edge_rule_removed": int(removed_edge.sum()),
                   "capped": capped, "percentiles_5_25_50_75_95_99_max": [round(float(v), 2) for v in pct]},
        "areas_ha": {"window": float(window.sum() * RES_M**2 / 1e4),
                     "report": float(report.sum() * RES_M**2 / 1e4)},
        "grown_buildings": [],
    }
    for i in np.nonzero(grown)[0]:
        cells = raise_ & (labels == i)
        if not cells.any():
            continue
        p = polys[i - 1].representative_point()
        lo, la = to_ll(p.x, p.y)
        meta["grown_buildings"].append({
            "height_m": round(float(fp_h[i]), 1), "year": recs[i - 1]["year"],
            "lidar_p99_m": None if not np.isfinite(top17[i]) else round(float(top17[i]), 1),
            "cells_raised": int(cells.sum()), "lonlat": [round(lo, 6), round(la, 6)]})
    meta["grown_buildings"].sort(key=lambda b: -b["height_m"])
    flag_suspects(meta["grown_buildings"], args.footprints, grid, labels, top17, log=log)
    for b in meta["grown_buildings"][:15]:
        log(f"  grown: {b['height_m']:6.1f} m (2017 survey {b['lidar_p99_m']} m), "
            f"built {b['year']}, {b['cells_raised']:,} cells, {b['lonlat'][1]:.5f}, "
            f"{b['lonlat'][0]:.5f}{'  SUSPECT: ' + b['suspect'] if b['suspect'] else ''}"
            f"{'  (' + b['note'] + ')' if b['note'] else ''}")
    sus = [b for b in meta["grown_buildings"] if b["suspect"]]
    log(f"  {len(sus)} of {len(meta['grown_buildings'])} grown buildings look like footprint "
        f"artefacts ({sum(b['cells_raised'] for b in sus):,} of {int(raise_.sum()):,} raised cells); "
        "kept, because the shadow study keeps them")

    layers = {"dem": dem, "dsm_2017": dsm17, "dsm_today": dsmnow, "cdsm": cdsm, "landcover": lc}
    masks = {"window": window, "report": report, "park": park, "water": water,
             "building": fp, "nodata": nodata, "changed": changed, "raised": raise_,
             "explained": explained, "dist59": dist59.astype(np.float32)}
    return grid, layers, masks, meta


# -------------------------------------------------------------- invariants
def check_invariants(layers, masks, res_m=RES_M):
    """Every rule the inputs must satisfy. Returns [(name, ok, detail)]."""
    dem, d17, dnow = layers["dem"], layers["dsm_2017"], layers["dsm_today"]
    cdsm, lc = layers["cdsm"], layers["landcover"]
    b, w = masks["building"], masks["water"]
    out = []
    finite = all(np.isfinite(a).all() for a in (dem, d17, dnow, cdsm))
    out.append(("no NaN or inf in DEM, DSMs, CDSM", finite, ""))
    for nm, d in (("2017", d17), ("today", dnow)):
        lo = float((d - dem).min())
        out.append((f"DSM {nm} >= DEM", lo >= 0.0, f"min DSM - DEM {lo:.3f} m"))
    out.append(("CDSM >= 0", float(cdsm.min()) >= 0.0, f"min {float(cdsm.min()):.3f}"))
    out.append(("CDSM = 0 on building and water cells",
                not (cdsm[b | w] != 0).any(), f"{int((cdsm[b | w] != 0).sum())} cells"))
    out.append(("CDSM <= 40 m", float(cdsm.max()) <= CANOPY_MAX_M, f"max {float(cdsm.max()):.2f}"))
    lo = float((dnow - d17).min())
    out.append(("DSM today >= DSM 2017", lo >= 0.0, f"min today - 2017 {lo:.3f} m"))
    ch = dnow != d17
    stray = ch & ~masks["explained"]
    out.append(("today differs from 2017 only on grown or new buildings",
                not stray.any(), f"{int(ch.sum()):,} cells differ, {int(stray.sum())} unexplained"))
    out.append(("same DEM, CDSM and land cover in both scenarios", True,
                "one array of each is written and both runs read it"))
    codes = set(np.unique(lc).tolist())
    out.append(("land cover codes in {1, 2, 5, 7}", codes <= {1, 2, 5, 7}, str(sorted(codes))))
    out.append(("land cover 2 exactly on footprints", bool(((lc == 2) == b).all()), ""))
    out.append(("pixel size 2.0 m", abs(res_m - 2.0) < 1e-9, f"{res_m}"))
    return out


def save(grid, layers, masks, meta, out_dir=INPUTS_DIR, log=print):
    import rasterio
    os.makedirs(out_dir, exist_ok=True)
    tf = metric_transform(grid)
    for name, arr in layers.items():
        path = os.path.join(out_dir, f"{name}.tif")
        dtype = "uint8" if arr.dtype == np.uint8 else "float32"
        with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                           count=1, dtype=dtype, crs=GEOTIFF_CRS, transform=tf,
                           compress="deflate", tiled=True) as dst:
            dst.write(arr.astype(dtype), 1)
    np.savez_compressed(os.path.join(out_dir, "masks.npz"), **masks)
    with open(os.path.join(out_dir, "domain.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    log(f"wrote {len(layers)} GeoTIFFs, masks.npz and domain.json to {out_dir}")


def load(in_dir=INPUTS_DIR):
    """Layers, masks and metadata as written by save()."""
    import rasterio
    layers = {}
    for name in ("dem", "dsm_2017", "dsm_today", "cdsm", "landcover"):
        with rasterio.open(os.path.join(in_dir, f"{name}.tif")) as src:
            layers[name] = src.read(1)
            transform, crs = src.transform, src.crs
    z = np.load(os.path.join(in_dir, "masks.npz"))
    masks = {k: z[k] for k in z.files}
    with open(os.path.join(in_dir, "domain.json"), encoding="utf-8") as f:
        meta = json.load(f)
    meta["_transform"], meta["_crs"] = transform, crs
    return layers, masks, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laz", default=os.path.join("data", "nyc"))
    ap.add_argument("--footprints", default=os.path.join("data", "nyc", "footprints_crop.geojson"))
    ap.add_argument("--resource", default=os.path.join("data", "nyc", "park.geojson"))
    ap.add_argument("--resource-where", default="signname=Central Park")
    ap.add_argument("--north-pad-m", type=float, default=None,
                    help="cut the domain's north side to this many metres past the "
                         "window. Default: the full 500 m plus 300 m margin")
    ap.add_argument("--out", default=INPUTS_DIR)
    args = ap.parse_args()
    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    grid, layers, masks, meta = build(args, log=log)
    checks = check_invariants(layers, masks, grid.res_m)
    bad = [c for c in checks if not c[1]]
    for nm, ok, det in checks:
        log(f"  {'PASS' if ok else 'FAIL'}  {nm}  {det}")
    meta["invariants"] = [{"check": nm, "pass": bool(ok), "detail": det} for nm, ok, det in checks]
    save(grid, layers, masks, meta, args.out, log=log)
    if bad:
        raise SystemExit(f"{len(bad)} input invariant(s) failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
