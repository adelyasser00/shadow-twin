"""
Inputs for any city: ground, buildings, trees and land cover on a north-up
2 m grid, for each scenario, in the format the heat, wind and shade engines
already read (pipeline.heat_inputs.save).

    TWIN_CITY=rotterdam .venv-heat/Scripts/python -m pipeline.city_inputs
    TWIN_CITY=nyc       .venv-heat/Scripts/python -m pipeline.city_inputs --out data/cities/nyc/inputs_check

A city is described in cities/<id>.json. What goes in:

  buildings  vector outlines with a year, and either measured roofs (a survey
             surface) or heights per part (prisms, building:part style)
  surface    a LiDAR point cloud or a DSM raster, one per survey; optional
  ground     the survey's ground, or a DEM
  trees      from the survey surface, or a canopy height raster
  city       projection, clock, weather file, analysis days, study area and
             the reference spot for wind (see pipeline.city)

Builders, by what the data allows:

  nyc_lidar     New York. Point clouds and NYC Building Footprints, through
                pipeline.heat_inputs exactly as published. Kept as it is, so
                the published numbers reproduce; run with --out to compare.
  survey_grown  Two surveys, one before and one after (Rotterdam: AHN3 and
                AHN5). The older survey is the "before" city, measured. Today
                is that surface raised where a building grew by more than
                GROWN_M, to the newer survey's measured roof. The same rule
                as New York's, with a measured roof in place of a prism.
  vector_parts  No survey (Dubai today): every building part stands as a
                prism from its base to its top, on a DEM, with trees from a
                canopy raster. One scenario.

Rules shared with New York, from pipeline.heat_inputs: canopy is surface
minus ground off buildings and water, 2 to 40 m, and nothing within 4 m of a
footprint (roofs spill past their walls); grown means more than 6 m above
what the older survey measured in the footprint; land cover is paved 1,
building 2, grass 5, water 7.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np

from . import city
from .heat_inputs import (CANOPY_MAX_M, CANOPY_MIN_M, EDGE_SKIP_M, GROWN_M, LC_BUILDING,
                          LC_GRASS, LC_PAVED, LC_WATER, shadow_reaches)

RES_M = 2.0
# Trees in only one of the two surveys are not counted as trees: what stands
# in one survey and not the other is a crane, a ship, a parked truck or a
# building that came down. See canopy_two_surveys.
GREEN_KINDS = {"groenvoorziening", "grasland overig", "grasland agrarisch", "loofbos",
               "naaldbos", "gemengd bos", "struiken", "houtwal", "heide", "moeras",
               "rietland", "kwelder", "duin"}


# ------------------------------------------------------------------ grid
def make_grid(crs, bounds, window_bounds, centre_lonlat, res_m=RES_M):
    """A north-up metric Grid on a res_m lattice, covering bounds, frame on the window."""
    from .grid import Grid, convergence_deg
    lon, lat = centre_lonlat
    conv = convergence_deg(crs, lon, lat)
    xmin = math.floor(bounds[0] / res_m) * res_m
    ymin = math.floor(bounds[1] / res_m) * res_m
    xmax = math.ceil(bounds[2] / res_m) * res_m
    ymax = math.ceil(bounds[3] / res_m) * res_m
    cols, rows = int(round((xmax - xmin) / res_m)), int(round((ymax - ymin) / res_m))
    # bearing -conv makes grid "up" the projection's own north, so rows and
    # columns are the projection's axes and the GeoTIFF is north-up.
    g = Grid(crs=crs, unit_m=1.0, res_m=res_m, bearing_deg=-conv, conv_deg=conv,
             x0=xmin, y0=ymax, rows=rows, cols=cols, frame=(0, rows, 0, cols),
             centre_lat=lat, centre_lon=lon)
    wx0, wy0, wx1, wy1 = window_bounds
    r0, c0 = g.crs_to_rowcol(wx0, wy1)
    r1, c1 = g.crs_to_rowcol(wx1, wy0)
    g.frame = (max(0, int(math.floor(r0))), min(rows, int(math.ceil(r1))),
               max(0, int(math.floor(c0))), min(cols, int(math.ceil(c1))))
    return g


# --------------------------------------------------------------- rasters
def read_raster(paths, grid, how="max", cache=None, log=print):
    """
    One float32 layer on the grid from GeoTIFF tiles in the grid's CRS, finer
    than the grid by a whole factor (AHN: 0.5 m into 2 m is 4 x 4). Each grid
    cell takes the max, mean or min of the valid source pixels in it: max is a
    surface's highest return, as the New York LiDAR is gridded. NaN where a
    cell has no valid pixel. Remote tiles (/vsicurl/) are read window by
    window; `cache` keeps the result as a .npy.
    """
    import rasterio
    from rasterio.windows import from_bounds
    if cache and os.path.exists(cache):
        arr = np.load(cache)
        if arr.shape == (grid.rows, grid.cols):
            log(f"  {os.path.basename(cache)} from cache")
            return arr
    tf = grid.transform
    x0, y1 = tf.c, tf.f
    x1, y0 = x0 + grid.cols * grid.res_m, y1 - grid.rows * grid.res_m
    acc = np.full((grid.rows, grid.cols), np.nan, np.float32)
    for p in paths:
        try:
            src = rasterio.open(p)
        except Exception as e:
            log(f"  skip {p}: {type(e).__name__}")
            continue
        with src:
            b = src.bounds
            ix0, iy0, ix1, iy1 = max(b.left, x0), max(b.bottom, y0), min(b.right, x1), min(b.top, y1)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            res = src.res[0]
            k = int(round(grid.res_m / res))
            if abs(k * res - grid.res_m) > 1e-6:
                raise SystemExit(f"{p}: {res} m pixels do not divide {grid.res_m} m cells")
            # Snap the overlap to whole grid cells.
            cx0 = math.ceil((ix0 - x0) / grid.res_m - 1e-6)
            cx1 = math.floor((ix1 - x0) / grid.res_m + 1e-6)
            cy0 = math.ceil((y1 - iy1) / grid.res_m - 1e-6)
            cy1 = math.floor((y1 - iy0) / grid.res_m + 1e-6)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            win = from_bounds(x0 + cx0 * grid.res_m, y1 - cy1 * grid.res_m,
                              x0 + cx1 * grid.res_m, y1 - cy0 * grid.res_m, src.transform)
            win = win.round_offsets().round_lengths()
            t0 = time.time()
            a = src.read(1, window=win).astype(np.float32)
            nd = src.nodata
            bad = ~np.isfinite(a) | (a > 1e30) | (a < -1e4)
            if nd is not None and np.isfinite(nd):
                bad |= a == nd
            a[bad] = np.nan
            h, w = (cy1 - cy0), (cx1 - cx0)
            a = a[:h * k, :w * k].reshape(h, k, w, k)
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    if how == "max":
                        r = np.nanmax(a, axis=(1, 3))
                    elif how == "min":
                        r = np.nanmin(a, axis=(1, 3))
                    else:
                        r = np.nanmean(a, axis=(1, 3))
            sub = acc[cy0:cy1, cx0:cx1]
            take = np.isfinite(r) & ~np.isfinite(sub)
            sub[take] = r[take]
            log(f"  {os.path.basename(p)}: {h} x {w} cells in {time.time() - t0:.0f} s")
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, acc)
    return acc


def fill_nearest(arr):
    """NaN cells take the nearest finite value, as the New York ground gaps are filled."""
    from scipy import ndimage
    bad = ~np.isfinite(arr)
    if not bad.any():
        return arr
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return arr[idx[0], idx[1]]


def burn(geoms, grid, values=None, dtype="int32", all_touched=False):
    """Rasterise polygons; with values, later (higher) values win."""
    from rasterio import features
    if not len(geoms):
        return np.zeros((grid.rows, grid.cols), dtype)
    if values is None:
        values = np.arange(1, len(geoms) + 1)
    order = np.argsort(values, kind="stable")
    shapes = [(geoms[i], values[i]) for i in order]
    return features.rasterize(shapes, out_shape=(grid.rows, grid.cols), transform=grid.transform,
                              fill=0, dtype=dtype, all_touched=all_touched)


# ------------------------------------------------------------- buildings
def load_3dbag(tiles_dir, bbox, log=print):
    """
    3DBAG footprints (layer pand) inside bbox, one row per building: geometry,
    id, year, ground (b3_h_maaiveld, NAP) and which survey 3DBAG fitted it to.
    Heights from 3DBAG's roof parts are not used: for buildings finished
    around the newest survey they can be wildly off (Zalmhaven's podium
    carries a 965 m roof part; AHN5 measures 70 m), so roofs come from the
    surveys themselves.
    """
    import geopandas as gpd
    import pandas as pd
    cols = ["identificatie", "oorspronkelijkbouwjaar", "b3_h_maaiveld", "b3_pw_bron",
            "b3_pw_datum", "b3_pw_selectie_reden", "status"]
    parts = []
    for f in sorted(glob.glob(os.path.join(tiles_dir, "*.gpkg"))):
        parts.append(gpd.read_file(f, layer="pand", columns=cols, bbox=tuple(bbox)))
    df = pd.concat(parts, ignore_index=True).drop_duplicates("identificatie")
    df = df[df.geometry.notna() & ~df.geometry.is_empty]
    df = df[~df["status"].astype(str).str.contains("gesloopt|niet gerealiseerd|ingetrokken", case=False)]
    df = df.set_crs("EPSG:28992", allow_override=True).reset_index(drop=True)
    df["year"] = pd.to_numeric(df["oorspronkelijkbouwjaar"], errors="coerce").fillna(0).astype(int)
    log(f"  3DBAG: {len(df):,} buildings in the box")
    return df


def label_stat(values, labels, n, q):
    """Per-label quantile of finite values (NaN where none)."""
    from .nyc_footprints import label_quantile
    v = np.where(np.isfinite(values), values, np.nan).astype(np.float32)
    return label_quantile(v, labels, n, q)


# ------------------------------------------------------------- the study
def study_polygon(cfg, crs, grown_points=None, log=print):
    """The analysis window: the named areas, clipped near the grown towers."""
    import geopandas as gpd
    from shapely.ops import unary_union
    st = cfg["study"]
    g = gpd.read_file(city.path(st["polygons"])).to_crs(crs)
    sel = g[g[st["name_field"]].isin(st["names"])]
    missing = set(st["names"]) - set(sel[st["name_field"]])
    if missing:
        raise SystemExit(f"study areas not found: {sorted(missing)}")
    poly = unary_union(list(sel.geometry))
    if st.get("within_m_of_grown") and grown_points:
        near = unary_union([p.buffer(st["within_m_of_grown"]) for p in grown_points])
        poly = poly.intersection(near)
    log(f"study: {', '.join(st['names'])}"
        + (f", within {st['within_m_of_grown']:.0f} m of the {len(grown_points)} towers of "
           f"{st['grown_min_m']:.0f} m or more that grew" if grown_points else "")
        + f": {poly.area / 1e4:.1f} ha (polygon, water included)")
    return poly


def december_suns(lat, lon, utc_offset, step_min=5, pad_min=15):
    """Sun positions through the December window (1.5 h after sunrise to 1.5 h before sunset)."""
    from .city_time import day_window
    from datetime import timedelta, timezone
    from .solar import sun_position
    w0, w1, _, _ = day_window("2026-12-21", lat, lon, utc_offset)
    t, out = w0 - timedelta(minutes=pad_min), []
    while t <= w1 + timedelta(minutes=pad_min):
        p = sun_position((t - timedelta(hours=utc_offset)).replace(tzinfo=timezone.utc), lat, lon)
        if p.altitude > 0:
            out.append(p)
        t += timedelta(minutes=step_min)
    return out


# ------------------------------------------------------------ survey_grown
def build_survey_grown(cfg, log=print):
    from pyproj import Transformer
    from scipy import ndimage
    from shapely.geometry import Point, box
    from .grid import convergence_deg

    crs = cfg["crs"]
    hc = cfg["heat"]
    src = cfg["sources"]
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    survey_year = int(cfg["survey_year"])

    # ---- buildings in a wide box around the study areas -----------------
    import geopandas as gpd
    st = gpd.read_file(city.path(cfg["study"]["polygons"])).to_crs(crs)
    st = st[st[cfg["study"]["name_field"]].isin(cfg["study"]["names"])]
    sb = st.total_bounds
    reach = hc.get("south_reach_m", 3000.0)
    wide = (sb[0] - reach, sb[1] - reach, sb[2] + reach, sb[3] + reach)
    bag = load_3dbag(city.path(src["buildings"]["tiles"]), wide, log=log)

    # Heights for choosing the domain, from the 5 m surveys (cheap to read
    # for the whole box). Today's survey decides which towers grew and which
    # far towers can shade the window.
    wide_grid = make_grid(crs, wide, wide, to_ll((sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2), res_m=10.0)
    log(f"domain search on a {wide_grid.rows} x {wide_grid.cols} grid of 10 m cells")
    cache = city.path(src["survey_today"].get("cache", os.path.join(cfg["paths"]["data"], "cache")))
    s5_today = read_raster([p.replace("03a_DSM_50cm/2023_R_", "03b_DSM_5m/2023_R5_") for p in src["survey_today"]["dsm"]],
                           wide_grid, "max", cache=os.path.join(cache, "search_today_10m.npy"), log=log)
    s5_before = read_raster(src["survey_before"]["dsm"], wide_grid, "max",
                            cache=os.path.join(cache, "search_before_10m.npy"), log=log)
    labs_w = burn(list(bag.geometry), wide_grid)
    n = len(bag)
    gnd = np.concatenate([[0.0], bag["b3_h_maaiveld"].fillna(0).to_numpy(np.float32)])
    h_today = label_stat(s5_today - gnd[labs_w], labs_w, n, 0.95)
    h_before = label_stat(s5_before - gnd[labs_w], labs_w, n, 0.95)
    bag["h_today_m"] = np.nan_to_num(h_today[1:], nan=0.0)
    bag["h_before_m"] = h_before[1:]
    tall_grown = bag[(bag["h_today_m"] >= cfg["study"]["grown_min_m"]) & np.isfinite(bag["h_before_m"])
                     & (bag["h_today_m"] > bag["h_before_m"] + GROWN_M)]
    log(f"  {len(tall_grown)} buildings of {cfg['study']['grown_min_m']:.0f} m or more grew "
        f"between the surveys (5 m check):")
    for _, r in tall_grown.sort_values("h_today_m", ascending=False).iterrows():
        c = r.geometry.centroid
        lo, la = to_ll(c.x, c.y)
        log(f"    {r['h_today_m']:6.1f} m (before {r['h_before_m']:.1f}), built {r['year']}, "
            f"{la:.5f}, {lo:.5f}  {r['identificatie']}")

    # Kept for the wind canvas, which reaches past the heat grid.
    wide_path = os.path.join(city.data_dir(), "buildings_wide.gpkg")
    bag[["identificatie", "year", "b3_h_maaiveld", "h_today_m", "h_before_m", "geometry"]].to_file(
        wide_path, driver="GPKG")
    log(f"  {len(bag):,} buildings with measured heights written to {wide_path}")

    # ---- window and casters --------------------------------------------
    window_poly = study_polygon(cfg, crs, [r.geometry.centroid for _, r in tall_grown.iterrows()], log=log)
    wc = window_poly.centroid
    wll = to_ll(wc.x, wc.y)
    near_poly = window_poly.buffer(hc["near_m"])
    suns = december_suns(wll[1], wll[0], city.utc_offset())
    conv = convergence_deg(crs, *wll)
    far = []
    for i, r in bag.iterrows():
        g = r.geometry
        if g.intersects(near_poly):
            continue
        h = r["h_today_m"]
        if h < hc["tall_m"]:
            continue
        if g.centroid.distance(wc) > reach + 2000:
            continue
        if shadow_reaches(g, h, suns, window_poly, 1.0, conv):
            far.append(i)
    log(f"casters: {int(bag.geometry.intersects(near_poly).sum()):,} buildings within "
        f"{hc['near_m']:.0f} m of the window; {len(far)} taller ones beyond can shade it in December")
    xmin, ymin, xmax, ymax = near_poly.bounds
    for i in far:
        bx0, by0, bx1, by1 = bag.geometry[i].bounds
        xmin, ymin, xmax, ymax = min(xmin, bx0), min(ymin, by0), max(xmax, bx1), max(ymax, by1)
    m = hc["margin_m"]
    grid = make_grid(crs, (xmin - m, ymin - m, xmax + m, ymax + m), window_poly.bounds, wll)
    log(f"domain {grid.cols * RES_M / 1000:.2f} km east-west by {grid.rows * RES_M / 1000:.2f} km "
        f"north-south, {grid.rows:,} x {grid.cols:,} cells at {RES_M} m, north-up ({crs})")

    # ---- surveys on the grid --------------------------------------------
    cover = survey_bounds(src["survey_before"]["dsm"])
    gx0, gy1 = grid.x0, grid.y0
    gx1, gy0 = gx0 + grid.cols * RES_M, gy1 - grid.rows * RES_M
    if gx0 < cover[0] or gy0 < cover[1] or gx1 > cover[2] or gy1 > cover[3]:
        raise SystemExit(f"the domain {gx0:.0f}, {gy0:.0f}, {gx1:.0f}, {gy1:.0f} runs past the older "
                         f"survey's tiles {cover}; add the neighbouring tiles to sources.survey_before")
    cdir = os.path.join(city.data_dir(), "cache", f"grid_{grid.rows}x{grid.cols}_{int(grid.x0)}_{int(grid.y0)}")
    log("reading the surveys")
    dsm_b = read_raster(src["survey_before"]["dsm"], grid, "max", os.path.join(cdir, "dsm_before.npy"), log)
    dtm_b = read_raster(src["survey_before"]["dtm"], grid, "mean", os.path.join(cdir, "dtm_before.npy"), log)
    dsm_t = read_raster(src["survey_today"]["dsm"], grid, "max", os.path.join(cdir, "dsm_today.npy"), log)
    ground = fill_nearest(dtm_b).astype(np.float32)
    log(f"  ground measured on {np.isfinite(dtm_b).mean() * 100:.1f}% of cells, the rest filled from "
        f"the nearest measured ground; before-surface on {np.isfinite(dsm_b).mean() * 100:.1f}%, "
        f"today's on {np.isfinite(dsm_t).mean() * 100:.1f}%")

    # Water and green from the BGT, the national large-scale topography.
    water = np.zeros((grid.rows, grid.cols), bool)
    green = np.zeros((grid.rows, grid.cols), bool)
    dom_ll = grid.bounds_lonlat((0, grid.rows, 0, grid.cols), pad_deg=0.001)
    if src.get("water"):
        w = gpd.read_file(city.path(src["water"]["path"]), bbox=dom_ll).to_crs(crs)
        # Dry ditches are mapped as water parts but hold water only some of the time.
        w = w[~w["type"].astype(str).str.contains("droge sloot")]
        water = burn(list(w.geometry), grid, np.ones(len(w), int), "uint8") > 0
    if src.get("green") and not os.path.exists(city.path(src["green"]["path"])):
        log(f"  WARNING: no green layer at {src['green']['path']}; everything open is paved")
    elif src.get("green"):
        gr = gpd.read_file(city.path(src["green"]["path"]), bbox=dom_ll).to_crs(crs)
        gr = gr[gr["fysiek_voorkomen"].isin(GREEN_KINDS)]
        green = burn(list(gr.geometry), grid, np.ones(len(gr), int), "uint8") > 0

    # Buildings on this grid.
    inside = bag[bag.geometry.intersects(box(*grid.domain_polygon_crs().bounds))].reset_index(drop=True)
    labels = burn(list(inside.geometry), grid)
    n = len(inside)
    fp = labels > 0
    gl = np.concatenate([[0.0], inside["b3_h_maaiveld"].to_numpy(np.float32)])
    log(f"  {n:,} footprints on the grid, {fp.mean() * 100:.1f}% of it")

    nodata = np.zeros_like(fp)        # both surveys cover the whole domain
    surf_b = np.where(np.isfinite(dsm_b), dsm_b, ground).astype(np.float32)
    surf_b = np.maximum(surf_b, ground)
    surf_t = np.where(np.isfinite(dsm_t), dsm_t, surf_b).astype(np.float32)
    hb = surf_b - ground
    ht = surf_t - ground

    # ---- grown, New York's rule with measured roofs ----------------------
    top_b = label_stat(hb, labels, n, 0.99)
    top_t = label_stat(ht, labels, n, 0.99)
    grown = np.isfinite(top_t) & np.isfinite(top_b) & (top_t > top_b + GROWN_M)
    grown[0] = False
    raise_all = fp & (surf_t > surf_b + 3.0)
    raise_ = raise_all & grown[labels]
    surf_now = np.where(raise_, surf_t, surf_b).astype(np.float32)
    log(f"today: {int(grown[1:].sum())} buildings stand over {GROWN_M:.0f} m above what the "
        f"{cfg['scenarios'][0]} survey measured in them, {int(raise_.sum()):,} cells raised to "
        f"their {cfg.get('today_label', 'today')} roofs")
    late = inside["year"].to_numpy() >= 2024
    if late.any():
        log(f"  {int(late.sum())} buildings on the grid were finished in 2024 or later, after the "
            "newest survey: they stand as the survey saw them")

    # ---- the layers ------------------------------------------------------
    dem = ground
    dsm_before = np.where(fp, surf_b, dem).astype(np.float32)
    dsm_today = np.where(fp, surf_now, dem).astype(np.float32)
    d_fp = ndimage.distance_transform_edt(~fp) * RES_M
    edge = (~fp) & (d_fp <= EDGE_SKIP_M)
    raw = canopy_two_surveys(hb, ht, fp | water)
    canopy = raw >= CANOPY_MIN_M
    removed_edge = canopy & edge
    cdsm = np.where(canopy & ~edge, np.minimum(raw, CANOPY_MAX_M), 0.0).astype(np.float32)
    one_only = (~fp & ~water & (np.maximum(hb, ht) >= CANOPY_MIN_M) & (raw < CANOPY_MIN_M))
    veg = cdsm > 0
    pct = np.percentile(cdsm[veg], [5, 25, 50, 75, 95, 99, 100]) if veg.any() else [0] * 7
    log(f"canopy: {int(veg.sum()):,} cells ({veg.mean() * 100:.1f}%); the edge rule removed "
        f"{int(removed_edge.sum()):,}; {int(one_only.sum()):,} cells tall in one survey only were "
        "not counted as trees (cranes, ships, vehicles, buildings gone)")

    lc = np.full((grid.rows, grid.cols), LC_PAVED, np.uint8)
    lc[green & ~water & ~fp] = LC_GRASS
    lc[water & ~fp] = LC_WATER
    lc[fp] = LC_BUILDING

    # ---- masks -----------------------------------------------------------
    win = burn([window_poly], grid, np.ones(1, int), "uint8") > 0
    window = win & ~nodata
    rr, cc = np.mgrid[0:grid.rows, 0:grid.cols]
    mc = int(round(m / RES_M))
    inner = (rr >= mc) & (rr < grid.rows - mc) & (cc >= mc) & (cc < grid.cols - mc)
    report = window & inner & ~water & ~fp
    changed = dsm_today != dsm_before
    explained = raise_.copy()
    d_changed = (ndimage.distance_transform_edt(~changed) * RES_M).astype(np.float32)
    fx, fy = to_crs(*cfg["focus"]["lonlat"])
    cx_, cy_ = grid.cell_centres_crs((0, grid.rows, 0, grid.cols))
    d_focus = np.hypot(cx_ - fx, cy_ - fy).astype(np.float32)
    log(f"window {window.sum() * RES_M ** 2 / 1e4:.1f} ha on the grid; reported "
        f"{report.sum() * RES_M ** 2 / 1e4:.1f} ha (open ground: no water, no buildings, "
        f"{m:.0f} m from the edges)")

    meta = {
        "city": cfg["id"],
        "grid": {"crs_build": crs, "crs_geotiff": cfg["crs_geotiff"], "res_m": RES_M,
                 "rows": grid.rows, "cols": grid.cols, "x0_m": grid.x0, "y0_m": grid.y0,
                 "conv_deg": grid.conv_deg, "transform_m": list(grid.transform)[:6]},
        "site": {"lat": wll[1], "lon": wll[0], "utc_offset": city.utc_offset()},
        "focus": {**cfg["focus"], "xy_m": [fx, fy]},
        "window_desc": cfg["study"]["label"], "near_m": hc["near_m"], "margin_m": m,
        "casters_far": [{"height_m": round(float(bag["h_today_m"][i]), 1), "year": int(bag["year"][i]),
                         "id": bag["identificatie"][i],
                         "lonlat": [round(v, 6) for v in to_ll(bag.geometry[i].centroid.x, bag.geometry[i].centroid.y)]}
                        for i in sorted(far, key=lambda i: -bag["h_today_m"][i])],
        "surveys": {"before": src["survey_before"]["credit"], "today": src["survey_today"]["credit"],
                    "buildings": src["buildings"]["credit"]},
        "heights": {cfg["scenarios"][0]: "measured: the older survey's surface inside each footprint",
                    "today": "the older survey, raised to the newer survey's roof where a building grew"},
        "scenario": {"buildings_grown": int(grown[1:].sum()), "cells_raised": int(raise_.sum()),
                     "cells_changed": int(changed.sum())},
        "canopy": {"cells": int(veg.sum()), "edge_rule_removed": int(removed_edge.sum()),
                   "one_survey_only": int(one_only.sum()),
                   "percentiles_5_25_50_75_95_99_max": [round(float(v), 2) for v in pct]},
        "areas_ha": {"window": float(window.sum() * RES_M ** 2 / 1e4),
                     "report": float(report.sum() * RES_M ** 2 / 1e4)},
        "grown_buildings": [],
    }
    for i in np.nonzero(grown)[0]:
        cells = raise_ & (labels == i)
        if not cells.any():
            continue
        r = inside.iloc[i - 1]
        p = r.geometry.representative_point()
        lo, la = to_ll(p.x, p.y)
        meta["grown_buildings"].append({
            "height_m": round(float(top_t[i]), 1), "year": int(r["year"]), "id": r["identificatie"],
            "lidar_p99_m": None if not np.isfinite(top_b[i]) else round(float(top_b[i]), 1),
            "cells_raised": int(cells.sum()), "lonlat": [round(lo, 6), round(la, 6)]})
    meta["grown_buildings"].sort(key=lambda b: -b["height_m"])
    for b in meta["grown_buildings"][:15]:
        log(f"  grown: {b['height_m']:6.1f} m (before {b['lidar_p99_m']} m), built {b['year']}, "
            f"{b['cells_raised']:,} cells, {b['lonlat'][1]:.5f}, {b['lonlat'][0]:.5f}")

    inside = inside.assign(label=np.arange(1, n + 1), grown=grown[1:])
    meta["_buildings"] = inside[["label", "identificatie", "year", "b3_h_maaiveld", "grown", "geometry"]]
    before = cfg["scenarios"][0]
    layers = {"dem": dem, f"dsm_{before}": dsm_before, "dsm_today": dsm_today, "cdsm": cdsm, "landcover": lc}
    masks = {"window": window, "report": report, "park": lc == LC_GRASS, "water": water & ~fp,
             "building": fp, "nodata": nodata, "changed": changed, "raised": raise_,
             "explained": explained, "dist_changed": d_changed, "dist_focus": d_focus,
             "labels": labels}
    return grid, layers, masks, meta


def survey_bounds(paths):
    """Union bounding box of a survey's tiles (they tile a rectangle)."""
    import rasterio
    bs = []
    for p in paths:
        with rasterio.open(p) as src:
            bs.append(tuple(src.bounds))
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def canopy_two_surveys(h_before, h_today, exclude):
    """
    Tree height where both surveys see something tall off buildings and water:
    the lower of the two. A tree stands in both; a crane, a ship or a building
    that came down stands in one.
    """
    raw = np.minimum(h_before, h_today)
    return np.where(exclude, 0.0, raw).astype(np.float32)


# --------------------------------------------------------------- checks
def check_invariants(layers, masks, scenarios, res_m=RES_M):
    """Every rule the inputs must satisfy, for any city. Returns [(name, ok, detail)]."""
    before = scenarios[0]
    dem, d0, dnow = layers["dem"], layers[f"dsm_{before}"], layers["dsm_today"]
    cdsm, lc = layers["cdsm"], layers["landcover"]
    b, w = masks["building"], masks["water"]
    out = [("no NaN or inf in DEM, DSMs, CDSM",
            all(np.isfinite(a).all() for a in (dem, d0, dnow, cdsm)), "")]
    for nm, d in ((before, d0), ("today", dnow)):
        lo = float((d - dem).min())
        out.append((f"DSM {nm} >= DEM", lo >= 0.0, f"min DSM - DEM {lo:.3f} m"))
    out.append(("CDSM >= 0", float(cdsm.min()) >= 0.0, f"min {float(cdsm.min()):.3f}"))
    out.append(("CDSM = 0 on building and water cells", not (cdsm[b | w] != 0).any(),
                f"{int((cdsm[b | w] != 0).sum())} cells"))
    out.append(("CDSM <= 40 m", float(cdsm.max()) <= CANOPY_MAX_M, f"max {float(cdsm.max()):.2f}"))
    lo = float((dnow - d0).min())
    out.append((f"DSM today >= DSM {before}", lo >= 0.0, f"min today - {before} {lo:.3f} m"))
    ch = dnow != d0
    stray = ch & ~masks["explained"]
    out.append((f"today differs from {before} only on grown or new buildings", not stray.any(),
                f"{int(ch.sum()):,} cells differ, {int(stray.sum())} unexplained"))
    codes = set(np.unique(lc).tolist())
    out.append(("land cover codes in {1, 2, 5, 7}", codes <= {1, 2, 5, 7}, str(sorted(codes))))
    out.append(("land cover 2 exactly on footprints", bool(((lc == 2) == b).all()), ""))
    out.append(("pixel size 2.0 m", abs(res_m - 2.0) < 1e-9, f"{res_m}"))
    return out


def save(grid, layers, masks, meta, out_dir, crs, log=print):
    """Same files as pipeline.heat_inputs.save, in the city's own metric CRS."""
    import rasterio
    os.makedirs(out_dir, exist_ok=True)
    tf = grid.transform
    if abs(tf.b) > 1e-9 or abs(tf.d) > 1e-9 or tf.a <= 0 or tf.e >= 0 or grid.unit_m != 1.0:
        raise SystemExit("city grid is not north-up in metres")
    for name, arr in layers.items():
        dtype = "uint8" if arr.dtype == np.uint8 else "float32"
        with rasterio.open(os.path.join(out_dir, f"{name}.tif"), "w", driver="GTiff",
                           height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                           crs=crs, transform=tf, compress="deflate", tiled=True) as dst:
            dst.write(arr.astype(dtype), 1)
    np.savez_compressed(os.path.join(out_dir, "masks.npz"), **masks)
    bld = meta.pop("_buildings", None)
    if bld is not None:
        bld.to_file(os.path.join(out_dir, "buildings.gpkg"), driver="GPKG")
    with open(os.path.join(out_dir, "domain.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1, default=float)
    log(f"wrote {len(layers)} GeoTIFFs, masks.npz and domain.json to {out_dir}")


BUILDERS = {"survey_grown": build_survey_grown}


def build(cfg, log=print):
    kind = cfg["builder"]
    if kind == "vector_parts":
        from .city_vector import build_vector_parts
        return build_vector_parts(cfg, log=log)
    if kind not in BUILDERS:
        raise SystemExit(f"no builder '{kind}'")
    return BUILDERS[kind](cfg, log=log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="default: the city's heat inputs folder")
    args = ap.parse_args()
    cfg = city.config()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    if cfg["builder"] == "nyc_lidar":
        # New York's published inputs, built the way they were published.
        from . import heat_inputs as H
        out = args.out or H.INPUTS_DIR
        if os.path.abspath(out) == os.path.abspath(H.INPUTS_DIR) and os.path.exists(os.path.join(out, "domain.json")):
            raise SystemExit(f"{out} holds the published inputs; give --out to rebuild them elsewhere")
        ns = argparse.Namespace(laz=os.path.join("data", "nyc"),
                                footprints=os.path.join("data", "nyc", "footprints_crop.geojson"),
                                resource=os.path.join("data", "nyc", "park.geojson"),
                                resource_where="signname=Central Park", north_pad_m=None)
        grid, layers, masks, meta = H.build(ns, log=log)
        checks = H.check_invariants(layers, masks, grid.res_m)
        meta["invariants"] = [{"check": nm, "pass": bool(ok), "detail": det} for nm, ok, det in checks]
        H.save(grid, layers, masks, meta, out, log=log)
        return 0 if all(c[1] for c in checks) else 1

    grid, layers, masks, meta = build(cfg, log=log)
    checks = check_invariants(layers, masks, city.scenarios(), grid.res_m)
    for nm, ok, det in checks:
        log(f"  {'PASS' if ok else 'FAIL'}  {nm}  {det}")
    meta["invariants"] = [{"check": nm, "pass": bool(ok), "detail": det} for nm, ok, det in checks]
    out = args.out or os.path.join(city.heat_dir(), "inputs")
    save(grid, layers, masks, meta, out, cfg["crs_geotiff"], log=log)
    return 0 if all(c[1] for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
