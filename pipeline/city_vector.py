"""
The vector_parts builder: a city with no survey, built from building parts.

    TWIN_CITY=dubai .venv-heat/Scripts/python -m pipeline.city_inputs

Dubai today has no open LiDAR (DUBAI_NOTES.md). Until official 3D buildings
arrive, the city is built from open vectors and rasters:

  buildings  Overture buildings and building parts (OpenStreetMap for the
             named towers). Where a building has parts, only the parts are
             drawn, each a prism from its min_height to its height: that is
             the OpenStreetMap Simple 3D Buildings rule, and it is how Burj
             Khalifa keeps its setbacks. A height comes from `height`, else
             from `num_floors` times FLOOR_M, else the building is a gap: it
             stands at GAP_M and is listed, never hidden.
  ground     Copernicus GLO-30. It is a surface model (buildings and trees
             included), so cells within GROUND_CLEAR_M of a building are
             dropped and filled from the open ground around them, then
             smoothed. Downtown is nearly flat.
  trees      Meta / WRI canopy height v2 (about 1 m), highest value per 2 m
             cell, New York's rules after that (2 to 40 m, none within 4 m of
             a building).

One scenario, today. The official data can replace any source by changing
cities/dubai.json; the grid, rules and outputs stay the same.
"""

from __future__ import annotations

import os

import numpy as np

from . import city
from .heat_inputs import (CANOPY_MAX_M, CANOPY_MIN_M, EDGE_SKIP_M, LC_BUILDING, LC_GRASS,
                          LC_PAVED, LC_WATER)

RES_M = 2.0
FLOOR_M = 3.5           # storey height for buildings tagged with floors only
GAP_M = 10.0            # a building with no height and no floors, until it gets one
GROUND_CLEAR_M = 30.0   # one GLO-30 pixel around every building


def reproject_to(paths, grid, resampling="bilinear", log=print):
    """Several rasters in any CRS onto the grid (first valid value wins)."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    out = np.full((grid.rows, grid.cols), np.nan, np.float32)
    for p in paths:
        with rasterio.open(p) as src:
            tmp = np.full((grid.rows, grid.cols), np.nan, np.float32)
            reproject(source=rasterio.band(src, 1), destination=tmp, src_transform=src.transform,
                      src_crs=src.crs, src_nodata=src.nodata, dst_transform=grid.transform,
                      dst_crs=grid.crs, dst_nodata=np.nan, resampling=getattr(Resampling, resampling))
        take = np.isfinite(tmp) & ~np.isfinite(out)
        out[take] = tmp[take]
        log(f"  {os.path.basename(p)}: {take.mean() * 100:.0f}% of the grid")
    return out


def load_parts(cfg, bbox_ll, crs, log=print):
    """Prisms: one row per building without parts, one per part otherwise."""
    import geopandas as gpd
    import pandas as pd
    src = cfg["sources"]["buildings"]
    b = gpd.read_file(city.path(src["buildings"]), bbox=bbox_ll)
    p = gpd.read_file(city.path(src["parts"]), bbox=bbox_ll)
    with_parts = set(p["building_id"].dropna())
    b_alone = b[~b["id"].isin(with_parts)].copy()
    b_alone["building_id"] = b_alone["id"]
    cols = ["building_id", "name", "height", "min_height", "num_floors", "min_floor", "source", "geometry"]
    for df in (b_alone, p):
        for c in cols:
            if c not in df.columns:
                df[c] = None
    parts = pd.concat([b_alone[cols].assign(kind="building"), p[cols].assign(kind="part")],
                      ignore_index=True)
    # A part is named after its building when it has no name of its own.
    names = dict(zip(b["id"], b["name"]))
    parts["name"] = parts["name"].where(parts["name"].notna(), parts["building_id"].map(names))
    parts = gpd.GeoDataFrame(parts, geometry="geometry", crs=b.crs).to_crs(crs)
    h = pd.to_numeric(parts["height"], errors="coerce")
    fl = pd.to_numeric(parts["num_floors"], errors="coerce")
    how = np.where(h.notna(), "height", np.where(fl.notna(), "floors", "gap"))
    top = np.where(h.notna(), h, np.where(fl.notna(), fl * FLOOR_M, GAP_M))
    base = pd.to_numeric(parts["min_height"], errors="coerce")
    mfl = pd.to_numeric(parts["min_floor"], errors="coerce")
    base = np.where(base.notna(), base, np.where(mfl.notna(), mfl * FLOOR_M, 0.0))
    parts["top_m"], parts["base_m"], parts["height_from"] = top, base, how
    parts["area_m2"] = parts.geometry.area
    log(f"  {len(b):,} buildings, {len(p):,} parts; drawn as {len(parts):,} prisms "
        f"({len(with_parts):,} buildings drawn by their parts)")
    for k in ("height", "floors", "gap"):
        sel = parts["height_from"] == k
        log(f"    height from {k}: {int(sel.sum()):,} prisms, {parts.loc[sel, 'area_m2'].sum() / 1e4:.1f} ha")
    return parts


def build_vector_parts(cfg, log=print):
    import geopandas as gpd
    from pyproj import Transformer
    from scipy import ndimage
    from shapely.geometry import box
    from .city_inputs import burn, fill_nearest, make_grid, check_invariants  # noqa: F401
    from .grid import convergence_deg  # noqa: F401
    from .heat_inputs import shadow_reaches
    from .city_inputs import december_suns

    crs = cfg["crs"]
    hc = cfg["heat"]
    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    w, s, e, n = cfg["study"]["bbox_lonlat"]
    x0, y0 = to_crs(w, s)
    x1, y1 = to_crs(e, n)
    window_poly = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    wc = window_poly.centroid
    wll = to_ll(wc.x, wc.y)
    log(f"study: {cfg['study']['label']}, {window_poly.area / 1e4:.0f} ha")

    reach = hc.get("south_reach_m", 3000.0)
    big = window_poly.buffer(reach)
    bll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    bx0, by0 = bll(big.bounds[0], big.bounds[1])
    bx1, by1 = bll(big.bounds[2], big.bounds[3])
    parts = load_parts(cfg, (bx0, by0, bx1, by1), crs, log=log)

    # Casters: everything near the window, and tall parts whose shadow reaches it.
    near_poly = window_poly.buffer(hc["near_m"])
    suns = december_suns(wll[1], wll[0], city.utc_offset())
    from .grid import convergence_deg as _conv
    conv = _conv(crs, *wll)
    far = []
    for i, r in parts.iterrows():
        g = r.geometry
        if g is None or g.is_empty or g.intersects(near_poly) or r["top_m"] < hc["tall_m"]:
            continue
        if shadow_reaches(g, float(r["top_m"]), suns, window_poly, 1.0, conv):
            far.append(i)
    xmin, ymin, xmax, ymax = near_poly.bounds
    for i in far:
        a, b_, c, d = parts.geometry[i].bounds
        xmin, ymin, xmax, ymax = min(xmin, a), min(ymin, b_), max(xmax, c), max(ymax, d)
    m = hc["margin_m"]
    grid = make_grid(crs, (xmin - m, ymin - m, xmax + m, ymax + m), window_poly.bounds, wll)
    log(f"casters: {len(far)} tall parts beyond {hc['near_m']:.0f} m can shade the window in December; "
        f"domain {grid.cols * RES_M / 1000:.2f} x {grid.rows * RES_M / 1000:.2f} km, "
        f"{grid.rows:,} x {grid.cols:,} cells")

    dom = box(*grid.domain_polygon_crs().bounds)
    parts = parts[parts.geometry.intersects(dom)].reset_index(drop=True)

    # Buildings: the highest prism top in each cell, and a label per building.
    top = burn(list(parts.geometry), grid, parts["top_m"].to_numpy(np.float32), "float32")
    bids = {b: i + 1 for i, b in enumerate(parts["building_id"].astype(str).unique())}
    lab_vals = parts["building_id"].astype(str).map(bids).to_numpy()
    labels = burn(list(parts.geometry), grid, lab_vals.astype(np.int32), "int32")
    fp = top > 0

    # Ground from GLO-30 with the buildings taken out.
    src = cfg["sources"]
    log("ground")
    dsm30 = reproject_to([city.path(p) for p in src["dem"]["paths"]], grid, "bilinear", log=log)
    near_b = ndimage.distance_transform_edt(~fp) * RES_M <= GROUND_CLEAR_M
    g = np.where(near_b, np.nan, dsm30)
    log(f"  {np.isfinite(g).mean() * 100:.0f}% of cells are open ground {GROUND_CLEAR_M:.0f} m from any "
        "building; the rest filled from them")
    g = fill_nearest(g)
    g = ndimage.median_filter(g, size=31)          # about 60 m
    ground = g.astype(np.float32)

    log("trees")
    chm = reproject_to([city.path(p) for p in src["canopy"]["paths"]], grid, "max", log=log)
    water = np.zeros_like(fp)
    if src.get("water"):
        wtr = gpd.read_file(city.path(src["water"]["path"])).to_crs(crs)
        water = burn(list(wtr.geometry), grid, np.ones(len(wtr), int), "uint8") > 0
    d_fp = ndimage.distance_transform_edt(~fp) * RES_M
    edge = (~fp) & (d_fp <= EDGE_SKIP_M)
    raw = np.where(fp | water, 0.0, np.nan_to_num(chm, nan=0.0)).astype(np.float32)
    canopy = raw >= CANOPY_MIN_M
    cdsm = np.where(canopy & ~edge, np.minimum(raw, CANOPY_MAX_M), 0.0).astype(np.float32)

    dsm = np.where(fp, ground + top, ground).astype(np.float32)
    lc = np.full((grid.rows, grid.cols), LC_PAVED, np.uint8)
    lc[(cdsm > 0) & ~fp] = LC_GRASS            # irrigated planting under the trees
    lc[water & ~fp] = LC_WATER
    lc[fp] = LC_BUILDING

    win = burn([window_poly], grid, np.ones(1, int), "uint8") > 0
    rr, cc = np.mgrid[0:grid.rows, 0:grid.cols]
    mc = int(round(m / RES_M))
    inner = (rr >= mc) & (rr < grid.rows - mc) & (cc >= mc) & (cc < grid.cols - mc)
    report = win & inner & ~water & ~fp
    fx, fy = to_crs(*cfg["focus"]["lonlat"])
    cx_, cy_ = grid.cell_centres_crs((0, grid.rows, 0, grid.cols))
    d_focus = np.hypot(cx_ - fx, cy_ - fy).astype(np.float32)
    zero = np.zeros_like(fp)

    gaps = parts[parts["height_from"] == "gap"].sort_values("area_m2", ascending=False)
    in_win = parts[parts.geometry.intersects(window_poly)]
    meta = {
        "city": cfg["id"],
        "grid": {"crs_build": crs, "crs_geotiff": cfg["crs_geotiff"], "res_m": RES_M,
                 "rows": grid.rows, "cols": grid.cols, "x0_m": grid.x0, "y0_m": grid.y0,
                 "conv_deg": grid.conv_deg, "transform_m": list(grid.transform)[:6]},
        "site": {"lat": wll[1], "lon": wll[0], "utc_offset": city.utc_offset()},
        "focus": {**cfg["focus"], "xy_m": [fx, fy]},
        "window_desc": cfg["study"]["label"], "near_m": hc["near_m"], "margin_m": m,
        "sources": {k: v.get("credit") for k, v in src.items() if isinstance(v, dict)},
        "heights": {"rule": f"height, else num_floors x {FLOOR_M} m, else a gap at {GAP_M} m",
                    "prisms": int(len(parts)),
                    "from_height_ha": round(float(parts.loc[parts.height_from == "height", "area_m2"].sum() / 1e4), 1),
                    "from_floors_ha": round(float(parts.loc[parts.height_from == "floors", "area_m2"].sum() / 1e4), 1),
                    "gap_ha": round(float(gaps["area_m2"].sum() / 1e4), 1),
                    "window_gap_share_of_area": round(float(in_win.loc[in_win.height_from == "gap", "area_m2"].sum()
                                                            / max(in_win["area_m2"].sum(), 1.0)), 3),
                    "largest_gaps": [{"name": r["name"], "area_m2": round(float(r["area_m2"])),
                                      "lonlat": [round(v, 6) for v in to_ll(r.geometry.centroid.x, r.geometry.centroid.y)]}
                                     for _, r in gaps.head(25).iterrows()]},
        "tallest": [{"name": r["name"], "top_m": round(float(r["top_m"]), 1), "kind": r["kind"]}
                    for _, r in parts.sort_values("top_m", ascending=False).head(15).iterrows()],
        "ground": {"source": "Copernicus GLO-30, buildings removed",
                   "min_m": round(float(ground.min()), 1), "max_m": round(float(ground.max()), 1)},
        "canopy": {"cells": int((cdsm > 0).sum()), "share": round(float((cdsm > 0).mean()), 4)},
        "areas_ha": {"window": float(win.sum() * RES_M ** 2 / 1e4),
                     "report": float(report.sum() * RES_M ** 2 / 1e4)},
        "grown_buildings": [],
    }
    log(f"tallest: " + ", ".join(f"{t['name'] or '?'} {t['top_m']:.0f} m" for t in meta["tallest"][:6]))
    log(f"heights: {meta['heights']['from_height_ha']} ha from height tags, {meta['heights']['from_floors_ha']} ha "
        f"from floors, {meta['heights']['gap_ha']} ha gaps; in the window {meta['heights']['window_gap_share_of_area'] * 100:.0f}% "
        "of building area is a gap")
    log(f"ground {meta['ground']['min_m']} to {meta['ground']['max_m']} m; trees on "
        f"{meta['canopy']['share'] * 100:.1f}% of cells; window {meta['areas_ha']['window']:.0f} ha, "
        f"reported {meta['areas_ha']['report']:.0f} ha")
    layers = {"dem": ground, "dsm_today": dsm, "cdsm": cdsm, "landcover": lc}
    masks = {"window": win, "report": report, "park": lc == LC_GRASS, "water": water & ~fp,
             "building": fp, "nodata": zero, "changed": zero, "raised": zero, "explained": zero,
             "dist_changed": np.full(fp.shape, 1e6, np.float32), "dist_focus": d_focus, "labels": labels}
    keep = parts.drop(columns=["geometry"]).copy()
    meta["_buildings"] = gpd.GeoDataFrame(
        keep.assign(label=lab_vals.astype(np.int32), year=0, grown=False,
                    identificatie=parts["building_id"].astype(str)),
        geometry=parts.geometry, crs=crs)[["label", "identificatie", "name", "top_m", "base_m",
                                           "height_from", "year", "grown", "geometry"]]
    return grid, layers, masks, meta
