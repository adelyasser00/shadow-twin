"""
Rasterise LiDAR point clouds straight into the grid the solver needs.

NYC's downloader hands you classified LAS/LAZ point clouds named by tile, like
990217.laz. Not derived rasters. That turns out to be better, because gridding
the points yourself means you control the resolution, you never build a giant
1 foot intermediate, and you can see exactly how the surface was made.

What comes out, on the whole domain (frame plus buffer, see grid.py):

  DSM   the highest return in each cell. Buildings, trees, everything. This is
        the surface that casts shadows.
  DEM   the lowest ground-classified return in each cell. Street level.

LAS classification is an ASPRS standard. Class 2 is ground, 6 is building,
1 is unclassified, 9 is water. Only class 2 is trusted for the DEM.

Three kinds of empty cell, kept apart on purpose
------------------------------------------------
  Small gaps, under 6 m from a return: dark roofs, occlusion. Filled from the
  nearest measured cell.

  Water inside the survey: the Reservoir, the Lake, the Pond. Near-infrared
  laser light is absorbed by water, so it comes back empty. These are real
  open surfaces that get real sun, so they are set to the surrounding ground
  level and kept in every statistic.

  Outside the survey: no tile there, nothing measured. Marked no-data. The
  runner can stand footprint prisms on it so buildings there still cast
  shadows, but no statistic is ever computed on it.

Water is told apart from missing tiles by two tests together: the empty area
must sit inside a downloaded tile's box, and it must be enclosed by measured
cells on every side. A missing tile fails the first test, an unmeasured corner
fails the second.

Memory
------
A 1 foot NYC tile holds tens of millions of points and will not fit in a laptop
all at once. Points are read in chunks and gridded straight into the output
array at the target resolution, so peak memory is one chunk plus the final
grid, not the whole cloud. Tiles that do not touch the domain are skipped
without reading a point.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import time

import numpy as np

# ASPRS standard classification codes, which is the whole reason to read the
# point cloud rather than a derived raster. The tile already knows what is a
# building and what is a tree; guessing from height alone turns every tree
# canopy in Central Park into a skyscraper.
GROUND_CLASS = 2
BUILDING_CLASS = 6
VEG_CLASSES = {3, 4, 5}   # low, medium and high vegetation
WATER_CLASS = 9
NOISE_CLASSES = {7, 18}   # low noise and high noise

CLASS_NAMES = {
    1: "unclassified", 2: "ground", 3: "low vegetation", 4: "medium vegetation",
    5: "high vegetation", 6: "building", 7: "low noise", 9: "water",
    10: "rail", 11: "road surface", 13: "wire guard", 14: "wire conductor",
    15: "transmission tower", 17: "bridge deck", 18: "high noise",
}


def _crs_units(crs) -> str:
    """
    laspy hands back a pyproj CRS, rasterio hands back its own. They expose the
    unit name differently, so ask both ways rather than assume.
    """
    if crs is None:
        return ""
    u = getattr(crs, "linear_units", None)
    if u:
        return str(u)
    info = getattr(crs, "axis_info", None)
    if info:
        for ax in info:
            name = getattr(ax, "unit_name", None)
            if name:
                return str(name)
    return ""


def _unit_to_metres(crs) -> float:
    units = _crs_units(crs).lower()
    if "metre" in units or "meter" in units:
        return 1.0
    if "foot" in units or "feet" in units or units == "ft":
        return 1200.0 / 3937.0 if "us" in units else 0.3048
    return None


def _fill_holes(grid: np.ndarray, valid: np.ndarray, max_iter: int = 60):
    """
    Fill empty cells from their nearest filled neighbour.

    Gaps happen under dense canopy for the ground surface, and over water where
    the laser gets no return at all. Filling by nearest neighbour is crude but
    it is honest about what it is: a guess, in cells that had no measurement.
    """
    from scipy import ndimage

    if valid.all():
        return grid
    idx = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    return grid[tuple(idx)]


def _buildings_without_classes(heights, hits_total, hits_single, res_m, log):
    """
    Find buildings in a tile that was never classified.

    NYC's 2017 delivery classifies ground and leaves nearly everything else as
    class 1, so the ASPRS building code is not available.

    The logic is subtractive, and the direction matters. In a dense city almost
    everything tall is a building, so the job is to take everything above
    street level and remove the vegetation, rather than to prove each cell is a
    building. Proving it the other way round rejects real towers: Midtown roofs
    carry water tanks, plant rooms and setbacks, and pulses grazing a tall
    facade come back more than once. Both of those look like a tree to a strict
    test, and the tallest buildings fail hardest.

    Vegetation is identified by two properties together, not either alone:

      Multiple returns. A pulse hitting a roof reflects once. A pulse hitting a
      canopy passes through gaps and reflects several times.

      Roughness. A roof is flat to within its parapet. A canopy varies by
      metres over a few metres.

    A cell has to look like vegetation on both counts before it is removed.
    """
    from scipy import ndimage

    tall = heights > 3.0

    with np.errstate(invalid="ignore", divide="ignore"):
        single_frac = np.where(
            hits_total > 0, hits_single / np.maximum(hits_total, 1), 1.0
        )
    have_returns = hits_total.sum() > 0

    # Roughness over roughly a 7 m window: wider than a parapet, narrower than
    # a building.
    w = max(3, int(round(7.0 / res_m)) | 1)
    mean = ndimage.uniform_filter(heights, size=w, mode="nearest")
    mean_sq = ndimage.uniform_filter(heights * heights, size=w, mode="nearest")
    rough = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))

    log("  no classification in this tile, removing vegetation from the tall mask")
    log(f"    tall (>3 m)            {tall.mean()*100:5.1f}%")

    if have_returns:
        veg = (single_frac < 0.55) & (rough > 3.0)
        log(f"    looks like vegetation  {(tall & veg).mean()*100:5.1f}%"
            "   (multi-return AND rough)")
    else:
        veg = rough > 5.0
        log(f"    looks like vegetation  {(tall & veg).mean()*100:5.1f}%"
            "   (rough only, no return counts in file)")

    built = tall & ~veg

    # Close gaps left by chimneys and lift rooms, then drop specks smaller than
    # a shed, which are vehicles, scaffolding and street furniture.
    built = ndimage.binary_closing(built, structure=np.ones((3, 3)))
    lab, nlab = ndimage.label(built)
    if nlab:
        sizes = ndimage.sum(np.ones_like(built, np.float32), lab, range(1, nlab + 1))
        min_cells = max(4, int(round(60.0 / (res_m * res_m))))
        keep = np.zeros(nlab + 1, dtype=bool)
        keep[1:] = sizes >= min_cells
        built = keep[lab]

    log(f"    buildings after cleanup {built.mean()*100:5.1f}% of the grid")
    if built.mean() < 0.15:
        log("    warning: under 15% in a dense city looks too low. "
            "Check the viewer against the basemap.")
    return built


# Bumped whenever the gridding changes, so a stale cache is never reused.
CACHE_VERSION = 3


def list_tiles(path):
    """Every .laz and .las in a folder, or a single file, sorted by name."""
    if isinstance(path, (list, tuple)):
        return list(path)
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.laz"))
                      + glob.glob(os.path.join(path, "*.las")))
    return [path]


def _open(path):
    """
    Open a point cloud, with parallel LAZ decompression where laspy offers it.
    Decompression is most of the time spent here, and it spreads across cores.
    """
    import laspy
    backend = getattr(getattr(laspy, "LazBackend", None), "LazrsParallel", None)
    if backend is not None and path.lower().endswith(".laz"):
        try:
            return laspy.open(path, laz_backend=backend)
        except Exception:
            pass
    return laspy.open(path)


def probe(paths):
    """
    Header-only look at every tile: CRS, units, point count, and bounding box
    in CRS units. Instant, even on 400 MB tiles.
    """
    try:
        import laspy  # noqa: F401
    except ImportError:
        raise SystemExit("pip install laspy lazrs")
    paths = list_tiles(paths)
    if not paths:
        raise SystemExit("no .laz or .las files found")
    tiles, crs = [], None
    for p in paths:
        with _open(p) as f:
            h = f.header
            c = h.parse_crs()
            if crs is None and c is not None:
                crs = c
            tiles.append({
                "path": p, "name": os.path.splitext(os.path.basename(p))[0],
                "points": int(h.point_count),
                "xmin": float(h.mins[0]), "ymin": float(h.mins[1]),
                "xmax": float(h.maxs[0]), "ymax": float(h.maxs[1]),
                "zmin": float(h.mins[2]), "zmax": float(h.maxs[2]),
                "has_crs": c is not None,
            })
    if crs is None:
        raise SystemExit(
            "This point cloud carries no CRS. NYC tiles normally declare "
            "EPSG:2263. Open an issue with the filename.")
    unit_m = _unit_to_metres(crs)
    if unit_m is None:
        raise SystemExit(f"Cannot work out the units of {crs}")
    return {"crs": crs, "unit_m": unit_m, "tiles": tiles}


def _cache_key(tiles, grid, derive_buildings):
    parts = [f"v{CACHE_VERSION}", f"b{int(derive_buildings)}",
             f"{grid.res_m:.4f}", f"{grid.bearing_deg:.6f}",
             f"{grid.x0:.3f}", f"{grid.y0:.3f}", f"{grid.rows}", f"{grid.cols}"]
    for t in tiles:
        st = os.stat(t["path"])
        parts.append(f"{t['name']}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def rasterise(paths, grid, chunk: int = 4_000_000, log=print,
              cache_dir: str | None = None, derive_buildings: bool = False):
    """
    Grid LAS/LAZ tiles into a DSM and a DEM over grid's whole domain.

    `grid` is a pipeline.grid.Grid. The result has the same keys the runner
    has always used, plus `water`, `in_tiles` and `grid`.

    If cache_dir is given, the gridded surface is saved there and reused by
    the next run with the same tiles and the same grid, so the 2017 run and
    the today run read the point clouds once between them.
    """
    info = probe(paths)
    crs, unit_m = info["crs"], info["unit_m"]
    log(f"CRS {crs.to_string()}  1 unit = {unit_m:.6f} m")
    if abs(unit_m - grid.unit_m) > 1e-9:
        raise SystemExit("grid and tiles disagree on units, rebuild the grid "
                         "from the same tiles")

    # Only tiles that touch the domain are worth decompressing.
    from shapely.geometry import box
    dom = grid.domain_polygon_crs()
    tiles = [t for t in info["tiles"]
             if box(t["xmin"], t["ymin"], t["xmax"], t["ymax"]).intersects(dom)]
    skipped = len(info["tiles"]) - len(tiles)
    if not tiles:
        raise SystemExit("None of these tiles touches the requested area. "
                         "Run tile_info to see what they cover.")
    log(f"{len(tiles)} tile(s) touch the domain"
        + (f", {skipped} skipped without reading" if skipped else ""))

    key = _cache_key(tiles, grid, derive_buildings)
    if cache_dir:
        cpath = os.path.join(cache_dir, f"lidar_{key}.npz")
        if os.path.exists(cpath):
            log(f"reusing gridded LiDAR from {cpath}")
            z = np.load(cpath, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            out = {k: z[k] for k in z.files if k != "meta"}
            out.update(meta)
            out["built"] = out["built"].astype(bool)
            for k in ("nodata", "water", "in_tiles"):
                out[k] = out[k].astype(bool)
            out["crs"] = crs
            out["grid"] = grid
            out["cellsize"] = grid.res_m
            out["transform"] = grid.transform
            out["bounds"] = grid.bounds_lonlat()
            _summarise(out, log)
            return out

    t_read = time.time()
    H, W = grid.rows, grid.cols
    N = H * W
    (rx, ry), (dx, dy) = grid.right, grid.down
    res_u = grid.res_u
    x0, y0 = grid.x0, grid.y0

    dsm = np.full(N, -np.inf, dtype=np.float32)   # every return: what casts shadow
    bld = np.full(N, -np.inf, dtype=np.float32)   # class 6 only: what is a building
    dem = np.full(N, np.inf, dtype=np.float32)    # class 2 only: street level
    # Fallback discriminator for unclassified tiles. A laser pulse that hits a
    # roof comes back once. A pulse that hits a tree punches through the canopy
    # and comes back several times.
    hits_total = np.zeros(N, dtype=np.int32)
    hits_single = np.zeros(N, dtype=np.int32)
    n_used = n_ground = n_bld = 0
    class_hist = {}

    def scatter(grid_flat, flat, vals, op):
        """Vectorised per-cell min or max without the slow ufunc.at path."""
        if flat.size == 0:
            return
        # Sorting by (flat, val) puts the winner for each cell last.
        order = np.lexsort((vals if op is np.maximum else -vals, flat))
        fs, vs = flat[order], vals[order]
        last = np.empty(fs.shape, dtype=bool)
        last[-1] = True
        np.not_equal(fs[1:], fs[:-1], out=last[:-1])
        f_u, v_u = fs[last], vs[last]
        grid_flat[f_u] = op(grid_flat[f_u], v_u)

    def count(acc, flat):
        """Per-cell counts, bincount over just the range this chunk touches."""
        if flat.size == 0:
            return
        lo, hi = int(flat.min()), int(flat.max())
        acc[lo:hi + 1] += np.bincount(flat - lo, minlength=hi - lo + 1).astype(np.int32)

    noise = list(NOISE_CLASSES)
    for t in tiles:
        with _open(t["path"]) as f:
            log(f"reading {t['name']}  {f.header.point_count:,} points")
            for pts in f.chunk_iterator(chunk):
                x = np.asarray(pts.x)
                y = np.asarray(pts.y)
                ex = x - x0
                ey = y - y0
                col = np.floor((ex * rx + ey * ry) / res_u).astype(np.int64)
                row = np.floor((ex * dx + ey * dy) / res_u).astype(np.int64)
                inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
                if not inb.any():
                    continue
                cls = np.asarray(pts.classification)
                clean = inb & ~np.isin(cls, noise)
                if not clean.any():
                    continue
                z = np.asarray(pts.z)
                flat = row[clean] * W + col[clean]
                zz = z[clean].astype(np.float32)
                scatter(dsm, flat, zz, np.maximum)
                n_used += int(clean.sum())

                u, ct = np.unique(cls[clean], return_counts=True)
                for k, v in zip(u.tolist(), ct.tolist()):
                    class_hist[k] = class_hist.get(k, 0) + v

                g = clean & (cls == GROUND_CLASS)
                if g.any():
                    scatter(dem, row[g] * W + col[g], z[g].astype(np.float32), np.minimum)
                    n_ground += int(g.sum())

                nret = getattr(pts, "number_of_returns", None)
                if nret is not None:
                    nr = np.asarray(nret)[clean]
                    count(hits_total, flat)
                    one = nr <= 1
                    if one.any():
                        count(hits_single, flat[one])

                b = clean & (cls == BUILDING_CLASS)
                if b.any():
                    scatter(bld, row[b] * W + col[b], z[b].astype(np.float32), np.maximum)
                    n_bld += int(b.sum())

    dsm = dsm.reshape(H, W)
    dem = dem.reshape(H, W)
    bld = bld.reshape(H, W)
    hits_total = hits_total.reshape(H, W)
    hits_single = hits_single.reshape(H, W)
    res_m = grid.res_m
    log(f"{n_used:,} points gridded in {time.time() - t_read:.0f} s")
    if n_used:
        log("  point classes present:")
        for k in sorted(class_hist, key=lambda k: -class_hist[k])[:8]:
            log(f"    {k:>3} {CLASS_NAMES.get(k, 'unknown'):<20} "
                f"{class_hist[k]:>12,}  {class_hist[k]/n_used*100:5.1f}%")
    if n_used == 0:
        raise SystemExit(
            "No points from these tiles landed inside the requested area. "
            "Either the tiles do not cover it, or the centre is wrong. "
            "Run tile_info.")
    if n_bld == 0:
        log("  no class 6 points. NYC's 2017 delivery classifies ground and "
            "little else, so buildings come from the footprints dataset.")

    # Outlier ceiling. A single bird return makes the tallest-object readout
    # nonsense and, worse, casts a shadow across the whole frame. Anything far
    # above the 99.99th percentile of the surface is not architecture.
    finite = dsm[np.isfinite(dsm)]
    ceiling = np.percentile(finite, 99.99)
    spread = ceiling - np.percentile(finite, 50)
    limit = ceiling + max(20.0 / unit_m, 0.15 * spread)
    n_out = int((dsm > limit).sum())
    if n_out:
        log(f"  rejected {n_out} cells above {limit*unit_m:.0f} m as outliers "
            f"(birds, aircraft or multipath)")
        dsm[dsm > limit] = -np.inf

    from scipy import ndimage

    dsm_valid = np.isfinite(dsm)
    dem_valid = np.isfinite(dem)

    # Which cells are empty, and why. See the module docstring.
    void = ndimage.distance_transform_edt(~dsm_valid) * res_m > 6.0
    in_tiles = _tile_boxes_mask(tiles, grid)
    k = max(1, int(round(20.0 / res_m)))           # 20 m blocks
    hc, wc = -(-H // k), -(-W // k)
    pad = np.zeros((hc * k, wc * k), dtype=bool)
    pad[:H, :W] = dsm_valid
    coarse = pad.reshape(hc, k, wc, k).any(axis=(1, 3))
    enclosed = ndimage.binary_fill_holes(coarse)
    enclosed = np.repeat(np.repeat(enclosed, k, axis=0), k, axis=1)[:H, :W]
    water = void & in_tiles & enclosed
    nodata = void & ~water
    frame_sl = grid.frame_slice
    log(f"surface coverage {dsm_valid[frame_sl].mean()*100:.1f}% of frame cells got a return")
    if water.any():
        log(f"  {water[frame_sl].mean()*100:.1f}% of the frame is enclosed open "
            f"surface with no return (water). Kept, at ground level.")
    if nodata[frame_sl].any():
        log(f"  {nodata[frame_sl].mean()*100:.1f}% of the frame lies outside the "
            f"LiDAR tiles. Marked no-data, never counted.")

    # Ground: lowest class 2 return, gaps filled from the nearest measured
    # ground cell. That reaches across the buffer too, which is what lets a
    # footprint prism stand on something sensible where there is no survey.
    if dem_valid.sum() >= 0.05 * dsm_valid.sum():
        dem_f = _fill_holes(np.where(dem_valid, dem, 0.0), dem_valid)
        ground_source = ("lowest ASPRS class 2 ground return per cell, gaps filled "
                         "from the nearest measured ground")
    else:
        from .nyc import estimate_ground
        base = _fill_holes(np.where(dsm_valid, dsm, 0.0), dsm_valid)
        dem_f = estimate_ground(base * unit_m, res_m) / unit_m
        ground_source = "estimated from block minima, no ground-classified points available"

    dsm_f = _fill_holes(np.where(dsm_valid, dsm, 0.0), dsm_valid)
    dsm_f[void] = dem_f[void]

    dsm_m = (dsm_f * unit_m).astype(np.float32)
    ground_m = np.minimum(dem_f * unit_m, dsm_m).astype(np.float32)
    heights = (dsm_m - ground_m).astype(np.float32)

    if n_bld > 0:
        built = np.isfinite(bld) & (heights > 2.0)
        log(f"building cells from ASPRS class 6: {built.mean()*100:.1f}% of the domain")
    elif derive_buildings:
        built = _buildings_without_classes(heights, hits_total, hits_single, res_m, log)
    else:
        built = np.zeros((H, W), dtype=bool)

    out = {
        "surface": dsm_m, "ground": ground_m, "heights": heights, "built": built,
        "nodata": nodata, "water": water, "in_tiles": in_tiles,
        "cellsize": res_m, "crs": crs, "grid": grid,
        "transform": grid.transform, "bounds": grid.bounds_lonlat(),
        "resample_factor": 1, "native_res_m": res_m,
        "ground_source": ground_source, "n_tiles": len(tiles),
        "tiles_used": [t["name"] for t in tiles],
        "coverage": float(dsm_valid[frame_sl].mean()),
        "source_desc": (f"NYC 2017 airborne LiDAR point cloud, {len(tiles)} tile(s), "
                        f"{n_used:,} returns gridded to {res_m} m by highest return per cell"),
    }
    _summarise(out, log)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        meta = {k: out[k] for k in ("cellsize", "resample_factor", "native_res_m",
                                     "ground_source", "n_tiles", "tiles_used",
                                     "coverage", "source_desc")}
        np.savez(cpath, surface=dsm_m, ground=ground_m, heights=heights,
                 built=built.astype(np.uint8), nodata=nodata.astype(np.uint8),
                 water=water.astype(np.uint8), in_tiles=in_tiles.astype(np.uint8),
                 meta=np.array(json.dumps(meta)))
        log(f"cached the gridded LiDAR at {cpath}")
    return out


def _tile_boxes_mask(tiles, grid):
    """True inside any downloaded tile's header box, on the domain grid."""
    from rasterio import features
    from shapely.geometry import box
    shapes = [(box(t["xmin"], t["ymin"], t["xmax"], t["ymax"]), 1) for t in tiles]
    return features.rasterize(shapes, out_shape=(grid.rows, grid.cols),
                              transform=grid.transform, fill=0,
                              dtype="uint8").astype(bool)


def _summarise(out, log):
    g = out["grid"]
    sl = g.frame_slice
    h = out["heights"][sl]
    ok = ~out["nodata"][sl]
    top = float(h[ok].max()) if ok.any() else 0.0
    log(f"surface {out['surface'].min():.1f} to {out['surface'].max():.1f} m above datum")
    log(f"tallest object in the frame, LiDAR as surveyed, {top:.1f} m above street "
        f"({top*3.28084:.0f} ft)")
