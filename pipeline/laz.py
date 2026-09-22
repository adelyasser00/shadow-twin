"""
Rasterise LiDAR point clouds straight into the grid the solver needs.

NYC's downloader hands you classified LAS/LAZ point clouds named by tile, like
990217.laz. Not derived rasters. That turns out to be better, because gridding
the points yourself means you control the resolution, you never build a giant
1 foot intermediate, and you can see exactly how the surface was made.

What comes out:

  DSM   the highest return in each cell. Buildings, trees, everything. This is
        the surface that casts shadows.
  DEM   the lowest ground-classified return in each cell. Street level.

LAS classification is an ASPRS standard. Class 2 is ground, 6 is building,
1 is unclassified, 9 is water. Only class 2 is trusted for the DEM.

Memory
------
A 1 foot NYC tile holds tens of millions of points and will not fit in a laptop
all at once. Points are read in chunks and gridded straight into the output
array at the target resolution, so peak memory is one chunk plus the final
grid, not the whole cloud.
"""

from __future__ import annotations

import math
import os
import glob

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


def rasterise(
    paths,
    centre_lat: float,
    centre_lon: float,
    span_m: float,
    res_m: float,
    chunk: int = 4_000_000,
    log=print,
):
    """
    Grid one or more LAS/LAZ tiles into a DSM and a DEM at res_m.

    Returns a dict with the same shape as pipeline.nyc.prepare, so the runner
    does not care which route the data came in by.
    """
    try:
        import laspy
    except ImportError:
        raise SystemExit("pip install laspy lazrs")
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_transform, transform_bounds

    if isinstance(paths, str):
        if os.path.isdir(paths):
            paths = sorted(
                glob.glob(os.path.join(paths, "*.laz"))
                + glob.glob(os.path.join(paths, "*.las"))
            )
        else:
            paths = [paths]
    if not paths:
        raise SystemExit("no .laz or .las files found")

    # Work out the CRS and units from the first file.
    with laspy.open(paths[0]) as f:
        crs = f.header.parse_crs()
        total_pts = f.header.point_count
    if crs is None:
        raise SystemExit(
            "This point cloud carries no CRS. NYC tiles normally declare "
            "EPSG:2263. Tell me the file and I will add an override."
        )
    unit_m = _unit_to_metres(crs)
    if unit_m is None:
        raise SystemExit(f"Cannot work out the units of {crs}")
    log(f"CRS {crs.to_string()}  1 unit = {unit_m:.6f} m")

    # Grid geometry, in CRS units, centred on the requested point.
    xs, ys = warp_transform("EPSG:4326", crs, [centre_lon], [centre_lat])
    cx, cy = xs[0], ys[0]
    span_u = span_m / unit_m
    res_u = res_m / unit_m
    n = int(round(span_u / res_u))
    x0, y1 = cx - span_u / 2, cy + span_u / 2
    transform = from_origin(x0, y1, res_u, res_u)
    log(f"target grid {n} x {n} at {res_m} m ({res_u:.2f} CRS units)")

    dsm = np.full((n, n), -np.inf, dtype=np.float32)   # every return: what casts shadow
    bld = np.full((n, n), -np.inf, dtype=np.float32)   # class 6 only: what is a building
    # Fallback discriminator for unclassified tiles. A laser pulse that hits a
    # roof comes back once. A pulse that hits a tree punches through the canopy
    # and comes back several times. Counting single-return pulses per cell
    # separates hard surfaces from vegetation without any classification.
    hits_total = np.zeros((n, n), dtype=np.int32)
    hits_single = np.zeros((n, n), dtype=np.int32)
    dem = np.full((n, n), np.inf, dtype=np.float32)    # class 2 only: street level
    n_used = 0
    n_ground = 0
    n_bld = 0
    class_hist = {}

    def scatter(grid, rows, cols, vals, op):
        """Vectorised per-cell min or max without the slow ufunc.at path."""
        if rows.size == 0:
            return
        flat = rows.astype(np.int64) * n + cols
        # Sorting by (flat, val) puts the winner for each cell last.
        order = np.lexsort((vals if op is np.maximum else -vals, flat))
        fs, vs = flat[order], vals[order]
        last = np.empty(fs.shape, dtype=bool)
        last[-1] = True
        np.not_equal(fs[1:], fs[:-1], out=last[:-1])
        f_u, v_u = fs[last], vs[last]
        # Fancy indexing returns a copy, so combine then write back.
        flat_grid = grid.reshape(-1)
        flat_grid[f_u] = op(flat_grid[f_u], v_u)

    for path in paths:
        with laspy.open(path) as f:
            log(f"reading {os.path.basename(path)}  {f.header.point_count:,} points")
            for pts in f.chunk_iterator(chunk):
                x = np.asarray(pts.x); y = np.asarray(pts.y); z = np.asarray(pts.z)
                cls = np.asarray(pts.classification)

                col = ((x - x0) / res_u).astype(np.int64)
                row = ((y1 - y) / res_u).astype(np.int64)
                inb = (col >= 0) & (col < n) & (row >= 0) & (row < n)
                clean = inb & ~np.isin(cls, list(NOISE_CLASSES))
                if not clean.any():
                    continue
                r, c = row[clean], col[clean]
                zz = z[clean].astype(np.float32)
                scatter(dsm, r, c, zz, np.maximum)
                n_used += int(clean.sum())

                u, ct = np.unique(cls[clean], return_counts=True)
                for k, v in zip(u.tolist(), ct.tolist()):
                    class_hist[k] = class_hist.get(k, 0) + v

                g = clean & (cls == GROUND_CLASS)
                if g.any():
                    scatter(dem, row[g], col[g], z[g].astype(np.float32), np.minimum)
                    n_ground += int(g.sum())

                nret = getattr(pts, "number_of_returns", None)
                if nret is not None:
                    nr = np.asarray(nret)[clean]
                    np.add.at(hits_total, (r, c), 1)
                    one = nr <= 1
                    if one.any():
                        np.add.at(hits_single, (r[one], c[one]), 1)

                b = clean & (cls == BUILDING_CLASS)
                if b.any():
                    scatter(bld, row[b], col[b], z[b].astype(np.float32), np.maximum)
                    n_bld += int(b.sum())

    log(f"{n_used:,} points gridded")
    log("  point classes present:")
    for k in sorted(class_hist, key=lambda k: -class_hist[k])[:8]:
        log(f"    {k:>3} {CLASS_NAMES.get(k, 'unknown'):<20} "
            f"{class_hist[k]:>12,}  {class_hist[k]/n_used*100:5.1f}%")
    if n_bld == 0:
        log("  no class 6 points in this tile. NYC's 2017 delivery classifies "
            "ground and little else, so buildings are found from return count "
            "and surface roughness instead.")
    if n_used == 0:
        raise SystemExit(
            f"No points from these tiles landed inside a {span_m:.0f} m square at "
            f"{centre_lat}, {centre_lon}.\n"
            "Either the tiles do not cover that point, or the centre is wrong. "
            "Check the tile footprint on the downloader map."
        )

    # Outlier ceiling. A single bird return makes the tallest-object readout
    # nonsense and, worse, casts a shadow across the whole frame. Anything far
    # above the 99.99th percentile of the surface is not architecture.
    finite = dsm[np.isfinite(dsm)]
    if finite.size:
        ceiling = np.percentile(finite, 99.99)
        spread = ceiling - np.percentile(finite, 50)
        limit = ceiling + max(20.0 / unit_m, 0.15 * spread)
        n_out = int((dsm > limit).sum())
        if n_out:
            log(f"  rejected {n_out} cells above {limit*unit_m:.0f} m as outliers "
                f"(birds, aircraft or multipath)")
            dsm[dsm > limit] = -np.inf

    dsm_valid = np.isfinite(dsm)
    dem_valid = np.isfinite(dem)
    cover = dsm_valid.mean()
    log(f"surface coverage {cover*100:.1f}% of cells got a return")

    # NYC tiles are squares rotated to the Manhattan street grid, so a north-up
    # frame always has corners with no LiDAR at all. Filling those from the
    # nearest measured cell used to extend real buildings outward as fake walls
    # that then cast fake shadows. Small gaps (dark roofs, water, occlusion) are
    # still filled; anything more than 6 m from a measurement is marked no-data,
    # set to street level so it blocks nothing, and blanked in every output.
    from scipy import ndimage
    dist_cells = ndimage.distance_transform_edt(~dsm_valid)
    nodata = dist_cells * res_m > 6.0
    if nodata.any():
        log(f"  {nodata.mean()*100:.1f}% of the frame has no LiDAR (outside the "
            f"tiles). Marked no-data, not filled.")
    if dem_valid.mean() >= 0.05:
        ground_level = float(np.median(dem[dem_valid]))
    else:
        ground_level = float(np.percentile(dsm[dsm_valid], 5))
    dsm = _fill_holes(np.where(dsm_valid, dsm, 0.0), dsm_valid)
    dsm[nodata] = ground_level

    if dem_valid.mean() >= 0.05:
        dem = _fill_holes(np.where(dem_valid, dem, 0.0), dem_valid)
        dem[nodata] = ground_level
        ground_source = "lowest ASPRS class 2 ground return per cell, small gaps filled from nearest neighbour"
    else:
        from .nyc import estimate_ground
        dem = estimate_ground(dsm * unit_m, res_m) / unit_m
        ground_source = "estimated from block minima, no ground-classified points available"

    dsm_m = dsm * unit_m
    ground_m = np.minimum(dem * unit_m, dsm_m)
    heights = dsm_m - ground_m

    west, south, east, north = transform_bounds(
        crs, "EPSG:4326", x0, y1 - n * res_u, x0 + n * res_u, y1, densify_pts=21
    )
    # Buildings come from the classification, not from a height threshold.
    # A height threshold cannot tell a plane tree from a brownstone, and in a
    # tile that is mostly park that difference is the whole picture.
    if n_bld > 0:
        built = np.isfinite(bld) & (heights > 2.0)
        log(f"building cells from ASPRS class 6: {built.mean()*100:.1f}% of the grid")
    else:
        built = _buildings_without_classes(
            heights, hits_total, hits_single, res_m, log
        )

    log(f"surface {dsm_m.min():.1f} to {dsm_m.max():.1f} m above datum")
    log(f"tallest object {heights.max():.1f} m above street "
        f"({heights.max()*3.28084:.0f} ft)")
    log(f"built fraction {built.mean()*100:.1f}%")

    return {
        "surface": dsm_m.astype(np.float32),
        "ground": ground_m.astype(np.float32),
        "heights": heights.astype(np.float32),
        "built": built,
        "cellsize": res_m,
        "bounds": (west, south, east, north),
        "crs": crs,
        "transform": transform,
        "resample_factor": 1,
        "native_res_m": res_m,
        "ground_source": ground_source,
        "n_tiles": len(paths),
        "coverage": float(cover),
        "nodata": nodata,
        "source_desc": (
            f"NYC 2017 airborne LiDAR point cloud, {len(paths)} tile(s), "
            f"{n_used:,} returns gridded to {res_m} m by highest return per cell"
        ),
    }
