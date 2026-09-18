"""
Read NYC 1-foot LiDAR rasters and prepare them for the solver.

The NYC 2017 capture ships as GeoTIFF tiles in NAD83 New York Long Island
(EPSG:2263), with US survey feet on all three axes. Everything in this module
exists to turn that into metres on a square grid the solver can use.

Two files matter per tile:
    hh_NYC_###.tif    highest hit, the digital surface model. Buildings, trees,
                      everything. This is the one that casts shadows.
    be_NYC_###.tif    bare earth. Optional. Used to work out how tall each
                      building is above the street.

If you only have the hh tiles, ground level is estimated instead. Manhattan is
flat enough that this works, with the caveat noted in estimate_ground.
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np

US_SURVEY_FOOT = 1200.0 / 3937.0  # 0.30480060960121924 metres exactly
INTL_FOOT = 0.3048


def _unit_to_metres(crs) -> float:
    """
    Metres per horizontal unit of a CRS.

    NYC data is in US survey feet. Getting this wrong by the 0.0002 percent
    between survey and international feet does not matter. Getting it wrong by
    the factor of 3.28 between feet and metres makes every shadow wrong.
    """
    units = (crs.linear_units or "").lower()
    if "metre" in units or "meter" in units:
        return 1.0
    if "us" in units and "foot" in units:
        return US_SURVEY_FOOT
    if "foot" in units or "feet" in units or units == "ft":
        return INTL_FOOT
    raise SystemExit(
        f"Cannot work out the units of this CRS: {crs}\n"
        f"linear_units reported as {units!r}. Reproject it or tell me the units."
    )


def load_mosaic(path_or_dir: str, pattern: str):
    """
    Open one GeoTIFF, or mosaic every GeoTIFF matching a pattern in a folder.

    Returns (array, transform, crs, nodata).
    """
    import rasterio
    from rasterio.merge import merge

    if os.path.isfile(path_or_dir):
        files = [path_or_dir]
    else:
        files = sorted(glob.glob(os.path.join(path_or_dir, pattern)))
        if not files:
            files = sorted(glob.glob(os.path.join(path_or_dir, "*.tif")))
    if not files:
        raise SystemExit(f"no GeoTIFFs found in {path_or_dir} matching {pattern}")

    srcs = [rasterio.open(f) for f in files]
    try:
        if len(srcs) == 1:
            arr = srcs[0].read(1).astype(np.float32)
            return arr, srcs[0].transform, srcs[0].crs, srcs[0].nodata, files
        mosaic, transform = merge(srcs)
        return (
            mosaic[0].astype(np.float32),
            transform,
            srcs[0].crs,
            srcs[0].nodata,
            files,
        )
    finally:
        for s in srcs:
            s.close()


def crop_to_centre(
    arr: np.ndarray, transform, crs, centre_lat: float, centre_lon: float, span_m: float
):
    """
    Cut a square of span_m metres a side, centred on a WGS84 point.

    Returns (cropped array, new transform). Raises if the point is outside the
    raster, which is the single most likely thing to go wrong when someone
    downloads the wrong tile.
    """
    from rasterio.transform import Affine
    from rasterio.warp import transform as warp_transform

    unit_m = _unit_to_metres(crs)
    xs, ys = warp_transform("EPSG:4326", crs, [centre_lon], [centre_lat])
    cx, cy = xs[0], ys[0]

    inv = ~transform
    col_c, row_c = inv * (cx, cy)
    h, w = arr.shape
    if not (0 <= col_c < w and 0 <= row_c < h):
        raise SystemExit(
            f"The point {centre_lat}, {centre_lon} is not inside this raster.\n"
            f"  raster is {w} x {h} cells\n"
            f"  the point lands at column {col_c:.0f}, row {row_c:.0f}\n"
            "You almost certainly downloaded a tile that does not cover your area. "
            "Check the tile on the finder map before downloading."
        )

    cell_m = abs(transform.a) * unit_m
    half_cells = int(round((span_m / 2.0) / cell_m))
    r0 = max(0, int(row_c) - half_cells)
    r1 = min(h, int(row_c) + half_cells)
    c0 = max(0, int(col_c) - half_cells)
    c1 = min(w, int(col_c) + half_cells)

    got_m = min(r1 - r0, c1 - c0) * cell_m
    if got_m < span_m * 0.8:
        print(
            f"  warning: only {got_m:.0f} m of the requested {span_m:.0f} m is "
            f"covered by the tiles you have. Download the neighbouring tiles "
            f"for the full square."
        )

    out = arr[r0:r1, c0:c1]
    new_transform = transform * Affine.translation(c0, r0)
    return out, new_transform


def resample(arr: np.ndarray, factor: int, how: str = "max") -> np.ndarray:
    """
    Reduce a raster by an integer factor using block pooling.

    max is the default and it is the right choice for a surface model headed
    into a shadow solver: it keeps roof heights and building edges intact.
    Averaging would shave the tops off towers and round every corner, which
    shortens every shadow. The cost of max is that buildings grow by up to one
    output cell in each direction, which errs toward more shadow rather than
    less. Say so if anyone asks.
    """
    if factor <= 1:
        return arr
    h, w = arr.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    a = arr[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor)
    if how == "max":
        return a.max(axis=(1, 3))
    if how == "mean":
        return a.mean(axis=(1, 3))
    if how == "min":
        return a.min(axis=(1, 3))
    raise ValueError(f"unknown resample mode {how}")


def estimate_ground(dsm: np.ndarray, cellsize_m: float, window_m: float = 150.0):
    """
    Estimate street level when no bare-earth raster is available.

    Takes the minimum surface height in coarse blocks, then smooths it back up
    to full resolution. In a dense grid city every block contains some street,
    so the block minimum is street level.

    Where this is wrong: large continuous built areas with no open ground, and
    real terrain relief inside one block. Central Park has genuine rock outcrops
    with tens of feet of relief, so building heights derived this way near the
    park edge carry more error than elsewhere. Download the be_ tiles if you
    care about that.
    """
    from scipy.ndimage import zoom, uniform_filter

    block = max(1, int(round(window_m / cellsize_m)))
    coarse = resample(dsm, block, "min")
    if coarse.size == 0:
        return np.zeros_like(dsm)
    coarse = uniform_filter(coarse, size=3, mode="nearest")
    zy = dsm.shape[0] / coarse.shape[0]
    zx = dsm.shape[1] / coarse.shape[1]
    ground = zoom(coarse, (zy, zx), order=1)[: dsm.shape[0], : dsm.shape[1]]
    if ground.shape != dsm.shape:
        pad_y = dsm.shape[0] - ground.shape[0]
        pad_x = dsm.shape[1] - ground.shape[1]
        ground = np.pad(ground, ((0, pad_y), (0, pad_x)), mode="edge")
    return np.minimum(ground, dsm).astype(np.float32)


def prepare(
    dsm_path: str,
    centre_lat: float,
    centre_lon: float,
    span_m: float,
    target_res_m: float,
    dem_path: str | None = None,
    dsm_pattern: str = "hh_*.tif",
    dem_pattern: str = "be_*.tif",
    log=print,
):
    """
    Full NYC preparation. Returns a dict the runner can hand to the solver.
    """
    from rasterio.warp import transform_bounds
    from rasterio.transform import array_bounds

    dsm_raw, transform, crs, nodata, files = load_mosaic(dsm_path, dsm_pattern)
    log(f"loaded {len(files)} surface tile(s), {dsm_raw.shape[0]} x {dsm_raw.shape[1]} cells")
    log(f"CRS {crs.to_string()}  units {crs.linear_units!r}")

    unit_m = _unit_to_metres(crs)
    native_res_m = abs(transform.a) * unit_m
    log(f"native cell {abs(transform.a):.3f} CRS units = {native_res_m:.3f} m")

    if nodata is not None:
        dsm_raw = np.where(dsm_raw == nodata, np.nan, dsm_raw)
    # NYC tiles also use large negative sentinels in places.
    dsm_raw = np.where(dsm_raw < -1000, np.nan, dsm_raw)

    dsm_c, transform_c = crop_to_centre(
        dsm_raw, transform, crs, centre_lat, centre_lon, span_m
    )
    log(f"cropped to {dsm_c.shape[0]} x {dsm_c.shape[1]} cells")

    factor = max(1, int(round(target_res_m / native_res_m)))
    dsm_r = resample(dsm_c, factor, "max")
    cellsize_m = native_res_m * factor
    log(f"resampled by {factor} (max pooling) to {dsm_r.shape[0]} x {dsm_r.shape[1]} "
        f"at {cellsize_m:.2f} m")

    # Heights are in the same units as the horizontal axes for these products.
    dsm_m = np.nan_to_num(dsm_r, nan=0.0) * unit_m

    if dem_path:
        dem_raw, dtransform, dcrs, dnodata, dfiles = load_mosaic(dem_path, dem_pattern)
        if dnodata is not None:
            dem_raw = np.where(dem_raw == dnodata, np.nan, dem_raw)
        dem_raw = np.where(dem_raw < -1000, np.nan, dem_raw)
        dem_c, _ = crop_to_centre(
            dem_raw, dtransform, dcrs, centre_lat, centre_lon, span_m
        )
        dem_r = resample(dem_c, factor, "min")
        # Guard against the two crops landing a row apart.
        ry = min(dem_r.shape[0], dsm_r.shape[0])
        rx = min(dem_r.shape[1], dsm_r.shape[1])
        dsm_m = dsm_m[:ry, :rx]
        ground_m = np.nan_to_num(dem_r[:ry, :rx], nan=0.0) * unit_m
        log(f"using {len(dfiles)} bare-earth tile(s) for ground level")
        ground_source = "NYC bare earth LiDAR (be_ tiles)"
    else:
        ground_m = estimate_ground(dsm_m, cellsize_m)
        log("no bare-earth tiles given, estimating street level from block minima")
        ground_source = "estimated from block minima of the surface model"

    ground_m = np.minimum(ground_m, dsm_m)
    heights = dsm_m - ground_m

    # A surface model that includes terrain is what the shadow sweep needs. A
    # height-above-ground array is what the 3D view needs. Keep both.
    n_rows, n_cols = dsm_m.shape
    west, south, east, north = transform_bounds(
        crs,
        "EPSG:4326",
        *array_bounds(n_rows * factor, n_cols * factor, transform_c),
        densify_pts=21,
    )

    built = heights > 3.0
    log(f"surface {dsm_m.min():.1f} to {dsm_m.max():.1f} m above datum")
    log(f"tallest object {heights.max():.1f} m above street "
        f"({heights.max()*3.2808:.0f} ft)")
    log(f"built fraction {built.mean()*100:.1f}%")

    return {
        "surface": dsm_m,
        "ground": ground_m,
        "heights": heights,
        "built": built,
        "cellsize": cellsize_m,
        "bounds": (west, south, east, north),
        "crs": crs,
        "transform": transform_c,
        "resample_factor": factor,
        "native_res_m": native_res_m,
        "ground_source": ground_source,
        "n_tiles": len(files),
    }
