"""
Turn a building mask into 3D footprints for the viewer.

Footprints are derived from the same raster that cast the shadows. If the
shapes on screen came from a different source than the shapes in the solver,
the picture would be showing you one building while the shadow came from
another. Deriving both from one array makes that impossible.

Heights are the median height above ground inside each connected component,
not the value at the centroid. An L-shaped block has a centroid sitting in
the courtyard, and sampling there would report a tower as one storey tall.
"""

from __future__ import annotations

import numpy as np


def from_mask(
    built: np.ndarray,
    heights: np.ndarray,
    transform,
    crs,
    resample_factor: int = 1,
    max_features: int = 2500,
    min_area_m2: float = 60.0,
    min_height_m: float = 4.0,
    log=print,
):
    """
    built            bool mask, True where there is a building
    heights          metres above local ground, same shape as built
    transform        affine of the raster BEFORE resampling
    resample_factor  how much the mask was reduced relative to that transform
    """
    from rasterio import features
    from rasterio.warp import transform as warp_transform
    from rasterio.transform import Affine
    from scipy import ndimage
    from shapely.geometry import shape

    # The transform came from the native-resolution crop, so scale it to match
    # the array we actually have. Skipping this puts every building in the
    # wrong place by a factor of the resample ratio, which looks like the whole
    # city slid into the harbour.
    t = transform * Affine.scale(float(resample_factor), float(resample_factor))

    labels, n = ndimage.label(built)
    if n == 0:
        log("  no buildings found in the mask")
        return []
    log(f"  {n} connected components")

    idx = np.arange(1, n + 1)
    med_h = ndimage.median(heights, labels, index=idx)
    counts = ndimage.sum(np.ones_like(built, dtype=np.float32), labels, index=idx)

    cell_area = abs(t.a * t.e)
    keep = set()
    for i in idx:
        h = med_h[i - 1]
        area = counts[i - 1] * cell_area
        if np.isnan(h) or h < min_height_m or area < min_area_m2:
            continue
        keep.add(int(i))
    log(f"  {len(keep)} pass the height and area filters")

    # Biggest first, so a cap keeps the buildings that matter visually.
    ranked = sorted(keep, key=lambda i: counts[i - 1], reverse=True)[:max_features]
    ranked_set = set(ranked)

    out = []
    simplify_m = max(1.0, abs(t.a) * 1.2)
    for geom, val in features.shapes(
        labels.astype(np.int32), mask=built, transform=t
    ):
        lab = int(val)
        if lab not in ranked_set:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        poly = poly.simplify(simplify_m, preserve_topology=True)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        ring = list(poly.exterior.coords)
        if len(ring) < 4:
            continue
        xs = [c[0] for c in ring]
        ys = [c[1] for c in ring]
        lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)
        out.append(
            {
                "ring": [[round(a, 6), round(b, 6)] for a, b in zip(lons, lats)],
                "height": round(float(med_h[lab - 1]), 1),
            }
        )

    out.sort(key=lambda b: b["height"], reverse=True)
    if out:
        log(f"  tallest footprint {out[0]['height']:.0f} m, "
            f"median {np.median([b['height'] for b in out]):.0f} m")
    return out
