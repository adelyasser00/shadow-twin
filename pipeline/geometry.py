"""
Shadow casting and Sky View Factor on a digital surface model.

This is the geometric half of an urban thermal model. It answers two
questions for every pixel in the grid:

  1. Is this pixel in direct sun at this instant, given the sun's position?
  2. How much of the sky hemisphere can this pixel see?

Neither question needs weather data, which is why this stage can run before
any meteorology is wired up. Both are inputs to a radiation budget later.

Method, and where it comes from
-------------------------------
Shadow: the shift-and-subtract sweep of Ratti and Richens (1999), which is
the same approach used inside UMEP and SOLWEIG. March away from each pixel
toward the sun in one-cell steps. The sun ray from a pixel at height H rises
by cellsize * tan(altitude) per cell of horizontal distance. If any surface
along that march pokes above the ray, the pixel is shaded.

Sky View Factor: horizon scanning with the discretised Steyn (1980) form.
For N evenly spaced azimuths, find the maximum horizon elevation angle
beta_i, then

    SVF = (1/N) * sum_i cos^2(beta_i)

This is the radiatively weighted SVF, that is, weighted by the cosine of
the zenith angle so it corresponds to the fraction of isotropic sky
radiation actually reaching a horizontal surface. An unobstructed pixel
gives 1.0; a pixel walled in on all sides gives 0.0.

Known limits, state these when you present results
--------------------------------------------------
- Integer cell stepping means oblique azimuths alias slightly. Error is
  bounded by half a cell in shadow edge placement.
- Horizon search is truncated at max_distance_m. Obstructions beyond that
  are ignored. The default derives the distance from the tallest object in
  the grid so nothing that can matter is missed inside the domain.
- Objects outside the raster edge are invisible to the sweep, so results
  within max_distance_m of the boundary are optimistic. Always clip a
  margin off before you publish numbers.
- This models geometry only. It says nothing about air temperature, wind,
  humidity or surface material. It is not thermal comfort on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NODATA = -9999.0


def _shift(arr: np.ndarray, row_off: int, col_off: int, fill: float) -> np.ndarray:
    """
    Shift a 2D array by whole cells without wrapping.

    A positive row_off moves content down (southward in a north-up raster),
    a positive col_off moves content right (eastward). Vacated cells take
    the fill value.
    """
    out = np.full_like(arr, fill)
    h, w = arr.shape
    r0_src, r1_src = max(0, -row_off), min(h, h - row_off)
    c0_src, c1_src = max(0, -col_off), min(w, w - col_off)
    if r0_src >= r1_src or c0_src >= c1_src:
        return out
    r0_dst, r1_dst = r0_src + row_off, r1_src + row_off
    c0_dst, c1_dst = c0_src + col_off, c1_src + col_off
    out[r0_dst:r1_dst, c0_dst:c1_dst] = arr[r0_src:r1_src, c0_src:c1_src]
    return out


def _look(arr: np.ndarray, row_off: int, col_off: int, fill: float) -> np.ndarray:
    """
    Sample the neighbour that sits at (row + row_off, col + col_off).

    This is the inverse of _shift and it is the one you want when marching
    outward from every pixel at once. Getting these two confused flips every
    shadow to the wrong side of its building, so the sign lives here and
    nowhere else.
    """
    return _shift(arr, -row_off, -col_off, fill)


def _ray_offsets(azimuth_deg: float, n_steps: int) -> list[tuple[int, int, float]]:
    """
    Whole-cell offsets along a horizontal ray, walking toward a compass
    bearing. Returns (row_off, col_off, distance_in_cells), deduplicated and
    ordered by increasing distance.

    Azimuth is clockwise from north. In a north-up raster, moving north
    decreases the row index and moving east increases the column index.
    """
    az = math.radians(azimuth_deg)
    east = math.sin(az)
    north = math.cos(az)
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int, float]] = []
    for n in range(1, n_steps + 1):
        col_off = int(round(n * east))
        row_off = int(round(-n * north))
        if (row_off, col_off) in seen or (row_off == 0 and col_off == 0):
            continue
        seen.add((row_off, col_off))
        dist_cells = math.hypot(row_off, col_off)
        out.append((row_off, col_off, dist_cells))
    return out


def sweep(
    dsm: np.ndarray,
    cellsize: float,
    altitude_deg: float,
    azimuth_deg: float,
    window: tuple | None = None,
    max_distance_m: float | None = None,
    max_steps: int = 4000,
) -> np.ndarray:
    """
    Binary sunlit mask for the cells inside `window`, marching through `dsm`.

    window   (r0, r1, c0, c1). Only these cells get an answer, but the march
             reads the whole surface, so buildings outside the window still
             cast shadows into it. This is what the buffer is for. None means
             the whole array, which is the original behaviour exactly.

    azimuth_deg is relative to the grid's "up" direction. On a north-up grid
    that is the compass azimuth; on a rotated grid, see Grid.azimuth_to_grid.

    Each step compares whole sub-arrays in place, so there is no per-step
    allocation. Cells whose neighbour at that offset would fall off the surface
    see nothing there, which is the same as the old NODATA fill.
    """
    H, W = dsm.shape
    r0, r1, c0, c1 = window if window is not None else (0, H, 0, W)
    fr = dsm[r0:r1, c0:c1]
    if altitude_deg <= 0.0:
        return np.zeros(fr.shape, dtype=np.float32)

    # Beyond this distance nothing on the surface is tall enough to reach the
    # ray from the lowest cell in the window, so the sweep can stop.
    relief = float(np.nanmax(dsm) - np.nanmin(fr))
    tan_alt = math.tan(math.radians(altitude_deg))
    reach = relief / max(tan_alt, 1e-6)
    if max_distance_m is not None:
        reach = min(reach, max_distance_m)
    n_steps = int(math.ceil(reach / cellsize)) + 1
    n_steps = max(1, min(n_steps, max_steps))

    blocked = np.zeros(fr.shape, dtype=bool)
    ray = np.empty(fr.shape, dtype=np.float32)
    hit = np.empty(fr.shape, dtype=bool)
    for row_off, col_off, dist_cells in _ray_offsets(azimuth_deg, n_steps):
        # Neighbour rows and columns for every window cell, clipped to dsm.
        a0, a1 = max(r0 + row_off, 0), min(r1 + row_off, H)
        b0, b1 = max(c0 + col_off, 0), min(c1 + col_off, W)
        if a0 >= a1 or b0 >= b1:
            continue
        wr0, wr1 = a0 - row_off - r0, a1 - row_off - r0
        wc0, wc1 = b0 - col_off - c0, b1 - col_off - c0
        rsub = ray[wr0:wr1, wc0:wc1]
        hsub = hit[wr0:wr1, wc0:wc1]
        np.add(fr[wr0:wr1, wc0:wc1], np.float32(dist_cells * cellsize * tan_alt), out=rsub)
        np.greater(dsm[a0:a1, b0:b1], rsub, out=hsub)
        bsub = blocked[wr0:wr1, wc0:wc1]
        np.logical_or(bsub, hsub, out=bsub)

    return (~blocked).astype(np.float32)


def cast_shadow(
    dsm: np.ndarray,
    cellsize: float,
    altitude_deg: float,
    azimuth_deg: float,
    max_distance_m: float | None = None,
) -> np.ndarray:
    """
    Binary sunlit mask for one instant, over the whole array.

    Returns a float array where 1.0 means the pixel receives direct sun and
    0.0 means it is shaded by something inside the domain. Below the horizon
    the whole grid returns 0.0.

    dsm            surface height in metres, buildings and vegetation included
    cellsize       ground size of one pixel in metres
    altitude_deg   solar altitude, degrees above horizon
    azimuth_deg    solar azimuth, degrees clockwise from grid up (north on a
                   north-up grid)
    """
    return sweep(dsm, cellsize, altitude_deg, azimuth_deg,
                 max_distance_m=max_distance_m)


def sky_view_factor(
    dsm: np.ndarray,
    cellsize: float,
    n_azimuths: int = 24,
    max_distance_m: float = 300.0,
    window: tuple | None = None,
) -> np.ndarray:
    """
    Radiatively weighted Sky View Factor per pixel, in the range 0 to 1.

    n_azimuths      number of compass directions scanned. 16 is coarse but
                    usable, 24 is a reasonable default, 32 and above buys
                    little for urban grids and costs linearly.
    max_distance_m  horizon search radius. 300 m is generous for a mid-rise
                    district: a 50 m building at 300 m subtends under 10
                    degrees and contributes under 3 percent to the sum.
    window          (r0, r1, c0, c1): answer only these cells, reading the
                    whole surface for the horizon. None is the whole array.

    SVF scans every direction evenly, so it does not care which way the grid
    faces.
    """
    H, W = dsm.shape
    r0, r1, c0, c1 = window if window is not None else (0, H, 0, W)
    fr = dsm[r0:r1, c0:c1]
    n_steps = int(math.ceil(max_distance_m / cellsize))
    acc = np.zeros(fr.shape, dtype=np.float64)

    for k in range(n_azimuths):
        azimuth = 360.0 * k / n_azimuths
        max_tan = np.zeros(fr.shape, dtype=np.float32)
        for row_off, col_off, dist_cells in _ray_offsets(azimuth, n_steps):
            a0, a1 = max(r0 + row_off, 0), min(r1 + row_off, H)
            b0, b1 = max(c0 + col_off, 0), min(c1 + col_off, W)
            if a0 >= a1 or b0 >= b1:
                continue
            wr0, wr1 = a0 - row_off - r0, a1 - row_off - r0
            wc0, wc1 = b0 - col_off - c0, b1 - col_off - c0
            tan_beta = (dsm[a0:a1, b0:b1] - fr[wr0:wr1, wc0:wc1]) / (dist_cells * cellsize)
            sub = max_tan[wr0:wr1, wc0:wc1]
            np.maximum(sub, tan_beta, out=sub)
        np.clip(max_tan, 0.0, None, out=max_tan)
        beta = np.arctan(max_tan)
        acc += np.cos(beta) ** 2

    return (acc / n_azimuths).astype(np.float32)


@dataclass
class DayShadowResult:
    """Everything the viewer needs from one modelled day."""

    hours: list[str]                 # local clock labels, e.g. "06:00"
    altitudes: list[float]
    azimuths: list[float]
    sunlit: np.ndarray               # (n_hours, rows, cols), 1 sunlit 0 shaded
    sun_hours: np.ndarray            # (rows, cols), total hours in direct sun
    svf: np.ndarray | None           # (rows, cols), 0 to 1, None if skipped


def run_day(
    dsm: np.ndarray,
    cellsize: float,
    positions,
    labels: list[str],
    n_azimuths: int = 24,
    svf_radius_m: float = 300.0,
    progress=None,
    window: tuple | None = None,
    bearing_deg: float = 0.0,
    compute_svf: bool = True,
    backend=None,
) -> DayShadowResult:
    """
    Cast shadows for every supplied sun position and accumulate sun hours.

    positions    a sequence of SunPosition objects from pipeline.solar
    labels       local clock labels matching positions, same length
    window       frame window inside dsm, see sweep. None is the whole array.
    bearing_deg  true bearing of the grid's "up". Sun azimuths are turned into
                 grid terms with it. 0 on a north-up grid.
    compute_svf  sky view factor is slow and the runner computes it once, not
                 per day, so it can switch this off.
    backend      optional callable with sweep's signature, for the GPU path.

    Sun hours assume the supplied positions are evenly spaced in time and
    each represents its own interval. Hourly positions give hours directly.
    Altitudes and azimuths in the result are the true compass values.
    """
    assert len(positions) == len(labels)
    fn = backend or sweep
    frames = []
    alts, azs = [], []
    for p, lab in zip(positions, labels):
        if progress:
            progress(f"shadow {lab}  alt {p.altitude:5.1f}  az {p.azimuth:6.1f}")
        az_grid = (p.azimuth - bearing_deg) % 360.0
        frames.append(fn(dsm, cellsize, p.altitude, az_grid, window=window))
        alts.append(p.altitude)
        azs.append(p.azimuth)

    sunlit = np.stack(frames).astype(np.float32)
    sun_hours = sunlit.sum(axis=0).astype(np.float32)

    svf = None
    if compute_svf:
        if progress:
            progress(f"sky view factor, {n_azimuths} azimuths, {svf_radius_m:.0f} m radius")
        svf = sky_view_factor(dsm, cellsize, n_azimuths, svf_radius_m, window=window)

    return DayShadowResult(
        hours=list(labels),
        altitudes=alts,
        azimuths=azs,
        sunlit=sunlit,
        sun_hours=sun_hours,
        svf=svf,
    )
