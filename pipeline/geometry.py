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


def cast_shadow(
    dsm: np.ndarray,
    cellsize: float,
    altitude_deg: float,
    azimuth_deg: float,
    max_distance_m: float | None = None,
) -> np.ndarray:
    """
    Binary sunlit mask for one instant.

    Returns a float array where 1.0 means the pixel receives direct sun and
    0.0 means it is shaded by something inside the domain. Below the horizon
    the whole grid returns 0.0.

    dsm            surface height in metres, buildings and vegetation included
    cellsize       ground size of one pixel in metres
    altitude_deg   solar altitude, degrees above horizon
    azimuth_deg    solar azimuth, degrees clockwise from north
    """
    if altitude_deg <= 0.0:
        return np.zeros_like(dsm, dtype=np.float32)

    relief = float(np.nanmax(dsm) - np.nanmin(dsm))
    tan_alt = math.tan(math.radians(altitude_deg))
    # Beyond this distance nothing in the domain is tall enough to reach the
    # ray, so the sweep can stop.
    reach = relief / max(tan_alt, 1e-6)
    if max_distance_m is not None:
        reach = min(reach, max_distance_m)
    n_steps = int(math.ceil(reach / cellsize)) + 1
    n_steps = max(1, min(n_steps, 4000))

    blocked = np.zeros(dsm.shape, dtype=bool)
    for row_off, col_off, dist_cells in _ray_offsets(azimuth_deg, n_steps):
        ray_height = dsm + dist_cells * cellsize * tan_alt
        neighbour = _look(dsm, row_off, col_off, NODATA)
        blocked |= neighbour > ray_height

    return (~blocked).astype(np.float32)


def sky_view_factor(
    dsm: np.ndarray,
    cellsize: float,
    n_azimuths: int = 24,
    max_distance_m: float = 300.0,
) -> np.ndarray:
    """
    Radiatively weighted Sky View Factor per pixel, in the range 0 to 1.

    n_azimuths      number of compass directions scanned. 16 is coarse but
                    usable, 24 is a reasonable default, 32 and above buys
                    little for urban grids and costs linearly.
    max_distance_m  horizon search radius. 300 m is generous for a mid-rise
                    district: a 50 m building at 300 m subtends under 10
                    degrees and contributes under 3 percent to the sum.
    """
    n_steps = int(math.ceil(max_distance_m / cellsize))
    acc = np.zeros(dsm.shape, dtype=np.float64)

    for k in range(n_azimuths):
        azimuth = 360.0 * k / n_azimuths
        max_tan = np.zeros(dsm.shape, dtype=np.float32)
        for row_off, col_off, dist_cells in _ray_offsets(azimuth, n_steps):
            neighbour = _look(dsm, row_off, col_off, NODATA)
            rise = neighbour - dsm
            valid = neighbour > NODATA / 2
            tan_beta = np.where(valid, rise / (dist_cells * cellsize), 0.0)
            np.maximum(max_tan, tan_beta, out=max_tan)
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
    svf: np.ndarray                  # (rows, cols), 0 to 1


def run_day(
    dsm: np.ndarray,
    cellsize: float,
    positions,
    labels: list[str],
    n_azimuths: int = 24,
    svf_radius_m: float = 300.0,
    progress=None,
) -> DayShadowResult:
    """
    Cast shadows for every supplied sun position, accumulate sun hours, and
    compute SVF once.

    positions  a sequence of SunPosition objects from pipeline.solar
    labels     local clock labels matching positions, same length

    Sun hours assume the supplied positions are evenly spaced in time and
    each represents its own interval. Hourly positions give hours directly.
    """
    assert len(positions) == len(labels)
    frames = []
    alts, azs = [], []
    for p, lab in zip(positions, labels):
        if progress:
            progress(f"shadow {lab}  alt {p.altitude:5.1f}  az {p.azimuth:6.1f}")
        frames.append(cast_shadow(dsm, cellsize, p.altitude, p.azimuth))
        alts.append(p.altitude)
        azs.append(p.azimuth)

    sunlit = np.stack(frames).astype(np.float32)
    sun_hours = sunlit.sum(axis=0).astype(np.float32)

    if progress:
        progress(f"sky view factor, {n_azimuths} azimuths, {svf_radius_m:.0f} m radius")
    svf = sky_view_factor(dsm, cellsize, n_azimuths, svf_radius_m)

    return DayShadowResult(
        hours=list(labels),
        altitudes=alts,
        azimuths=azs,
        sunlit=sunlit,
        sun_hours=sun_hours,
        svf=svf,
    )
