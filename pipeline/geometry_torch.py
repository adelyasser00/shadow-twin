"""
GPU versions of the shadow sweep and sky view factor.

Same maths as geometry.py, same conventions, same results. The only difference
is that the arrays live on a GPU. Use this on Kaggle or Modal, not on a laptop.

The CPU path in geometry.py stays the reference implementation and is the one
the self-tests run against. This module is checked against it by
verify_torch.py, which runs both and compares.

    from pipeline.geometry_torch import available, run_day_gpu
    if available():
        res = run_day_gpu(dsm, cellsize, positions, labels)
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import DayShadowResult, _ray_offsets

NODATA = -9999.0


def available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def device_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "cpu"
    except Exception:
        return "torch not installed"


def _look(t, row_off: int, col_off: int, fill: float):
    """Sample the neighbour at (row+row_off, col+col_off). Mirrors geometry._look."""
    import torch
    out = torch.full_like(t, fill)
    h, w = t.shape
    r0s, r1s = max(0, row_off), min(h, h + row_off)
    c0s, c1s = max(0, col_off), min(w, w + col_off)
    if r0s >= r1s or c0s >= c1s:
        return out
    r0d, r1d = r0s - row_off, r1s - row_off
    c0d, c1d = c0s - col_off, c1s - col_off
    out[r0d:r1d, c0d:c1d] = t[r0s:r1s, c0s:c1s]
    return out


def cast_shadow_gpu(dsm_t, cellsize, altitude_deg, azimuth_deg, max_distance_m=None):
    import torch
    if altitude_deg <= 0.0:
        return torch.zeros_like(dsm_t)
    relief = float(dsm_t.max() - dsm_t.min())
    tan_alt = math.tan(math.radians(altitude_deg))
    reach = relief / max(tan_alt, 1e-6)
    if max_distance_m is not None:
        reach = min(reach, max_distance_m)
    n_steps = max(1, min(int(math.ceil(reach / cellsize)) + 1, 6000))

    blocked = torch.zeros_like(dsm_t, dtype=torch.bool)
    for row_off, col_off, dist_cells in _ray_offsets(azimuth_deg, n_steps):
        ray = dsm_t + dist_cells * cellsize * tan_alt
        blocked |= _look(dsm_t, row_off, col_off, NODATA) > ray
    return (~blocked).to(dsm_t.dtype)


def sky_view_factor_gpu(dsm_t, cellsize, n_azimuths=24, max_distance_m=300.0):
    import torch
    n_steps = int(math.ceil(max_distance_m / cellsize))
    acc = torch.zeros_like(dsm_t, dtype=torch.float32)
    for k in range(n_azimuths):
        az = 360.0 * k / n_azimuths
        max_tan = torch.zeros_like(dsm_t, dtype=torch.float32)
        for row_off, col_off, dist_cells in _ray_offsets(az, n_steps):
            nb = _look(dsm_t, row_off, col_off, NODATA)
            rise = nb - dsm_t
            valid = nb > NODATA / 2
            tan_beta = torch.where(valid, rise / (dist_cells * cellsize),
                                   torch.zeros_like(rise))
            max_tan = torch.maximum(max_tan, tan_beta)
        beta = torch.atan(max_tan.clamp(min=0.0))
        acc += torch.cos(beta) ** 2
    return acc / n_azimuths


def run_day_gpu(dsm, cellsize, positions, labels, n_azimuths=24,
                svf_radius_m=300.0, skip_svf=False, progress=None):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if progress:
        progress(f"torch backend on {device_name()}")
    t = torch.from_numpy(np.ascontiguousarray(dsm, dtype=np.float32)).to(dev)

    frames, alts, azs = [], [], []
    for p, lab in zip(positions, labels):
        if progress:
            progress(f"shadow {lab}  alt {p.altitude:5.1f}  az {p.azimuth:6.1f}")
        frames.append(cast_shadow_gpu(t, cellsize, p.altitude, p.azimuth).cpu().numpy())
        alts.append(p.altitude)
        azs.append(p.azimuth)

    sunlit = np.stack(frames).astype(np.float32)
    svf = None
    if not skip_svf:
        if progress:
            progress(f"sky view factor on GPU, {n_azimuths} azimuths")
        svf = sky_view_factor_gpu(t, cellsize, n_azimuths, svf_radius_m).cpu().numpy()

    del t
    torch.cuda.empty_cache() if dev == "cuda" else None

    return DayShadowResult(
        hours=list(labels), altitudes=alts, azimuths=azs,
        sunlit=sunlit, sun_hours=sunlit.sum(axis=0).astype(np.float32),
        svf=svf if svf is not None else np.zeros(dsm.shape, dtype=np.float32),
    )
