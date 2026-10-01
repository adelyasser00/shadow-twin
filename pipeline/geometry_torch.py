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


class TorchSweeper:
    """
    The windowed sweep of geometry.sweep, on a GPU.

    Holds the surface on the device once, then answers calls with the same
    signature as geometry.sweep, so the runner can pass it as run_day's
    backend. verify_torch.py checks it against the CPU sweep cell for cell.
    """

    def __init__(self, dsm):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.np_dsm = dsm
        self.t = torch.from_numpy(np.ascontiguousarray(dsm, dtype=np.float32)).to(self.dev)
        self.dmax = float(self.t.max())

    def __call__(self, dsm, cellsize, altitude_deg, azimuth_deg, window=None,
                 max_distance_m=None, max_steps=4000):
        torch = self.torch
        t = self.t
        H, W = t.shape
        r0, r1, c0, c1 = window if window is not None else (0, H, 0, W)
        fr = t[r0:r1, c0:c1]
        if altitude_deg <= 0.0:
            return np.zeros(tuple(fr.shape), dtype=np.float32)
        relief = self.dmax - float(fr.min())
        tan_alt = math.tan(math.radians(altitude_deg))
        reach = relief / max(tan_alt, 1e-6)
        if max_distance_m is not None:
            reach = min(reach, max_distance_m)
        n_steps = max(1, min(int(math.ceil(reach / cellsize)) + 1, max_steps))
        blocked = torch.zeros(fr.shape, dtype=torch.bool, device=self.dev)
        for row_off, col_off, dist_cells in _ray_offsets(azimuth_deg, n_steps):
            a0, a1 = max(r0 + row_off, 0), min(r1 + row_off, H)
            b0, b1 = max(c0 + col_off, 0), min(c1 + col_off, W)
            if a0 >= a1 or b0 >= b1:
                continue
            wr0, wr1 = a0 - row_off - r0, a1 - row_off - r0
            wc0, wc1 = b0 - col_off - c0, b1 - col_off - c0
            ray = fr[wr0:wr1, wc0:wc1] + float(np.float32(dist_cells * cellsize * tan_alt))
            blocked[wr0:wr1, wc0:wc1] |= t[a0:a1, b0:b1] > ray
        return (~blocked).to(torch.float32).cpu().numpy()


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
