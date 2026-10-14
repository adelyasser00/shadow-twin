"""
The solve step of a nested wind box, on its own: everything between "the box
is voxelised" and "the street layer is saved". tools/wind_nest.py runs it on
this machine; tools/nest_cloud.py packs its inputs so it can run on a cloud
GPU, where only numpy, scipy and a GPU library are needed (CuPy for CUDA, or
pyopencl).

A pack (dict of arrays) holds: solid, veg, prof, iu, gk, U_in (3, nz, ny)
and fluid_in (nz, ny), the 8 m run's wind and open fraction on the inlet
plane; side_u (3, nz, 2, nx) or None; kfirst, roof; the nesting-check plane
(ic, kz, check_parent, check_fluid); res_m, z0_m, facade, spinup and average
in flow-throughs, accumulate_every, inlet_skip, seed.
"""

from __future__ import annotations

import time

import numpy as np


def street_layer(arr, kfirst):
    """arr (c, nz, ny, nx) at the first open cell of each column -> (c, ny, nx). As run_wind.street_layer."""
    k = np.broadcast_to(kfirst[None, None].astype(np.int64), (arr.shape[0], 1) + kfirst.shape)
    return np.take_along_axis(arr, k, axis=1)[:, 0]


def pick_backend(backend="auto"):
    """'cuda' (CuPy), 'opencl' (pyopencl), or 'auto': CUDA if CuPy sees a GPU, else OpenCL."""
    if backend in ("auto", "cuda"):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() > 0:
                return "cuda"
        except Exception:
            if backend == "cuda":
                raise
    return "opencl"


def solve(p, log=print, backend="auto"):
    from .wind_lbm import InflowTurbulence, Solver
    t0 = time.time()
    solid = p["solid"]
    nz, ny, nx = solid.shape
    prof, gk = p["prof"], int(p["gk"])
    turb = InflowTurbulence(prof, gk, float(p["res_m"]), ny, iu=p["iu"], seed=int(p.get("seed", 11)))
    f = turb.field
    f *= p["fluid_in"][None, None].astype(np.float32)        # no gusts inside the 8 m run's buildings
    f[0] += (p["U_in"][0] - prof[:, None]).astype(np.float32)[None]
    f[1] += p["U_in"][1].astype(np.float32)[None]
    f[2] += p["U_in"][2].astype(np.float32)[None]
    kind = pick_backend(backend)
    if kind == "cuda":
        from .wind_lbm_cuda import CudaSolver as S
    else:
        S = Solver
    s = S(solid, prof, veg=p["veg"], res_m=float(p["res_m"]), z0_m=float(p["z0_m"]), turbulence=turb,
          log=log, facade_drag=bool(p["facade"]), sides=p.get("side_u"))
    acc_every = int(p["accumulate_every"])
    ft = nx / float(prof[gk:].mean())
    n_spin = int(float(p["spinup"]) * ft)
    n_avg = int(float(p["average"]) * ft)
    n_avg -= n_avg % (2 * acc_every)
    log(f"  flow-through {ft:,.0f} steps; {n_spin:,} to settle, {n_avg:,} averaged ({kind})")
    mlups = [s.run(n_spin, log_every=max(1, n_spin // 2))]
    if not np.isfinite(s.max_speed()):
        raise SystemExit("the solver blew up while settling")
    mlups.append(s.run(n_avg // 2, accumulate_every=acc_every))
    half, k_half = s.sums()
    half = half[4].copy()                                    # only |u| is needed from the first half
    mlups.append(s.run(n_avg // 2, accumulate_every=acc_every))
    full, k_full = s.sums()
    if not np.isfinite(full).all():
        raise SystemExit("the solver blew up while averaging")
    kf = p["kfirst"]
    sp1 = street_layer((half / k_half)[None], kf)[0]
    sp2 = street_layer(((full[4] - half) / (k_full - k_half))[None], kf)[0]
    mean = full / k_full
    del full
    ground_cols = ~p["roof"].copy()
    skip = int(p["inlet_skip"])
    ground_cols[:, :skip] = False
    ground_cols[:, -skip:] = False
    d = np.abs(sp1 - sp2)[ground_cols]
    typical = float(np.median((sp1 + sp2)[ground_cols] / 2))
    conv = {"median_abs_diff_over_median_speed": round(float(np.median(d)) / typical, 4),
            "p90_abs_diff_over_median_speed": round(float(np.percentile(d, 90)) / typical, 4)}
    ic, kz = int(p["ic"]), p["kz"]
    child = np.sqrt((mean[:3, :, :, ic][:, kz] ** 2).sum(axis=0))
    par = p["check_parent"]
    ok = (~solid[kz, :, ic]) & (p["check_fluid"] > 0.999) & (par > 1e-4)
    nest_err = float(np.median(np.abs(child[ok] / par[ok] - 1)))
    nest_bias = float(np.median(child[ok] / par[ok] - 1))
    street = street_layer(mean, kf)
    above = street_layer(mean[4:5], np.minimum(kf + 1, nz - 1))[0]
    out = {"street_speed": street[4].astype(np.float32), "street_speed_above": above.astype(np.float32),
           "street_half1": sp1.astype(np.float32), "street_half2": sp2.astype(np.float32),
           "street_u": street[:3].astype(np.float32), "u_low": mean[:3, :41].astype(np.float16),
           "check_child": child.astype(np.float32), "check_parent": par.astype(np.float32), "check_ok": ok,
           "meta": {"convergence_street_speed": conv, "nest_err": nest_err, "nest_bias": nest_bias,
                    "flow_through_steps": round(ft), "spinup_steps": n_spin, "average_steps": n_avg,
                    "samples": int(k_full), "mlups": [round(x) for x in mlups], "device": s.device_name,
                    "backend": kind, "wall_drag": round(s.cd_wall, 5), "gusts_calibrated": turb.calibrated,
                    "solve_s": round(time.time() - t0)}}
    log(f"  solved in {out['meta']['solve_s']} s at {np.mean(mlups):.0f} MLUPs on {s.device_name}; street "
        f"halves differ by {conv['median_abs_diff_over_median_speed'] * 100:.1f}%; nesting "
        f"{nest_err * 100:.1f}%")
    del s
    return out
