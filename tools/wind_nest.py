"""
A 2 m wind box around Central Park Tower, fed by the 8 m city run
(POST_NOTES.md, A6 step 2; the rules are in A8, fixed before the first run).

One-way nesting: what enters the 2 m box at its inlet face is the 8 m run's
time-mean wind on that plane, plus gusts the size of the 8 m run's resolved
turbulence there. Inside, the city is voxelised from the same 2 m canvas at
2 m, with the wall settings that passed the wind tunnel checks at 2 m (facade
friction on). Sides periodic and lid free slip, as in the city runs.

    .venv-heat/Scripts/python -I tools/wind_nest.py run [--scenario 2017 today] [--dir 315] [--width 480]
    .venv-heat/Scripts/python -I tools/wind_nest.py score [--dir 315] [--width 480]

--width 400 is the blockage check (A8): the same pair in a narrower box.
--average 6: flow-throughs averaged (default 3, as the city runs). The box is
only 600 m long, so 3 average about 5 times less real time than a city run.

Output (data/, not committed):
  data/wind/nest/<scenario>_dec_<dir>_2m_fw[_w<width>]/street.npz, run.json
  data/wind/nest/score_<dir>[_w<width>].json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pipeline.run_wind import (ACC_EVERY, ALPHA, AVERAGE_HERO, INLET_SKIP, RUNS_DIR, SPINUP, Z0_M,  # noqa: E402
                               run_name as parent_name, street_layer)
from pipeline.wind_domain import BOX, Box, load_canvas, voxelise  # noqa: E402
from pipeline.wind_lbm import InflowTurbulence, Solver  # noqa: E402
from tools.aij_case_d import KAPPA_UVW  # noqa: E402

TOWER = "Central Park Tower"
TOWER_LONLAT = (-73.98102, 40.76644)        # as in tools/wind_story.py
RES = 2.0
PARENT_RES = 8.0
NEST = {"length_m": 600.0, "width_m": 480.0, "height_m": 600.0, "roi_from_inlet_m": 250.0}
OUT = os.path.join(HERE, "data", "wind", "nest")
CHECK_X_M = 40.0                # the nesting check: this far downstream of the inlet,
CHECK_Z_M = (150.0, 450.0)      # at these heights above the box's lowest ground (A8)
RING_M = (20.0, 80.0)           # street ring around the tower, as in tools/wind_story.py


def tower_xy():
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32118", always_xy=True).transform
    return tf(*TOWER_LONLAT)


def load_parent(scenario, theta):
    d = os.path.join(RUNS_DIR, parent_name(scenario, "dec", theta, PARENT_RES))
    with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
        m = json.load(f)
    z = np.load(os.path.join(d, "field.npz"))
    return m, z["u"].astype(np.float32), z["uu"].astype(np.float32)


def parent_box(info, m):
    pb = Box(info, m["theta_from_deg"], PARENT_RES, m["box_size"])
    if not np.allclose(pb.p0, m["box"]["p0"]) or abs(pb.x0c - m["box"]["x0c"]) > 1e-9:
        raise SystemExit("the 8 m run's box does not match the canvas")
    return pb


def parent_at(m, u, uu, pb, x, y, zabs):
    """The 8 m run's mean wind (3, ...), resolved k and open fraction at map points and absolute heights."""
    from scipy.ndimage import map_coordinates
    shape = np.broadcast(x, y, zabs).shape
    x, y, zabs = (np.broadcast_to(a, shape) for a in (x, y, zabs))
    ip, jp = pb.from_map(x, y)
    top = u.shape[1] - 1
    gk = m["ground_layer"]
    kp = (zabs - m["gmin_m"]) / PARENT_RES + 0.5
    coords = [np.minimum(kp, top).ravel(), jp.ravel(), ip.ravel()]
    U = np.stack([map_coordinates(u[c], coords, order=1, mode="nearest") for c in range(3)]).reshape((3,) + shape)
    UU = map_coordinates(uu, coords, order=1, mode="nearest").reshape(shape)
    fluid = map_coordinates((uu > 0).astype(np.float32), coords, order=1, mode="nearest").reshape(shape)
    # Above the saved layers (520 m up): the city runs' inlet power law, from the top saved layer.
    f = np.where(kp > top, ((np.maximum(kp, top) - gk + 0.5) / (top - gk + 0.5)) ** ALPHA, 1.0)
    U = U * f[None]
    UU = UU * f ** 2
    return U, 0.5 * np.maximum(UU - (U ** 2).sum(axis=0), 0.0), np.clip(fluid, 0.0, 1.0)


def nest_box(width):
    return dict(NEST, width_m=float(width))


def nest_name(scenario, theta, facade=True, width=NEST["width_m"], average=AVERAGE_HERO):
    return (f"{scenario}_dec_{theta:05.1f}_{RES:g}m" + ("_fw" if facade else "")
            + ("" if width == NEST["width_m"] else f"_w{width:g}")
            + ("" if average == AVERAGE_HERO else f"_avg{average:g}"))


def run_one(scenario, theta, facade=True, log=print, force=False, width=NEST["width_m"], average=AVERAGE_HERO):
    out = os.path.join(OUT, nest_name(scenario, theta, facade, width, average))
    if os.path.exists(os.path.join(out, "run.json")) and not force:
        log(f"  {os.path.basename(out)}: already done")
        return out
    t0 = time.time()
    canvas, info = load_canvas()
    m, u, uu = load_parent(scenario, theta)
    pb = parent_box(info, m)
    tx, ty = tower_xy()
    info_c = dict(info, roi_centre=[float(tx), float(ty)])
    box = nest_box(width)
    solid, veg, cols, bx = voxelise(canvas, info_c, theta, RES, scenario, "dec", box)
    del canvas
    nz, ny, nx = solid.shape
    gmin = cols["gmin"]
    zabs = gmin + (np.arange(nz) - 0.5) * RES

    # Inlet plane from the 8 m run.
    xi, yi = bx.to_map(np.zeros(ny), np.arange(ny))
    U, k_res, fluid = parent_at(m, u, uu, pb, xi[None, :], yi[None, :], zabs[:, None])
    open_in = ~solid[:, :, 0]
    prof = np.zeros(nz, np.float32)
    kl = np.zeros(nz, np.float64)
    for k in range(1, nz):
        if open_in[k].any():
            prof[k] = U[0, k][open_in[k]].mean()
            kl[k] = k_res[k][open_in[k]].mean()
    gk = int(np.median(cols["kfirst"][:, 0]))
    iu = np.where(prof > 1e-4, np.sqrt(kl / KAPPA_UVW) / np.maximum(prof, 1e-4), 0.0)
    turb = InflowTurbulence(prof, gk, RES, ny, iu=iu)
    f = turb.field                                   # (3, slices, nz, ny)
    f *= fluid[None, None].astype(np.float32)        # no gusts inside the 8 m run's buildings
    f[0] += (U[0] - prof[:, None]).astype(np.float32)[None]
    f[1] += U[1].astype(np.float32)[None]
    f[2] += U[2].astype(np.float32)[None]
    log(f"  inlet from {parent_name(scenario, 'dec', theta, PARENT_RES)}: lid {prof[-1]:.4f}, "
        f"Iu {iu[gk + 5]:.2f} at {zabs[gk + 5] - gmin:.0f} m, {iu[nz // 2]:.2f} at {zabs[nz // 2] - gmin:.0f} m")

    s = Solver(solid, prof, veg=veg, res_m=RES, z0_m=Z0_M, turbulence=turb, log=log, facade_drag=facade)
    ft = nx / float(prof[gk:].mean())
    n_spin = int(SPINUP * ft)
    n_avg = int(average * ft)
    n_avg -= n_avg % (2 * ACC_EVERY)
    log(f"  {os.path.basename(out)}: flow-through {ft:,.0f} steps; {n_spin:,} to settle, {n_avg:,} averaged")
    mlups = [s.run(n_spin, log_every=max(1, n_spin // 2))]
    if not np.isfinite(s.max_speed()):
        raise SystemExit("the solver blew up while settling")
    mlups.append(s.run(n_avg // 2, accumulate_every=ACC_EVERY))
    half, k_half = s.sums()
    half = half.copy()
    mlups.append(s.run(n_avg // 2, accumulate_every=ACC_EVERY))
    full, k_full = s.sums()
    if not np.isfinite(full).all():
        raise SystemExit("the solver blew up while averaging")
    kf = cols["kfirst"]
    mean = full / k_full
    sp1 = street_layer((half / k_half)[4:5], kf)[0]
    sp2 = street_layer(((full - half) / (k_full - k_half))[4:5], kf)[0]
    ground_cols = ~cols["roof"]
    ground_cols[:, :INLET_SKIP] = False
    ground_cols[:, -INLET_SKIP:] = False
    d = np.abs(sp1 - sp2)[ground_cols]
    typical = float(np.median((sp1 + sp2)[ground_cols] / 2))
    conv = {"median_abs_diff_over_median_speed": round(float(np.median(d)) / typical, 4),
            "p90_abs_diff_over_median_speed": round(float(np.percentile(d, 90)) / typical, 4)}

    # The nesting check: the 2 m box against the 8 m run on one plane, well above the street.
    ic = int(round(CHECK_X_M / RES))
    kz = np.nonzero((zabs - gmin >= CHECK_Z_M[0]) & (zabs - gmin <= CHECK_Z_M[1]))[0]
    xc, yc = bx.to_map(np.full(ny, ic), np.arange(ny))
    Up, _, flp = parent_at(m, u, uu, pb, xc[None, :], yc[None, :], zabs[kz][:, None])
    child = np.sqrt((mean[:3, :, :, ic][:, kz] ** 2).sum(axis=0))      # (len(kz), ny)
    par = np.sqrt((Up ** 2).sum(axis=0))
    ok = (~solid[kz, :, ic]) & (flp > 0.999) & (par > 1e-4)
    nest_err = float(np.median(np.abs(child[ok] / par[ok] - 1)))
    nest_bias = float(np.median(child[ok] / par[ok] - 1))

    street = street_layer(mean, kf)
    above = street_layer(mean[4:5], np.minimum(kf + 1, nz - 1))[0]     # the next cell up, 2 to 4 m
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "street.npz"), street_speed=street[4].astype(np.float32),
                        street_speed_above=above.astype(np.float32),
                        street_u=street[:3].astype(np.float32), kfirst=kf, roof=cols["roof"],
                        building=cols["building"].astype(np.float32), ground=cols["ground"].astype(np.float32),
                        u_low=mean[:3, :41].astype(np.float16), profile=prof, iu=iu.astype(np.float32),
                        check_child=child.astype(np.float32), check_parent=par.astype(np.float32), check_ok=ok)
    meta = {"scenario": scenario, "season": "dec", "theta_from_deg": float(theta), "res_m": RES,
            "box": bx.meta(), "box_size": box, "gmin_m": gmin, "ground_layer": gk, "tower": TOWER,
            "tower_xy": [float(tx), float(ty)], "parent": parent_name(scenario, "dec", theta, PARENT_RES),
            "smagorinsky": 0.17, "z0_m": Z0_M, "wall_cd": round(s.cd_wall, 5),
            "walls": "slip + wall-law drag" + (" on ground, roofs and facades" if facade else " on ground and roofs"),
            "sides": "periodic", "lid": "free slip", "gusts_calibrated": turb.calibrated,
            "flow_through_steps": round(ft), "spinup_steps": n_spin, "average_steps": n_avg,
            "samples": int(k_full), "average_flow_throughs": average, "accumulate_every": ACC_EVERY,
            "mlups": [round(x) for x in mlups],
            "device": s.device_name, "convergence_street_speed": conv,
            "nesting_check": {"x_from_inlet_m": CHECK_X_M, "z_m": CHECK_Z_M, "n": int(ok.sum()),
                              "median_abs_rel_diff": round(nest_err, 4), "median_rel_diff": round(nest_bias, 4)},
            "wall_s": round(time.time() - t0), "generated": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
    log(f"  {os.path.basename(out)}: done in {meta['wall_s']} s at {np.mean(mlups):.0f} MLUPs; street halves differ "
        f"by {conv['median_abs_diff_over_median_speed'] * 100:.1f}%; 2 m vs 8 m on the check plane: "
        f"median |diff| {nest_err * 100:.1f}%, median diff {nest_bias * 100:+.1f}%")
    del s
    return out


def tower_ring():
    """Heat-grid points on open ground 20 to 80 m from the tower's footprint, as tools/wind_story.py does."""
    from scipy import ndimage
    from pipeline.wind_export import heat_points
    x, y, layers, masks, meta = heat_points()
    tx, ty = tower_xy()
    bld = masks["building"]
    lab, _ = ndimage.label(bld)
    r = int(round((y[0, 0] - ty) / 2.0))
    c = int(round((tx - x[0, 0]) / 2.0))
    if not lab[r, c]:
        raise SystemExit("no building at the tower's position")
    dist = ndimage.distance_transform_edt(lab != lab[r, c]) * 2.0
    ring = ~bld & ~masks["water"] & (dist >= RING_M[0]) & (dist <= RING_M[1])
    return x, y, ring


def score(theta, facade=True, log=print, width=NEST["width_m"], average=AVERAGE_HERO):
    from pipeline.wind_export import RunBox, resample
    x, y, ring = tower_ring()
    res = {}
    for sc in ("2017", "today"):
        d = os.path.join(OUT, nest_name(sc, theta, facade, width, average))
        with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
            m = json.load(f)
        z = np.load(os.path.join(d, "street.npz"))
        ok = (~z["roof"]) & np.isfinite(z["street_speed"])
        v = resample(z["street_speed"], ok, RunBox(m), x[ring], y[ring])
        va = resample(z["street_speed_above"], ok, RunBox(m), x[ring], y[ring])
        k8 = np.load(os.path.join(RUNS_DIR, parent_name(sc, "dec", theta, PARENT_RES), "k_heatgrid.npy"))[ring]
        res[sc] = {"median_2m": float(np.nanmedian(v)), "p90_2m": float(np.nanpercentile(v, 90)),
                   "median_2m_next_cell_up": float(np.nanmedian(va)),
                   "n_2m": int(np.isfinite(v).sum()), "median_8m_x_lawn": float(np.nanmedian(k8)),
                   "nesting_check": m["nesting_check"], "convergence": m["convergence_street_speed"]}
    a, b = res["2017"], res["today"]
    out = {"tower": TOWER, "theta": theta, "width_m": width, "average_flow_throughs": average,
           "ring_m": RING_M, "ring_cells": int(ring.sum()),
           "runs": res,
           "change_pct_2m": round((b["median_2m"] / a["median_2m"] - 1) * 100, 1),
           "change_p90_pct_2m": round((b["p90_2m"] / a["p90_2m"] - 1) * 100, 1),
           "change_pct_8m": round((b["median_8m_x_lawn"] / a["median_8m_x_lawn"] - 1) * 100, 1),
           "change_pct_2m_next_cell_up": round((b["median_2m_next_cell_up"] / a["median_2m_next_cell_up"] - 1) * 100,
                                               1),
           "generated": datetime.now().isoformat(timespec="seconds")}
    tag = (("" if width == NEST["width_m"] else f"_w{width:g}")
           + ("" if average == AVERAGE_HERO else f"_avg{average:g}"))
    with open(os.path.join(OUT, f"score_{theta:05.1f}{tag}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    log(f"  {TOWER}, {theta:g} deg, {width:g} m wide box, {average:g} flow-throughs, street ring {RING_M[0]:g}-{RING_M[1]:g} m: change since 2017 "
        f"{out['change_pct_2m']:+.1f}% at 2 m (windiest tenth {out['change_p90_pct_2m']:+.1f}%), "
        f"{out['change_pct_8m']:+.1f}% at 8 m; next cell up (2-4 m, does not decide) "
        f"{out['change_pct_2m_next_cell_up']:+.1f}%")
    for sc, r in res.items():
        log(f"    {sc}: nesting check {r['nesting_check']['median_abs_rel_diff'] * 100:.1f}%, "
            f"street halves {r['convergence']['median_abs_diff_over_median_speed'] * 100:.1f}%, "
            f"ring cells sampled {r['n_2m']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("run", "score"))
    ap.add_argument("--scenario", nargs="*", default=["2017", "today"])
    ap.add_argument("--dir", type=float, default=315.0)
    ap.add_argument("--width", type=float, default=NEST["width_m"], help="box width, m (400: the blockage check)")
    ap.add_argument("--average", type=float, default=AVERAGE_HERO, help="flow-throughs averaged")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    if args.what == "run":
        for sc in args.scenario:
            run_one(sc, args.dir, log=log, force=args.force, width=args.width, average=args.average)
    score(args.dir, log=log, width=args.width, average=args.average)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
