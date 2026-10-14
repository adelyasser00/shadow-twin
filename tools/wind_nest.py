"""
A 2 m wind box around Central Park Tower, fed by the 8 m city run
(POST_NOTES.md, A6 step 2; the rules are in A8, fixed before the first run).

One-way nesting: what enters the 2 m box at its inlet face is the 8 m run's
time-mean wind on that plane, plus gusts the size of the 8 m run's resolved
turbulence there. Inside, the city is voxelised from the same 2 m canvas at
2 m, with the wall settings that passed the wind tunnel checks at 2 m (facade
friction on). Sides periodic and lid free slip, as in the city runs, or
with --sides parent the side faces held at the 8 m run's time-mean wind
(one-way nesting on the sides too; New York's periodic box failed its
blockage check, POST_NOTES A8).

    .venv-heat/Scripts/python -I tools/wind_nest.py run [--scenario 2017 today] [--dir 315] [--width 480]
    .venv-heat/Scripts/python -I tools/wind_nest.py score [--dir 315] [--width 480]
    TWIN_CITY=rotterdam .venv-heat/Scripts/python -I tools/wind_nest.py run --dir 247.5 --sides parent --average 6

Any city: the tower is the city's focus (cities/<id>.json), New York's is
Central Park Tower.

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

from pipeline import city as _city  # noqa: E402
from pipeline.run_wind import (ACC_EVERY, ALPHA, AVERAGE_HERO, HERO, INLET_SKIP, RUNS_DIR, SPINUP, Z0_M,  # noqa: E402
                               run_name as parent_name)
from pipeline.wind_domain import Box, load_canvas, voxelise  # noqa: E402
from tools.aij_case_d import KAPPA_UVW  # noqa: E402

NYC = _city.city_id() == "nyc"
BEFORE = _city.before()
TOWER = "Central Park Tower" if NYC else _city.get("focus.name")
TOWER_LONLAT = (-73.98102, 40.76644) if NYC else tuple(_city.get("focus.lonlat"))   # NYC as in tools/wind_story.py
CRS_M = _city.get("crs_geotiff", "EPSG:32118")
RES = 2.0
PARENT_RES = 8.0
NEST = {"length_m": 600.0, "width_m": 480.0, "height_m": 600.0, "roi_from_inlet_m": 250.0}
OUT = os.path.join(_city.wind_dir(), "nest")       # data/wind/nest for New York
CHECK_X_M = 40.0                # the nesting check: this far downstream of the inlet,
CHECK_Z_M = (150.0, 450.0)      # at these heights above the box's lowest ground (A8)
RING_M = (20.0, 80.0)           # street ring around the tower, as in tools/wind_story.py


def tower_xy():
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", CRS_M, always_xy=True).transform
    return tf(*TOWER_LONLAT)


def load_parent(scenario, theta, parent_tag=""):
    d = os.path.join(RUNS_DIR, parent_name(scenario, "dec", theta, PARENT_RES, parent_tag))
    with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
        m = json.load(f)
    z = np.load(os.path.join(d, "field.npz"))
    return m, z["u"].astype(np.float32), z["uu"].astype(np.float32)


def parent_box(info, m):
    # The 8 m run may be centred elsewhere than the canvas default (--centre-focus).
    info = dict(info, roi_centre=list(m["box"]["p0"]))
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


SEED = 11                       # the inlet gusts' random seed; every run before D23 used 11


def nest_box(width, length=NEST["length_m"], height=NEST["height_m"]):
    return dict(NEST, width_m=float(width), length_m=float(length), height_m=float(height))


def nest_name(scenario, theta, facade=True, width=NEST["width_m"], average=AVERAGE_HERO, sides="periodic",
              parent_tag="", length=NEST["length_m"], height=NEST["height_m"], seed=SEED):
    return (f"{scenario}_dec_{theta:05.1f}_{RES:g}m" + ("_fw" if facade else "")
            + ("" if width == NEST["width_m"] else f"_w{width:g}")
            + ("" if length == NEST["length_m"] else f"_l{length:g}")
            + ("" if height == NEST["height_m"] else f"_h{height:g}")
            + ("" if average == AVERAGE_HERO else f"_avg{average:g}")
            + ("" if sides == "periodic" else "_sides")
            + (f"_p{parent_tag.strip('_')}" if parent_tag else "")
            + ("" if seed == SEED else f"_s{seed}"))


def prepare(scenario, theta, facade=True, log=print, width=NEST["width_m"], average=AVERAGE_HERO,
            sides="periodic", parent_tag="", length=NEST["length_m"], height=NEST["height_m"], seed=SEED):
    """Everything the solve needs, as one dict of arrays (pipeline/nest_solve.py), and the meta to save."""
    canvas, info = load_canvas()
    m, u, uu = load_parent(scenario, theta, parent_tag)
    pb = parent_box(info, m)
    tx, ty = tower_xy()
    info_c = dict(info, roi_centre=[float(tx), float(ty)])
    box = nest_box(width, length, height)
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
    pname = parent_name(scenario, "dec", theta, PARENT_RES, parent_tag)
    log(f"  inlet from {pname}: lid {prof[-1]:.4f}, "
        f"Iu {iu[gk + 5]:.2f} at {zabs[gk + 5] - gmin:.0f} m, {iu[nz // 2]:.2f} at {zabs[nz // 2] - gmin:.0f} m")

    side_u = None
    if sides == "parent":
        # Both side faces held at the 8 m run's time-mean wind, all along the box.
        side_u = np.zeros((3, nz, 2, nx), np.float32)
        for q, j in enumerate((0, ny - 1)):
            xs, ys = bx.to_map(np.arange(nx), np.full(nx, j))
            Us, _, _ = parent_at(m, u, uu, pb, xs[None, :], ys[None, :], zabs[:, None])
            side_u[:, :, q, :] = Us
        side_u[:, :gk] = 0.0
        log(f"  sides from the 8 m run: mean along-wind {float(side_u[0, gk:].mean()):.4f}, "
            f"across {float(np.abs(side_u[1, gk:]).mean()):.4f} (lattice units)")

    # The nesting-check plane, from the 8 m run (the 2 m side comes after the solve).
    ic = int(round(CHECK_X_M / RES))
    kz = np.nonzero((zabs - gmin >= CHECK_Z_M[0]) & (zabs - gmin <= CHECK_Z_M[1]))[0]
    xc, yc = bx.to_map(np.full(ny, ic), np.arange(ny))
    Up, _, flp = parent_at(m, u, uu, pb, xc[None, :], yc[None, :], zabs[kz][:, None])
    pack = {"solid": solid, "veg": veg, "prof": prof, "iu": iu, "gk": gk,
            "U_in": U, "fluid_in": fluid, "side_u": side_u,
            "kfirst": cols["kfirst"], "roof": cols["roof"], "ic": ic, "kz": kz,
            "check_parent": np.sqrt((Up ** 2).sum(axis=0)).astype(np.float32),
            "check_fluid": flp.astype(np.float32),
            "res_m": RES, "z0_m": Z0_M, "facade": bool(facade), "spinup": SPINUP, "average": float(average),
            "accumulate_every": ACC_EVERY, "inlet_skip": INLET_SKIP, "seed": int(seed)}
    meta = {"scenario": scenario, "season": "dec", "theta_from_deg": float(theta), "res_m": RES,
            "box": bx.meta(), "box_size": box, "gmin_m": gmin, "ground_layer": gk, "tower": TOWER,
            "tower_xy": [float(tx), float(ty)], "parent": pname, "gust_seed": int(seed),
            "smagorinsky": 0.17, "z0_m": Z0_M,
            "walls": "slip + wall-law drag" + (" on ground, roofs and facades" if facade else " on ground and roofs"),
            "sides": "periodic" if sides == "periodic" else "the 8 m run's time-mean wind",
            "lid": "free slip", "average_flow_throughs": average, "accumulate_every": ACC_EVERY}
    extra = {"building": cols["building"].astype(np.float32), "ground": cols["ground"].astype(np.float32)}
    return pack, meta, extra


def finish(out, pack, meta, extra, res, t0, log=print):
    """Write street.npz and run.json exactly as the original run_one did."""
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "street.npz"), street_speed=res["street_speed"],
                        street_speed_above=res["street_speed_above"],
                        street_u=res["street_u"], kfirst=pack["kfirst"], roof=pack["roof"],
                        building=extra["building"], ground=extra["ground"],
                        u_low=res["u_low"], profile=pack["prof"], iu=np.asarray(pack["iu"], np.float32),
                        check_child=res["check_child"], check_parent=res["check_parent"], check_ok=res["check_ok"])
    r = res["meta"]
    meta = dict(meta)
    meta.update({"wall_cd": r["wall_drag"], "gusts_calibrated": r["gusts_calibrated"],
                 "flow_through_steps": r["flow_through_steps"], "spinup_steps": r["spinup_steps"],
                 "average_steps": r["average_steps"], "samples": r["samples"], "mlups": r["mlups"],
                 "device": r["device"], "backend": r.get("backend"),
                 "convergence_street_speed": r["convergence_street_speed"],
                 "nesting_check": {"x_from_inlet_m": CHECK_X_M, "z_m": CHECK_Z_M, "n": int(res["check_ok"].sum()),
                                   "median_abs_rel_diff": round(r["nest_err"], 4),
                                   "median_rel_diff": round(r["nest_bias"], 4)},
                 "wall_s": round(time.time() - t0), "generated": datetime.now().isoformat(timespec="seconds")})
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
    conv = r["convergence_street_speed"]
    log(f"  {os.path.basename(out)}: done in {meta['wall_s']} s at {np.mean(r['mlups']):.0f} MLUPs; street halves "
        f"differ by {conv['median_abs_diff_over_median_speed'] * 100:.1f}%; 2 m vs 8 m on the check plane: "
        f"median |diff| {r['nest_err'] * 100:.1f}%, median diff {r['nest_bias'] * 100:+.1f}%")
    return out


def run_one(scenario, theta, facade=True, log=print, force=False, width=NEST["width_m"], average=AVERAGE_HERO,
            sides="periodic", parent_tag="", backend="opencl", length=NEST["length_m"], height=NEST["height_m"],
            seed=SEED):
    from pipeline.nest_solve import solve
    out = os.path.join(OUT, nest_name(scenario, theta, facade, width, average, sides, parent_tag, length, height,
                                      seed))
    if os.path.exists(os.path.join(out, "run.json")) and not force:
        log(f"  {os.path.basename(out)}: already done")
        return out
    t0 = time.time()
    pack, meta, extra = prepare(scenario, theta, facade, log, width, average, sides, parent_tag, length, height,
                                seed)
    log(f"  {os.path.basename(out)}: solving")
    res = solve(pack, log=log, backend=backend)
    return finish(out, pack, meta, extra, res, t0, log=log)


def tower_ring():
    """Heat-grid points on open ground 20 to 80 m from the tower's footprint, as tools/wind_story.py does."""
    from scipy import ndimage
    from pipeline.wind_export import heat_points
    x, y, layers, masks, meta = heat_points()
    tx, ty = tower_xy()
    bld = masks["building"]
    if NYC or "labels" not in masks:
        lab, _ = ndimage.label(bld)
    else:
        # The tower's own outline (BAG), as tools/city_wind_story.py rings it.
        lab = masks["labels"]
    r = int(round((y[0, 0] - ty) / 2.0))
    c = int(round((tx - x[0, 0]) / 2.0))
    if not lab[r, c]:
        raise SystemExit("no building at the tower's position")
    dist = ndimage.distance_transform_edt(lab != lab[r, c]) * 2.0
    ring = ~bld & ~masks["water"] & (dist >= RING_M[0]) & (dist <= RING_M[1])
    return x, y, ring


def parent_ring_speed(sc, theta, parent_tag, x, y, ring):
    """The 8 m run's street speed on the ring: k_heatgrid if wind_export made one, else resampled here."""
    from pipeline.wind_export import RunBox, load_run, resample
    d = os.path.join(RUNS_DIR, parent_name(sc, "dec", theta, PARENT_RES, parent_tag))
    k = os.path.join(d, "k_heatgrid.npy")
    if os.path.exists(k):
        return np.load(k)[ring]
    with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
        m = json.load(f)
    m["_dir"] = d
    run = load_run(m)
    return resample(run["street_speed"], run["ok"], RunBox(m), x[ring], y[ring])


def score_path(theta, width=NEST["width_m"], average=AVERAGE_HERO, sides="periodic", parent_tag="",
               length=NEST["length_m"], height=NEST["height_m"], seed=SEED):
    tag = (("" if width == NEST["width_m"] else f"_w{width:g}")
           + ("" if length == NEST["length_m"] else f"_l{length:g}")
           + ("" if height == NEST["height_m"] else f"_h{height:g}")
           + ("" if average == AVERAGE_HERO else f"_avg{average:g}")
           + ("" if sides == "periodic" else "_sides")
           + (f"_p{parent_tag.strip('_')}" if parent_tag else "")
           + ("" if seed == SEED else f"_s{seed}"))
    return os.path.join(OUT, f"score_{theta:05.1f}{tag}.json")


def score(theta, facade=True, log=print, width=NEST["width_m"], average=AVERAGE_HERO, sides="periodic",
          parent_tag="", length=NEST["length_m"], height=NEST["height_m"], seed=SEED):
    from pipeline.wind_export import RunBox, resample
    x, y, ring = tower_ring()
    res = {}
    for sc in (BEFORE, "today"):
        d = os.path.join(OUT, nest_name(sc, theta, facade, width, average, sides, parent_tag, length, height, seed))
        with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
            m = json.load(f)
        z = np.load(os.path.join(d, "street.npz"))
        ok = (~z["roof"]) & np.isfinite(z["street_speed"])
        v = resample(z["street_speed"], ok, RunBox(m), x[ring], y[ring])
        va = resample(z["street_speed_above"], ok, RunBox(m), x[ring], y[ring])
        k8 = parent_ring_speed(sc, theta, parent_tag, x, y, ring)
        res[sc] = {"median_2m": float(np.nanmedian(v)), "p90_2m": float(np.nanpercentile(v, 90)),
                   "mean_2m": float(np.nanmean(v)), "p25_2m": float(np.nanpercentile(v, 25)),
                   "p75_2m": float(np.nanpercentile(v, 75)),
                   "median_2m_next_cell_up": float(np.nanmedian(va)),
                   "n_2m": int(np.isfinite(v).sum()), "median_8m_x_lawn": float(np.nanmedian(k8)),
                   "nesting_check": m["nesting_check"], "convergence": m["convergence_street_speed"]}
    a, b = res[BEFORE], res["today"]
    out = {"tower": TOWER, "theta": theta, "width_m": width, "length_m": length, "height_m": height, "gust_seed": seed,
           "average_flow_throughs": average, "sides": sides,
           "parent": parent_name(BEFORE, "dec", theta, PARENT_RES, parent_tag),
           "ring_m": RING_M, "ring_cells": int(ring.sum()),
           "runs": res,
           "change_pct_2m": round((b["median_2m"] / a["median_2m"] - 1) * 100, 1),
           "change_p90_pct_2m": round((b["p90_2m"] / a["p90_2m"] - 1) * 100, 1),
           "change_mean_pct_2m": round((b["mean_2m"] / a["mean_2m"] - 1) * 100, 1),
           "change_p25_pct_2m": round((b["p25_2m"] / a["p25_2m"] - 1) * 100, 1),
           "change_p75_pct_2m": round((b["p75_2m"] / a["p75_2m"] - 1) * 100, 1),
           "change_pct_8m": round((b["median_8m_x_lawn"] / a["median_8m_x_lawn"] - 1) * 100, 1),
           "change_pct_2m_next_cell_up": round((b["median_2m_next_cell_up"] / a["median_2m_next_cell_up"] - 1) * 100,
                                               1),
           "generated": datetime.now().isoformat(timespec="seconds")}
    with open(score_path(theta, width, average, sides, parent_tag, length, height, seed), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(out, f, indent=1)
    log(f"  {TOWER}, {theta:g} deg, box {length:g} x {width:g} x {height:g} m, {average:g} flow-throughs, street ring {RING_M[0]:g}-{RING_M[1]:g} m, sides {sides}: change since {BEFORE} "
        f"{out['change_pct_2m']:+.1f}% at 2 m (mean {out['change_mean_pct_2m']:+.1f}%, windiest tenth {out['change_p90_pct_2m']:+.1f}%), "
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
    ap.add_argument("--scenario", nargs="*", default=[BEFORE, "today"])
    ap.add_argument("--dir", type=float, default=HERO)
    ap.add_argument("--sides", choices=("periodic", "parent"), default="periodic",
                    help="parent: hold the side faces at the 8 m run's mean wind")
    ap.add_argument("--parent-tag", default="", help="the 8 m runs' folder suffix, e.g. _zh (run_wind --tag)")
    ap.add_argument("--backend", default="opencl", choices=("opencl", "cuda", "auto"))
    ap.add_argument("--width", type=float, default=NEST["width_m"], help="box width, m (400: the blockage check)")
    ap.add_argument("--length", type=float, default=NEST["length_m"], help="box length along the wind, m")
    ap.add_argument("--height", type=float, default=NEST["height_m"], help="box height, m")
    ap.add_argument("--average", type=float, default=AVERAGE_HERO, help="flow-throughs averaged")
    ap.add_argument("--seed", type=int, default=SEED, help="the inlet gusts' random seed (D23 noise test)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    if args.what == "run":
        for sc in args.scenario:
            run_one(sc, args.dir, log=log, force=args.force, width=args.width, average=args.average,
                    sides=args.sides, parent_tag=args.parent_tag, backend=args.backend, length=args.length,
                    height=args.height, seed=args.seed)
    score(args.dir, log=log, width=args.width, average=args.average, sides=args.sides, parent_tag=args.parent_tag,
          length=args.length, height=args.height, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
