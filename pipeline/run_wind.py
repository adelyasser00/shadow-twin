"""
Run the wind solver over the city for one or more wind directions.

    .venv-heat/Scripts/python -m pipeline.run_wind --scenario today --dir 315
    .venv-heat/Scripts/python -m pipeline.run_wind --queue hero
    .venv-heat/Scripts/python -m pipeline.run_wind --queue library

Each run lands in data/wind/runs/<scenario>_<season>_<dir>_<res>m/:

  field.npz   time-mean velocity in the box (lattice units, float16) for the
              lowest layers, mean speed and mean |u|^2 there, and the street
              level layer (the first open cell of every column) on its own
  run.json    geometry of the box, inlet profile, steps, speed, convergence

How long it runs: one "flow-through" is the time the inlet wind needs to
cross the box. The flow first settles for SPINUP flow-throughs, then is
averaged for AVERAGE flow-throughs. Convergence is checked by comparing the
first and second half of the average at street level.

Inlet: a power-law profile, u(z) = u_top (z / H)^ALPHA, the usual shape over
a city centre (exponent about 0.3; Davenport). The city itself reshapes it
within a few hundred metres, and every result is read as a ratio to the open
park at the same height, so the exact inlet shape matters little.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np

from .wind_domain import BOX, LAD, WIND_DIR, load_canvas, voxelise
from .wind_lbm import Solver, power_profile

RUNS_DIR = os.path.join(WIND_DIR, "runs")
ALPHA = 0.28
U_TOP = 0.09                    # lattice speed at the lid: Mach 0.16
SPINUP = 1.0                    # flow-throughs before averaging starts (the box starts at the inlet profile)
AVERAGE = 1.5                   # flow-throughs averaged
ACC_EVERY = 4                   # sample every 4th step
Z0_M = 0.1                      # roughness length of ground and roofs for the wall drag
SAVE_LAYERS_M = 520.0           # keep the 3D mean field up to this height

# The heat day (18 December of the typical year) blows from 302 to 325 degrees
# all day; the hero direction is its middle. The library covers the compass in
# 22.5 degree steps, most frequent winter directions first (Central Park
# typical year, December to February daytime: W, WNW and NW are 42% of hours).
HERO = 315.0
LIBRARY = [292.5, 337.5, 270.0, 0.0, 247.5, 22.5, 225.0, 45.0, 202.5, 67.5, 180.0, 90.0,
           157.5, 112.5, 135.0]
SMALL_BOX = {"length_m": 2600.0, "width_m": 2400.0, "height_m": 720.0, "roi_from_inlet_m": 1400.0}


def run_name(scenario, season, theta, res, tag=""):
    return f"{scenario}_{season}_{theta:05.1f}_{res:g}m{tag}"


def street_layer(arr, kfirst):
    """arr (c, nz, ny, nx) at the first open cell of each column -> (c, ny, nx)."""
    k = np.broadcast_to(kfirst[None, None].astype(np.int64), (arr.shape[0], 1) + kfirst.shape)
    return np.take_along_axis(arr, k, axis=1)[:, 0]


def one(scenario, theta, res=8.0, season="dec", box=BOX, tag="", log=print, force=False):
    out_dir = os.path.join(RUNS_DIR, run_name(scenario, season, theta, res, tag))
    if os.path.exists(os.path.join(out_dir, "run.json")) and not force:
        log(f"  {os.path.basename(out_dir)}: already done")
        return out_dir
    t0 = time.time()
    canvas, info = load_canvas()
    solid, veg, cols, bx = voxelise(canvas, info, theta, res, scenario, season, box)
    nz = bx.nz
    gk = int(np.median(cols["kfirst"][:, 0]))          # ground layer at the inlet
    h_lid = (nz - gk) * res
    prof = power_profile(nz, gk, U_TOP, (nz - gk), ALPHA)
    prof[:gk] = 0.0
    s = Solver(solid, prof, veg=veg, res_m=res, z0_m=Z0_M, log=log)
    u_mean = float(prof[gk:].mean())
    ft = bx.nx / u_mean
    n_spin = int(SPINUP * ft)
    n_avg = int(AVERAGE * ft)
    n_avg -= n_avg % (2 * ACC_EVERY)
    log(f"  {os.path.basename(out_dir)}: flow-through {ft:,.0f} steps; {n_spin:,} to settle, "
        f"{n_avg:,} averaged")
    mlups = [s.run(n_spin, log_every=max(1, n_spin // 3))]
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
    first = half / k_half
    second = (full - half) / (k_full - k_half)
    sp1 = street_layer(first[4:5], kf)[0]
    sp2 = street_layer(second[4:5], kf)[0]
    ground_cols = ~cols["roof"]
    ground_cols[:, :INLET_SKIP] = False
    ground_cols[:, -INLET_SKIP:] = False
    d = np.abs(sp1 - sp2)[ground_cols]
    typical = float(np.median((sp1 + sp2)[ground_cols] / 2))
    conv = {"median_abs_diff_over_median_speed": round(float(np.median(d)) / typical, 4),
            "p90_abs_diff_over_median_speed": round(float(np.percentile(d, 90)) / typical, 4)}
    nsave = min(nz, int(round(SAVE_LAYERS_M / res)) + 1 + gk)
    os.makedirs(out_dir, exist_ok=True)
    street = street_layer(mean, kf)
    np.savez_compressed(
        os.path.join(out_dir, "field.npz"),
        u=mean[:3, :nsave].astype(np.float16), uu=mean[3, :nsave].astype(np.float16),
        speed=mean[4, :nsave].astype(np.float16),
        street_u=street[:3].astype(np.float32), street_uu=street[3].astype(np.float32),
        street_speed=street[4].astype(np.float32),
        kfirst=kf, roof=cols["roof"], ground=cols["ground"].astype(np.float32),
        building=cols["building"].astype(np.float32), canopy=cols["canopy"],
        cover=cols["cover"], solid=np.packbits(solid[:nsave], axis=-1), profile=prof)
    meta = {"scenario": scenario, "season": season, "theta_from_deg": theta, "res_m": res,
            "box": bx.meta(), "box_size": box, "gmin_m": cols["gmin"], "ground_layer": gk,
            "lid_above_ground_m": h_lid, "profile": {"alpha": ALPHA, "u_top": U_TOP},
            "tau0": 0.5005, "smagorinsky": 0.17, "leaf_area_density": LAD[season],
            "walls": "slip + wall-law drag", "z0_m": Z0_M, "wall_cd": round(s.cd_wall, 5),
            "flow_through_steps": round(ft), "spinup_steps": n_spin, "average_steps": n_avg,
            "samples": int(k_full), "accumulate_every": ACC_EVERY, "saved_layers": nsave,
            "mlups": [round(m) for m in mlups], "device": s.device_name,
            "convergence_street_speed": conv, "wall_s": round(time.time() - t0),
            "generated": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    log(f"  {os.path.basename(out_dir)}: done in {meta['wall_s']} s at {np.mean(mlups):.0f} MLUPs; "
        f"street speed halves differ by {conv['median_abs_diff_over_median_speed'] * 100:.1f}% of the typical speed (median)")
    del s
    return out_dir


INLET_SKIP = 12


def queue(name):
    """Named batches, in the order they are worth running."""
    nw = [292.5, 337.5, 270.0]
    rest = [d for d in LIBRARY if d not in nw]
    q = {
        "hero": [("today", HERO, 8.0, BOX, "", "dec"), ("2017", HERO, 8.0, BOX, "", "dec")],
        "today-nw": [("today", d, 8.0, BOX, "", "dec") for d in nw],
        "grid": [("today", HERO, 8.0, SMALL_BOX, "_small", "dec"),
                 ("today", HERO, 6.0, SMALL_BOX, "_small", "dec")],
        "today-rest": [("today", d, 8.0, BOX, "", "dec") for d in rest],
        # The June control day blows from 250 to 270 degrees; trees in leaf.
        "june": [(sc, d, 8.0, BOX, "", "jun") for d in (247.5, 270.0) for sc in ("today", "2017")],
        "2017-lib": [("2017", d, 8.0, BOX, "", "dec") for d in LIBRARY],
    }
    if name == "all":
        return [j for k in ("hero", "today-nw", "grid", "today-rest", "june", "2017-lib") for j in q[k]]
    if name not in q:
        raise SystemExit(f"unknown queue {name}; one of {', '.join(q)} or all")
    return q[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=("2017", "today"))
    ap.add_argument("--dir", type=float, nargs="*")
    ap.add_argument("--res", type=float, default=8.0)
    ap.add_argument("--season", default="dec", choices=("dec", "jun"))
    ap.add_argument("--queue", nargs="*", help="hero, today-nw, grid, today-rest, june, 2017-lib, or all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    jobs = []
    for q in args.queue or []:
        jobs += queue(q)
    if args.scenario:
        jobs += [(args.scenario, d, args.res, BOX, "", args.season) for d in (args.dir or [HERO])]
    for sc, d, res, box, tag, season in jobs:
        one(sc, d, res, season, box, tag, log=log, force=args.force)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
