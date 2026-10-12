"""
The second wind tunnel check: AIJ Case A, one 1:1:2 building (b x b x 2b) in
a turbulent boundary layer, wind square to a face. The independent case for a
change made after Case D failed (POST_NOTES.md, A7): a change counts only if
it holds on a geometry it was not chosen on. The rules (size, box, probes,
thresholds) were fixed in POST_NOTES.md, A7, before the first run.

    .venv-heat/Scripts/python -I tools/aij_case_a.py run [--res 2] [--facade] [--plain]
    .venv-heat/Scripts/python -I tools/aij_case_a.py score

Input: data/wind/aij/wayback_a/CaseA(1_1_2)_*.xls (from the Internet Archive,
like Case D). Its Inflow sheet gives U and k against z/b; the pedestrian plane
sheet gives mean U, V, W at 60 points at z = 0.125b.

Full scale: b = 25 m, the Case D tower's width, so 2 m cells put the same
12.5 cells across both. Frame: x downwind, y across, building centre at the
origin, in units of b in the file.

Output (data/, not committed):
  data/wind/aij/case_a/runs/<res>m[_fw]/plane.npz, run.json
  data/wind/aij/case_a/score.json, score_probes.csv
  data/wind/figures/aij/aij_case_a_*.png (internal)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pipeline.run_wind import ACC_EVERY, AVERAGE_HERO, SPINUP, U_TOP, Z0_M  # noqa: E402
from pipeline.wind_lbm import InflowTurbulence, Solver  # noqa: E402
from tools.aij_case_d import KAPPA_UVW, PASS_NEAR_MAE, PASS_R  # noqa: E402

AIJ = os.path.join(HERE, "data", "wind", "aij")
OUT = os.path.join(AIJ, "case_a")
RUNS = os.path.join(OUT, "runs")
FIG = os.path.join(HERE, "data", "wind", "figures", "aij")

B_M = 25.0                      # full-scale b (POST_NOTES.md, A7)
H_M = 2 * B_M
PROBE_Z_B = 0.125               # the pedestrian plane, z / b
BOX = {"length_m": 600.0, "width_m": 400.0, "height_m": 200.0, "building_from_inlet_m": 200.0}
MIN_MEAS = 0.2                  # relative error only where the tunnel reads at least this / UH


def data_file():
    found = sorted(glob.glob(os.path.join(AIJ, "wayback_a", "CaseA(1_1_2)_*.xls")))
    if not found:
        raise SystemExit("no AIJ Case A data file in data/wind/aij/wayback_a/ (see POST_NOTES.md, A7)")
    return found[-1]


def read_data():
    """Inflow (z m, U m/s, k m2/s2) and probes (number, x m, y m, measured |mean U| m/s)."""
    import xlrd
    book = xlrd.open_workbook(data_file())
    sh = book.sheet_by_name("Inflow")
    inflow = []
    for r in range(sh.nrows):
        v = sh.row_values(r)
        if isinstance(v[1], float):
            inflow.append([v[1] * B_M, v[2], v[6]])
    sh = [s for s in book.sheets() if s.name.startswith("Results(Hol. sec.; z=0.125b")][0]
    probes = []
    for r in range(sh.nrows):
        v = sh.row_values(r)
        if not isinstance(v[0], float) or not isinstance(v[4], float):
            continue
        if abs(v[3] - PROBE_Z_B) > 1e-6:
            raise SystemExit(f"probe {v[0]:g} is not at z = {PROBE_Z_B} b")
        probes.append([v[0], v[1] * B_M, v[2] * B_M, float(np.sqrt(v[4] ** 2 + v[5] ** 2 + v[6] ** 2))])
    return np.array(inflow), np.array(probes)


class Box:
    def __init__(self, res):
        self.res = float(res)
        self.nx = int(round(BOX["length_m"] / res))
        self.ny = int(round(BOX["width_m"] / res))
        self.nz = int(round(BOX["height_m"] / res)) + 1     # + the solid ground layer
        self.ic = int(round(BOX["building_from_inlet_m"] / res))
        self.jc = self.ny // 2
        self.kp = int(PROBE_Z_B * B_M // res) + 1           # the layer holding the probe height

    def meta(self):
        return {"res_m": self.res, "nx": self.nx, "ny": self.ny, "nz": self.nz, "ic": self.ic, "jc": self.jc,
                "probe_layer": self.kp}


def voxelise(bx, sub=4):
    """Solid (nz, ny, nx) by the NYC rules: median of sub x sub samples per column, solid below the top."""
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    x = (np.arange(bx.nx) - bx.ic) * bx.res
    y = (np.arange(bx.ny) - bx.jc) * bx.res
    hs = []
    for oa in offs:
        for ob in offs:
            inx = np.abs(x + oa * bx.res) < B_M / 2
            iny = np.abs(y + ob * bx.res) < B_M / 2
            hs.append(np.where(iny[:, None] & inx[None, :], H_M, 0.0))
    bh = np.median(np.stack(hs), axis=0)
    zc = (np.arange(bx.nz, dtype=np.float32) - 0.5) * bx.res
    solid = zc[:, None, None] < bh[None]
    solid[0] = True
    return solid, bh


def inlet(inflow, bx):
    """Lattice inlet speed per layer, the measured intensity per layer, and UH (U at 2b) in lattice units."""
    z, u, k = inflow[:, 0], inflow[:, 1], inflow[:, 2]

    def at(h, v):
        return np.interp(np.log(np.maximum(h, z[0])), np.log(z), v)

    h = (np.arange(bx.nz) - 0.5) * bx.res
    ratio = at(h, u)
    scale = U_TOP / ratio[-1]
    prof = (ratio * scale).astype(np.float32)
    prof[0] = 0.0
    iu = (np.sqrt(at(h, k) / KAPPA_UVW) / ratio).astype(np.float32)
    iu[0] = 0.0
    return prof, iu, float(at(np.array([H_M]), u)[0] * scale), float(at(np.array([H_M]), u)[0])


def run_name(res, facade):
    return f"{res:g}m" + ("_fw" if facade else "")


def run_one(res, facade, log=print, force=False):
    out = os.path.join(RUNS, run_name(res, facade))
    if os.path.exists(os.path.join(out, "run.json")) and not force:
        log(f"  {os.path.basename(out)}: already done")
        return out
    t0 = time.time()
    inflow, _ = read_data()
    bx = Box(res)
    solid, bh = voxelise(bx)
    prof, iu, uh, uh_ms = inlet(inflow, bx)
    turb = InflowTurbulence(prof, 1, res, bx.ny, iu=iu)
    s = Solver(solid, prof, res_m=res, z0_m=Z0_M, turbulence=turb, log=log, facade_drag=facade)
    ft = bx.nx / float(prof[1:].mean())
    n_spin = int(SPINUP * ft)
    n_avg = int(AVERAGE_HERO * ft)
    n_avg -= n_avg % (2 * ACC_EVERY)
    log(f"  {os.path.basename(out)}: flow-through {ft:,.0f} steps; {n_spin:,} to settle, {n_avg:,} averaged")
    mlups = [s.run(n_spin, log_every=max(1, n_spin // 2))]
    if not np.isfinite(s.max_speed()):
        raise SystemExit("the solver blew up while settling")
    mlups.append(s.run(n_avg // 2, accumulate_every=ACC_EVERY))
    half, k_half = s.sums()
    half = half[:3, bx.kp].copy()
    mlups.append(s.run(n_avg // 2, accumulate_every=ACC_EVERY))
    full, k_full = s.sums()
    if not np.isfinite(full).all():
        raise SystemExit("the solver blew up while averaging")
    mean = full[:3] / k_full
    plane = mean[:, bx.kp]
    sp1 = np.sqrt(((half / k_half) ** 2).sum(axis=0))
    sp2 = np.sqrt((((full[:3, bx.kp] - half) / (k_full - k_half)) ** 2).sum(axis=0))
    open_ = ~solid[bx.kp]
    open_[:, :10] = False
    open_[:, -10:] = False
    d = np.abs(sp1 - sp2)[open_]
    conv = round(float(np.median(d) / np.median((sp1 + sp2)[open_] / 2)), 4)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "plane.npz"), u=plane.astype(np.float32), solid=solid[bx.kp],
                        u_low=mean[:, :8].astype(np.float16), profile=prof, iu=iu, building=bh.astype(np.float32))
    meta = {"box": bx.meta(), "box_size": BOX, "b_m": B_M, "uh_lattice": uh, "uh_tunnel_ms": uh_ms,
            "probe_z_m": PROBE_Z_B * B_M, "smagorinsky": 0.17, "z0_m": Z0_M, "wall_cd": round(s.cd_wall, 5),
            "walls": "slip + wall-law drag" + (" on ground, roofs and facades" if facade else " on ground and roofs"),
            "inflow": "measured U and k (AIJ Case A, Inflow sheet)", "gusts_calibrated": turb.calibrated,
            "flow_through_steps": round(ft),
            "spinup_steps": n_spin, "average_steps": n_avg, "samples": int(k_full),
            "mlups": [round(m) for m in mlups], "device": s.device_name,
            "convergence_plane_speed": conv, "data_file": os.path.basename(data_file()),
            "wall_s": round(time.time() - t0), "generated": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    log(f"  {os.path.basename(out)}: done in {meta['wall_s']} s at {np.mean(mlups):.0f} MLUPs; "
        f"halves differ by {conv * 100:.1f}% of the typical speed (median)")
    del s
    return out


def sample(run_dir, probes):
    """Model |mean u| / UH at each probe, in the cell holding the probe height."""
    with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
        meta = json.load(f)
    b = meta["box"]
    z = np.load(os.path.join(run_dir, "plane.npz"))
    sp = np.sqrt((z["u"].astype(np.float64) ** 2).sum(axis=0))
    i = np.rint(probes[:, 1] / b["res_m"] + b["ic"]).astype(int)
    j = np.rint(probes[:, 2] / b["res_m"] + b["jc"]).astype(int)
    if z["solid"][j, i].any():
        raise SystemExit("a probe falls inside the building")
    return sp[j, i] / meta["uh_lattice"], meta


def score(log=print):
    inflow, probes = read_data()
    uh_ms = float(np.interp(np.log(H_M), np.log(inflow[:, 0]), inflow[:, 1]))
    meas = probes[:, 3] / uh_ms
    use = meas >= MIN_MEAS
    results, rows = {}, []
    for run_dir in sorted(glob.glob(os.path.join(RUNS, "*"))):
        if not os.path.exists(os.path.join(run_dir, "run.json")):
            continue
        sim, meta = sample(run_dir, probes)
        rel = sim / meas - 1
        m = {"n": len(meas), "r": round(float(np.corrcoef(sim, meas)[0, 1]), 3),
             "err_n": int(use.sum()), "mean_abs_rel_err": round(float(np.mean(np.abs(rel[use]))), 3),
             "mean_rel_bias": round(float(np.mean(rel[use])), 3),
             "hit_rate_25pct": round(float(np.mean(np.abs(rel[use]) <= 0.25)), 3),
             "convergence": meta["convergence_plane_speed"]}
        m["passes"] = bool(m["r"] >= PASS_R and m["mean_abs_rel_err"] <= PASS_NEAR_MAE)
        name = os.path.basename(run_dir)
        results[name] = m
        log(f"  case A {name:8s} r {m['r']:.3f}  |sim/meas-1| {m['mean_abs_rel_err']:.3f} (n {m['err_n']})  "
            f"bias {m['mean_rel_bias']:+.3f}  hit {m['hit_rate_25pct']:.2f}  conv {m['convergence'] * 100:.1f}%  "
            f"{'pass' if m['passes'] else 'fail'}")
        for p, s_, ms in zip(probes, sim, meas):
            rows.append([name, int(p[0]), p[1], p[2], round(float(ms), 4), round(float(s_), 4), bool(ms >= MIN_MEAS)])
    out = {"thresholds": {"r_min": PASS_R, "mean_abs_rel_err_max": PASS_NEAR_MAE,
                          "error_where_meas_at_least": MIN_MEAS},
           "uh_tunnel_ms": uh_ms, "runs": results, "data_file": os.path.basename(data_file()),
           "generated": datetime.now().isoformat(timespec="seconds")}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "score.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(OUT, "score_probes.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "probe", "x_m", "y_m", "measured_over_uh", "simulated_over_uh", "in_error_metric"])
        w.writerows(rows)
    figure(probes, meas, results)
    return out


def figure(probes, meas, results):
    """Simulated vs measured, and both along the probe lines; internal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=110)
    styles = {"2m": dict(color="#2e8b57", marker="^", s=24, label="2 m, slip facades"),
              "2m_fw": dict(color="#c0392b", marker="v", s=24, label="2 m, facade friction")}
    for name, st in styles.items():
        if name not in results:
            continue
        sim, _ = sample(os.path.join(RUNS, name), probes)
        a1.scatter(meas, sim, **st)
        m = results[name]
        st = dict(st)
        a2.scatter(probes[:, 1] / B_M + 0.04 * (name == "2m_fw"), sim, color=st["color"], marker=st["marker"], s=14)
    a2.scatter(probes[:, 1] / B_M - 0.04, meas, color="k", marker="o", s=14, label="wind tunnel")
    a1.plot([0, 1.3], [0, 1.3], color="k", lw=0.8)
    a1.fill_between([0, 1.3], [0, 0.975], [0, 1.625], color="k", alpha=0.06, lw=0)
    a1.text(0.03, 1.27, "\n".join(f"{n}: r {m['r']:.2f}, err {m['mean_abs_rel_err']:.2f}" for n, m in results.items()),
            va="top", fontsize=9)
    a1.set_xlim(0, 1.3)
    a1.set_ylim(0, 1.3)
    a1.set_aspect("equal")
    a1.set_xlabel("wind tunnel |mean U| / UH")
    a1.set_ylabel("model |mean U| / UH")
    a1.legend(loc="lower right", fontsize=9)
    a2.set_xlabel("x / b (building from -0.5 to 0.5)")
    a2.set_ylabel("|mean U| / UH")
    a2.legend(fontsize=9)
    fig.suptitle("AIJ Case A, 1:1:2 building, pedestrian plane z = 0.125b (shaded: within 25%)")
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, f"aij_case_a_{datetime.now():%Y%m%d_%H%M%S}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("run", "score"))
    ap.add_argument("--res", type=float, nargs="*", default=[2.0])
    ap.add_argument("--facade", action="store_true", help="facade friction (the A7 change)")
    ap.add_argument("--plain", action="store_true", help="also the reference run without it")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    if args.what == "run":
        for res in args.res:
            if args.facade:
                run_one(res, True, log, args.force)
            if args.plain or not args.facade:
                run_one(res, False, log, args.force)
    score(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
