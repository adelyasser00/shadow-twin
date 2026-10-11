"""
The wind tunnel check for the wind layer: AIJ Case D, one 100 m tower among
regular low-rise blocks (Architectural Institute of Japan benchmark, 1/400
model, Niigata Institute of Technology). The NYC solver runs on the tunnel's
geometry and inflow with the NYC settings, and street-level wind is compared
with the tunnel's probes. The rules (resolution, thresholds, how probes are
read) were fixed in POST_NOTES.md, Part A, before the first run.

    .venv-heat/Scripts/python -I tools/aij_case_d.py check
    .venv-heat/Scripts/python -I tools/aij_case_d.py run [--res 8 4] [--dir 0 22.5 45] [--mirror]
    .venv-heat/Scripts/python -I tools/aij_case_d.py score

Input: the AIJ data file, data/wind/aij/wayback/CaseD(Highrise+Blocks)_*.xls
(from the Internet Archive; aij.or.jp serves a busy page). It holds the probe
table, the measured speeds for 0, 22.5 and 45 degrees and the inflow
profile; its Fig. 3 gives the block layout written out below.

Output (data/, not committed):
  data/wind/aij/runs/<dir>_<res>m[_mirror]/street.npz, run.json
  data/wind/aij/score.json, score_probes.csv
  data/wind/figures/aij/aij_check_*.png, aij_scatter_*.png (internal)

Frame: the probe table's, model mm times 0.4 = metres full scale, tower centre
at the origin. Wind direction theta means the flow heads toward
(cos theta, sin theta): anticlockwise from +x (POST_NOTES.md, A5, gives the
evidence from the measurements). --mirror runs the opposite turn, as a
diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pipeline.run_wind import ACC_EVERY, AVERAGE_HERO, INLET_SKIP, SPINUP, U_TOP, Z0_M, street_layer  # noqa: E402
from pipeline.wind_lbm import InflowTurbulence, Solver  # noqa: E402

AIJ = os.path.join(HERE, "data", "wind", "aij")
RUNS = os.path.join(AIJ, "runs")
FIG = os.path.join(HERE, "data", "wind", "figures", "aij")
SCALE = 0.4                     # metres full scale per mm of model (1/400)
DIRS = (0.0, 22.5, 45.0)

# Fig. 3, model mm, tower centre at the origin: the left edges of the nine
# block columns (blocks 100 mm, streets alternately 75 and 50 mm) and how many
# rows each column has either side of the centre row (row pitch 125 mm,
# streets 25 mm). The tower stands in place of the central block.
COL_LEFT_MM = (-700, -525, -375, -200, -50, 125, 275, 450, 600)
COL_ROWS = (2, 4, 4, 5, 5, 5, 4, 4, 2)
BLOCK_MM, ROW_PITCH_MM = 100.0, 125.0
BLOCK_H_M = 10.0                # not labelled in Fig. 3; SimScale's write-up of the case
TOWER_HALF_M, TOWER_H_M = 12.5, 100.0
Z_REF_M = 100.0                 # UH: the approach wind at the tower's height

BOX = {"length_m": 1600.0, "width_m": 1000.0, "height_m": 300.0, "tower_from_inlet_m": 600.0}
KAPPA_UVW = 0.5 * (1 + 0.75 ** 2 + 0.5 ** 2)    # k = this * sigma_u^2 for our gust ratios

# The gate (POST_NOTES.md, A1), decided at 8 m only.
PASS_R = 0.7
PASS_NEAR_MAE = 0.25
NEAR_M = 50.0
HIT = 0.25


def data_file():
    found = sorted(glob.glob(os.path.join(AIJ, "wayback", "CaseD(Highrise+Blocks)_*.xls")))
    if not found:
        raise SystemExit("no AIJ Case D data file in data/wind/aij/wayback/ (see POST_NOTES.md, A4)")
    return found[-1]


def read_data():
    """Inflow (z m, U/UH, k/UH^2) and probes (number, x m, y m, Vs/UH per direction, NaN if not measured)."""
    import xlrd
    book = xlrd.open_workbook(data_file())
    sh = book.sheet_by_name("Inflow")
    inflow = [sh.row_values(r) for r in range(sh.nrows)]
    inflow = np.array([[r[1] * SCALE, r[4], r[6]] for r in inflow if isinstance(r[1], float)])
    sh = book.sheet_by_name("Results (Exp_T)")
    probes = []
    for r in range(sh.nrows):
        v = sh.row_values(r)
        if not isinstance(v[0], float) or not isinstance(v[1], float):
            continue
        meas = [x if isinstance(x, float) else np.nan for x in v[7:10]]
        probes.append([v[0], v[1] * SCALE, v[2] * SCALE] + meas)
    return inflow, np.array(probes)


def buildings():
    """(x0, y0, width, depth, height) in metres, probe frame."""
    out = []
    for x0, n in zip(COL_LEFT_MM, COL_ROWS):
        for k in range(-n, n + 1):
            if x0 == -50 and k == 0:
                continue
            out.append((x0 * SCALE, (ROW_PITCH_MM * k - BLOCK_MM / 2) * SCALE,
                        BLOCK_MM * SCALE, BLOCK_MM * SCALE, BLOCK_H_M))
    out.append((-TOWER_HALF_M, -TOWER_HALF_M, 2 * TOWER_HALF_M, 2 * TOWER_HALF_M, TOWER_H_M))
    return out


def height_at(x, y):
    h = np.zeros(np.shape(x), np.float32)
    for x0, y0, w, d, hh in buildings():
        inside = (x >= x0) & (x < x0 + w) & (y >= y0) & (y < y0 + d)
        h[inside] = np.maximum(h[inside], hh)
    return h


class AijBox:
    """The turned box: x downwind, y to its left; the tower centre on a cell centre."""

    def __init__(self, theta, res, mirror=False):
        self.theta, self.res, self.mirror = float(theta), float(res), bool(mirror)
        self.nx = int(round(BOX["length_m"] / res))
        self.ny = int(round(BOX["width_m"] / res))
        self.nz = int(round(BOX["height_m"] / res)) + 1      # + the solid ground layer
        self.ic = int(round(BOX["tower_from_inlet_m"] / res))
        self.jc = self.ny // 2
        t = math.radians(-theta if mirror else theta)
        self.ex = np.array([math.cos(t), math.sin(t)])
        self.ey = np.array([-self.ex[1], self.ex[0]])

    def to_probe(self, i, j):
        """Probe-frame metres of cell centres (i, j); fractional cells welcome."""
        a = (np.asarray(i, float) - self.ic) * self.res
        b = (np.asarray(j, float) - self.jc) * self.res
        return a * self.ex[0] + b * self.ey[0], a * self.ex[1] + b * self.ey[1]

    def from_probe(self, x, y):
        a = np.asarray(x, float) * self.ex[0] + np.asarray(y, float) * self.ex[1]
        b = np.asarray(x, float) * self.ey[0] + np.asarray(y, float) * self.ey[1]
        return a / self.res + self.ic, b / self.res + self.jc

    def meta(self):
        return {"theta": self.theta, "res_m": self.res, "mirror": self.mirror, "nx": self.nx,
                "ny": self.ny, "nz": self.nz, "ic": self.ic, "jc": self.jc}


def voxelise(bx, sub=4):
    """Solid (nz, ny, nx), column building height and first open layer, by the NYC rules."""
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    jj, ii = np.meshgrid(np.arange(bx.ny), np.arange(bx.nx), indexing="ij")
    hs = np.stack([height_at(*bx.to_probe(ii + oa, jj + ob)) for oa in offs for ob in offs])
    bh = np.median(hs, axis=0)
    zc = (np.arange(bx.nz, dtype=np.float32) - 0.5) * bx.res
    solid = zc[:, None, None] < bh[None]
    solid[0] = True
    return solid, bh, np.argmin(solid, axis=0).astype(np.int16)


def inlet(inflow, bx):
    """Lattice inlet speed per layer, the measured intensity per layer, and UH in lattice units."""
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
    return prof, iu, float(at(np.array([Z_REF_M]), u)[0] * scale)


def run_name(theta, res, mirror):
    return f"d{theta:04.1f}_{res:g}m" + ("_mirror" if mirror else "")


def run_one(theta, res, mirror=False, log=print, force=False):
    out = os.path.join(RUNS, run_name(theta, res, mirror))
    if os.path.exists(os.path.join(out, "run.json")) and not force:
        log(f"  {os.path.basename(out)}: already done")
        return out
    t0 = time.time()
    inflow, _ = read_data()
    bx = AijBox(theta, res, mirror)
    solid, bh, kfirst = voxelise(bx)
    prof, iu, uh = inlet(inflow, bx)
    turb = InflowTurbulence(prof, 1, res, bx.ny, iu=iu)
    s = Solver(solid, prof, res_m=res, z0_m=Z0_M, turbulence=turb, log=log)
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
    half = half.copy()
    mlups.append(s.run(n_avg // 2, accumulate_every=ACC_EVERY))
    full, k_full = s.sums()
    if not np.isfinite(full).all():
        raise SystemExit("the solver blew up while averaging")
    mean = full / k_full
    sp1 = street_layer((half / k_half)[4:5], kfirst)[0]
    sp2 = street_layer(((full - half) / (k_full - k_half))[4:5], kfirst)[0]
    ground = kfirst == 1
    ground[:, :INLET_SKIP] = False
    ground[:, -INLET_SKIP:] = False
    d = np.abs(sp1 - sp2)[ground]
    typical = float(np.median((sp1 + sp2)[ground] / 2))
    conv = {"median_abs_diff_over_median_speed": round(float(np.median(d)) / typical, 4),
            "p90_abs_diff_over_median_speed": round(float(np.percentile(d, 90)) / typical, 4)}
    street = street_layer(mean, kfirst)
    nkeep = int(round(160.0 / res)) + 1
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "street.npz"), speed=street[4].astype(np.float32),
                        u=street[:3].astype(np.float32), kfirst=kfirst, building=bh.astype(np.float32),
                        speed_low=mean[4, :nkeep].astype(np.float16), u_low=mean[:3, :nkeep].astype(np.float16),
                        profile=prof, iu=iu)
    meta = {"box": bx.meta(), "box_size": BOX, "uh_lattice": uh, "z_ref_m": Z_REF_M,
            "tau0": 0.5005, "smagorinsky": 0.17, "walls": "slip + wall-law drag", "z0_m": Z0_M,
            "wall_cd": round(s.cd_wall, 5), "inflow": "measured U/UH and k (AIJ Case D, Inflow sheet)",
            "flow_through_steps": round(ft), "spinup_steps": n_spin, "average_steps": n_avg,
            "samples": int(k_full), "accumulate_every": ACC_EVERY,
            "mlups": [round(m) for m in mlups], "device": s.device_name,
            "convergence_street_speed": conv, "tower_top_m": float((kfirst.max() - 1) * res),
            "data_file": os.path.basename(data_file()), "wall_s": round(time.time() - t0),
            "generated": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    log(f"  {os.path.basename(out)}: done in {meta['wall_s']} s at {np.mean(mlups):.0f} MLUPs; street speed "
        f"halves differ by {conv['median_abs_diff_over_median_speed'] * 100:.1f}% of the typical speed (median)")
    del s
    return out


def near_tower(x, y):
    return np.hypot(np.maximum(np.abs(x) - TOWER_HALF_M, 0), np.maximum(np.abs(y) - TOWER_HALF_M, 0)) <= NEAR_M


def sample(run_dir, probes):
    """Street-level mean speed / UH at each probe, and whether the probe had to move off a building."""
    with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
        meta = json.load(f)
    b = meta["box"]
    bx = AijBox(b["theta"], b["res_m"], b["mirror"])
    z = np.load(os.path.join(run_dir, "street.npz"))
    street = z["kfirst"] == 1
    fi, fj = bx.from_probe(probes[:, 1], probes[:, 2])
    jj, ii = np.nonzero(street)
    sim, moved = [], []
    for i, j in zip(fi, fj):
        ci, cj = int(round(float(i))), int(round(float(j)))
        if not street[cj, ci]:
            k = int(np.argmin((ii - i) ** 2 + (jj - j) ** 2))
            ci, cj = int(ii[k]), int(jj[k])
            moved.append(True)
        else:
            moved.append(False)
        sim.append(float(z["speed"][cj, ci]) / meta["uh_lattice"])
    return np.array(sim), np.array(moved), meta


def metrics(meas, sim, near):
    ok = np.isfinite(meas)
    rel = sim / meas - 1
    return {"n": int(ok.sum()), "r": round(float(np.corrcoef(sim[ok], meas[ok])[0, 1]), 3),
            "near_n": int((ok & near).sum()),
            "near_mean_abs_rel_err": round(float(np.mean(np.abs(rel[ok & near]))), 3),
            "hit_rate_25pct": round(float(np.mean(np.abs(rel[ok]) <= HIT)), 3),
            "mean_rel_bias": round(float(np.mean(rel[ok])), 3),
            "ratio_of_means": round(float(sim[ok].mean() / meas[ok].mean()), 3)}


def score(log=print):
    _, probes = read_data()
    near = near_tower(probes[:, 1], probes[:, 2])
    results, rows = {}, []
    for run_dir in sorted(glob.glob(os.path.join(RUNS, "d*"))):
        if not os.path.exists(os.path.join(run_dir, "run.json")):
            continue
        sim, moved, meta = sample(run_dir, probes)
        b = meta["box"]
        meas = probes[:, 3 + DIRS.index(b["theta"])]
        m = metrics(meas, sim, near)
        m["moved_off_buildings"] = int(moved[np.isfinite(meas)].sum())
        m["convergence"] = meta["convergence_street_speed"]["median_abs_diff_over_median_speed"]
        m["passes"] = bool(m["r"] >= PASS_R and m["near_mean_abs_rel_err"] <= PASS_NEAR_MAE)
        name = os.path.basename(run_dir)
        results[name] = m
        log(f"  {name:16s} r {m['r']:.3f}  near |sim/meas-1| {m['near_mean_abs_rel_err']:.3f}  "
            f"hit {m['hit_rate_25pct']:.2f}  bias {m['mean_rel_bias']:+.3f}  moved {m['moved_off_buildings']}  "
            f"conv {m['convergence'] * 100:.1f}%  {'pass' if m['passes'] else 'fail'}")
        for p, s, mv, nr, ms in zip(probes, sim, moved, near, meas):
            rows.append([name, int(p[0]), p[1], p[2], b["theta"], b["res_m"], b["mirror"],
                         None if np.isnan(ms) else round(float(ms), 4), round(float(s), 4), bool(mv), bool(nr)])
    gate = [results.get(run_name(d, 8.0, False)) for d in DIRS]
    verdict = "not run" if None in gate else ("pass" if all(g["passes"] for g in gate) else "fail")
    out = {"verdict_8m": verdict, "thresholds": {"r_min": PASS_R, "near_mean_abs_rel_err_max": PASS_NEAR_MAE,
                                                 "near_m": NEAR_M},
           "runs": results, "data_file": os.path.basename(data_file()),
           "generated": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(AIJ, "score.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(AIJ, "score_probes.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "probe", "x_m", "y_m", "dir_deg", "res_m", "mirror", "measured_vs_uh",
                    "simulated_vs_uh", "moved_off_building", "within_50m"])
        w.writerows(rows)
    log(f"  verdict at 8 m: {verdict}")
    figure(probes, results)
    return out


def figure(probes, results):
    """Simulated vs measured, one panel per direction; internal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), dpi=110)
    for ax, d in zip(axs, DIRS):
        meas = probes[:, 3 + DIRS.index(d)]
        for res, mirror, style in ((8.0, False, dict(color="#1f4e8c", marker="o", s=26, label="8 m")),
                                   (4.0, False, dict(color="#d08a00", marker="s", s=18, label="4 m")),
                                   (8.0, True, dict(color="#999999", marker="x", s=18, label="8 m, mirrored turn"))):
            name = run_name(d, res, mirror)
            if name not in results:
                continue
            sim, _, _ = sample(os.path.join(RUNS, name), probes)
            ax.scatter(meas, sim, **style)
            style["label"] = None
        ax.plot([0, 1.2], [0, 1.2], color="k", lw=0.8)
        ax.fill_between([0, 1.2], [0, 0.9], [0, 1.5], color="k", alpha=0.06, lw=0)
        txt = [f"{lab}: r {results[n]['r']:.2f}, near err {results[n]['near_mean_abs_rel_err']:.2f}"
               for lab, n in (("8 m", run_name(d, 8.0, False)), ("4 m", run_name(d, 4.0, False)),
                              ("mirror", run_name(d, 8.0, True))) if n in results]
        ax.text(0.03, 1.15, "\n".join(txt), va="top", fontsize=9)
        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, 1.2)
        ax.set_aspect("equal")
        ax.set_title(f"{d:g} degrees")
        ax.set_xlabel("wind tunnel, Vs / UH")
    axs[0].set_ylabel("model, street-level mean speed / UH")
    axs[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("AIJ Case D, 2 m probes: model vs wind tunnel (shaded: within 25%)")
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, f"aij_scatter_{datetime.now():%Y%m%d_%H%M%S}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


def check():
    """Draw the voxelised blocks and the probes for each direction at 8 m; internal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    inflow, probes = read_data()
    fig, axs = plt.subplots(2, 3, figsize=(16, 10), dpi=100)
    for col, d in enumerate(DIRS):
        for row, res in enumerate((8.0, 4.0)):
            bx = AijBox(d, res)
            _, bh, kfirst = voxelise(bx)
            ax = axs[row, col]
            ax.imshow(bh, origin="lower", cmap="Greys", vmin=0, vmax=110, interpolation="nearest")
            fi, fj = bx.from_probe(probes[:, 1], probes[:, 2])
            ok = np.isfinite(probes[:, 3])
            ax.scatter(fi[ok], fj[ok], s=8, c="#c0392b")
            ax.scatter(fi[~ok], fj[~ok], s=8, c="#2980b9")
            r = 380 / res
            ax.set_xlim(bx.ic - r, bx.ic + r)
            ax.set_ylim(bx.jc - r, bx.jc + r)
            ax.set_title(f"{d:g} deg, {res:g} m cells, tower top {(kfirst.max() - 1) * res:.0f} m", fontsize=10)
    prof, iu, uh = inlet(inflow, AijBox(0, 8.0))
    fig.suptitle(f"AIJ Case D voxels (wind left to right); red: measured probes, blue: not measured. "
                 f"UH {uh:.4f} lattice; Iu at 4 m {iu[1]:.2f}, at 100 m {iu[13]:.2f}")
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, f"aij_check_{datetime.now():%Y%m%d_%H%M%S}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("check", "run", "score"))
    ap.add_argument("--res", type=float, nargs="*", default=[8.0, 4.0])
    ap.add_argument("--dir", type=float, nargs="*", default=list(DIRS))
    ap.add_argument("--mirror", action="store_true", help="also run the opposite turn at 8 m (diagnostic)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    if args.what == "check":
        check()
    elif args.what == "run":
        for res in args.res:
            for d in args.dir:
                run_one(d, res, log=log, force=args.force)
        if args.mirror:
            for d in args.dir:
                if d:
                    run_one(d, 8.0, mirror=True, log=log, force=args.force)
        score(log)
    else:
        score(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
