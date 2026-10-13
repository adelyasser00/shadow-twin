"""
Run a nested 2 m wind box on a cloud GPU (Kaggle's free tier: one P100 or
two T4s, 16 GB each), for boxes too big for this laptop's 4 GB.

    TWIN_CITY=rotterdam .venv-heat/Scripts/python -I tools/nest_cloud.py pack --dir 247.5 --width 2220 --parent-tag _zh --sides parent --average 6
        builds every input here (city data stays here) and writes an upload
        folder: data/cities/<id>/wind/nest_cloud/upload/
    python run_cloud.py <upload folder> <output folder>           (on the cloud GPU)
    TWIN_CITY=rotterdam .venv-heat/Scripts/python -I tools/nest_cloud.py unpack <downloaded output folder>
        writes street.npz and run.json into the usual nest folders, then scores
    .venv-cuda/Scripts/python -I tools/nest_cloud.py selftest
        CUDA (CuPy) against OpenCL on a small box with gusts, facade friction
        and held sides

The solve is pipeline/nest_solve.py, the same code tools/wind_nest.py runs
here; only where it runs changes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

CODE = ["pipeline/__init__.py", "pipeline/wind_lbm.py", "pipeline/wind_lbm_cuda.py", "pipeline/nest_solve.py"]

RUN_CLOUD = r'''"""
Solve every pack in <upload folder> on this machine's GPU(s); results go to <output folder>.

    python run_cloud.py /kaggle/input/<dataset> /kaggle/working/out [--backend cuda|opencl|auto]

Two GPUs (Kaggle's T4 x2): two packs at a time, one per GPU. Needs numpy,
scipy and CuPy (pip install cupy-cuda12x) or pyopencl.
"""
import argparse, glob, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_pack(path):
    z = np.load(path, allow_pickle=False)
    p = {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
    p.setdefault("side_u", None)
    return p


def solve_one(pack_path, out_dir, backend):
    from pipeline.nest_solve import solve
    name = os.path.basename(os.path.dirname(pack_path))
    out = os.path.join(out_dir, name)
    if os.path.exists(os.path.join(out, "result.json")):
        print(f"{name}: already solved", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    print(f"{name}: loading", flush=True)
    res = solve(load_pack(pack_path), log=lambda m: print(f"[{time.time() - t0:7.1f}s] {name} {m}", flush=True),
                backend=backend)
    meta = res.pop("meta")
    np.savez_compressed(os.path.join(out, "result.npz"), **res)
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{name}: done in {time.time() - t0:.0f} s", flush=True)


def gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return max(1, len([l for l in out.splitlines() if l.startswith("GPU")]))
    except Exception:
        return 1


def ram_gb():
    try:
        with open("/proc/meminfo") as f:
            return int(f.readline().split()[1]) / 2**20
    except Exception:
        return 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--backend", default="auto", choices=("auto", "cuda", "opencl"))
    ap.add_argument("--one", action="store_true", help="src is one pack.npz (used for two GPUs)")
    a = ap.parse_args()
    if a.one:
        solve_one(a.src, a.out, a.backend)
        sys.exit(0)
    packs = sorted(glob.glob(os.path.join(a.src, "**", "pack.npz"), recursive=True))
    print(f"{len(packs)} packs, {gpus()} GPU(s), {ram_gb():.0f} GB RAM", flush=True)
    n = min(gpus(), len(packs)) if ram_gb() >= 24 else 1
    if n == 1:
        for p in packs:
            solve_one(p, a.out, a.backend)
    else:
        procs = []
        for i, p in enumerate(packs):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % n))
            procs.append(subprocess.Popen([sys.executable, __file__, p, a.out, "--one", "--backend", a.backend],
                                          env=env))
            if len(procs) == n:
                for q in procs:
                    q.wait()
                procs = []
        for q in procs:
            q.wait()
    print("all done", flush=True)
'''


NOTEBOOK_CELL = r'''# Nested 2 m wind box (tools/nest_cloud.py). Settings: Accelerator GPU P100 or GPU T4 x2; Internet on.
import glob, os, shutil, subprocess, sys, zipfile
hits = glob.glob("/kaggle/input/**/run_cloud.py", recursive=True)
if not hits:                                    # the upload stayed a zip: unpack it outside the output folder
    with zipfile.ZipFile(glob.glob("/kaggle/input/**/*.zip", recursive=True)[0]) as z:
        z.extractall("/tmp/upload")
    hits = glob.glob("/tmp/upload/**/run_cloud.py", recursive=True)
src = os.path.dirname(hits[0])
try:
    import cupy
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"], check=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv
!free -g | head -2
!python -u {src}/run_cloud.py {src} /kaggle/working/out --backend cuda
shutil.make_archive("/kaggle/working/nest_results", "zip", "/kaggle/working/out")
print("download /kaggle/working/nest_results.zip")
'''


def notebook(path):
    lines = NOTEBOOK_CELL.splitlines(keepends=True)
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1)


def cloud_dir(args):
    if args.root:
        return args.root
    from pipeline import city
    return os.path.join(city.wind_dir(), "nest_cloud")


def pack(args):
    import tools.wind_nest as N
    from pipeline import city
    root = cloud_dir(args)
    up = os.path.join(root, "upload")
    os.makedirs(up, exist_ok=True)
    for sc in args.scenario or [city.before(), "today"]:
        name = N.nest_name(sc, args.dir, True, args.width, args.average, args.sides, args.parent_tag, args.length,
                           args.height)
        local = os.path.join(root, "local", name)
        t0 = time.time()
        p, meta, extra = N.prepare(sc, args.dir, True, print, args.width, args.average, args.sides, args.parent_tag,
                                   args.length, args.height)
        os.makedirs(local, exist_ok=True)
        with open(os.path.join(local, "meta.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(meta, f, indent=1)
        np.savez_compressed(os.path.join(local, "extra.npz"), **extra,
                            kfirst=p["kfirst"], roof=p["roof"], prof=p["prof"], iu=p["iu"])
        arrays = {k: (np.asarray(v) if not isinstance(v, np.ndarray) else v) for k, v in p.items() if v is not None}
        os.makedirs(os.path.join(up, name), exist_ok=True)
        np.savez_compressed(os.path.join(up, name, "pack.npz"), **arrays)
        nz, ny, nx = p["solid"].shape
        print(f"{name}: {nx} x {ny} x {nz} = {nx * ny * nz / 1e6:.1f} M cells, about "
              f"{nx * ny * nz * 113 / 2**30:.1f} GB on the GPU; packed in {time.time() - t0:.0f} s, "
              f"{os.path.getsize(os.path.join(up, name, 'pack.npz')) / 2**20:.0f} MB", flush=True)
    for c in CODE:
        dst = os.path.join(up, c)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(HERE, c), dst)
    with open(os.path.join(up, "run_cloud.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(RUN_CLOUD)
    notebook(os.path.join(root, "kaggle_nest.ipynb"))
    shutil.make_archive(os.path.join(root, "nest_upload"), "zip", up)
    print(f"upload folder: {up}\nzipped: {os.path.join(root, 'nest_upload.zip')}\n"
          f"notebook: {os.path.join(root, 'kaggle_nest.ipynb')}")


def unpack(args):
    import tools.wind_nest as N
    root = cloud_dir(args)
    done = []
    for rj in sorted(glob.glob(os.path.join(args.results, "**", "result.json"), recursive=True)):
        rdir = os.path.dirname(rj)
        name = os.path.basename(rdir)
        local = os.path.join(root, "local", name)
        with open(os.path.join(local, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        ex = np.load(os.path.join(local, "extra.npz"))
        z = np.load(os.path.join(rdir, "result.npz"))
        with open(rj, encoding="utf-8") as f:
            rmeta = json.load(f)
        res = {k: z[k] for k in z.files}
        res["meta"] = rmeta
        pk = {"kfirst": ex["kfirst"], "roof": ex["roof"], "prof": ex["prof"], "iu": ex["iu"]}
        extra = {"building": ex["building"], "ground": ex["ground"]}
        out = os.path.join(args.out or N.OUT, name)
        if os.path.exists(os.path.join(out, "run.json")):
            print(f"{name}: already unpacked")
        else:
            N.finish(out, pk, meta, extra, res, time.time() - rmeta.get("solve_s", 0))
        done.append(meta)
    if args.out:
        return
    for key in sorted({(m["theta_from_deg"], m["box_size"]["width_m"], m["box_size"]["length_m"],
                        m["box_size"]["height_m"], m["average_flow_throughs"], m["sides"], m["parent"].split("8m", 1)[-1])
                       for m in done}):
        theta, width, length, height, average, sides, tag = key
        sides = "periodic" if sides == "periodic" else "parent"
        sp = N.score_path(theta, width, average, sides, tag, length, height)
        if os.path.exists(sp):
            print(f"  {os.path.basename(sp)}: already scored")
            continue
        try:
            N.score(theta, width=width, average=average, sides=sides, parent_tag=tag, length=length, height=height)
        except (OSError, SystemExit) as e:
            print(f"  not scored yet ({e}): both years are needed")


def selftest(args):
    from pipeline.wind_lbm import InflowTurbulence, Solver, power_profile
    from pipeline.wind_lbm_cuda import CudaSolver
    nz, ny, nx = 24, 30, 48
    solid = np.zeros((nz, ny, nx), bool)
    solid[0] = True
    solid[1:12, 12:17, 18:24] = True
    prof = power_profile(nz, 1, 0.06, nz - 1, 0.28)
    prof[:1] = 0
    sides = np.zeros((3, nz, 2, nx), np.float32)
    sides[0] = prof[:, None, None]
    # The fields are stored in half precision, so a one-bit rounding difference
    # anywhere grows to about 0.2% of the largest speed within a few hundred
    # steps. Two codes doing the same arithmetic in a different order can't be
    # bitwise equal; the bar is that CUDA strays from OpenCL no further than
    # OpenCL strays from itself when only its compiler's math flag changes.
    def run(S, steps, gusts, fast=True):
        t = InflowTurbulence(prof, 1, 2.0, ny) if gusts else None
        s = S(solid, prof, res_m=2.0, turbulence=t, facade_drag=True, sides=sides, log=lambda m: None,
              fast_math=fast)
        s.run(steps, accumulate_every=2)
        return s.velocity(), s.sums()[0][4]

    def gap(a, b):
        return (float(np.abs(a[0] - b[0]).max() / np.abs(a[0]).max()),
                float(np.abs(a[1] - b[1]).max() / np.abs(a[1]).max()))

    ok = bool(np.array_equal(run(Solver, 1, True)[0], run(CudaSolver, 1, True)[0]))
    print(f"after 1 step, CUDA and OpenCL identical: {ok}")
    for gusts in (False, True):
        ref = run(Solver, 300, gusts)
        cu = gap(ref, run(CudaSolver, 300, gusts))
        own = gap(ref, run(Solver, 300, gusts, fast=False))
        good = cu[0] <= 1.5 * own[0] and cu[1] <= 1.5 * own[1]
        ok &= good
        print(f"300 steps {'with gusts' if gusts else 'steady'}: CUDA vs OpenCL {cu[0] * 100:.3f}% of the largest "
              f"speed ({cu[1] * 100:.3f}% in the summed speed); OpenCL vs itself without fast math "
              f"{own[0] * 100:.3f}% ({own[1] * 100:.3f}%): {'PASS' if good else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    a = sub.add_parser("pack")
    a.add_argument("--scenario", nargs="*")
    a.add_argument("--dir", type=float, required=True)
    a.add_argument("--width", type=float, default=480.0)
    a.add_argument("--length", type=float, default=600.0)
    a.add_argument("--height", type=float, default=600.0)
    a.add_argument("--average", type=float, default=6.0)
    a.add_argument("--sides", choices=("periodic", "parent"), default="parent")
    a.add_argument("--parent-tag", default="")
    a.add_argument("--root", help="instead of data/cities/<id>/wind/nest_cloud (for checks)")
    b = sub.add_parser("unpack")
    b.add_argument("results")
    b.add_argument("--root", help="as given to pack")
    b.add_argument("--out", help="write here instead of the nest folder, and don't score (for checks)")
    sub.add_parser("selftest")
    args = ap.parse_args()
    return {"pack": pack, "unpack": unpack, "selftest": selftest}[args.what](args)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
