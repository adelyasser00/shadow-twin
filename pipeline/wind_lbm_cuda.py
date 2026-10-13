"""
The wind solver on CUDA, through CuPy: the same kernel as pipeline/wind_lbm.py
for machines with an NVIDIA GPU and no OpenCL (cloud notebooks such as
Kaggle's), and no 1 GB cap per buffer (NVIDIA's OpenCL caps one buffer at a
quarter of the card, which limits a 2 m box to about 27 M cells on 4 GB).

    from pipeline.wind_lbm_cuda import CudaSolver
    s = CudaSolver(solid, profile, ...)       # same arguments as wind_lbm.Solver

The kernel text is wind_lbm.KERNEL, translated mechanically: OpenCL's half
loads and stores become cuda_fp16 conversions, the global id becomes the
block and thread index, and indices are 64-bit. Host preparation (flags,
drag per cell, side faces) is wind_lbm.host_arrays, shared with the OpenCL
solver. The two are checked against each other on a small box
(tools/nest_cloud.py selftest).
"""

from __future__ import annotations

import re
import time

import numpy as np

from .wind_lbm import (C, CS_SMAGORINSKY, IDX, KERNEL, MZ, NACC, OPP, SOLID, W, _csv, host_arrays)

PREAMBLE = r"""
#include <cuda_fp16.h>
typedef unsigned char uchar;
__device__ inline float mix(float a, float b, float t) { return a + (b - a) * t; }
"""


def cuda_source(**fmt):
    src = KERNEL % fmt
    src = src.replace("__constant int", "__constant__ int").replace("__constant float", "__constant__ float")
    src = src.replace("inline float load_f(__global const half* f, int i, long n) {\n"
                      "    return vload_half(i * N + n, f) + WT[i];\n}",
                      "__device__ inline float load_f(const __half* f, int i, long n) {\n"
                      "    return __half2float(f[i * N + n]) + WT[i];\n}")
    src = src.replace("inline void store_f(__global half* f, int i, long n, float v) {\n"
                      "    vstore_half_rte(v - WT[i], i * N + n, f);\n}",
                      "__device__ inline void store_f(__half* f, int i, long n, float v) {\n"
                      "    f[i * N + n] = __float2half_rn(v - WT[i]);\n}")
    src = src.replace("inline float feq(", "__device__ inline float feq(")
    src = src.replace("__kernel void", 'extern "C" __global__ void')
    src = src.replace("__global const half*", "const __half*").replace("__global half*", "__half*")
    src = src.replace("__global const ", "const ").replace("__global ", "")
    src = src.replace("get_global_id(0)", "((long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x)")
    if "vload_half" in src or "vstore_half" in src or re.search(r"__global\b(?!__)", src):
        raise RuntimeError("OpenCL left in the CUDA source")
    src = re.sub(r"\blong\b", "long long", src)
    src = re.sub(r"(\d+)L\b", r"\1LL", src)
    return PREAMBLE + src


class CudaSolver:
    """wind_lbm.Solver's interface on CUDA (CuPy)."""

    def __init__(self, solid, profile, veg=None, tau0=0.5005, smag=CS_SMAGORINSKY,
                 slip=True, z0_m=0.1, res_m=8.0, turbulence=None, device=None, log=print,
                 facade_drag=False, sides=None, fast_math=True):
        import cupy as cp
        self.cp = cp
        h = host_arrays(solid, profile, veg, slip, z0_m, res_m, facade_drag, sides)
        flag, prof, veg, sides = h["flag"], h["prof"], h["veg"], h["sides"]
        nz, ny, nx = self.shape = h["shape"]
        self.n = n = h["n"]
        self.log = log
        self.flag = flag
        self.cd_wall = h["cd_wall"]
        self.facade_drag = h["facade_drag"]
        self.sides = sides is not None
        if device is not None:
            cp.cuda.Device(device).use()
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        self.device_name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        src = cuda_source(nx=nx, ny=ny, nz=nz, n=n, tau0=tau0, smag=smag, slip=int(bool(slip)),
                          sides=int(sides is not None), idx=_csv(IDX),
                          cx=_csv(C[:, 0]), cy=_csv(C[:, 1]), cz=_csv(C[:, 2]),
                          w=_csv(W, "{:.9f}f"), opp=_csv(OPP), mz=_csv(MZ))
        self.module = cp.RawModule(code=src, options=("--use_fast_math", "-std=c++14") if fast_math else ("-std=c++14",))
        self.k_step = self.module.get_function("step")
        self.k_init = self.module.get_function("init")
        self.fa = cp.empty(19 * n, dtype=cp.float16)
        self.fb = cp.empty(19 * n, dtype=cp.float16)
        self.d_flag = cp.asarray(flag.ravel())
        self.d_veg = cp.asarray(veg.ravel())
        self.d_prof = cp.asarray(prof)
        self.d_u = cp.zeros(3 * n, dtype=cp.float32)
        self.d_acc = cp.zeros(NACC * n, dtype=cp.float32)
        self.turb = turbulence
        self.d_ta = cp.zeros(3 * ny * nz, dtype=cp.float32)
        self.d_tb = cp.zeros(3 * ny * nz, dtype=cp.float32)
        self.d_side = cp.asarray(sides.ravel() if sides is not None else np.zeros(1, np.float32))
        self._slice = None
        self.block = (128,)
        self.grid = (int(-(-n // 128)),)
        self.k_init(self.grid, self.block, (self.fa, self.d_u, self.d_flag, self.d_prof))
        cp.cuda.Device().synchronize()
        self.steps_done = 0
        self.samples = 0
        self.mem_mb = (2 * 19 * 2 * n + (3 + NACC) * 4 * n + 5 * n) / 2**20
        log(f"  lbm (cuda): {nx} x {ny} x {nz} = {n / 1e6:.1f} M cells on {self.device_name}, "
            f"{self.mem_mb:.0f} MB, tau0 {tau0}, Cs {smag}, "
            + (f"slip walls, wall drag {self.cd_wall:.4f} (z0 {z0_m} m)" if slip else "no-slip walls")
            + (", on facades too" if self.facade_drag else "")
            + (", sides held at a coarser run's wind" if self.sides else ""))

    def run(self, steps, accumulate_every=0, log_every=0):
        cp = self.cp
        t = time.time()
        for s in range(steps):
            acc = 1 if accumulate_every and (s % accumulate_every == 0) else 0
            tw = self._inflow(self.steps_done + s)
            self.k_step(self.grid, self.block,
                        (self.fa, self.fb, self.d_flag, self.d_veg, self.d_u, self.d_prof, self.d_acc,
                         np.int32(acc), self.d_ta, self.d_tb, np.float32(tw), self.d_side))
            self.fa, self.fb = self.fb, self.fa
            self.samples += acc
            if log_every and (s + 1) % log_every == 0:
                cp.cuda.Device().synchronize()
                el = time.time() - t
                self.log(f"    step {self.steps_done + s + 1:,}: "
                         f"{(s + 1) * self.n / el / 1e6:.0f} MLUPs, max |u| {self.max_speed():.3f}")
        cp.cuda.Device().synchronize()
        self.steps_done += steps
        return steps * self.n / max(time.time() - t, 1e-9) / 1e6

    def _inflow(self, step):
        if self.turb is None:
            return 0.0
        pos = step / self.turb.steps_per_slice
        k = int(pos)
        if k != self._slice:
            if self._slice is not None and k == self._slice + 1:
                self.d_ta, self.d_tb = self.d_tb, self.d_ta
                self.d_tb.set(self.turb.slice(k + 1))
            else:
                self.d_ta.set(self.turb.slice(k))
                self.d_tb.set(self.turb.slice(k + 1))
            self._slice = k
        return pos - k

    def velocity(self):
        return self.d_u.get().reshape(3, *self.shape)

    def max_speed(self):
        u = self.velocity()
        s = np.sqrt((u ** 2).sum(axis=0))
        s[self.flag == SOLID] = 0
        if not np.isfinite(s).all():
            return float("nan")
        return float(s.max())

    def sums(self):
        return self.d_acc.get().reshape(NACC, *self.shape), self.samples

    def mean_velocity(self):
        s, k = self.sums()
        s /= max(1, k)
        return s[:3], s[3], s[4]
