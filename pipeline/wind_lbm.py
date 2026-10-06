"""
A 3D wind solver for the GPU: lattice Boltzmann, D3Q19, with a Smagorinsky
large eddy model and regularised collisions. Runs through OpenCL, so any GPU
with a current driver works (NVIDIA, AMD, Intel), and the kernel is compiled
by the driver itself.

    from pipeline.wind_lbm import Solver
    s = Solver(solid, veg=None, profile=u_of_z)
    s.run(steps, average_from=...)
    mean_u = s.mean_velocity()            # (3, nz, ny, nx), lattice units

Why lattice Boltzmann
---------------------
Wind between buildings is a large eddy problem: separation at sharp corners,
wakes and downwash at the foot of towers. Lattice Boltzmann solves it on a
plain voxel grid, which a city made of rasters already is, and it is one of
the fastest ways to run large eddy simulation on a GPU. Commercial urban wind
tools (SimScale's pedestrian wind comfort, XFlow, PowerFLOW) use the same
family of methods, validated on the AIJ wind tunnel benchmarks.

Set-up, like a wind tunnel with a turntable
-------------------------------------------
The flow always enters at x = 0 and leaves at x = nx - 1. The city is turned
under it for each wind direction (pipeline/wind_domain.py), so there is one
inlet face and one outlet face.

  inlet     equilibrium at the prescribed profile u(z), density 1
  outlet    equilibrium at density 1 and the velocity one cell upstream
  sides     periodic
  top       free slip (mirror)
  ground    solid, halfway bounce-back, as are buildings
  trees     a drag force per cell, F = -c |u| u, applied implicitly with the
            exact difference method (Kupershtokh), so dense canopy cannot
            reverse the flow

Distributions are stored as 16-bit floats, shifted by their lattice weights
(FP16S in Lehmann 2022, "Accuracy and performance of the lattice Boltzmann
method with 64-bit, 32-bit, and customized 16-bit number formats"), which
halves memory and bandwidth. Arithmetic is 32-bit.

Units: lattice. Velocities are fractions of one cell per step, so a field is
only ever read as a ratio to a reference speed. That is how wind tunnel and
pedestrian wind studies report it too: at these Reynolds numbers the flow
around sharp-edged buildings scales with the wind speed.
"""

from __future__ import annotations

import os
import time

import numpy as np

# D3Q19, FluidX3D order: rest, 6 faces, 12 edges; i and i+1 are opposites.
C = np.array([[0, 0, 0],
              [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, -1], [0, 1, 1], [0, -1, -1],
              [1, -1, 0], [-1, 1, 0], [1, 0, -1], [-1, 0, 1], [0, 1, -1], [0, -1, 1]], np.int32)
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, np.float64)
OPP = np.array([0] + [i + 1 if i % 2 else i - 1 for i in range(1, 19)], np.int32)
# Mirror in z for the free-slip top: same cx, cy, opposite cz.
MZ = np.array([int(np.nonzero((C[:, 0] == c[0]) & (C[:, 1] == c[1]) & (C[:, 2] == -c[2]))[0][0])
               for c in C], np.int32)

FLUID, SOLID, INLET, OUTLET = 0, 1, 2, 3
NACC = 5                        # summed: u, v, w, |u|^2, |u|
CS_SMAGORINSKY = 0.17


KERNEL = r"""
#define NX %(nx)d
#define NY %(ny)d
#define NZ %(nz)d
#define N  %(n)dL
#define TAU0 %(tau0)ff
#define SMAG %(smag)ff

__constant int CX[19] = {%(cx)s};
__constant int CY[19] = {%(cy)s};
__constant int CZ[19] = {%(cz)s};
__constant float WT[19] = {%(w)s};
__constant int OP[19] = {%(opp)s};
__constant int MZ[19] = {%(mz)s};

inline float load_f(__global const half* f, int i, long n) {
    return vload_half(i * N + n, f) + WT[i];
}
inline void store_f(__global half* f, int i, long n, float v) {
    vstore_half_rte(v - WT[i], i * N + n, f);
}
inline float feq(int i, float rho, float ux, float uy, float uz) {
    const float cu = CX[i] * ux + CY[i] * uy + CZ[i] * uz;
    const float uu = ux * ux + uy * uy + uz * uz;
    return WT[i] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * uu);
}

__kernel void init(__global half* f, __global float* u, __global const uchar* flag,
                   __global const float* prof) {
    const long n = get_global_id(0);
    if (n >= N) return;
    const int z = (int)(n / (NX * NY));
    const float ux = flag[n] == 1 ? 0.0f : prof[z];
    for (int i = 0; i < 19; i++) store_f(f, i, n, feq(i, 1.0f, ux, 0.0f, 0.0f));
    u[n] = ux; u[N + n] = 0.0f; u[2 * N + n] = 0.0f;
}

__kernel void step(__global const half* fa, __global half* fb,
                   __global const uchar* flag, __global const float* veg,
                   __global float* u, __global const float* prof,
                   __global float* acc, const int accumulate) {
    const long n = get_global_id(0);
    if (n >= N) return;
    const uchar fl = flag[n];
    if (fl == 1) return;
    const int x = (int)(n %% NX);
    const int y = (int)((n / NX) %% NY);
    const int z = (int)(n / (NX * NY));
    float rho, ux, uy, uz, f[19];

    if (fl == 2 || fl == 3) {
        rho = 1.0f;
        if (fl == 2) { ux = prof[z]; uy = 0.0f; uz = 0.0f; }
        else {
            const long m = n - 1;
            ux = fmax(u[m], 0.0f); uy = u[N + m]; uz = u[2 * N + m];
        }
        for (int i = 0; i < 19; i++) store_f(fb, i, n, feq(i, rho, ux, uy, uz));
        u[n] = ux; u[N + n] = uy; u[2 * N + n] = uz;
        return;
    }

    // Pull streaming with halfway bounce-back and a mirror at the top.
    for (int i = 0; i < 19; i++) {
        int xs = x - CX[i];
        int ys = y - CY[i]; ys = ys < 0 ? ys + NY : (ys >= NY ? ys - NY : ys);
        int zs = z - CZ[i];
        if (zs >= NZ) {                       // from above the lid: mirror
            const long m = (long)xs + NX * ((long)ys + NY * (long)z);
            f[i] = flag[m] == 1 ? load_f(fa, OP[i], n) : load_f(fa, MZ[i], m);
            continue;
        }
        const long m = (long)xs + NX * ((long)ys + NY * (long)zs);
        f[i] = flag[m] == 1 ? load_f(fa, OP[i], n) : load_f(fa, i, m);
    }

    rho = 0.0f; ux = 0.0f; uy = 0.0f; uz = 0.0f;
    for (int i = 0; i < 19; i++) {
        rho += f[i]; ux += CX[i] * f[i]; uy += CY[i] * f[i]; uz += CZ[i] * f[i];
    }
    ux /= rho; uy /= rho; uz /= rho;

    // Tree drag, implicit: u' = u / (1 + c |u|).
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    const float c = veg[n];
    if (c > 0.0f) {
        const float s = 1.0f / (1.0f + c * sqrt(ux * ux + uy * uy + uz * uz));
        dx = ux * (s - 1.0f); dy = uy * (s - 1.0f); dz = uz * (s - 1.0f);
    }

    // Non-equilibrium stress, Smagorinsky relaxation time, regularised
    // collision (projected onto the second-order Hermite basis).
    float pxx = 0.0f, pyy = 0.0f, pzz = 0.0f, pxy = 0.0f, pxz = 0.0f, pyz = 0.0f;
    for (int i = 0; i < 19; i++) {
        const float fn = f[i] - feq(i, rho, ux, uy, uz);
        pxx += CX[i] * CX[i] * fn; pyy += CY[i] * CY[i] * fn; pzz += CZ[i] * CZ[i] * fn;
        pxy += CX[i] * CY[i] * fn; pxz += CX[i] * CZ[i] * fn; pyz += CY[i] * CZ[i] * fn;
    }
    const float q = pxx * pxx + pyy * pyy + pzz * pzz + 2.0f * (pxy * pxy + pxz * pxz + pyz * pyz);
    const float tau = 0.5f * (TAU0 + sqrt(TAU0 * TAU0 + 18.0f * 1.41421356f * SMAG * SMAG * sqrt(q) / rho));
    const float k = 1.0f - 1.0f / tau;
    const float vx = ux + dx, vy = uy + dy, vz = uz + dz;
    for (int i = 0; i < 19; i++) {
        const float cx = CX[i], cy = CY[i], cz = CZ[i];
        const float qp = (cx * cx - 1.0f / 3.0f) * pxx + (cy * cy - 1.0f / 3.0f) * pyy
                       + (cz * cz - 1.0f / 3.0f) * pzz
                       + 2.0f * (cx * cy * pxy + cx * cz * pxz + cy * cz * pyz);
        store_f(fb, i, n, feq(i, rho, vx, vy, vz) + k * 4.5f * WT[i] * qp);
    }
    // Physical velocity with a body force: halfway through the step.
    ux += 0.5f * dx; uy += 0.5f * dy; uz += 0.5f * dz;
    u[n] = ux; u[N + n] = uy; u[2 * N + n] = uz;
    if (accumulate) {
        acc[n] += ux; acc[N + n] += uy; acc[2 * N + n] += uz;
        const float uu = ux * ux + uy * uy + uz * uz;
        acc[3 * N + n] += uu;
        acc[4 * N + n] += sqrt(uu);
    }
}
"""


def _csv(a, fmt="{}"):
    return ", ".join(fmt.format(v) for v in a)


class Solver:
    """
    solid    bool (nz, ny, nx); z = 0 must be solid everywhere (the ground)
    veg      float (nz, ny, nx) drag coefficient per cell in lattice units, or None
    profile  float (nz,) inlet speed along +x at each height, lattice units
    tau0     molecular relaxation time; the Smagorinsky model adds to it
    """

    def __init__(self, solid, profile, veg=None, tau0=0.5005, smag=CS_SMAGORINSKY,
                 device=None, log=print):
        import pyopencl as cl
        solid = np.ascontiguousarray(solid, dtype=bool)
        nz, ny, nx = solid.shape
        if not solid[0].all():
            raise ValueError("the bottom layer must be solid ground")
        self.shape = (nz, ny, nx)
        self.n = n = nx * ny * nz
        self.log = log
        flag = np.where(solid, SOLID, FLUID).astype(np.uint8)
        flag[1:, :, 0][~solid[1:, :, 0]] = INLET
        flag[1:, :, -1][~solid[1:, :, -1]] = OUTLET
        self.flag = flag
        prof = np.asarray(profile, np.float32)
        if prof.shape != (nz,):
            raise ValueError("profile must have one speed per layer")
        if veg is None:
            veg = np.zeros(solid.shape, np.float32)
        veg = np.where(solid, 0.0, veg).astype(np.float32)

        self.ctx = cl.Context([device or pick_device()])
        self.queue = cl.CommandQueue(self.ctx)
        dev = self.ctx.devices[0]
        self.device_name = dev.name.strip()
        mf = cl.mem_flags
        lattice_bytes = 19 * 2 * n
        if lattice_bytes > dev.max_mem_alloc_size:
            raise MemoryError(f"{n:,} cells need {lattice_bytes / 2**20:.0f} MB per lattice, "
                              f"the device allows {dev.max_mem_alloc_size / 2**20:.0f} MB per buffer")
        src = KERNEL % dict(nx=nx, ny=ny, nz=nz, n=n, tau0=tau0, smag=smag,
                            cx=_csv(C[:, 0]), cy=_csv(C[:, 1]), cz=_csv(C[:, 2]),
                            w=_csv(W, "{:.9f}f"), opp=_csv(OPP), mz=_csv(MZ))
        self.prg = cl.Program(self.ctx, src).build(options=["-cl-fast-relaxed-math"])
        self.k_step = cl.Kernel(self.prg, "step")
        self.fa = cl.Buffer(self.ctx, mf.READ_WRITE, lattice_bytes)
        self.fb = cl.Buffer(self.ctx, mf.READ_WRITE, lattice_bytes)
        self.d_flag = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flag.ravel())
        self.d_veg = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=veg.ravel())
        self.d_prof = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=prof)
        self.d_u = cl.Buffer(self.ctx, mf.READ_WRITE, 3 * 4 * n)
        self.d_acc = cl.Buffer(self.ctx, mf.READ_WRITE, NACC * 4 * n)
        cl.enqueue_fill_buffer(self.queue, self.d_acc, np.float32(0), 0, NACC * 4 * n)
        self.gsize = (int(-(-n // 128) * 128),)
        self.lsize = (128,)
        self.prg.init(self.queue, self.gsize, self.lsize, self.fa, self.d_u, self.d_flag, self.d_prof)
        self.queue.finish()
        self.steps_done = 0
        self.samples = 0
        self.mem_mb = (2 * lattice_bytes + (3 + NACC) * 4 * n + 5 * n) / 2**20
        log(f"  lbm: {nx} x {ny} x {nz} = {n / 1e6:.1f} M cells on {self.device_name}, "
            f"{self.mem_mb:.0f} MB, tau0 {tau0}, Cs {smag}")

    def run(self, steps, accumulate_every=0, log_every=0):
        """Advance `steps` steps; sum velocity every `accumulate_every` steps (0: never)."""
        t = time.time()
        k = self.k_step
        for s in range(steps):
            acc = 1 if accumulate_every and (s % accumulate_every == 0) else 0
            k(self.queue, self.gsize, self.lsize, self.fa, self.fb, self.d_flag, self.d_veg,
              self.d_u, self.d_prof, self.d_acc, np.int32(acc))
            self.fa, self.fb = self.fb, self.fa
            self.samples += acc
            if log_every and (s + 1) % log_every == 0:
                self.queue.finish()
                el = time.time() - t
                self.log(f"    step {self.steps_done + s + 1:,}: "
                         f"{(s + 1) * self.n / el / 1e6:.0f} MLUPs, max |u| {self.max_speed():.3f}")
        self.queue.finish()
        self.steps_done += steps
        el = time.time() - t
        return steps * self.n / max(el, 1e-9) / 1e6

    def velocity(self):
        import pyopencl as cl
        out = np.empty(3 * self.n, np.float32)
        cl.enqueue_copy(self.queue, out, self.d_u)
        return out.reshape(3, *self.shape)

    def max_speed(self):
        u = self.velocity()
        s = np.sqrt((u ** 2).sum(axis=0))
        s[self.flag == SOLID] = 0
        if not np.isfinite(s).all():
            return float("nan")
        return float(s.max())

    def sums(self):
        """Raw sums (NACC, nz, ny, nx): u, v, w, |u|^2, |u|, and the sample count."""
        import pyopencl as cl
        out = np.empty(NACC * self.n, np.float32)
        cl.enqueue_copy(self.queue, out, self.d_acc)
        return out.reshape(NACC, *self.shape), self.samples

    def mean_velocity(self):
        """Time means over the samples: velocity (3, ...), |u|^2 and |u| (nz, ny, nx)."""
        s, k = self.sums()
        s /= max(1, k)
        return s[:3], s[3], s[4]


def pick_device():
    """The GPU with the most memory, else whatever OpenCL offers."""
    import pyopencl as cl
    want = os.environ.get("WIND_CL_DEVICE")
    devs = [d for p in cl.get_platforms() for d in p.get_devices()]
    if want:
        devs = [d for d in devs if want.lower() in d.name.lower()] or devs
    gpus = [d for d in devs if d.type & cl.device_type.GPU]
    pool = gpus or devs
    # Prefer a discrete card: integrated GPUs share system memory and are slower.
    pool.sort(key=lambda d: (("nvidia" in d.vendor.lower() or "amd" in d.vendor.lower()
                              or "advanced micro" in d.vendor.lower()), d.global_mem_size),
              reverse=True)
    return pool[0]


def power_profile(nz, ground_k, u_ref, z_ref_cells, alpha):
    """Inlet speed per layer: u_ref (h / z_ref)^alpha, h measured from ground_k."""
    k = np.arange(nz, dtype=np.float64)
    h = np.maximum(k - ground_k + 0.5, 0.5)
    return (u_ref * (h / z_ref_cells) ** alpha).astype(np.float32)
