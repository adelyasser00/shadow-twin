"""
Check the GPU path against the CPU reference.

    python -m pipeline.verify_torch

Run this once on Kaggle before trusting a GPU render. If the two backends
disagree, the CPU one is right, because it is the one with analytical tests.
"""
import numpy as np

from .geometry import cast_shadow, sky_view_factor
from .solar import sun_position
from datetime import datetime, timezone


def main():
    try:
        import torch
    except ImportError:
        print("torch is not installed, nothing to compare")
        return 0
    from .geometry_torch import cast_shadow_gpu, sky_view_factor_gpu, device_name

    print("comparing CPU and torch backends on " + device_name())
    rng = np.random.default_rng(11)
    dsm = np.zeros((220, 220), dtype=np.float32)
    for _ in range(40):
        r, c = rng.integers(10, 200, 2)
        h, w = rng.integers(8, 30, 2)
        dsm[r:r + h, c:c + w] = rng.uniform(10, 180)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.from_numpy(dsm).to(dev)
    ok = True

    for alt, az in [(12.0, 140.0), (35.0, 200.0), (62.0, 265.0), (25.8, 180.0)]:
        a = cast_shadow(dsm, 2.0, alt, az)
        b = cast_shadow_gpu(t, 2.0, alt, az).cpu().numpy()
        diff = int((a != b).sum())
        good = diff == 0
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  shadow alt {alt:5.1f} az {az:6.1f}  "
              f"{diff} cells differ of {a.size}")

    a = sky_view_factor(dsm, 2.0, 16, 200)
    b = sky_view_factor_gpu(t, 2.0, 16, 200).cpu().numpy()
    err = float(np.abs(a - b).max())
    good = err < 1e-5
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  sky view factor  max difference {err:.2e}")

    print("BACKENDS AGREE" if ok else "BACKENDS DISAGREE, USE THE CPU PATH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
