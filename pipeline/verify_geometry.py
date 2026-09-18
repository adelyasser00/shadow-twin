"""
Self-tests for geometry.py against cases you can work out on paper.

Run:  python -m pipeline.verify_geometry
"""

import math

import numpy as np

from .geometry import cast_shadow, sky_view_factor

CELL = 2.0  # metres


def report(label, expected, got, tol, unit=""):
    err = abs(expected - got)
    ok = err <= tol
    print(
        f"{'PASS' if ok else 'FAIL'}  {label:<52} "
        f"expected {expected:8.3f}  got {got:8.3f}  err {err:6.3f} {unit}"
    )
    return ok


def flat_ground(n=120):
    return np.zeros((n, n), dtype=np.float32)


def with_tower(n=200, height=30.0, size_cells=10):
    dsm = np.zeros((n, n), dtype=np.float32)
    c = n // 2
    h = size_cells // 2
    dsm[c - h : c + h, c - h : c + h] = height
    return dsm, c, h


def main():
    ok = True
    print("geometry self-test\n" + "-" * 92)

    # 1. Flat ground sees the entire sky and never shades itself.
    flat = flat_ground()
    svf_flat = sky_view_factor(flat, CELL, n_azimuths=16, max_distance_m=100)
    ok &= report("SVF of flat ground (interior)", 1.0, float(svf_flat[20:-20, 20:-20].mean()), 1e-6)

    lit = cast_shadow(flat, CELL, altitude_deg=45.0, azimuth_deg=180.0)
    ok &= report("sunlit fraction of flat ground", 1.0, float(lit.mean()), 1e-6)

    lit_night = cast_shadow(flat, CELL, altitude_deg=-5.0, azimuth_deg=270.0)
    ok &= report("sunlit fraction below the horizon", 0.0, float(lit_night.mean()), 1e-6)

    print()

    # 2. Shadow length from a tower. A tower of height H under a sun at
    #    altitude a throws a shadow H / tan(a) long, pointing away from the sun.
    #    Sun due south (azimuth 180) means the shadow runs due north, which is
    #    toward decreasing row index.
    for height, alt in [(30.0, 45.0), (30.0, 30.0), (60.0, 60.0)]:
        dsm, c, h = with_tower(height=height)
        lit = cast_shadow(dsm, CELL, altitude_deg=alt, azimuth_deg=180.0)
        col = c  # centre column, runs through the tower
        north_edge = c - h  # first ground row north of the tower
        run = 0
        r = north_edge - 1
        while r >= 0 and lit[r, col] == 0.0:
            run += 1
            r -= 1
        measured_m = run * CELL
        expected_m = height / math.tan(math.radians(alt))
        # Whole-cell ray stepping places the shadow tip to the nearest cell,
        # so one cell of slack is the correct tolerance, not a fudge. UMEP
        # interpolates sub-pixel and would land closer. At 2 m cells this is
        # a 2 m uncertainty on a 30 m shadow, well inside the error already
        # carried by model-estimated building heights.
        ok &= report(
            f"shadow length, H={height:.0f} m, sun {alt:.0f} deg",
            expected_m,
            measured_m,
            CELL * 1.001,
            "m",
        )

    print()

    # 3. Shadow direction follows the sun. Sun in the east throws the shadow
    #    west, sun in the west throws it east.
    dsm, c, h = with_tower(height=30.0)
    lit_e = cast_shadow(dsm, CELL, altitude_deg=30.0, azimuth_deg=90.0)
    lit_w = cast_shadow(dsm, CELL, altitude_deg=30.0, azimuth_deg=270.0)
    shade_west_of_tower = (lit_e[c, : c - h] == 0).sum()
    shade_east_of_tower = (lit_e[c, c + h :] == 0).sum()
    east_ok = shade_west_of_tower > 10 and shade_east_of_tower == 0
    print(
        f"{'PASS' if east_ok else 'FAIL'}  sun in the east throws the shadow west"
        f"            west cells {shade_west_of_tower}, east cells {shade_east_of_tower}"
    )
    ok &= east_ok
    shade_east2 = (lit_w[c, c + h :] == 0).sum()
    shade_west2 = (lit_w[c, : c - h] == 0).sum()
    west_ok = shade_east2 > 10 and shade_west2 == 0
    print(
        f"{'PASS' if west_ok else 'FAIL'}  sun in the west throws the shadow east"
        f"            east cells {shade_east2}, west cells {shade_west2}"
    )
    ok &= west_ok

    # Overhead sun shades nothing but the footprint footprint edges.
    lit_noon = cast_shadow(dsm, CELL, altitude_deg=89.9, azimuth_deg=180.0)
    ok &= report(
        "sunlit fraction with the sun overhead", 1.0, float(lit_noon.mean()), 0.002
    )

    print()

    # 4. SVF at the foot of a wall. For a pixel hard against an infinitely
    #    long wall of height H, exactly half the compass is blocked at a
    #    horizon angle approaching 90 degrees, so SVF approaches 0.5.
    n = 160
    walled = np.zeros((n, n), dtype=np.float32)
    walled[:, n // 2 :] = 400.0  # a very tall half-plane to the east
    svf_w = sky_view_factor(walled, CELL, n_azimuths=32, max_distance_m=200)
    at_wall = float(svf_w[n // 2, n // 2 - 1])
    ok &= report("SVF hard against a very tall wall", 0.5, at_wall, 0.05)

    # 5. SVF falls as walls get taller, and never leaves [0, 1].
    prev = 1.01
    monotonic = True
    for wall_h in [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]:
        canyon = np.zeros((100, 100), dtype=np.float32)
        canyon[:, :40] = wall_h
        canyon[:, 60:] = wall_h
        v = float(sky_view_factor(canyon, CELL, 16, 120)[50, 50])
        if not (0.0 <= v <= 1.0) or v > prev + 1e-6:
            monotonic = False
        print(f"      canyon wall {wall_h:5.1f} m  ->  SVF {v:.4f}")
        prev = v
    print(
        f"{'PASS' if monotonic else 'FAIL'}  SVF decreases monotonically with wall height and stays in [0,1]"
    )
    ok &= monotonic

    print("-" * 92)
    print("ALL CHECKS PASSED" if ok else "SOMETHING FAILED, STOP HERE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
