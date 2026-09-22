"""
Facade solver check on a case with a known answer.

    python -m pipeline.verify_facade

One isolated 200 m tower on flat ground, footprint rotated 29 degrees like
Manhattan's grid. With nothing around it, every wall that faces the sun must be
lit for every hour it faces it, and no wall may be lit while facing away. The
expected hours are computed from sun geometry alone, independently of the ray
march, so this catches self-shadowing and orientation bugs.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np

from . import facade
from .solar import sun_position

LAT, LON = 40.7645, -73.9790
WALLS = {"N": 29, "E": 119, "S": 209, "W": 299}


def main():
    n, cell = 300, 2.0
    yy, xx = np.mgrid[:n, :n]
    a = math.radians(29)
    e, nth = xx - n / 2, -(yy - n / 2)
    u = e * math.cos(a) - nth * math.sin(a)
    v = e * math.sin(a) + nth * math.cos(a)
    built = (abs(u) < 20) & (abs(v) < 12)
    ground = np.zeros((n, n), np.float32)
    heights = np.where(built, 200.0, 0).astype(np.float32)
    labels = built.astype(np.int32)

    ok = True
    print("isolated rotated tower, facade self-test\n" + "-" * 70)
    for date, label in [((2026, 3, 21), "21 March"), ((2026, 6, 21), "21 June"),
                        ((2026, 12, 21), "21 December")]:
        pos = []
        for hh in range(8, 17):
            utc = datetime(*date, hh, 0, tzinfo=timezone.utc) + timedelta(hours=5)
            p = sun_position(utc, LAT, LON)
            if p.above_horizon:
                pos.append(p)
        pts = facade.wall_samples(built, heights, labels, cell, levels=4,
                                  log=lambda *_: None)
        r = facade.solve(pts, ground + heights, ground, cell, pos,
                         [str(i) for i in range(len(pos))], log=None,
                         labels_grid=labels)
        ro = facade.per_building(pts, r, 1)
        for o, normal in WALLS.items():
            expect = sum(1 for p in pos if math.cos(math.radians(p.azimuth - normal)) > 0.02)
            got = float(ro[o]["sun_hours"][1])
            # Wall-bin averaging mixes in corner cells, so allow 0.6 h.
            good = abs(got - expect) <= 0.6
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  {label:<12} {o} wall ({normal:3d} deg)  "
                  f"expected {expect:>2} h  got {got:4.1f} h")
    print("-" * 70)
    print("ALL CHECKS PASSED" if ok else "SOMETHING FAILED, STOP HERE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
