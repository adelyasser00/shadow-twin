"""
Self-tests for grid.py: where cells are, which way they face, and how the
overlays are turned north-up for the browser.

    python -m pipeline.verify_grid

Every check works in EPSG:2263, the NYC LiDAR's own CRS, so a unit mistake
between feet and metres shows up here and not in a published number.
"""

import math

import numpy as np

from . import grid as G

LAT, LON = 40.7812, -73.9665      # middle of Central Park


def report(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label:<58} {detail}")
    return ok


def main():
    from pyproj import CRS, Transformer
    from rasterio import features
    from shapely.geometry import Polygon

    crs = CRS.from_epsg(2263)
    unit = 1200.0 / 3937.0
    ok = True
    print("grid self-test\n" + "-" * 92)

    g = G.rotated(crs, unit, LAT, LON, length_m=4000, width_m=1200,
                  bearing_deg=28.9, res_m=2.0, buffer_m=500)
    fh, fw = g.frame_shape
    ok &= report("frame is 4000 m by 1200 m at 2 m", (fh, fw) == (2000, 600),
                 f"got {fh} x {fw} cells")
    ok &= report("buffer is 500 m on every side", abs(g.buffer_m - 500) < 1e-6,
                 f"got {g.buffer_m:.1f} m")

    # Round trip, and the affine agrees with the explicit maths.
    rr = np.array([0.0, 10.5, 1234.25, g.rows - 1.0])
    cc = np.array([0.0, 99.5, 17.75, g.cols - 1.0])
    x, y = g.rowcol_to_crs(rr, cc)
    r2, c2 = g.crs_to_rowcol(x, y)
    err = float(max(np.abs(r2 - rr).max(), np.abs(c2 - cc).max()))
    ok &= report("row/col to CRS and back", err < 1e-6, f"max error {err:.2e} cells")
    xa, ya = g.transform * (cc, rr)
    err = float(max(np.abs(np.asarray(xa) - x).max(), np.abs(np.asarray(ya) - y).max()))
    ok &= report("affine transform matches rowcol_to_crs", err < 1e-6,
                 f"max error {err:.2e} ft")

    # Cell size is 2 m on the ground in both directions.
    x0, y0 = g.rowcol_to_crs(100, 100)
    x1, y1 = g.rowcol_to_crs(100, 101)
    x2, y2 = g.rowcol_to_crs(101, 100)
    dc = math.hypot(x1 - x0, y1 - y0) * unit
    dr = math.hypot(x2 - x0, y2 - y0) * unit
    ok &= report("one cell is 2 m along rows and columns",
                 abs(dc - 2) < 1e-9 and abs(dr - 2) < 1e-9, f"{dc:.6f} m, {dr:.6f} m")

    # Grid up points along the bearing, measured on the ground in true terms.
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xa, ya = g.rowcol_to_crs(1000, 300)
    xb, yb = g.rowcol_to_crs(900, 300)      # 100 cells toward grid up
    la, pa = to_ll.transform(xa, ya)
    lb, pb = to_ll.transform(xb, yb)
    # Geodesic bearing on the ellipsoid. A spherical shortcut is off by a
    # tenth of a degree at this latitude, which is bigger than the thing
    # being tested.
    from pyproj import Geod
    brg = Geod(ellps="WGS84").inv(la, pa, lb, pb)[0]
    ok &= report("grid up is 28.9 degrees true", abs(brg - 28.9) < 0.05,
                 f"measured {brg:.3f} deg")

    # Rasterising through a rotated affine: a 100 m square drawn in CRS space,
    # square to the grid, must come out as 50 x 50 cells, 2500 of them.
    cx, cy = g.rowcol_to_crs(1000.0, 300.0)
    (rx, ry), (dx, dy) = g.right, g.down
    s = 50.0 / unit
    pts = [(cx + a * s * rx + b * s * dx, cy + a * s * ry + b * s * dy)
           for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    m = features.rasterize([(Polygon(pts), 1)], out_shape=(g.rows, g.cols),
                           transform=g.transform, fill=0, dtype="uint8")
    n = int(m.sum())
    rows, cols = np.nonzero(m)
    ok &= report("rotated rasterise of a square 100 m block", n == 2500,
                 f"{n} cells, rows {rows.min()}-{rows.max()}, cols {cols.min()}-{cols.max()}")
    ok &= report("  and it lands where it was drawn",
                 rows.min() == 975 and cols.min() == 275, "")

    # North-up resampling: pick a frame cell, find its centre in lon/lat, and
    # check the output pixel there maps back to that same cell.
    idx, (w, so, e, no) = g.northup_index()
    r0, r1, c0, c1 = g.frame
    fr, fc = 700, 211
    xc, yc = g.rowcol_to_crs(r0 + fr + 0.5, c0 + fc + 0.5)
    lon, lat = to_ll.transform(xc, yc)
    H, W = idx.shape
    pr = int((no - lat) / (no - so) * H)
    pc = int((lon - w) / (e - w) * W)
    got = int(idx[pr, pc])
    ok &= report("north-up image pixel maps back to its frame cell",
                 got == fr * fw + fc, f"expected {fr * fw + fc}, got {got}")
    inside = float((idx >= 0).mean())
    expect = (fh * fw * 4.0) / (H * W * 4.0)
    ok &= report("share of the north-up image covered by the frame",
                 abs(inside - expect) < 0.01, f"{inside:.3f} vs {expect:.3f} by area")

    # The legacy square reproduces the original north-up CRS grid exactly.
    sq = G.square(crs, unit, 40.7668, -73.9790, 700, 2.0)
    cxs, cys = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(-73.9790, 40.7668)
    span_u, res_u = 700 / unit, 2.0 / unit
    ok &= report("legacy square: corner, size and orientation",
                 abs(sq.x0 - (cxs - span_u / 2)) < 1e-6 and abs(sq.y0 - (cys + span_u / 2)) < 1e-6
                 and sq.rows == 350 and abs(sq.transform.b) < 1e-12
                 and abs(sq.transform.a - res_u) < 1e-12,
                 f"x0 off by {sq.x0 - (cxs - span_u / 2):.2e} ft")

    print("-" * 92)
    print("ALL CHECKS PASSED" if ok else "SOMETHING FAILED, STOP HERE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
