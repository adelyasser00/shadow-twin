"""
Write GeoTIFFs in the exact format NYC ships, to test the reader before the
real tiles land.

    python -m pipeline.make_nyc_test_tifs --out /tmp/nyc_test

EPSG:2263, NAD83 New York Long Island, US survey feet on every axis, 1 foot
cells, values in feet above datum. Synthetic geometry, real file format.

The towers have exact known heights so the unit conversion can be checked
against them. If the reader reports 472 m for the 1550 ft tower, feet to metres
is right. If it reports 1550, it is reading feet as metres and every shadow in
the project is 3.28 times too long.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

CENTRE_LAT, CENTRE_LON = 40.7670, -73.9770
FT = 1200.0 / 3937.0  # metres per US survey foot

# height in feet, offset from centre in feet (east, north), footprint feet
TOWERS = [
    ("supertall A", 1550.0, (-300.0, 900.0), 150.0),
    ("supertall B", 1428.0, (200.0, 950.0), 130.0),
    ("supertall C", 1005.0, (700.0, 880.0), 160.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/nyc_test")
    ap.add_argument("--span-m", type=float, default=700.0)
    args = ap.parse_args()

    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_transform

    os.makedirs(args.out, exist_ok=True)

    span_ft = args.span_m / FT
    n = int(span_ft)  # 1 ft cells
    print(f"building {n} x {n} cells at 1 ft ({args.span_m:.0f} m a side)")

    xs, ys = warp_transform("EPSG:4326", "EPSG:2263", [CENTRE_LON], [CENTRE_LAT])
    cx, cy = xs[0], ys[0]
    # North-up transform with the centre in the middle.
    transform = from_origin(cx - span_ft / 2, cy + span_ft / 2, 1.0, 1.0)

    ground_ft = np.full((n, n), 40.0, dtype=np.float32)  # roughly Midtown datum
    dsm_ft = ground_ft.copy()

    def col_row(east_ft, north_ft):
        return int(n / 2 + east_ft), int(n / 2 - north_ft)

    # A park strip across the north third: open ground, a few trees.
    park_r1 = int(n * 0.36)
    rng = np.random.default_rng(3)
    for _ in range(90):
        tr = rng.integers(10, park_r1)
        tc = rng.integers(10, n - 10)
        rad = int(rng.integers(10, 26))
        rr, cc = np.ogrid[:n, :n]
        crown = (rr - tr) ** 2 + (cc - tc) ** 2 <= rad**2
        dsm_ft[crown] = np.maximum(dsm_ft[crown], 40.0 + rng.uniform(30, 70))

    # Street grid south of the park: long east-west blocks, avenues north-south.
    block_h = int(200 / 1.0)   # 200 ft deep blocks
    block_w = int(600 / 1.0)   # 600 ft wide blocks
    street = int(60 / 1.0)
    r = park_r1 + street
    while r + block_h < n:
        c = 20
        while c + block_w < n:
            h = float(rng.uniform(120, 420))
            if rng.random() < 0.12:
                h = float(rng.uniform(500, 780))
            split = int(block_w * rng.uniform(0.3, 0.7))
            dsm_ft[r:r + block_h, c:c + split] = 40.0 + h
            dsm_ft[r:r + block_h, c + split:c + block_w] = 40.0 + h * rng.uniform(0.7, 1.2)
            c += block_w + street
        r += block_h + street

    # The named towers, placed just south of the park edge.
    for name, h_ft, (e, nth), fp in TOWERS:
        cc, rr = col_row(e, nth)
        half = int(fp / 2)
        r0, r1 = max(0, rr - half), min(n, rr + half)
        c0, c1 = max(0, cc - half), min(n, cc + half)
        dsm_ft[r0:r1, c0:c1] = 40.0 + h_ft
        print(f"  {name}: {h_ft:.0f} ft = {h_ft*FT:.1f} m at row {rr}, col {cc}")

    profile = dict(
        driver="GTiff", height=n, width=n, count=1, dtype="float32",
        crs="EPSG:2263", transform=transform, nodata=-9999.0, compress="deflate",
    )
    for name, arr in [("hh_NYC_TEST.tif", dsm_ft), ("be_NYC_TEST.tif", ground_ft)]:
        with rasterio.open(os.path.join(args.out, name), "w", **profile) as ds:
            ds.write(arr, 1)
        print(f"wrote {os.path.join(args.out, name)}  "
              f"{arr.min():.0f} to {arr.max():.0f} ft")

    print(f"\nexpected after conversion: tallest object "
          f"{TOWERS[0][1]*FT:.1f} m above street")
    print(f"\n  python -m pipeline.run_nyc --dsm {args.out} --dem {args.out} \\")
    print(f"      --span {args.span_m*0.85:.0f} --res 2.0 --skip-svf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
