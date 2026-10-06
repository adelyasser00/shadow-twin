"""
The city as voxels for the wind solver, turned to each wind direction.

    .venv-heat/Scripts/python -m pipeline.wind_domain            # build the canvas once
    .venv-heat/Scripts/python -m pipeline.wind_domain --show 315 # look at one direction

Two steps.

Canvas: one north-up 2 m raster in metres (EPSG:32118, the heat layer's own
projection), wider than any turned domain, with ground, building height above
ground for 2017 and today, and tree canopy height. Inside the heat grid it is
the heat layer's own surface, so wind and heat see the same buildings:

  ground    the heat DEM (LiDAR ground); beyond it, the nearest DEM value
  buildings LiDAR roofs where the 2017 survey measured them (the heat DSMs,
            "today" raised by the grown-building rule). Everywhere else, and
            where the survey has no data, NYC Building Footprints as prisms at
            their recorded roof height; for 2017, only those built before 2017.
  trees     the heat layer's canopy (LiDAR, leaf-on); none beyond it

Domain: for a wind from direction theta (degrees from north, where the wind
comes from, as weather stations report it), a box whose x axis points
downwind, like a wind tunnel turntable. Each cell is sampled from 16 points of
the canvas, 2 m apart. A column is as tall as the median of its 16 samples,
so a building covering half a cell or more fills it and a thinner sliver does
not. Height steps are whole cells.

Trees are drag, not walls: within the crown (the top three quarters of the
canopy height), each cell gets c = Cd * LAD * dx * cover, the standard canopy
drag (Cd 0.2, leaf area density LAD per m3 of crown). Bare in December (LAD
0.3, mostly branches), in leaf in June (LAD 1.0).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from .heat_inputs import HEAT_DIR, INPUTS_DIR

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIND_DIR = os.path.join(HERE, "data", "wind")
CANVAS = os.path.join(WIND_DIR, "canvas.npz")
FOOTPRINTS = os.path.join(HERE, "data", "nyc", "footprints_crop.geojson")

RES_CANVAS = 2.0
CANVAS_HALF_M = 3300.0          # canvas spans 6.6 km, enough for a 3.2 x 2.9 km box at any angle
SURVEY_YEAR = 2017
CD_TREE = 0.2
LAD = {"dec": 0.3, "jun": 1.0}  # m2 of leaf (or branch) per m3 of crown
CROWN_BASE = 0.25               # crown starts at a quarter of the tree height
INLET_CLEAR = 6                 # cells kept free of buildings at the inlet
OUTLET_CLEAR = 4

# The box around the study area. x runs downwind. The study area (the south
# end of the park and the towers on 57th Street) sits around ROI_FROM_INLET_M
# from the inlet, so at least 600 m of city lies upwind of it whatever the
# direction.
BOX = {"length_m": 3200.0, "width_m": 2880.0, "height_m": 960.0, "roi_from_inlet_m": 1700.0}


def roi_centre(meta, masks):
    """Study centre in EPSG:32118 metres: middle of the park window, moved south to the towers."""
    t = meta["_transform"]
    rr, cc = np.nonzero(masks["window"])
    x = t.c + (cc.mean() + 0.5) * t.a
    y = t.f + (rr.mean() + 0.5) * t.e
    # The window runs 1.5 km up the avenues from 59th Street; the towers stand
    # 150 to 500 m south of it. Pull the centre 250 m back down the avenues.
    ax = meta["avenue_axis"]
    return float(x - 250.0 * ax["ux"]), float(y - 250.0 * ax["uy"])


def build_canvas(log=print):
    import rasterio
    from pyproj import Transformer
    from rasterio import features
    from scipy import ndimage
    from shapely.ops import transform as shp_tf

    from .heat_inputs import load
    from . import nyc_footprints as NF

    layers, masks, meta = load()
    t = meta["_transform"]
    cx, cy = roi_centre(meta, masks)
    n = int(round(2 * CANVAS_HALF_M / RES_CANVAS))
    x0, y1 = cx - CANVAS_HALF_M, cy + CANVAS_HALF_M
    ctf = rasterio.transform.from_origin(x0, y1, RES_CANVAS, RES_CANVAS)
    log(f"  canvas {n} x {n} at {RES_CANVAS} m around ({cx:.0f}, {cy:.0f}) EPSG:32118")

    # Where the heat grid sits on the canvas (same 2 m pixels, offset by a shift).
    hc0 = (t.c - x0) / RES_CANVAS
    hr0 = (y1 - t.f) / RES_CANVAS
    if abs(hc0 - round(hc0)) > 1e-3 or abs(hr0 - round(hr0)) > 1e-3:
        # Snap the canvas to the heat grid's pixel lattice.
        x0 += (hc0 - round(hc0)) * RES_CANVAS
        y1 -= (hr0 - round(hr0)) * RES_CANVAS
        ctf = rasterio.transform.from_origin(x0, y1, RES_CANVAS, RES_CANVAS)
        hc0, hr0 = (t.c - x0) / RES_CANVAS, (y1 - t.f) / RES_CANVAS
    hc0, hr0 = int(round(hc0)), int(round(hr0))
    H, W = layers["dem"].shape
    inside = np.zeros((n, n), bool)
    inside[hr0:hr0 + H, hc0:hc0 + W] = True

    def paste(a, fill):
        out = np.full((n, n), fill, np.float32)
        out[hr0:hr0 + H, hc0:hc0 + W] = a
        return out

    dem = paste(layers["dem"], np.nan)
    idx = ndimage.distance_transform_edt(np.isnan(dem), return_distances=False, return_indices=True)
    ground = dem[idx[0], idx[1]]
    canopy = paste(layers["cdsm"], 0.0)
    lidar = paste(~masks["nodata"], False).astype(bool) & inside
    hbld = {s: paste(np.where(masks["building"], layers[f"dsm_{s}"] - layers["dem"], 0.0), 0.0)
            for s in ("2017", "today")}

    # Footprint prisms for everything the survey did not measure.
    to_ll = Transformer.from_crs("EPSG:32118", "EPSG:4326", always_xy=True).transform
    to_m = Transformer.from_crs("EPSG:4326", "EPSG:32118", always_xy=True).transform
    lons, lats = to_ll([x0, x0 + n * RES_CANVAS], [y1 - n * RES_CANVAS, y1])
    recs = NF.read(FOOTPRINTS, bbox_lonlat=(min(lons), min(lats), max(lons), max(lats)), log=log)
    recs = [r for r in recs if r["height_m"]]
    polys = [shp_tf(to_m, r["poly"]) for r in recs]
    lab = features.rasterize([(p, i + 1) for i, p in enumerate(polys)], out_shape=(n, n),
                             transform=ctf, fill=0, dtype="int32")
    hts = np.array([0.0] + [r["height_m"] for r in recs], np.float32)
    yrs = np.array([0] + [r["year"] or 0 for r in recs], np.int32)
    prism_today = hts[lab]
    prism_2017 = np.where((yrs[lab] > 0) & (yrs[lab] < SURVEY_YEAR), prism_today, 0.0)
    # A building with no year is kept in 2017: almost all of them are old.
    prism_2017 = np.where((lab > 0) & (yrs[lab] == 0), prism_today, prism_2017)
    hb = {"today": np.where(lidar, hbld["today"], prism_today),
          "2017": np.where(lidar, hbld["2017"], prism_2017)}
    out = {"ground": ground.astype(np.float32), "canopy": canopy.astype(np.float32),
           "b_today": hb["today"].astype(np.float32), "b_2017": hb["2017"].astype(np.float32),
           "lidar": lidar, "heat": inside}
    os.makedirs(WIND_DIR, exist_ok=True)
    info = {"crs": "EPSG:32118", "res_m": RES_CANVAS, "x0": x0, "y1": y1, "n": n,
            "roi_centre": [cx, cy], "heat_offset_rc": [hr0, hc0], "heat_shape": [H, W],
            "footprints": len(recs), "avenue_axis": meta["avenue_axis"],
            "site": meta["site"]}
    np.savez_compressed(CANVAS, **out)
    with open(CANVAS.replace(".npz", ".json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=1)
    log(f"  canvas written: {len(recs):,} footprints, tallest {max(hb['today'].max(), 0):.0f} m today, "
        f"{hb['2017'].max():.0f} m in 2017, LiDAR on {lidar.mean() * 100:.0f}% of it")
    return out, info


def load_canvas():
    z = np.load(CANVAS)
    with open(CANVAS.replace(".npz", ".json"), encoding="utf-8") as f:
        info = json.load(f)
    return {k: z[k] for k in z.files}, info


def axes(theta_deg):
    """Unit vectors (east, north) of the box: x downwind, y to the left of it."""
    t = math.radians(theta_deg)
    ex = np.array([-math.sin(t), -math.cos(t)])
    ey = np.array([-ex[1], ex[0]])
    return ex, ey


class Box:
    """Geometry of one turned domain: cell (k, j, i) <-> EPSG:32118 metres."""

    def __init__(self, info, theta, res, box=BOX):
        self.theta, self.res = float(theta), float(res)
        self.nx = int(round(box["length_m"] / res))
        self.ny = int(round(box["width_m"] / res))
        self.nz = int(round(box["height_m"] / res)) + 1   # + the solid ground layer
        self.p0 = np.array(info["roi_centre"], float)
        self.x0c = box["roi_from_inlet_m"] / res
        self.ex, self.ey = axes(theta)

    def to_map(self, i, j):
        """Map metres of the centre of cell column (j, i); arrays welcome."""
        a = (np.asarray(i, float) + 0.5 - self.x0c) * self.res
        b = (np.asarray(j, float) + 0.5 - self.ny / 2.0) * self.res
        return (self.p0[0] + a * self.ex[0] + b * self.ey[0],
                self.p0[1] + a * self.ex[1] + b * self.ey[1])

    def from_map(self, x, y):
        """Fractional (i, j) cell coordinates of map points; cell centres are integers."""
        dx, dy = np.asarray(x, float) - self.p0[0], np.asarray(y, float) - self.p0[1]
        i = (dx * self.ex[0] + dy * self.ex[1]) / self.res + self.x0c - 0.5
        j = (dx * self.ey[0] + dy * self.ey[1]) / self.res + self.ny / 2.0 - 0.5
        return i, j

    def meta(self):
        return {"theta": self.theta, "res_m": self.res, "nx": self.nx, "ny": self.ny, "nz": self.nz,
                "p0": self.p0.tolist(), "x0c": self.x0c, "ex": self.ex.tolist(), "ey": self.ey.tolist()}


def sample_columns(canvas, info, box, scenario, sub=4):
    """Per column: ground, building height (median rule), canopy height and cover."""
    n, res_c = info["n"], info["res_m"]
    x0, y1 = info["x0"], info["y1"]
    offs = (np.arange(sub) + 0.5) / sub - 0.5          # sub-sample offsets in cells
    jj, ii = np.meshgrid(np.arange(box.ny), np.arange(box.nx), indexing="ij")
    g = np.zeros((sub * sub, box.ny, box.nx), np.float32)
    b = np.zeros_like(g)
    c = np.zeros_like(g)
    bkey = f"b_{scenario}"
    k = 0
    for oa in offs:
        for ob in offs:
            x, y = box.to_map(ii + oa, jj + ob)
            col = np.clip(((x - x0) / res_c).astype(np.int64), 0, n - 1)
            row = np.clip(((y1 - y) / res_c).astype(np.int64), 0, n - 1)
            g[k] = canvas["ground"][row, col]
            b[k] = canvas[bkey][row, col]
            c[k] = canvas["canopy"][row, col]
            k += 1
    ground = g.mean(axis=0)
    bh = np.median(b, axis=0)
    tree = c > 2.0
    cover = tree.mean(axis=0)
    ch = np.where(tree.any(axis=0), np.where(tree, c, 0).sum(axis=0) / np.maximum(tree.sum(axis=0), 1), 0)
    return ground, bh, ch.astype(np.float32), cover.astype(np.float32)


def voxelise(canvas, info, theta, res, scenario, season="dec", box=BOX):
    """Solid mask (nz, ny, nx), tree drag (nz, ny, nx), column info and the Box."""
    bx = Box(info, theta, res, box)
    ground, bh, ch, cover = sample_columns(canvas, info, bx, scenario)
    gmin = float(ground.min())
    g = ground - gmin
    bh[:, :INLET_CLEAR] = 0.0
    bh[:, -OUTLET_CLEAR:] = 0.0
    ch[:, :INLET_CLEAR] = 0.0
    top = g + bh
    zc = (np.arange(bx.nz, dtype=np.float32) - 0.5) * res        # cell centre heights
    solid = zc[:, None, None] < top[None]
    solid[0] = True
    # Tree drag: the overlap of each cell with the crown, as a fraction of the cell.
    lo = g + CROWN_BASE * ch
    hi = g + ch
    zlo = (np.arange(bx.nz, dtype=np.float32) - 1.0) * res
    zhi = zlo + res
    ov = np.clip(np.minimum(zhi[:, None, None], hi[None]) - np.maximum(zlo[:, None, None], lo[None]), 0, None) / res
    veg = (CD_TREE * LAD[season] * res * cover[None] * ov).astype(np.float32)
    veg[solid] = 0.0
    # The first fluid layer of each column: where people walk (or a roof).
    kfirst = np.argmin(solid, axis=0).astype(np.int16)
    cols = {"ground": ground, "gmin": gmin, "building": bh, "canopy": ch, "cover": cover,
            "kfirst": kfirst, "roof": bh > 0.5 * res}
    return solid, veg, cols, bx


def show(theta, res, scenario, out_png, log=print):
    from PIL import Image
    canvas, info = load_canvas()
    solid, veg, cols, bx = voxelise(canvas, info, theta, res, scenario)
    h = (solid[1:].sum(axis=0) * res).astype(np.float32)
    img = np.zeros(h.shape + (3,), np.uint8)
    img[..., 0] = np.clip(h / 200 * 255, 0, 255)
    img[..., 1] = np.clip(veg.sum(axis=0) * 300, 0, 255)
    img[..., 2] = 60
    # Flow runs left to right in the picture (x is downwind).
    Image.fromarray(img[::-1]).save(out_png)
    log(f"  {theta} deg {scenario}: {bx.nx} x {bx.ny} x {bx.nz}, solid {solid[1:].mean() * 100:.1f}%, "
        f"tallest column {h.max():.0f} m, tree cells {(veg > 0).sum():,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", type=float, default=None, help="write a picture of one direction")
    ap.add_argument("--res", type=float, default=8.0)
    ap.add_argument("--scenario", default="today")
    ap.add_argument("--out", default=os.path.join(WIND_DIR, "domain.png"))
    args = ap.parse_args()
    t0 = time.time()
    if args.show is None or not os.path.exists(CANVAS):
        build_canvas()
    if args.show is not None:
        show(args.show, args.res, args.scenario, args.out)
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
