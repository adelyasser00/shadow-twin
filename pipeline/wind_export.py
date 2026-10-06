"""
Wind for the viewer: a street-level wind map, wind lines to animate, and the
anchor that turns the solver's ratios into metres per second.

    .venv-heat/Scripts/python -m pipeline.wind_export

Reads data/wind/runs/ and writes viewer/wind/: wind.json, one map image and
two line files per run. The page fetches them on demand; a build without them
shows no wind layer.

Anchor
------
The solver gives ratios, not speeds. Every run is read against the open lawns
of the study area (park ground with no tree canopy, the median of their
street-level wind): there the wind equals the weather file's wind, the Central
Park station, and everywhere else it is that wind times the simulated ratio.
That keeps the heat layer's meaning for open lawns, where it already used the
station wind, and only changes the pattern. Both years use the 2017 run's
lawn value, so a change on the lawns themselves would still show.

A single spot was tried first (the middle of Sheep Meadow) and dropped: on a
northwester it sits in the sheltered zone behind the Upper West Side, so every
other spot looked too windy against it.

Lines
-----
  street   evenly spaced lines that follow the wind 2.5 m above the ground
           (the first open cell of each column, about 4 m up at 8 m cells)
  air      3D lines released upwind of every tower of 150 m or more and
           across the park, followed through the time-mean 3D wind: they show
           air thrown down the face of a tower and the wake behind it

Every line has one vertex per fixed slice of travel time, so its vertices sit
far apart where the wind is fast and close together where it is slow. The
browser moves light along them one vertex per tick: the light moves at the
simulated wind speed, sped up by a fixed factor.

Binary line file (little-endian): int32 count, int32 total vertices, uint16
vertices per line, then int16 east, north, up per vertex in decimetres from
the site origin.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime

import numpy as np

from .wind_domain import WIND_DIR, load_canvas
from .run_wind import RUNS_DIR, run_name

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "viewer", "wind")

SHEEP_MEADOW = (-73.9752, 40.7718)      # lon, lat, approximate centre of the lawn
REF_RADIUS_M = 40.0
STREET_UP_M = 2.5
VERTEX_CELLS = 1.5                      # vertex spacing at the Sheep Meadow speed, in cells
MAX_VERTICES = 160
MIN_VERTICES = 10
SLOW = 0.05                             # stop where the wind falls below 5% of Sheep Meadow
D_SEP_CELLS = 2.4                       # street lines stay about 20 m apart
TOWER_M = 150.0

# Street wind map: ratio to the open lawn, fixed breaks.
K_BREAKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 4.0]
K_COLOURS = ["#1a1035", "#2c2a6b", "#2f4f8f", "#2f7aa6", "#3aa3b0",
             "#6cc7a6", "#b5e08f", "#f1ef7a", "#fff6c9"]
K_ALPHA = 200

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


# --------------------------------------------------------------- geodesy
def _ecef(lon, lat, h):
    lon, lat = np.radians(lon), np.radians(lat)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return ((n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon),
            (n * (1 - WGS84_E2) + h) * np.sin(lat))


def enu(lon, lat, h, lon0, lat0):
    """East, north, up in metres from (lon0, lat0, 0) on the WGS84 ellipsoid."""
    x, y, z = _ecef(lon, lat, h)
    x0, y0, z0 = _ecef(lon0, lat0, 0.0)
    dx, dy, dz = x - x0, y - y0, z - z0
    lo, la = math.radians(lon0), math.radians(lat0)
    e = -math.sin(lo) * dx + math.cos(lo) * dy
    n = -math.sin(la) * math.cos(lo) * dx - math.sin(la) * math.sin(lo) * dy + math.cos(la) * dz
    u = math.cos(la) * math.cos(lo) * dx + math.cos(la) * math.sin(lo) * dy + math.sin(la) * dz
    return e, n, u


# ------------------------------------------------------------------ runs
class RunBox:
    """Box geometry back from run.json."""

    def __init__(self, m):
        b = m["box"]
        self.res, self.nx, self.ny, self.nz = b["res_m"], b["nx"], b["ny"], b["nz"]
        self.p0, self.x0c = np.array(b["p0"]), b["x0c"]
        self.ex, self.ey = np.array(b["ex"]), np.array(b["ey"])

    def to_map(self, i, j):
        a = (np.asarray(i, float) + 0.5 - self.x0c) * self.res
        b = (np.asarray(j, float) + 0.5 - self.ny / 2.0) * self.res
        return (self.p0[0] + a * self.ex[0] + b * self.ey[0],
                self.p0[1] + a * self.ex[1] + b * self.ey[1])

    def from_map(self, x, y):
        dx, dy = np.asarray(x, float) - self.p0[0], np.asarray(y, float) - self.p0[1]
        i = (dx * self.ex[0] + dy * self.ex[1]) / self.res + self.x0c - 0.5
        j = (dx * self.ey[0] + dy * self.ey[1]) / self.res + self.ny / 2.0 - 0.5
        return i, j


def runs_available():
    out = []
    if not os.path.isdir(RUNS_DIR):
        return out
    for d in sorted(os.listdir(RUNS_DIR)):
        p = os.path.join(RUNS_DIR, d, "run.json")
        if os.path.exists(p) and "_test" not in d:
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            m["_dir"] = os.path.join(RUNS_DIR, d)
            m["_name"] = d
            out.append(m)
    return out


def load_run(m):
    z = np.load(os.path.join(m["_dir"], "field.npz"))
    r = {k: z[k] for k in z.files}
    r["meta"], r["box"] = m, RunBox(m)
    r["ok"] = (~r["roof"]) & np.isfinite(r["street_speed"])
    return r


def resample(values, ok, box, x, y):
    """Bilinear sample of a column field at map points, from open-ground columns only."""
    from scipy.ndimage import map_coordinates
    i, j = box.from_map(x, y)
    v = np.where(ok, values, 0.0).astype(np.float64)
    num = map_coordinates(v, [j.ravel(), i.ravel()], order=1, mode="constant", cval=0.0)
    den = map_coordinates(ok.astype(np.float64), [j.ravel(), i.ravel()], order=1, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0.25, num / np.maximum(den, 1e-9), np.nan)
    return out.reshape(np.shape(x))


def heat_points():
    """EPSG:32118 centres of the heat grid cells, and the heat layers."""
    from .heat_inputs import load
    layers, masks, meta = load()
    t = meta["_transform"]
    H, W = layers["dem"].shape
    cc, rr = np.meshgrid(np.arange(W), np.arange(H))
    x = t.c + (cc + 0.5) * t.a
    y = t.f + (rr + 0.5) * t.e
    return x, y, layers, masks, meta


def sheep_meadow(layers, masks, meta, x, y):
    """The most open spot of the lawn near SHEEP_MEADOW: (x, y) in EPSG:32118 and a disc mask."""
    from pyproj import Transformer
    from scipy import ndimage
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32118", always_xy=True).transform
    sx, sy = tf(*SHEEP_MEADOW)
    lawn = masks["park"] & ~masks["building"] & ~masks["water"] & (layers["cdsm"] < 2.0)
    dist = ndimage.distance_transform_edt(lawn) * 2.0
    near = (x - sx) ** 2 + (y - sy) ** 2 < 250.0 ** 2
    k = np.argmax(np.where(near, dist, -1))
    r, c = np.unravel_index(k, dist.shape)
    cx, cy = float(x[r, c]), float(y[r, c])
    disc = ((x - cx) ** 2 + (y - cy) ** 2 < REF_RADIUS_M ** 2) & lawn
    return cx, cy, float(dist[r, c]), disc


def open_lawn(layers, masks):
    """Reported park ground with no canopy: the anchor of every ratio."""
    return masks["report"] & ~masks["building"] & ~masks["water"] & (layers["cdsm"] < 2.0)


def lawn_reference(run, x, y, lawn):
    """Median street-level speed (lattice units) over the open lawns, every 3rd cell."""
    rr, cc = np.nonzero(lawn)
    rr, cc = rr[::3], cc[::3]
    v = resample(run["street_speed"], run["ok"], run["box"], x[rr, cc], y[rr, cc])
    return float(np.nanmedian(v))


def reference_speed(run, cx, cy):
    """Mean street-level speed (lattice units) within REF_RADIUS_M of the anchor."""
    b = run["box"]
    g = np.linspace(-REF_RADIUS_M, REF_RADIUS_M, 9)
    gx, gy = np.meshgrid(g, g)
    keep = gx ** 2 + gy ** 2 <= REF_RADIUS_M ** 2
    v = resample(run["street_speed"], run["ok"], b, cx + gx[keep], cy + gy[keep])
    return float(np.nanmean(v))


# ----------------------------------------------------------------- lines
def _interp2(fields, ok, i, j):
    from scipy.ndimage import map_coordinates
    out = [map_coordinates(f, [j, i], order=1, mode="nearest") for f in fields]
    okv = map_coordinates(ok.astype(np.float32), [j, i], order=1, mode="constant", cval=0.0)
    return out, okv


def street_lines(run, u_ref, roi, rng, n_seeds=9000):
    """Evenly spaced street-level lines in box cell coordinates: list of (n, 2) arrays."""
    b = run["box"]
    ok = run["ok"]
    ux = np.where(ok, run["street_u"][0], 0.0).astype(np.float32)
    uy = np.where(ok, run["street_u"][1], 0.0).astype(np.float32)
    dtau = VERTEX_CELLS / u_ref
    sub = 6
    dt = dtau / sub
    jj, ii = np.nonzero(ok & roi)
    pick = rng.choice(len(ii), size=min(n_seeds, len(ii)), replace=False)
    p = np.stack([ii[pick] + rng.uniform(-0.5, 0.5, len(pick)),
                  jj[pick] + rng.uniform(-0.5, 0.5, len(pick))], axis=1).astype(np.float64)

    def trace(p0, sign):
        n = len(p0)
        pts = np.full((n, MAX_VERTICES, 2), np.nan)
        pts[:, 0] = p0
        alive = np.ones(n, bool)
        cur = p0.copy()
        length = np.ones(n, int)
        for v in range(1, MAX_VERTICES):
            for _ in range(sub):
                (a, c), o = _interp2((ux, uy), ok, cur[:, 0], cur[:, 1])
                mid = cur + sign * 0.5 * dt * np.stack([a, c], 1)
                (a2, c2), o2 = _interp2((ux, uy), ok, mid[:, 0], mid[:, 1])
                step = sign * dt * np.stack([a2, c2], 1)
                sp = np.hypot(a2, c2)
                bad = (sp < SLOW * u_ref) | (o2 < 0.5) | ~roi_at(roi, cur)
                alive &= ~bad
                cur = np.where(alive[:, None], cur + step, cur)
            pts[alive, v] = cur[alive]
            length[alive] = v + 1
            if not alive.any():
                break
        return pts, length

    back, lb = trace(p, -1.0)
    starts = back[np.arange(len(p)), lb - 1]
    fwd, lf = trace(starts, 1.0)
    # Greedy acceptance, longest first: a line is cut where it comes within
    # D_SEP_CELLS of a line already kept.
    occ_res = D_SEP_CELLS / 2.0
    W = int(b.nx / occ_res) + 2
    Hh = int(b.ny / occ_res) + 2
    occ = np.zeros((Hh, W), np.int32)
    order = np.argsort(-lf)
    kept = []
    for k in order:
        L = fwd[k, :lf[k]]
        if len(L) < MIN_VERTICES:
            continue
        ci = (L[:, 0] / occ_res).astype(int).clip(0, W - 1)
        cj = (L[:, 1] / occ_res).astype(int).clip(0, Hh - 1)
        hit = occ[cj, ci] > 0
        # Neighbouring cells count too, so lines do not run shoulder to shoulder.
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    hit |= occ[(cj + dj).clip(0, Hh - 1), (ci + di).clip(0, W - 1)] > 0
        if hit.any():
            # Keep the longest free run.
            free = ~hit
            best, bs, cur_s = 0, 0, None
            for t, f in enumerate(np.append(free, False)):
                if f and cur_s is None:
                    cur_s = t
                elif not f and cur_s is not None:
                    if t - cur_s > best:
                        best, bs = t - cur_s, cur_s
                    cur_s = None
            if best < MIN_VERTICES:
                continue
            L = L[bs:bs + best]
            ci, cj = ci[bs:bs + best], cj[bs:bs + best]
        occ[cj, ci] = 1
        kept.append(L)
    return kept, dtau


def roi_at(roi, p):
    i = np.clip(np.round(p[:, 0]).astype(int), 0, roi.shape[1] - 1)
    j = np.clip(np.round(p[:, 1]).astype(int), 0, roi.shape[0] - 1)
    return roi[j, i]


def air_lines(run, u_ref, roi, rng):
    """3D lines released upwind of the tall towers and across the park: list of (n, 3) arrays (i, j, k)."""
    from scipy.ndimage import map_coordinates
    b = run["box"]
    u = run["u"].astype(np.float32)
    nz = u.shape[1]
    solid = np.unpackbits(run["solid"], axis=-1)[..., :b.nx].astype(bool)[:nz]
    bh = run["building"]
    seeds = []
    # Towers: columns at least TOWER_M tall inside the study area, one seed rake each.
    tall = (bh >= TOWER_M) & roi
    from scipy import ndimage
    lab, nlab = ndimage.label(tall)
    for t in range(1, nlab + 1):
        jj, ii = np.nonzero(lab == t)
        h = float(bh[jj, ii].max())
        i0, j0 = ii.min(), jj.mean()
        width = max(2.0, jj.max() - jj.min() + 1)
        for zf in (0.2, 0.35, 0.5, 0.65, 0.8):
            for dj in np.linspace(-0.9, 0.9, 5):
                seeds.append((i0 - 6.0, j0 + dj * width * 0.7, run["kfirst"][int(j0), max(0, i0 - 6)] + zf * h / b.res))
    # The park: a low rake across the whole study area, upwind of it.
    jr = np.nonzero(roi.any(axis=1))[0]
    ir = np.nonzero(roi.any(axis=0))[0]
    for j in np.linspace(jr.min(), jr.max(), 34):
        for zm in (12.0, 40.0):
            i = ir.min() + 2
            seeds.append((i, j, run["kfirst"][int(j), int(i)] + zm / b.res))
    s = np.array(seeds, np.float64)
    dtau = VERTEX_CELLS / u_ref
    sub = 6
    dt = dtau / sub
    n = len(s)
    pts = np.full((n, MAX_VERTICES * 2, 3), np.nan)
    pts[:, 0] = s
    cur = s.copy()
    alive = np.ones(n, bool)
    length = np.ones(n, int)
    fields = [np.ascontiguousarray(u[c]) for c in range(3)]
    sol = solid.astype(np.float32)

    def vel(q):
        coords = [q[:, 2], q[:, 1], q[:, 0]]
        return np.stack([map_coordinates(f, coords, order=1, mode="nearest") for f in fields], 1), \
            map_coordinates(sol, coords, order=1, mode="nearest")

    for v in range(1, MAX_VERTICES * 2):
        for _ in range(sub):
            a, _s = vel(cur)
            mid = cur + 0.5 * dt * a
            a2, s2 = vel(mid)
            sp = np.linalg.norm(a2, axis=1)
            bad = ((sp < SLOW * u_ref) | (s2 > 0.5) | (mid[:, 0] < 1) | (mid[:, 0] > b.nx - 3)
                   | (mid[:, 1] < 1) | (mid[:, 1] > b.ny - 2) | (mid[:, 2] < 1) | (mid[:, 2] > nz - 2))
            alive &= ~bad
            cur = np.where(alive[:, None], cur + dt * a2, cur)
        pts[alive, v] = cur[alive]
        length[alive] = v + 1
        if not alive.any():
            break
    return [pts[k, :length[k]] for k in range(n) if length[k] >= MIN_VERTICES], dtau


def to_enu(run, ij, k=None, ground_m=None, origin=None):
    """Box (i, j[, k]) -> (east, north, up) metres from the site origin."""
    from pyproj import Transformer
    b = run["box"]
    x, y = b.to_map(ij[:, 0], ij[:, 1])
    lon, lat = Transformer.from_crs("EPSG:32118", "EPSG:4326", always_xy=True).transform(x, y)
    if k is None:
        up = np.full(len(x), STREET_UP_M)
    else:
        # Height above the local ground, as the viewer draws a flat city.
        from scipy.ndimage import map_coordinates
        g = map_coordinates(run["ground"] - run["meta"]["gmin_m"], [ij[:, 1], ij[:, 0]], order=1, mode="nearest")
        up = np.maximum((k - 0.5) * b.res - g, 1.0)
    e, n, u = enu(np.asarray(lon), np.asarray(lat), up, origin[0], origin[1])
    return np.stack([e, n, up], 1)


def write_lines(path, lines):
    lens = np.array([len(L) for L in lines], np.uint16)
    pts = np.concatenate(lines, 0) if lines else np.zeros((0, 3))
    q = np.clip(np.round(pts * 10), -32767, 32767).astype("<i2")
    with open(path, "wb") as f:
        f.write(np.array([len(lines), len(pts)], "<i4").tobytes())
        f.write(lens.astype("<u2").tobytes())
        f.write(q.tobytes())
    return os.path.getsize(path)


def main():
    import rasterio  # noqa: F401  (heat inputs)
    from .heat_export import heat_grid, write_png as _unused  # noqa: F401
    from . import export
    from .heat_inputs import US_FT  # noqa: F401

    runs = runs_available()
    if not runs:
        raise SystemExit("no wind runs in data/wind/runs")
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        if f.endswith((".png", ".bin")) or f == "wind.json":
            os.remove(os.path.join(OUT_DIR, f))
    version = datetime.now().strftime("%Y%m%d%H%M%S")
    x, y, layers, masks, meta = heat_points()
    grid = heat_grid(meta, masks)
    idx, bounds = grid.northup_index()
    painter = export.Painter(idx, bounds)
    fsl = grid.frame_slice
    origin = (meta["site"]["lon"], meta["site"]["lat"])
    cx, cy, open_m, disc = sheep_meadow(layers, masks, meta, x, y)
    lawn = open_lawn(layers, masks)
    print(f"anchor: open lawns, {lawn.sum() * 4 / 1e4:.1f} ha; Sheep Meadow at ({cx:.0f}, {cy:.0f}) "
          f"for reference", flush=True)
    # Study area for the lines: the heat display frame.
    r0, r1, c0, c1 = grid.frame
    fx, fy = x[r0:r1, c0:c1], y[r0:r1, c0:c1]
    rng = np.random.default_rng(7)
    refs = {}
    entries = []
    for m in sorted(runs, key=lambda m: (m["scenario"] != "2017", m["theta_from_deg"])):
        run = load_run(m)
        key = (m["season"], m["theta_from_deg"], m["res_m"], m["_name"].split("m", 1)[-1] if False else "")
        ref = lawn_reference(run, x, y, lawn)
        refs[(m["scenario"], m["season"], m["theta_from_deg"], m["res_m"], m["_name"].endswith("_small"))] = ref
        m["_sheep"] = reference_speed(run, cx, cy)
        entries.append((m, run, ref))
    out_runs = []
    for m, run, ref in entries:
        r17 = refs.get(("2017", m["season"], m["theta_from_deg"], m["res_m"], m["_name"].endswith("_small")), ref)
        b = run["box"]
        # Street map on the heat grid: ratio to Sheep Meadow (2017 run).
        k_map = resample(run["street_speed"], run["ok"], b, fx, fy) / r17
        full = np.full(x.shape, np.nan, np.float32)
        full[r0:r1, c0:c1] = k_map
        name = m["_name"]
        mask = masks["building"][fsl] | ~np.isfinite(full[fsl])
        png = painter.png(np.nan_to_num(full[fsl], nan=0.0), K_BREAKS, K_COLOURS, alpha=K_ALPHA, mask=mask)
        img_name = f"street_{name}.png"
        with open(os.path.join(OUT_DIR, img_name), "wb") as f:
            import base64
            f.write(base64.b64decode(png.split(",", 1)[1]))
        np.save(os.path.join(m["_dir"], "k_heatgrid.npy"), full)
        # Lines, in the box: the study area is the heat frame.
        fi, fj = b.from_map(fx[::4, ::4], fy[::4, ::4])
        roi = np.zeros((b.ny, b.nx), bool)
        ii = np.clip(np.round(fi).astype(int), 0, b.nx - 1)
        jj = np.clip(np.round(fj).astype(int), 0, b.ny - 1)
        roi[jj, ii] = True
        from scipy import ndimage
        roi = ndimage.binary_closing(roi, iterations=2)
        sl, dtau = street_lines(run, r17, roi, rng)
        st_enu = [to_enu(run, L, origin=origin) for L in sl]
        al, _ = air_lines(run, r17, roi, rng)
        air_enu = [to_enu(run, L[:, :2], k=L[:, 2], origin=origin) for L in al]
        # Random order, so drawing only the first part of a file still covers the park.
        st_enu = [st_enu[i] for i in rng.permutation(len(st_enu))]
        air_enu = [air_enu[i] for i in rng.permutation(len(air_enu))]
        s_bytes = write_lines(os.path.join(OUT_DIR, f"street_{name}.bin"), st_enu)
        a_bytes = write_lines(os.path.join(OUT_DIR, f"air_{name}.bin"), air_enu)
        # Seconds between vertices for 1 m/s at Sheep Meadow: divide by the wind.
        dtau_s = dtau * b.res * r17
        k_park = k_map[masks["report"][r0:r1, c0:c1]]
        out_runs.append({
            "name": name, "scenario": m["scenario"], "season": m["season"],
            "dir": m["theta_from_deg"], "res_m": m["res_m"],
            "img": f"wind/{img_name}?v={version}",
            "street": f"wind/street_{name}.bin?v={version}",
            "air": f"wind/air_{name}.bin?v={version}",
            "dtau_s_at_1ms": round(float(dtau_s), 4),
            "ref_lattice": round(ref, 5), "ref_2017_lattice": round(r17, 5),
            "sheep_meadow_ratio": round(m["_sheep"] / r17, 3),
            "park_k_median": round(float(np.nanmedian(k_park)), 3),
            "park_k_p90": round(float(np.nanpercentile(k_park, 90)), 3),
            "lines": {"street": len(st_enu), "air": len(air_enu), "bytes": s_bytes + a_bytes},
            "convergence": m.get("convergence_street_speed"),
        })
        print(f"{name}: open lawns {ref:.4f} (2017 {r17:.4f}), Sheep Meadow {m['_sheep'] / r17:.2f}, park median ratio "
              f"{out_runs[-1]['park_k_median']}, {len(st_enu)} street lines, {len(air_enu)} air lines, "
              f"{(s_bytes + a_bytes) / 1e6:.2f} MB", flush=True)
    w_, s_, e_, n_ = bounds
    bundle = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "origin": {"lon": origin[0], "lat": origin[1]},
        "bounds": {"west": w_, "south": s_, "east": e_, "north": n_},
        "anchor": {"name": "open lawns of the park", "ha": round(float(lawn.sum() * 4 / 1e4), 1),
                   "sheep_meadow": [cx, cy]},
        "legend": [{"min": K_BREAKS[i], "max": K_BREAKS[i + 1], "color": K_COLOURS[i]}
                   for i in range(len(K_COLOURS))],
        "street_up_m": STREET_UP_M,
        "spacing_k1_m": VERTEX_CELLS * 8.0,
        "runs": out_runs,
    }
    with open(os.path.join(OUT_DIR, "wind.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(bundle, f, separators=(",", ":"))
    size = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR))
    print(f"wrote {OUT_DIR}: {len(os.listdir(OUT_DIR))} files, {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
