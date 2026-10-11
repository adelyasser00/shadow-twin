"""
Data for the wind post: where the street got windier than in 2017, the air
that comes down Central Park Tower's windward face, and a vertical section
through the tower along the wind.

    .venv-heat/Scripts/python tools/post_wind_data.py

Uses the December hero pair (wind from 315 degrees, 3 flow-throughs each).

Windier than 2017
-----------------
Street-level wind of each year as a ratio to the open lawns (the 2017 run's
lawn value for both, as everywhere else), on the heat layer's 2 m grid in a
window around Billionaires' Row, each smoothed over 20 m (Gaussian sigma):
the scale from which the 6 m grid check agrees with 8 m. The change is
today / 2017 - 1.

Noise bar: the runs carry random averaging noise. It is measured where no
building changed nearby: open ground more than QUIET_M from every building
that changed since 2017 (there are 75 in the heat grid, several north of
59th Street), separately for the park and for the streets. Street canyons
are slower, so their ratios are noisier. The bar is three times the larger
of the two standard deviations of the change there. Only cells above the bar
get colour.

Lines down the face
-------------------
Seeds on a vertical rake 40 to 120 m upwind of the tower's windward face, 30
to 160 m up, across its width; followed through today's time-mean 3D wind.
Kept: lines that pass within 16 m of the windward face and reach 12 m above
the street within 150 m of the tower. Ranked by how far they fall, at most 12
kept; each is cut 30 vertices after it reaches the street.

Section
-------
The vertical plane along the wind (the box's x axis) through the tower's
middle: in-plane velocity of each year, the solid cells, and the tower's 2017
outline, for tools/post_wind_section.py.

Writes viewer/wind/post_windier.png (+ .json bounds), viewer/wind/post_lines_<year>.bin,
data/wind/post/section.npz and data/wind/post/post_data.json.
"""

from __future__ import annotations

import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pipeline.run_wind import RUNS_DIR, run_name  # noqa: E402
from pipeline.wind_export import (OUT_DIR, enu, load_run, resample, runs_available,  # noqa: E402
                                  write_lines)

CPT = (-73.98102, 40.766438)
TOWERS = {"Central Park Tower": CPT, "Steinway Tower": (-73.977562, 40.76496),
          "53 W 53rd": (-73.978155, 40.761827)}
WINDOW_M = (-700.0, 900.0, -800.0, 600.0)      # west, east, south, north of CPT, metres
SMOOTH_CELLS = 10
SIGMAS = 3.0
QUIET_M = 500.0
# Windier than 2017, percent: lower edges of the bins; first edge is replaced by the bar.
BINS = [None, 50, 100, 200]
# ColorBrewer Purples, 4 classes: sequential, colour-blind safe, clear of the amber.
COLOURS = ["#cbc9e2", "#9e9ac8", "#756bb1", "#54278f"]
POST_DIR = os.path.join(HERE, "data", "wind", "post")


def smooth(a, ok):
    from scipy.ndimage import gaussian_filter
    num = gaussian_filter(np.where(ok, a, 0.0), SMOOTH_CELLS)
    den = gaussian_filter(ok.astype(np.float32), SMOOTH_CELLS)
    return np.where(ok & (den > 0.2), num / np.maximum(den, 1e-6), np.nan)


def main():
    from pyproj import Transformer
    from PIL import Image
    from scipy import ndimage
    from pipeline.heat_inputs import load
    os.makedirs(POST_DIR, exist_ok=True)
    layers, masks, meta = load()
    t = meta["_transform"]
    H, W = layers["dem"].shape
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32118", always_xy=True).transform
    to_ll = Transformer.from_crs("EPSG:32118", "EPSG:4326", always_xy=True).transform
    runs = {m["scenario"]: m for m in runs_available() if m["_name"] in
            (run_name("today", "dec", 315.0, 8.0), run_name("2017", "dec", 315.0, 8.0))}
    r = {sc: load_run(m) for sc, m in runs.items()}
    with open(os.path.join(OUT_DIR, "wind.json"), encoding="utf-8") as f:
        wj = json.load(f)
    ref = next(x["ref_2017_lattice"] for x in wj["runs"] if x["name"] == run_name("2017", "dec", 315.0, 8.0))

    # ---------------------------------------------------------------- windier
    cx, cy = tf(*CPT)
    w_, e_, s_, n_ = WINDOW_M
    c0, c1 = int((cx + w_ - t.c) / t.a), int((cx + e_ - t.c) / t.a)
    r0, r1 = int((cy + n_ - t.f) / t.e), int((cy + s_ - t.f) / t.e)
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, H), min(c1, W)
    # The whole heat grid too, for the noise in quiet places.
    cc, rr = np.meshgrid(np.arange(W), np.arange(H))
    X = t.c + (cc + 0.5) * t.a
    Y = t.f + (rr + 0.5) * t.e
    k = {}
    for sc in ("2017", "today"):
        run = r[sc]
        a = resample(run["street_speed"], run["ok"], run["box"], X, Y) / ref
        k[sc] = smooth(a, np.isfinite(a) & ~masks["building"])
    with np.errstate(invalid="ignore", divide="ignore"):
        change = k["today"] / np.maximum(k["2017"], 0.02) - 1.0
    open_ground = ~masks["building"] & ~masks["water"]
    far = ndimage.distance_transform_edt(~masks["changed"]) * 2.0 > QUIET_M
    quiet_park = masks["park"] & open_ground & far
    quiet_streets = open_ground & ~masks["park"] & far
    print(f"quiet ground: park {quiet_park.sum() * 4 / 1e4:.1f} ha, streets {quiet_streets.sum() * 4 / 1e4:.1f} ha")
    sd_park = float(np.nanstd(change[quiet_park]))
    sd_streets = float(np.nanstd(change[quiet_streets]))
    bar = SIGMAS * max(sd_park, sd_streets)
    print(f"noise: park {sd_park * 100:.1f}%, streets {sd_streets * 100:.1f}%; bar {bar * 100:.1f}%")
    win = (slice(r0, r1), slice(c0, c1))
    ch = change[win] * 100.0
    edges = [bar * 100.0] + BINS[1:] + [1e9]
    idx = np.zeros(ch.shape, np.uint8)
    for i in range(len(COLOURS)):
        idx[(ch >= edges[i]) & (ch < edges[i + 1])] = i + 1
    idx[~open_ground[win] | ~np.isfinite(ch)] = 0
    rgba = np.zeros(ch.shape + (4,), np.uint8)
    for i, col in enumerate(COLOURS):
        sel = idx == i + 1
        rgba[sel] = [int(col[j:j + 2], 16) for j in (1, 3, 5)] + [235]
    Image.fromarray(rgba, "RGBA").save(os.path.join(OUT_DIR, "post_windier.png"), optimize=True)
    xs = [t.c + c0 * t.a, t.c + c1 * t.a]
    ys = [t.f + r0 * t.e, t.f + r1 * t.e]
    lons, lats = zip(*[to_ll(x, y) for x in xs for y in ys])
    bounds = {"west": min(lons), "east": max(lons), "south": min(lats), "north": max(lats)}
    # The biggest patch above the bar within 200 m of Central Park Tower: where
    # the "57th Street" label points.
    above = (idx > 0)
    lab_p, n_p = ndimage.label(above)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    dx = (t.c + (xx + 0.5) * t.a) - cx
    dy = (t.f + (yy + 0.5) * t.e) - cy
    near_cpt = np.hypot(dx, dy) < 200.0
    best, pool = 0, None
    for q in range(1, n_p + 1):
        m = (lab_p == q) & near_cpt
        if m.sum() > best:
            best = int(m.sum())
            # The patch cell with the strongest change, not the centroid (a patch can bend round a corner).
            sel = np.where(m, ch, -1e9)
            ii, jj = np.unravel_index(np.argmax(sel), sel.shape)
            pool = to_ll(t.c + (c0 + jj + 0.5) * t.a, t.f + (r0 + ii + 0.5) * t.e)
    with open(os.path.join(OUT_DIR, "post_windier.json"), "w", encoding="utf-8") as f:
        json.dump({"bounds": bounds, "bar_pct": round(bar * 100, 1), "edges_pct": [round(e, 1) for e in edges[:-1]],
                   "colours": COLOURS, "pool_lonlat": pool, "pool_m2": best * 4}, f, indent=1)
    # How much open ground near each tower clears the bar.
    lab, _ = ndimage.label(masks["building"])
    near = {}
    for nm, ll in TOWERS.items():
        x, y = tf(*ll)
        rr_, cc_ = int((y - t.f) / t.e), int((x - t.c) / t.a)
        fp = lab == lab[rr_, cc_]
        d = ndimage.distance_transform_edt(~fp) * 2.0
        ring = open_ground & (d <= 100) & np.isfinite(change)
        near[nm] = {"open_m2_within_100m": int(ring.sum() * 4),
                    "share_windier_above_bar": round(float((change[ring] >= bar).mean()), 3),
                    "share_calmer_below_bar": round(float((change[ring] <= -bar).mean()), 3)}
    print("near the towers:", json.dumps(near))

    # ------------------------------------------------------------- the lines
    from scipy.ndimage import map_coordinates
    out_lines = {}
    sect = {}
    for sc in ("today", "2017"):
        run = r[sc]
        b = run["box"]
        u = run["u"].astype(np.float32)
        nz = u.shape[1]
        solid = np.unpackbits(run["solid"], axis=-1)[..., :b.nx].astype(bool)[:nz]
        i_t, j_t = b.from_map(cx, cy)
        i_t, j_t = float(i_t), float(j_t)
        jt = int(round(j_t))
        bh = run["building"]
        # The tower's columns and its windward face along the box x axis (x runs
        # downwind), found once in today's run and reused for 2017: both runs
        # share the box, so the same plane and the same seeds compare the years.
        if sc == "today":
            lab2, _ = ndimage.label(bh > 100.0)
            tid = lab2[jt, int(round(i_t))]
            if tid == 0:
                # The label point can miss the tallest columns; take the biggest tall group nearby.
                win2 = lab2[jt - 6:jt + 7, int(i_t) - 6:int(i_t) + 7]
                ids, cnt = np.unique(win2[win2 > 0], return_counts=True)
                tid = int(ids[np.argmax(cnt)]) if ids.size else 0
            tj, ti = np.nonzero(lab2 == tid)
            face_i = ti.min()
            tower_cols = (tj, ti)
        else:
            tj, ti = tower_cols
            face_i = ti.min()
        g = run["ground"] - run["meta"]["gmin_m"]
        gk = int(run["kfirst"][jt, face_i - 2])
        seeds = []
        for di in (5, 8, 11, 15):
            for zm in np.arange(30.0, 170.0, 15.0):
                for jj in np.linspace(tj.min() - 1, tj.max() + 1, 7):
                    seeds.append((face_i - di, jj, gk + zm / b.res))
        s = np.array(seeds, float)
        fields = [np.ascontiguousarray(u[c]) for c in range(3)]
        sol = solid.astype(np.float32)
        dt = 0.5 / 0.06
        n_max = 900
        pts = np.full((len(s), n_max, 3), np.nan)
        pts[:, 0] = s
        cur = s.copy()
        alive = np.ones(len(s), bool)
        for v in range(1, n_max):
            co = [cur[:, 2], cur[:, 1], cur[:, 0]]
            a = np.stack([map_coordinates(f, co, order=1, mode="nearest") for f in fields], 1)
            mid = cur + 0.5 * dt * a
            co = [mid[:, 2], mid[:, 1], mid[:, 0]]
            a2 = np.stack([map_coordinates(f, co, order=1, mode="nearest") for f in fields], 1)
            sp = np.linalg.norm(a2, axis=1)
            stuck = map_coordinates(sol, co, order=1, mode="nearest") > 0.5
            alive &= (sp > 0.002) & ~stuck & (mid[:, 2] > 1) & (mid[:, 0] > 1) & (mid[:, 0] < b.nx - 2)
            cur = np.where(alive[:, None], cur + dt * a2, cur)
            pts[alive, v] = cur[alive]
        # Height above the local ground, metres.
        keep = []
        for q in range(len(s)):
            L = pts[q][np.isfinite(pts[q, :, 0])]
            if len(L) < 10:
                continue
            gl = map_coordinates(g, [L[:, 1], L[:, 0]], order=1, mode="nearest")
            hz = (L[:, 2] - 0.5) * b.res - gl
            dface = np.abs(L[:, 0] - face_i) * b.res
            lat_ok = (L[:, 1] >= tj.min() - 3) & (L[:, 1] <= tj.max() + 3)
            touch = np.nonzero((dface < 16) & lat_ok)[0]
            low = np.nonzero((hz < 12) & (np.hypot(L[:, 0] - i_t, L[:, 1] - j_t) * b.res < 150))[0]
            if not touch.size or not low.size or low[0] < touch[0]:
                continue
            cut = min(len(L), low[0] + 30)
            keep.append((float(hz[:touch[0] + 1].max() - hz[low[0]]), L[:cut], hz[:cut]))
        keep.sort(key=lambda x: -x[0])
        chosen = []
        for fall, L, hz in keep:
            # Spread: skip a line whose seed is within 2 cells of one already chosen.
            if any(np.linalg.norm(L[0] - c[0]) < 2.0 for c in chosen):
                continue
            chosen.append((L, hz))
            if len(chosen) >= 12:
                break
        print(f"{sc}: {len(keep)} lines come down the windward face; {len(chosen)} kept")
        enu_lines = []
        for L, hz in chosen:
            x, y = b.to_map(L[:, 0], L[:, 1])
            lon, lat = to_ll(x, y)
            e, nn, _ = enu(np.asarray(lon), np.asarray(lat), np.maximum(hz, 1.0), wj["origin"]["lon"], wj["origin"]["lat"])
            enu_lines.append(np.stack([e, nn, np.maximum(hz, 1.0)], 1))
        write_lines(os.path.join(OUT_DIR, f"post_lines_{sc}.bin"), enu_lines)
        out_lines[sc] = {"n": len(enu_lines), "candidates": len(keep),
                         "falls_m": [round(f, 1) for f, _, _ in keep[:len(chosen)]]}
        # Section along the wind through the tower's middle row.
        jm = int(round((tj.min() + tj.max()) / 2))
        i_lo, i_hi = max(0, face_i - 60), min(b.nx, face_i + 50)
        sect[sc] = {"ux": u[0, :, jm, i_lo:i_hi], "uz": u[2, :, jm, i_lo:i_hi],
                    "solid": solid[:, jm, i_lo:i_hi], "ground_k": run["kfirst"][jm, i_lo:i_hi],
                    "x0_m": (i_lo - face_i) * b.res, "face_i": face_i, "row": jm}
    # Free-stream reference for the section: today's run, 300 m above ground, upwind of the tower.
    run = r["today"]
    b = run["box"]
    jm = sect["today"]["row"]
    k300 = int(run["kfirst"][jm, 20] + 300 / b.res)
    u_ref = float(np.linalg.norm(run["u"][:, k300, jm, 10:30].astype(np.float32), axis=0).mean())
    np.savez_compressed(os.path.join(POST_DIR, "section.npz"), res=b.res, u_ref=u_ref,
                        **{f"{sc}_{k_}": v for sc in sect for k_, v in sect[sc].items()
                           if k_ not in ("face_i", "row")})
    with open(os.path.join(HERE, "data", "wind", "tower_heights_2017.json"), encoding="utf-8") as f:
        heights = json.load(f)
    info = {"noise_park_pct": round(sd_park * 100, 1), "noise_streets_pct": round(sd_streets * 100, 1),
            "bar_pct": round(bar * 100, 1), "near_towers": near, "lines": out_lines,
            "section_u_ref_lattice": u_ref, "heights": heights}
    with open(os.path.join(POST_DIR, "post_data.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(out_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
