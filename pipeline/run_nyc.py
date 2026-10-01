"""
Shadow study of a part of Manhattan, following NYC's CEQR method.

The whole of Central Park, today's skyline, one command each way:

    python -m pipeline.run_nyc --preset central-park --laz data/nyc --footprints data/nyc/footprints_crop.geojson --resource data/nyc/park.geojson
    python -m pipeline.run_nyc --preset central-park --laz data/nyc --footprints data/nyc/footprints_crop.geojson --resource data/nyc/park.geojson --burn-footprints

The first is the 2017 survey as flown, the second raises it to today's
recorded roof heights. Run build_viewer after each.

Method
------
New York City requires a shadow assessment for projects under City
Environmental Quality Review (CEQR Technical Manual, Chapter 8). It fixes the
method, and this runner follows it so the output is comparable with real
environmental impact statements:

  * Four analysis days: 21 March (also 21 September), 6 May (also 6 August),
    21 June and 21 December.
  * A timeframe window per day, 1.5 hours after sunrise to 1.5 hours before
    sunset, on Eastern Standard Time all year. Daylight saving is not used.
  * The subject of concern is sunlight-sensitive resources such as parks and
    open space, not streets or other buildings. Pass --resource to measure
    shadow on a park polygon specifically.

Street and facade statistics are still reported, because they matter for heat,
but they are not what CEQR regulates. Street means open ground outside
buildings and outside the resource, so the park is never counted twice.

Frame and buffer
----------------
The frame is what gets studied. The buffer around it only casts shadows in.
See grid.py. The central-park preset fits a frame to the park polygon along
the avenue grid, 28.9 degrees east of north, with 250 m of city on every side,
inside a 2 km buffer: enough for a 426 m tower's shadow at the lowest sun in
the December window. Beyond the last LiDAR tile, footprints stand in as
prisms so the buffer still casts.

Scenario
--------
--burn-footprints raises the 2017 surface to every footprint's current roof
height, across the whole domain, and that is "today".
--cap-height caps every building at a height, in metres, and re-solves. Run it
once with and once without, and the difference is the shadow those storeys
cast. That is the same "incremental shadow" CEQR asks for.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from . import export, nyc
from .geometry import run_day, sky_view_factor
from .solar import sun_position

DEFAULT_LAT = 40.7645
DEFAULT_LON = -73.9790

# CEQR Technical Manual, Chapter 8, analysis days and timeframe windows, EST.
CEQR_DAYS = [
    {"id": "dec", "label": "21 December", "short": "Dec 21",
     "date": "2026-12-21", "window": ("08:51", "14:53")},
    {"id": "mar", "label": "21 March / 21 September", "short": "Mar 21",
     "date": "2026-03-21", "window": ("07:36", "16:29")},
    {"id": "may", "label": "6 May / 6 August", "short": "May 6",
     "date": "2026-05-06", "window": ("06:27", "17:18")},
    {"id": "jun", "label": "21 June", "short": "Jun 21",
     "date": "2026-06-21", "window": ("05:57", "18:01")},
]
EST = -5.0
STEP_MIN = 30

EDGE_MARGIN_M = 80.0
GROWN_M = 6.0                 # two storeys: less than this is survey noise          # only used when there is no buffer
SVF_AZIMUTHS = 16
SVF_RADIUS_M = 250.0

# The published 700 m run, for putting new numbers on the same ground.
PUBLISHED_BOX = "40.7668,-73.9790,700"

PRESETS = {
    "central-park": {
        "bearing": 28.9, "frame_from_resource": True, "margin": 250.0,
        "buffer": 2000.0, "site_name": "Central Park, Manhattan",
        "resource_name": "Central Park", "compare_box": PUBLISHED_BOX,
        "camera": {"along": 0.16, "back_m": 1100.0, "height_m": 700.0, "pitch": -30.0},
    },
}


def frames_for(spec, lat, lon):
    """Sun positions every STEP_MIN minutes inside the CEQR window."""
    y, m, d = (int(x) for x in spec["date"].split("-"))
    h0, m0 = (int(x) for x in spec["window"][0].split(":"))
    h1, m1 = (int(x) for x in spec["window"][1].split(":"))
    start = h0 * 60 + m0
    end = h1 * 60 + m1
    t = int(math.ceil(start / STEP_MIN) * STEP_MIN)
    out = []
    while t <= end:
        local = datetime(y, m, d) + timedelta(minutes=t)
        utc = (local - timedelta(hours=EST)).replace(tzinfo=timezone.utc)
        p = sun_position(utc, lat, lon)
        if p.above_horizon:
            out.append((p, f"{t // 60:02d}:{t % 60:02d}"))
        t += STEP_MIN
    return out


def _grid_from_prep(prep, lat, lon):
    """A Grid describing the legacy raster path's north-up square."""
    from . import grid as G
    from rasterio.transform import Affine
    tf = prep["transform"] * Affine.scale(float(prep["resample_factor"]))
    unit_m = prep["cellsize"] / abs(tf.a)
    h, w = prep["surface"].shape
    conv = G.convergence_deg(prep["crs"], lon, lat)
    return G.Grid(crs=prep["crs"], unit_m=unit_m, res_m=prep["cellsize"],
                  bearing_deg=-conv, conv_deg=conv, x0=tf.c, y0=tf.f,
                  rows=h, cols=w, frame=(0, h, 0, w),
                  centre_lat=lat, centre_lon=lon)


def _camera(grid, cfg):
    """
    A starting view that looks up the avenues from behind the south end, the
    way the park is usually seen, rather than from the middle of the frame.
    """
    r0, r1, c0, c1 = grid.frame
    row = r1 - cfg["along"] * (r1 - r0)
    col = 0.5 * (c0 + c1)
    x, y = grid.rowcol_to_crs(row, col)
    lon, lat = grid.to_lonlat()(float(x), float(y))
    return {"lat": round(lat, 6), "lon": round(lon, 6),
            "heading": round(grid.bearing_deg, 2), "back_m": cfg["back_m"],
            "height_m": cfg["height_m"], "pitch": cfg["pitch"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laz", default=None)
    ap.add_argument("--dsm", default=None)
    ap.add_argument("--dem", default=None)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="central-park: whole park, avenue-aligned, 2 km buffer")
    ap.add_argument("--footprints", default=None,
                    help="NYC Building Footprints GeoJSON, ideally the crop")
    ap.add_argument("--resource", default=None,
                    help="GeoJSON of a sunlight-sensitive resource, such as a park, "
                         "to measure shadow on it the way CEQR does")
    ap.add_argument("--resource-name", default=None)
    ap.add_argument("--resource-where", default=None,
                    help="keep only resource features matching this, e.g. "
                         "\"Central Park\" or \"signname=Central Park\"")
    ap.add_argument("--burn-footprints", action="store_true",
                    help="scenario: raise the shadow surface to every footprint's "
                         "current roof height. The LiDAR is from May 2017; this adds "
                         "towers finished since, so the run shows today's shadows.")
    ap.add_argument("--burn-mode", choices=["grown", "all"], default="grown",
                    help="grown: raise only buildings that stand clearly taller than "
                         "the 2017 survey measured them. all: stamp every footprint at "
                         "its roof height, setbacks included, which is how the "
                         "September 2026 run was made")
    ap.add_argument("--cap-height", type=float, default=None,
                    help="scenario: cap every building at this many metres")
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--span", type=float, default=700.0,
                    help="square frame side, metres, when not fitting a resource")
    ap.add_argument("--length", type=float, default=None,
                    help="frame length along --bearing, metres")
    ap.add_argument("--width", type=float, default=None,
                    help="frame width across --bearing, metres")
    ap.add_argument("--bearing", type=float, default=None,
                    help="true bearing of the frame's long axis, degrees. "
                         "Manhattan's avenues run 28.9")
    ap.add_argument("--frame-from-resource", action="store_true", default=None,
                    help="fit the frame around the resource polygon")
    ap.add_argument("--margin", type=float, default=None,
                    help="metres of city kept around the resource, default 250")
    ap.add_argument("--buffer", type=float, default=None,
                    help="metres around the frame that cast shadows in but are "
                         "not reported. 0 reproduces the original runs")
    ap.add_argument("--compare-box", default=None,
                    help="lat,lon,span of an earlier square run, to report its "
                         "numbers on the same ground")
    ap.add_argument("--res", type=float, default=2.0)
    ap.add_argument("--site-name", default=None)
    ap.add_argument("--days", default="dec,mar,may,jun")
    ap.add_argument("--skip-svf", action="store_true")
    ap.add_argument("--skip-facades", action="store_true")
    ap.add_argument("--facade-levels", type=int, default=4)
    ap.add_argument("--max-buildings", type=int, default=2500)
    ap.add_argument("--context-min-height", type=float, default=100.0,
                    help="towers outside the frame at least this tall are drawn "
                         "as plain context")
    ap.add_argument("--context-ring", type=float, default=200.0,
                    help="metres of city drawn around the frame, plain, so it does "
                         "not end at a hard edge. 0 to turn off")
    ap.add_argument("--profile-step", type=float, default=250.0,
                    help="metres per band in the south-to-north park profile")
    ap.add_argument("--save-rasters", default=None,
                    help="write the frame's sun-hour rasters to this .npz, for "
                         "tools/compare_runs.py")
    ap.add_argument("--cache", default=None,
                    help="folder to cache the gridded LiDAR in, default data/cache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Presets fill in anything not given explicitly.
    preset = PRESETS.get(args.preset, {})
    def pick(name, fallback):
        v = getattr(args, name)
        if v is not None:
            return v
        return preset.get(name, fallback)
    bearing = pick("bearing", 0.0)
    frame_from_resource = bool(pick("frame_from_resource", False))
    margin = pick("margin", 250.0)
    buffer_m = pick("buffer", 0.0)
    site_name = pick("site_name", "Central Park South, Manhattan")
    resource_name = pick("resource_name", "the park")
    compare_box = pick("compare_box", None)
    camera_cfg = preset.get("camera")

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = None if args.no_cache else (args.cache or os.path.join(here, "data", "cache"))

    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    if not args.laz and not args.dsm:
        raise SystemExit("give me --laz (point clouds) or --dsm (rasters)")
    if frame_from_resource and not args.resource:
        raise SystemExit("--frame-from-resource needs --resource")

    # ---- resource, first, because it can decide the frame --------------
    res_geoms = None
    if args.resource:
        from . import resource as R
        log(f"reading {resource_name} from {args.resource}")
        res_geoms, res_names = R.load(args.resource, args.resource_where, log=log)
        if frame_from_resource:
            R.check_fit(res_geoms, res_names, args.resource_where)

    # ---- grid and surface ------------------------------------------------
    from . import grid as G
    if args.laz:
        from . import laz
        info = laz.probe(args.laz)
        crs, unit_m = info["crs"], info["unit_m"]
        grid = G.build(crs, unit_m, args.res, lat=args.lat, lon=args.lon,
                       span_m=args.span, length_m=args.length, width_m=args.width,
                       bearing_deg=bearing, buffer_m=buffer_m, margin_m=margin,
                       fit_geoms=res_geoms if frame_from_resource else None)
        L, Wd = grid.frame_size_m()
        log(f"frame {L:,.0f} m by {Wd:,.0f} m at {args.res} m, long axis "
            f"{grid.bearing_deg:.1f} deg, {grid.frame_shape[0]:,} x {grid.frame_shape[1]:,} cells")
        log(f"domain adds {grid.buffer_m:,.0f} m of buffer: {grid.rows:,} x {grid.cols:,} cells")
        if L * Wd > 40e6:
            log("  warning: a frame this big will be slow and heavy in the browser")
        log("rasterising point cloud tiles")
        prep = laz.rasterise(args.laz, grid, log=log, cache_dir=cache_dir,
                             derive_buildings=not args.footprints)
    else:
        if frame_from_resource or buffer_m or bearing:
            raise SystemExit("the GeoTIFF path only does the original north-up "
                             "square. Use --laz for frames, buffers and rotation.")
        log("preparing rasters")
        prep = nyc.prepare(args.dsm, args.lat, args.lon, args.span, args.res,
                           dem_path=args.dem, log=log)
        prep.setdefault("water", np.zeros(prep["surface"].shape, dtype=bool))
        grid = _grid_from_prep(prep, args.lat, args.lon)
        prep["grid"] = grid

    cellsize = grid.res_m
    shape = (grid.rows, grid.cols)
    fsl = grid.frame_slice
    fshape = grid.frame_shape
    nodata = prep.get("nodata", np.zeros(shape, dtype=bool))
    surface = prep["surface"]
    ground = prep["ground"]

    # ---- buildings -------------------------------------------------------
    labels, n_labels, recs, polys = None, 0, None, None
    h_label = None
    building_source = "derived from LiDAR returns"
    scenario = None
    if args.footprints:
        from . import nyc_footprints as NF
        log("reading NYC Building Footprints")
        recs = NF.read(args.footprints, grid.bounds_lonlat((0, grid.rows, 0, grid.cols),
                                                            pad_deg=0.002), log=log)
        labels, polys = NF.rasterise(recs, grid, log=log)
        n_labels = len(recs)
        h_label, building_source = NF.scenario_heights(
            recs, labels, prep["heights"], nodata, today=args.burn_footprints, log=log)

        # Casters beyond the survey: footprint prisms on no-data ground.
        stand = nodata & (labels > 0)
        if stand.any():
            surface = np.where(stand, ground + h_label[labels], surface).astype(np.float32)
            log(f"  {int((stand & (h_label[labels] > 0)).sum()):,} buffer cells beyond "
                f"the survey stand as footprint prisms")

        if args.burn_footprints:
            # A footprint's height_roof is its highest roof. Stamping that over
            # the whole footprint also raises every setback, terrace and lower
            # wing of a building that has not changed since 2017, which adds
            # mass that is not there. "grown" only raises buildings whose
            # recorded roof stands clearly above the tallest thing the survey
            # measured inside them: built or topped out since 2017.
            target = ground + h_label[labels]
            raise_all = (labels > 0) & (target > surface + 3.0)
            top17 = NF.label_quantile(prep["heights"], labels, n_labels, 0.99)
            fp_h = np.array([0.0] + [r["height_m"] or 0.0 for r in recs], dtype=np.float32)
            grown = ~np.isfinite(top17) | (fp_h > np.nan_to_num(top17, nan=0.0) + GROWN_M)
            grown[0] = False
            # Buildings beyond the survey already stand as prisms; they are not
            # "grown", there was simply nothing measured there.
            grown &= ~(NF.label_fraction(nodata, labels, n_labels) > 0.5)
            grown &= np.bincount(labels.ravel(), minlength=n_labels + 1) > 0
            raise_grown = raise_all & grown[labels]
            raise_ = raise_all if args.burn_mode == "all" else raise_grown
            surface = np.where(raise_, target, surface).astype(np.float32)
            scenario = {"type": "today", "burn_mode": args.burn_mode,
                        "cells_raised": int(raise_.sum()),
                        "cells_raised_in_frame": int(raise_[fsl].sum()),
                        "buildings_grown": int(grown[1:].sum())}
            log(f"SCENARIO: {int(grown[1:].sum()):,} buildings stand over {GROWN_M:.0f} m "
                f"above what the 2017 survey measured in them")
            log(f"  burn mode '{args.burn_mode}': raised {int(raise_.sum()):,} cells, "
                f"{int(raise_[fsl].sum()):,} in the frame. Stamping every footprint "
                f"would raise {int(raise_all[fsl].sum()):,} frame cells, only "
                f"{int(raise_grown[fsl].sum()):,} of them in buildings that grew.")
        built = (labels > 0) & (h_label[labels] >= 3.0)
        heights = h_label[labels]
        building_source = f"NYC Building Footprints (NYC OTI) outlines; heights: {building_source}"
    else:
        if args.burn_footprints:
            raise SystemExit("--burn-footprints needs --footprints")
        built = prep["built"]
        heights = prep["heights"]

    if args.cap_height is not None:
        cap = float(args.cap_height)
        tall = built & ((surface - ground) > cap)
        surface = np.where(tall, ground + cap, surface).astype(np.float32)
        heights = np.minimum(heights, cap)
        if h_label is not None:
            capped = int((h_label > cap).sum())
            h_label = np.minimum(h_label, cap)
        else:
            capped = 0
        scenario = {"type": "height cap", "cap_m": cap, "buildings_capped": capped}
        log(f"SCENARIO: every building capped at {cap:.0f} m "
            f"({capped} buildings, {int(tall.sum()):,} surface cells lowered)")

    if scenario:
        now = (surface - ground)[fsl][~nodata[fsl]]
        if now.size:
            log(f"tallest object in the frame after the scenario {now.max():.1f} m "
                f"({now.max()*3.28084:.0f} ft)")

    # ---- masks, all on the frame -----------------------------------------
    edge_m = EDGE_MARGIN_M if grid.buffer_m < 1.0 else 0.0
    blank = nodata[fsl].copy()
    if edge_m:
        m = max(1, int(edge_m / cellsize))
        edge = np.ones(fshape, dtype=bool)
        edge[m:-m, m:-m] = False
        blank |= edge
    resource = None
    if res_geoms is not None:
        from . import resource as R
        resource = R.rasterise(res_geoms, grid)[fsl] & ~blank
        log(f"{resource_name}: {int(resource.sum()):,} cells "
            f"({resource.sum()*cellsize*cellsize/1e4:.1f} ha) inside the frame")
        if not resource.any():
            log("  warning: the resource polygon does not overlap the frame")
            resource = None
    built_f = built[fsl]
    street = ~(built_f | blank)
    if resource is not None:
        street &= ~resource
    log(f"street statistics over {int(street.sum()):,} open cells outside "
        f"buildings{' and the park' if resource is not None else ''}")

    compare = None
    if compare_box and resource is not None:
        from . import resource as R
        lat_c, lon_c, span_c = (float(v) for v in compare_box.split(","))
        compare = R.window_mask(grid, lat_c, lon_c, span_c, EDGE_MARGIN_M) & resource
        if compare.any():
            log(f"published frame ({span_c:.0f} m at {lat_c}, {lon_c}): "
                f"{compare.sum()*cellsize*cellsize/1e4:.1f} ha of {resource_name} "
                "on the same ground as before")
        else:
            compare = None

    # Rows run south to north through the park on the avenue-aligned grid, so
    # a profile by distance from the park's south edge is just rows.
    profile_rows = None
    if resource is not None:
        rows_any = np.nonzero(resource.any(axis=1))[0]
        south_row = int(rows_any.max())
        dist = (south_row - np.arange(fshape[0]) + 0.5) * cellsize
        profile_rows = dist

    painter_idx, painter_bounds = grid.northup_index()
    painter = export.Painter(painter_idx, painter_bounds)
    log(f"overlay images {painter_idx.shape[1]} x {painter_idx.shape[0]} px, north-up")

    # ---- solve ---------------------------------------------------------
    backend = None
    if args.gpu:
        from .geometry_torch import TorchSweeper, available, device_name
        if available():
            backend = TorchSweeper(surface)
            log(f"GPU sweep on {device_name()}")
        else:
            log("--gpu given but no CUDA device, staying on the CPU")

    step_h = STEP_MIN / 60.0
    wanted = [d for d in CEQR_DAYS if d["id"] in args.days.split(",")]
    days_out, solved, rasters = [], [], {}
    for spec in wanted:
        fr = frames_for(spec, grid.centre_lat, grid.centre_lon)
        pos = [p for p, _ in fr]
        lab = [l for _, l in fr]
        log(f"{spec['label']}: CEQR window {spec['window'][0]}-{spec['window'][1]} EST, "
            f"{len(fr)} frames, peak sun {max(p.altitude for p in pos):.1f} deg")
        res = run_day(surface, cellsize, pos, lab, window=grid.frame,
                      bearing_deg=grid.bearing_deg, compute_svf=False, backend=backend)
        sun_h = res.sun_hours * step_h
        window_h = len(fr) * step_h
        solved.append((spec, pos, lab))
        rasters[spec["id"]] = sun_h.astype(np.float32)

        frames = []
        for i, l in enumerate(lab):
            f = {"hour": l, "altitude": round(res.altitudes[i], 2),
                 "azimuth": round(res.azimuths[i], 2),
                 "sunlit_fraction": float(res.sunlit[i][street].mean()),
                 "png": painter.png(res.sunlit[i], [0.0, 0.5, 1.01],
                                    export.SHADOW_COLORS, alpha=205, mask=blank)}
            if resource is not None:
                f["resource_sunlit"] = float(res.sunlit[i][resource].mean())
            frames.append(f)

        breaks = [round(window_h * i / 8, 3) for i in range(8)] + [window_h + 0.01]
        layer = export.layer(f"sunhours_{spec['id']}", "Hours of direct sun",
                             f"hours inside the CEQR window, {spec['label']}",
                             sun_h, breaks, export.SUNHOUR_COLORS,
                             mask=blank, stats_mask=~street, painter=painter)
        s = sun_h[street]
        stats = {"median_sun_hours": float(np.median(s)),
                 "mean_sun_hours": float(s.mean()),
                 "frac_street_no_sun": float((s < 0.25).mean()),
                 "frac_street_under_2h": float((s < 2).mean()),
                 "frac_street_over_half_day": float((s > window_h / 2).mean())}
        if resource is not None:
            r = sun_h[resource]
            stats["resource"] = {
                "name": resource_name,
                "area_ha": round(float(resource.sum() * cellsize * cellsize / 1e4), 2),
                "median_sun_hours": float(np.median(r)),
                "sunlit_area_time": float(res.sunlit[:, resource].mean()),
                "frac_under_2h": float((r < 2).mean()),
                "frac_no_sun": float((r < 0.25).mean()),
                "ha_no_sun": round(float((r < 0.25).sum() * cellsize * cellsize / 1e4), 2),
            }
            # South to north profile.
            prof = []
            step = args.profile_step
            top = float(profile_rows[resource.any(axis=1)].max())
            k = 0
            while k * step < top:
                band = (profile_rows >= k * step) & (profile_rows < (k + 1) * step)
                cells = resource & band[:, None]
                if cells.sum() * cellsize * cellsize >= 5000:     # at least half a hectare
                    rb = sun_h[cells]
                    prof.append({"from_m": round(k * step), "to_m": round((k + 1) * step),
                                 "area_ha": round(float(cells.sum() * cellsize**2 / 1e4), 2),
                                 "frac_no_sun": round(float((rb < 0.25).mean()), 4),
                                 "median_sun_hours": round(float(np.median(rb)), 2)})
                k += 1
            stats["resource"]["profile"] = prof
            if compare is not None:
                rc = sun_h[compare]
                stats["resource"]["published_frame"] = {
                    "area_ha": round(float(compare.sum() * cellsize**2 / 1e4), 2),
                    "frac_no_sun": float((rc < 0.25).mean()),
                    "median_sun_hours": float(np.median(rc))}
            log(f"  {resource_name}: median {np.median(r):.1f} h of sun, "
                f"{(r < 0.25).mean()*100:.1f}% never in sun inside the window "
                f"({(r < 0.25).sum()*cellsize*cellsize/1e4:.1f} ha)")
        log(f"  street: median {np.median(s):.1f} h, "
            f"{(s < 0.25).mean()*100:.1f}% never in sun inside the window")
        days_out.append({"id": spec["id"], "label": spec["label"], "short": spec["short"],
                         "date": spec["date"], "window": list(spec["window"]),
                         "step_h": step_h, "daylight_hours_modelled": window_h,
                         "facade_hours_max": window_h,
                         "peak_altitude": round(max(res.altitudes), 2),
                         "hours": frames, "sunhours": layer, "stats": stats})

    svf_layer = None
    if not args.skip_svf:
        log("sky view factor")
        svf = sky_view_factor(surface, cellsize, SVF_AZIMUTHS, SVF_RADIUS_M, window=grid.frame)
        svf_layer = export.layer("svf", "Sky view factor", "fraction of sky visible",
                                 svf, [0.0, 0.2, 0.32, 0.44, 0.56, 0.68, 0.8, 0.92, 1.0001],
                                 export.SVF_COLORS, mask=blank, stats_mask=~street,
                                 painter=painter)

    # ---- the buildings the browser draws ---------------------------------
    context = []
    if args.footprints:
        from . import nyc_footprints as NF
        buildings, context = NF.buildings_for_viewer(
            recs, polys, h_label, grid, context_min_m=args.context_min_height,
            ring_m=args.context_ring, log=log)
    else:
        from .footprints import from_mask
        buildings, lab_l, n_labels = from_mask(
            built, heights, grid.transform, grid.crs, 1,
            max_features=args.max_buildings, log=log, return_labels=True)
        labels = lab_l
    log(f"{len(buildings)} buildings")

    # ---- facades -------------------------------------------------------
    has_facades = False
    if not args.skip_facades and buildings and labels is not None:
        from . import facade
        log("solving sun on building walls")
        keep = np.zeros(n_labels + 1, dtype=bool)
        keep[[b["id"] for b in buildings]] = True
        pts = facade.wall_samples(built, heights, labels, cellsize,
                                  levels=args.facade_levels, log=log,
                                  window=grid.frame, keep_labels=keep)
        if pts is not None:
            for spec, pos, lab in solved:
                t_f = time.time()
                r = facade.solve(pts, surface, ground, cellsize, pos, lab, log=None,
                                 labels_grid=labels, bearing_deg=grid.bearing_deg)
                r["sun_hours"] = r["sun_hours"] * step_h
                r["gain_wh_m2"] = r["gain_wh_m2"] * step_h
                rolled = facade.per_building(pts, r, n_labels, bearing_deg=grid.bearing_deg)
                for b in buildings:
                    i = b["id"]
                    rec = {"h": round(float(rolled["best"]["sun_hours"][i]), 2),
                           "lo": round(float(rolled["low"]["sun_hours"][i]), 2),
                           "hi": round(float(rolled["high"]["sun_hours"][i]), 2),
                           "kwh": round(float(rolled["best"]["gain_wh_m2"][i]) / 1000.0, 3)}
                    for o, _ in facade.ORIENTS:
                        rec[o] = [round(float(rolled[o]["sun_hours"][i]), 1),
                                  round(float(rolled[o]["gain_wh_m2"][i]) / 1000.0, 2)]
                    b.setdefault("sun", {})[spec["id"]] = rec
                v = [b["sun"][spec["id"]]["h"] for b in buildings]
                log(f"  {spec['short']}: sunniest wall median {np.median(v):.1f} h, "
                    f"max {max(v):.1f} h  ({time.time() - t_f:.0f} s)")
            has_facades = True
    for b in buildings + context:
        b.pop("id", None)

    fh_arr = (surface - ground)[fsl]
    ok = ~blank
    tallest = float(fh_arr[ok].max()) if ok.any() else float(fh_arr.max())
    w_, s_, e_, n_ = painter.bounds
    Lm, Wm = grid.frame_size_m()
    site = {"name": site_name, "centre_lat": grid.centre_lat, "centre_lon": grid.centre_lon,
            "bounds": {"west": w_, "south": s_, "east": e_, "north": n_},
            "span_m": round(max(Lm, Wm)), "cellsize_m": round(cellsize, 3),
            "grid": list(fshape), "edge_margin_m": edge_m,
            "frame": grid.describe(),
            "tallest_m": round(tallest, 1), "tallest_ft": round(tallest * 3.28084),
            "nodata_frac": round(float(nodata[fsl].mean()), 3),
            "water_frac": round(float(prep.get("water", np.zeros(shape, bool))[fsl].mean()), 3)}
    if camera_cfg:
        site["camera"] = _camera(grid, camera_cfg)
    bundle = {
        "mode": "REAL", "warning": "",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "site": site,
        "model": {"solar_algorithm": "NOAA / Meeus low precision",
                  "shadow_method": "Ratti and Richens shift-and-subtract sweep, whole-cell stepping",
                  "svf_method": (f"horizon scan, {SVF_AZIMUTHS} azimuths, {SVF_RADIUS_M:.0f} m radius, "
                                 "Steyn cos^2 weighting") if svf_layer else None,
                  "surface_source": prep.get("source_desc", "NYC 2017 airborne LiDAR"),
                  "ground_source": prep["ground_source"],
                  "building_source": building_source,
                  "native_resolution_m": round(prep["native_res_m"], 3),
                  "resample": "none" if prep["resample_factor"] == 1 else
                              f"max pooling by {prep['resample_factor']}",
                  "time_basis": ("CEQR Technical Manual Chapter 8: four analysis days, "
                                 "1.5 h after sunrise to 1.5 h before sunset, Eastern "
                                 f"Standard Time year round, sampled every {STEP_MIN} min"),
                  "buffer_m": round(grid.buffer_m),
                  "buffer_desc": (
                      f"Buildings up to {grid.buffer_m/1000:.1f} km outside the frame cast "
                      "shadows into it: the LiDAR surface where the survey reaches, "
                      "footprint prisms at recorded roof heights beyond it"
                      + (", counting only buildings built before 2017"
                         if args.footprints and not args.burn_footprints else "")
                      + ".") if grid.buffer_m >= 1.0 else None,
                  "scenario": scenario,
                  "does_not_compute": ["air temperature", "surface temperature",
                                       "mean radiant temperature", "wind",
                                       "thermal comfort indices such as UTCI or PET",
                                       "cloud"]},
        "days": days_out, "svf": svf_layer, "buildings": buildings + context,
        "has_facades": has_facades,
    }
    path = args.out or os.path.join(here, "viewer", "data.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    size = export.write_bundle(path, bundle)
    log(f"wrote {path}  {size/1024/1024:.1f} MB")

    if args.save_rasters:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_rasters)), exist_ok=True)
        extra = {}
        if resource is not None:
            extra["resource"] = resource
        if compare is not None:
            extra["published"] = compare
        np.savez_compressed(
            args.save_rasters, street=street, blank=blank,
            **{f"sun_{k}": v for k, v in rasters.items()}, **extra,
            meta=np.array(json.dumps({
                "scenario": "today" if args.burn_footprints else "2017",
                "cellsize": cellsize, "frame": grid.describe(),
                "profile_rows": profile_rows.tolist() if profile_rows is not None else None,
                "windows_h": {d["id"]: d["daylight_hours_modelled"] for d in days_out},
                "resource_name": resource_name})))
        log(f"saved sun-hour rasters to {args.save_rasters}")

    print()
    print("  CEQR day        street never sun   " +
          (f"{resource_name} median sun   never sun" if resource is not None else ""))
    for d in days_out:
        st = d["stats"]
        line = f"  {d['short']:<14}  {st['frac_street_no_sun']*100:6.1f}%          "
        if "resource" in st:
            r = st["resource"]
            line += f"  {r['median_sun_hours']:5.1f} h            {r['frac_no_sun']*100:5.1f}%"
        print(line)
    if compare is not None:
        print(f"\n  On the published 700 m frame's ground, {resource_name} never in sun:")
        for d in days_out:
            pf = d["stats"]["resource"]["published_frame"]
            print(f"    {d['short']:<8} {pf['frac_no_sun']*100:5.1f}%")
    if resource is not None and days_out and days_out[0]["stats"]["resource"]["profile"]:
        d0 = days_out[0]
        print(f"\n  {resource_name} never in sun on {d0['label']}, by distance from its south edge:")
        for p in d0["stats"]["resource"]["profile"]:
            bar = "#" * int(round(p["frac_no_sun"] * 40))
            print(f"    {p['from_m']:>5}-{p['to_m']:<5} m  {p['frac_no_sun']*100:5.1f}%  {bar}")
    if scenario and scenario["type"] == "height cap":
        print(f"\n  SCENARIO: capped at {scenario['cap_m']:.0f} m. "
              "Compare against a run without --cap-height.")
    elif scenario and scenario["type"] == "today":
        print("\n  SCENARIO: today's skyline burned in. "
              "Compare against the plain 2017 run.")
    print(f"\n  total {time.time() - t0:.0f} s")
    print("\n  next: python -m pipeline.build_viewer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
