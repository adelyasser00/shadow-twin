"""
Shadow study for a block of Manhattan, following NYC's CEQR method.

    python -m pipeline.run_nyc --laz data/nyc --footprints data/nyc/footprints.geojson
    python -m pipeline.build_viewer

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
but they are not what CEQR regulates.

Scenario
--------
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
SVF_AZIMUTHS = 16
SVF_RADIUS_M = 250.0


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


def rasterise_polygons(path, crs, transform, shape):
    from pyproj import Transformer
    from rasterio import features
    from shapely.geometry import shape as shp
    from shapely.ops import transform as tf

    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"] if "features" in gj else gj
    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    shapes = []
    for ft in feats:
        g = ft.get("geometry")
        if g:
            shapes.append((tf(to_grid, shp(g)), 1))
    if not shapes:
        return np.zeros(shape, dtype=bool)
    return features.rasterize(shapes, out_shape=shape, transform=transform,
                              fill=0, dtype="uint8").astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laz", default=None)
    ap.add_argument("--dsm", default=None)
    ap.add_argument("--dem", default=None)
    ap.add_argument("--footprints", default=None,
                    help="NYC Building Footprints GeoJSON")
    ap.add_argument("--resource", default=None,
                    help="GeoJSON of a sunlight-sensitive resource, such as a park, "
                         "to measure shadow on it the way CEQR does")
    ap.add_argument("--resource-name", default="the park")
    ap.add_argument("--burn-footprints", action="store_true",
                    help="scenario: raise the shadow surface to every footprint's "
                         "current roof height. The LiDAR is from May 2017; this adds "
                         "towers finished since, so the run shows today's shadows.")
    ap.add_argument("--cap-height", type=float, default=None,
                    help="scenario: cap every building at this many metres")
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--span", type=float, default=700.0)
    ap.add_argument("--res", type=float, default=2.0)
    ap.add_argument("--site-name", default="Central Park South, Manhattan")
    ap.add_argument("--days", default="dec,mar,may,jun")
    ap.add_argument("--skip-svf", action="store_true")
    ap.add_argument("--skip-facades", action="store_true")
    ap.add_argument("--facade-levels", type=int, default=4)
    ap.add_argument("--max-buildings", type=int, default=2500)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    if not args.laz and not args.dsm:
        raise SystemExit("give me --laz (point clouds) or --dsm (rasters)")

    # ---- surface -------------------------------------------------------
    if args.laz:
        from . import laz
        log("rasterising point cloud tiles")
        prep = laz.rasterise(args.laz, args.lat, args.lon, args.span, args.res, log=log)
    else:
        log("preparing rasters")
        prep = nyc.prepare(args.dsm, args.lat, args.lon, args.span, args.res,
                           dem_path=args.dem, log=log)
    shape = prep["surface"].shape
    cellsize = prep["cellsize"]
    nodata = prep.get("nodata", np.zeros(shape, dtype=bool))
    from rasterio.transform import Affine
    grid_tf = prep["transform"] * Affine.scale(float(prep["resample_factor"]))

    # ---- buildings -----------------------------------------------------
    buildings, labels, n_labels = [], None, 0
    building_source = "derived from LiDAR returns"
    if args.footprints:
        from . import nyc_footprints
        log("reading NYC Building Footprints")
        buildings, labels, n_labels, fp_h, fp_built = nyc_footprints.load(
            args.footprints, prep["bounds"], prep["crs"], grid_tf, shape, log=log)
        prep["heights"] = fp_h
        prep["built"] = fp_built
        building_source = "NYC Building Footprints (NYC OTI), surveyed outlines and roof heights"

    # ---- scenario ------------------------------------------------------
    scenario = None
    if args.burn_footprints:
        if not args.footprints:
            raise SystemExit("--burn-footprints needs --footprints")
        target = prep["ground"] + prep["heights"]
        raise_ = prep["built"] & (target > prep["surface"] + 3.0)
        prep["surface"] = np.where(raise_, target, prep["surface"]).astype(np.float32)
        gained = (target - prep["surface"])[raise_]
        scenario = {"type": "today", "cells_raised": int(raise_.sum())}
        log(f"SCENARIO: surface raised to current footprint heights in "
            f"{int(raise_.sum()):,} cells (buildings newer than the 2017 survey)")
    if args.cap_height is not None:
        cap = float(args.cap_height)
        tall = prep["built"] & ((prep["surface"] - prep["ground"]) > cap)
        n_cells = int(tall.sum())
        prep["surface"] = np.where(tall, prep["ground"] + cap, prep["surface"]).astype(np.float32)
        prep["heights"] = np.minimum(prep["heights"], cap)
        capped = 0
        for b in buildings:
            if b["height"] > cap:
                b["height"] = round(cap, 1)
                capped += 1
        scenario = {"type": "height cap", "cap_m": cap, "buildings_capped": capped}
        log(f"SCENARIO: every building capped at {cap:.0f} m "
            f"({capped} buildings, {n_cells:,} surface cells lowered)")

    # ---- masks ---------------------------------------------------------
    m = max(1, int(EDGE_MARGIN_M / cellsize))
    edge = np.ones(shape, dtype=bool)
    edge[m:-m, m:-m] = False
    blank = edge | nodata
    street = ~(prep["built"] | blank)
    resource = None
    if args.resource:
        resource = rasterise_polygons(args.resource, prep["crs"], grid_tf, shape) & ~blank
        log(f"{args.resource_name}: {int(resource.sum()):,} cells "
            f"({resource.sum()*cellsize*cellsize/1e4:.1f} ha) inside the frame")
        if not resource.any():
            log("  warning: the resource polygon does not overlap the frame")
            resource = None
    log(f"street statistics over {int(street.sum()):,} open cells")

    # ---- solve ---------------------------------------------------------
    step_h = STEP_MIN / 60.0
    wanted = [d for d in CEQR_DAYS if d["id"] in args.days.split(",")]
    days_out, solved = [], []
    for spec in wanted:
        fr = frames_for(spec, args.lat, args.lon)
        pos = [p for p, _ in fr]
        lab = [l for _, l in fr]
        log(f"{spec['label']}: CEQR window {spec['window'][0]}-{spec['window'][1]} EST, "
            f"{len(fr)} frames, peak sun {max(p.altitude for p in pos):.1f} deg")
        res = run_day(prep["surface"], cellsize, pos, lab,
                      n_azimuths=SVF_AZIMUTHS, svf_radius_m=SVF_RADIUS_M)
        sun_h = res.sun_hours * step_h
        window_h = len(fr) * step_h
        solved.append((spec, pos, lab))

        frames = []
        for i, l in enumerate(lab):
            f = {"hour": l, "altitude": round(res.altitudes[i], 2),
                 "azimuth": round(res.azimuths[i], 2),
                 "sunlit_fraction": float(res.sunlit[i][street].mean()),
                 "png": export.to_data_uri(export.quantize(
                     res.sunlit[i], [0.0, 0.5, 1.01], export.SHADOW_COLORS,
                     alpha=205, mask=blank)[0])}
            if resource is not None:
                f["resource_sunlit"] = float(res.sunlit[i][resource].mean())
            frames.append(f)

        breaks = [round(window_h * i / 8, 3) for i in range(8)] + [window_h + 0.01]
        layer = export.layer(f"sunhours_{spec['id']}", "Hours of direct sun",
                             f"hours inside the CEQR window, {spec['label']}",
                             sun_h, breaks, export.SUNHOUR_COLORS,
                             mask=blank, stats_mask=~street)
        s = sun_h[street]
        stats = {"median_sun_hours": float(np.median(s)),
                 "mean_sun_hours": float(s.mean()),
                 "frac_street_no_sun": float((s < 0.25).mean()),
                 "frac_street_under_2h": float((s < 2).mean()),
                 "frac_street_over_half_day": float((s > window_h / 2).mean())}
        if resource is not None:
            r = sun_h[resource]
            stats["resource"] = {
                "name": args.resource_name,
                "area_ha": round(float(resource.sum() * cellsize * cellsize / 1e4), 2),
                "median_sun_hours": float(np.median(r)),
                "sunlit_area_time": float(res.sunlit[:, resource].mean()),
                "frac_under_2h": float((r < 2).mean()),
                "frac_no_sun": float((r < 0.25).mean()),
            }
            log(f"  {args.resource_name}: median {np.median(r):.1f} h of sun, "
                f"{(r < 0.25).mean()*100:.1f}% never in sun inside the window")
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
        svf = sky_view_factor(prep["surface"], cellsize, SVF_AZIMUTHS, SVF_RADIUS_M)
        svf_layer = export.layer("svf", "Sky view factor", "fraction of sky visible",
                                 svf, [0.0, 0.2, 0.32, 0.44, 0.56, 0.68, 0.8, 0.92, 1.0001],
                                 export.SVF_COLORS, mask=blank, stats_mask=~street)

    # ---- buildings from LiDAR if no footprints -------------------------
    if not args.footprints:
        from .footprints import from_mask
        buildings, labels, n_labels = from_mask(
            prep["built"], prep["heights"], prep["transform"], prep["crs"],
            prep["resample_factor"], max_features=args.max_buildings, log=log,
            return_labels=True)
    log(f"{len(buildings)} buildings")

    # ---- facades -------------------------------------------------------
    has_facades = False
    if not args.skip_facades and buildings and labels is not None:
        from . import facade
        log("solving sun on building walls")
        pts = facade.wall_samples(prep["built"], prep["heights"], labels, cellsize,
                                  levels=args.facade_levels, log=log)
        if pts is not None:
            for spec, pos, lab in solved:
                r = facade.solve(pts, prep["surface"], prep["ground"], cellsize,
                                 pos, lab, log=None, labels_grid=labels)
                r["sun_hours"] = r["sun_hours"] * step_h
                r["gain_wh_m2"] = r["gain_wh_m2"] * step_h
                rolled = facade.per_building(pts, r, n_labels)
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
                    f"max {max(v):.1f} h")
            has_facades = True
    for b in buildings:
        b.pop("id", None)

    heights = prep["heights"]
    b = prep["bounds"]
    bundle = {
        "mode": "REAL", "warning": "",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "site": {"name": args.site_name, "centre_lat": args.lat, "centre_lon": args.lon,
                 "bounds": {"west": b[0], "south": b[1], "east": b[2], "north": b[3]},
                 "span_m": round(max(shape) * cellsize), "cellsize_m": round(cellsize, 3),
                 "grid": list(shape), "edge_margin_m": EDGE_MARGIN_M,
                 "tallest_m": round(float(heights.max()), 1),
                 "tallest_ft": round(float(heights.max()) * 3.28084),
                 "nodata_frac": round(float(nodata.mean()), 3)},
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
                  "scenario": scenario,
                  "does_not_compute": ["air temperature", "surface temperature",
                                       "mean radiant temperature", "wind",
                                       "thermal comfort indices such as UTCI or PET",
                                       "cloud"]},
        "days": days_out, "svf": svf_layer, "buildings": buildings,
        "has_facades": has_facades,
    }
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = args.out or os.path.join(here, "viewer", "data.json")
    size = export.write_bundle(path, bundle)
    log(f"wrote {path}  {size/1024/1024:.1f} MB")

    print()
    print("  CEQR day        street never sun   " +
          (f"{args.resource_name} median sun   never sun" if resource is not None else ""))
    for d in days_out:
        st = d["stats"]
        line = f"  {d['short']:<14}  {st['frac_street_no_sun']*100:6.1f}%          "
        if "resource" in st:
            r = st["resource"]
            line += f"  {r['median_sun_hours']:5.1f} h            {r['frac_no_sun']*100:5.1f}%"
        print(line)
    if scenario and scenario["type"] == "height cap":
        print(f"\n  SCENARIO: capped at {scenario['cap_m']:.0f} m. "
              "Compare against a run without --cap-height.")
    elif scenario and scenario["type"] == "today":
        print("\n  SCENARIO: today's skyline burned in. "
              "Compare against the plain 2017 run.")
    print("\n  next: python -m pipeline.build_viewer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
