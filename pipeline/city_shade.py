"""
The shadow study for any city, on the shared builder's grid.

    TWIN_CITY=rotterdam .venv-heat/Scripts/python -m pipeline.city_shade
    TWIN_CITY=rotterdam .venv-heat/Scripts/python -m pipeline.city_shade --scenario today

New York keeps pipeline.run_nyc, which made the published numbers. This is
the same method on any city's inputs (pipeline.city_inputs):

  * The analysis days and windows of pipeline.city_time: CEQR's four days,
    1.5 h after sunrise to 1.5 h before sunset, standard time all year,
    every 30 minutes, plus any days the city adds.
  * The same shadow sweep (pipeline.geometry, Ratti and Richens) and the
    same NOAA sun.
  * The surface is the scenario's buildings plus the tree canopy, with trees
    solid, as New York's LiDAR surface treats them.
  * Statistics on the study window's open ground (no buildings, no water):
    the share that never gets direct sun inside the window, and median sun
    hours; then, between the two scenarios, the hectares that lost an hour
    or more, and all of it.

Writes <heat dir>/../shade/<scenario>.npz (sun-hour rasters), shade.json
(numbers) and a viewer bundle, <viewer dir>/data-<scenario>.json, in the
format pipeline.build_viewer reads.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import city
from . import export
from .city_time import analysis_days, frames_for
from .geometry import run_day

STEP_MIN = 30
NO_SUN_H = 0.25


def shade_dir():
    return os.path.join(os.path.dirname(city.heat_dir()), "shade")


def grid_from_meta(meta, masks):
    """A Grid for the heat inputs: north-up, metres, frame on the window's box."""
    from .grid import Grid
    t = meta["_transform"]
    rows, cols = masks["window"].shape
    rr, cc = np.nonzero(masks["window"])
    g = meta["grid"]
    return Grid(crs=meta["grid"]["crs_build"], unit_m=1.0, res_m=g["res_m"],
                bearing_deg=-g["conv_deg"], conv_deg=g["conv_deg"], x0=t.c, y0=t.f,
                rows=rows, cols=cols, frame=(int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1),
                centre_lat=meta["site"]["lat"], centre_lon=meta["site"]["lon"])


def surface_for(layers, scenario):
    dsm = layers[f"dsm_{scenario}"]
    return np.maximum(dsm, layers["dem"] + layers["cdsm"]).astype(np.float32)


def buildings_for_viewer(in_dir, grid, layers, masks, scenario, ring_m=200.0, tall_m=60.0, log=print):
    """Footprints with this scenario's height (99th percentile above ground), in lon/lat."""
    import geopandas as gpd
    from .nyc_footprints import label_quantile
    df = gpd.read_file(os.path.join(in_dir, "buildings.gpkg"))
    lab = masks["labels"]
    h = label_quantile((layers[f"dsm_{scenario}"] - layers["dem"]).astype(np.float32), lab, int(lab.max()), 0.99)
    df["height"] = [float(h[i]) if i < len(h) and np.isfinite(h[i]) else 0.0 for i in df["label"]]
    frame = grid.frame_polygon_crs()
    ring = frame.buffer(ring_m)
    ll = df.to_crs("EPSG:4326")
    out_f, out_c = [], []
    for (_, r), g in zip(df.iterrows(), ll.geometry):
        if r["height"] < 3.0 or g is None or g.is_empty:
            continue
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        ring_xy = [[round(c[0], 6), round(c[1], 6)] for c in g.exterior.coords]
        rec = {"ring": ring_xy, "height": round(r["height"], 1)}
        p = r.geometry.representative_point()
        if frame.contains(p):
            out_f.append(rec)
        elif ring.intersects(r.geometry) or r["height"] >= tall_m:
            out_c.append({**rec, "context": True})
    log(f"  {len(out_f):,} buildings in the frame, {len(out_c):,} around it")
    return out_f + out_c


def run(scenario, log=print, viewer=True):
    from .heat_inputs import INPUTS_DIR, load
    cfg = city.config()
    layers, masks, meta = load()
    grid = grid_from_meta(meta, masks)
    fsl = grid.frame_slice
    surf = surface_for(layers, scenario)
    report = masks["report"][fsl]
    blank = ~masks["window"][fsl] | masks["building"][fsl] | masks["water"][fsl]
    lat, lon = meta["site"]["lat"], meta["site"]["lon"]
    utc = city.utc_offset()
    days = analysis_days(cfg, lat, lon)
    painter_idx, painter_bounds = grid.northup_index()
    painter = export.Painter(painter_idx, painter_bounds)
    step_h = STEP_MIN / 60.0
    out_days, rasters, stats = [], {}, {}
    ha = grid.res_m ** 2 / 1e4
    for spec in days:
        fr = frames_for(spec, lat, lon, utc, STEP_MIN)
        pos, lab = [p for p, _ in fr], [l for _, l in fr]
        t0 = time.time()
        res = run_day(surf, grid.res_m, pos, lab, window=grid.frame, bearing_deg=grid.bearing_deg,
                      compute_svf=False)
        sun_h = (res.sun_hours * step_h).astype(np.float32)
        window_h = len(fr) * step_h
        rasters[spec["id"]] = sun_h
        r = sun_h[report]
        st = {"window": list(spec["window"]), "frames": len(fr), "window_h": window_h,
              "peak_altitude": round(max(res.altitudes), 2),
              "report_ha": round(float(report.sum() * ha), 1),
              "median_sun_h": round(float(np.median(r)), 2),
              "frac_no_sun": round(float((r < NO_SUN_H).mean()), 4),
              "ha_no_sun": round(float((r < NO_SUN_H).sum() * ha), 2)}
        stats[spec["id"]] = st
        log(f"  {scenario} {spec['label']}: window {spec['window'][0]}-{spec['window'][1]} "
            f"{city.clock_name()}, {len(fr)} frames, peak sun {st['peak_altitude']:.1f} deg; open ground "
            f"median {st['median_sun_h']:.1f} h of sun, {st['frac_no_sun'] * 100:.1f}% never in sun "
            f"({time.time() - t0:.0f} s)")
        if viewer:
            frames = [{"hour": l, "altitude": round(res.altitudes[i], 2), "azimuth": round(res.azimuths[i], 2),
                       "sunlit_fraction": float(res.sunlit[i][report].mean()),
                       "png": painter.png(res.sunlit[i], [0.0, 0.5, 1.01], export.SHADOW_COLORS,
                                          alpha=205, mask=blank)} for i, l in enumerate(lab)]
            breaks = [round(window_h * i / 8, 3) for i in range(8)] + [window_h + 0.01]
            layer = export.layer(f"sunhours_{spec['id']}", "Hours of direct sun",
                                 f"hours inside the window, {spec['label']}", sun_h, breaks,
                                 export.SUNHOUR_COLORS, mask=blank, stats_mask=~report, painter=painter)
            out_days.append({"id": spec["id"], "label": spec["label"], "short": spec["short"],
                             "date": spec["date"], "window": list(spec["window"]), "step_h": step_h,
                             "daylight_hours_modelled": window_h, "facade_hours_max": window_h,
                             "peak_altitude": st["peak_altitude"], "hours": frames, "sunhours": layer,
                             "stats": {"median_sun_hours": st["median_sun_h"],
                                       "mean_sun_hours": float(r.mean()),
                                       "frac_street_no_sun": st["frac_no_sun"],
                                       "frac_street_under_2h": float((r < 2).mean()),
                                       "frac_street_over_half_day": float((r > window_h / 2).mean())}})
    os.makedirs(shade_dir(), exist_ok=True)
    np.savez_compressed(os.path.join(shade_dir(), f"{scenario}.npz"), report=report,
                        **{f"sun_{k}": v for k, v in rasters.items()},
                        meta=np.array(json.dumps({"scenario": scenario, "frame": list(grid.frame),
                                                  "days": [d["id"] for d in days]})))
    if viewer:
        fh = (surf - layers["dem"])[fsl]
        tallest = float(fh.max())
        w_, s_, e_, n_ = painter.bounds
        L, W = grid.frame_size_m()
        bundle = {
            "mode": "REAL", "warning": "",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "site": {"name": cfg["name"], "centre_lat": lat, "centre_lon": lon,
                     "bounds": {"west": w_, "south": s_, "east": e_, "north": n_},
                     "span_m": round(max(L, W)), "cellsize_m": grid.res_m, "grid": list(grid.frame_shape),
                     "edge_margin_m": 0.0, "frame": grid.describe(),
                     "tallest_m": round(tallest, 1), "tallest_ft": round(tallest * 3.28084),
                     "nodata_frac": 0.0, "water_frac": round(float(masks["water"][fsl].mean()), 3),
                     "city": cfg["id"], "clock": city.clock_name()},
            "model": {"solar_algorithm": "NOAA / Meeus low precision",
                      "shadow_method": "Ratti and Richens shift-and-subtract sweep, whole-cell stepping",
                      "svf_method": None,
                      "surface_source": meta.get("surveys", {}).get("before"),
                      "ground_source": meta.get("surveys", {}).get("before"),
                      "building_source": meta.get("surveys", {}).get("buildings"),
                      "native_resolution_m": 0.5, "resample": "highest of 4 x 4 pixels per 2 m cell",
                      "time_basis": (f"CEQR method: 1.5 h after sunrise to 1.5 h before sunset, "
                                     f"{city.clock_name()} all year, sampled every {STEP_MIN} min"),
                      "buffer_m": None, "buffer_desc": None,
                      "scenario": {"type": "today"} if scenario == "today" else None,
                      "does_not_compute": ["air temperature", "surface temperature", "wind", "cloud"]},
            "days": out_days, "svf": None,
            "buildings": buildings_for_viewer(INPUTS_DIR, grid, layers, masks, scenario, log=log),
            "has_facades": False,
        }
        os.makedirs(city.viewer_dir(), exist_ok=True)
        path = os.path.join(city.viewer_dir(), f"data_{scenario}.json")
        size = export.write_bundle(path, bundle)
        log(f"  wrote {path}  {size / 1024 / 1024:.1f} MB")
    return stats, rasters, report, grid


def compare(log=print):
    """Before against today, cell by cell on the open ground of the window."""
    before = city.before()
    a = np.load(os.path.join(shade_dir(), f"{before}.npz"))
    b = np.load(os.path.join(shade_dir(), "today.npz"))
    report = a["report"]
    from .heat_inputs import load
    _, masks, meta = load()
    grid = grid_from_meta(meta, masks)
    fsl = grid.frame_slice
    d_focus = masks["dist_focus"][fsl]
    ha = 2.0 * 2.0 / 1e4
    out = {}
    for k in json.loads(str(a["meta"]))["days"]:
        s0, s1 = a[f"sun_{k}"][report], b[f"sun_{k}"][report]
        lost = s0 - s1
        df = d_focus[report]
        rec = {"report_ha": round(float(report.sum() * ha), 1),
               "frac_no_sun": {before: round(float((s0 < NO_SUN_H).mean()), 4),
                               "today": round(float((s1 < NO_SUN_H).mean()), 4)},
               "ha_lost_any": round(float((lost >= 0.25).sum() * ha), 2),
               "ha_lost_1h": round(float((lost >= 1.0).sum() * ha), 2),
               "ha_lost_all": round(float(((s0 >= NO_SUN_H) & (s1 < NO_SUN_H)).sum() * ha), 2),
               "ha_gained_any": round(float((lost <= -0.25).sum() * ha), 2),
               "lost_1h_by_distance_m": {f"{lo}-{hi}": round(float(((lost >= 1.0) & (df >= lo) & (df < hi)).sum() * ha), 2)
                                         for lo, hi in ((0, 250), (250, 500), (500, 750), (750, 1000), (1000, 2000))}}
        out[k] = rec
        log(f"  {k}: never in sun {rec['frac_no_sun'][before] * 100:.1f}% ({before}) -> "
            f"{rec['frac_no_sun']['today'] * 100:.1f}% (today); lost sun {rec['ha_lost_any']} ha, "
            f"an hour or more {rec['ha_lost_1h']} ha, all of it {rec['ha_lost_all']} ha; "
            f"gained {rec['ha_gained_any']} ha")
    return out


def towers(min_m=70.0, day="dec", log=print):
    """
    Each tower that grew, added on its own to the older city: what it alone
    takes from the open ground on one day. New York's tools/compare_heat.py
    --towers does the same for felt temperature. Overlaps between towers
    are not split, so the parts can add up to more than the whole.
    """
    from .heat_inputs import load
    cfg = city.config()
    layers, masks, meta = load()
    grid = grid_from_meta(meta, masks)
    fsl = grid.frame_slice
    report = masks["report"][fsl]
    before = surface_for(layers, city.before())
    now = surface_for(layers, "today")
    lab, raised = masks["labels"], masks["raised"]
    lat, lon = meta["site"]["lat"], meta["site"]["lon"]
    spec = next(d for d in analysis_days(cfg, lat, lon) if d["id"] == day)
    fr = frames_for(spec, lat, lon, city.utc_offset(), STEP_MIN)
    pos, lb = [p for p, _ in fr], [l for _, l in fr]
    step_h = STEP_MIN / 60.0
    base = run_day(before, grid.res_m, pos, lb, window=grid.frame, bearing_deg=grid.bearing_deg,
                   compute_svf=False).sun_hours * step_h
    from pyproj import Transformer
    to_m = Transformer.from_crs("EPSG:4326", meta["grid"]["crs_build"], always_xy=True).transform
    t = meta["_transform"]
    out = []
    ha = grid.res_m ** 2 / 1e4
    for b in meta["grown_buildings"]:
        if b["height_m"] < min_m:
            continue
        x, y = to_m(*b["lonlat"])
        r, c = int((y - t.f) / t.e), int((x - t.c) / t.a)
        i = int(lab[r, c])
        cells = raised & (lab == i)
        if not i or not cells.any():
            continue
        s = np.where(cells, now, before).astype(np.float32)
        sun = run_day(s, grid.res_m, pos, lb, window=grid.frame, bearing_deg=grid.bearing_deg,
                      compute_svf=False).sun_hours * step_h
        lost = (base - sun)[report]
        rec = {**b, "label": i, "day": day,
               "ha_lost_any": round(float((lost >= 0.25).sum() * ha), 2),
               "ha_lost_1h": round(float((lost >= 1.0).sum() * ha), 2),
               "ha_lost_all": round(float(((base[report] >= NO_SUN_H) & (sun[report] < NO_SUN_H)).sum() * ha), 2)}
        out.append(rec)
        log(f"  {b['height_m']:6.1f} m, built {b['year']}, {b['lonlat'][1]:.5f}, {b['lonlat'][0]:.5f}: "
            f"alone it takes sun from {rec['ha_lost_any']} ha, an hour or more from {rec['ha_lost_1h']} ha, "
            f"all of it from {rec['ha_lost_all']} ha")
    with open(os.path.join(shade_dir(), f"towers_{day}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=list(city.scenarios()) + ["both"], default="both")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--towers", action="store_true", help="each tower that grew, on its own, 21 December")
    args = ap.parse_args()
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    if args.towers:
        towers(log=log)
        return 0

    scs = city.scenarios() if args.scenario == "both" else (args.scenario,)
    result = {"city": city.city_id(), "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "stats": {}}
    for sc in scs:
        result["stats"][sc] = run(sc, log=log, viewer=not args.no_viewer)[0]
    if args.scenario == "both" and len(city.scenarios()) == 2:
        log("change between the scenarios, open ground of the window:")
        result["change"] = compare(log=log)
    with open(os.path.join(shade_dir(), "shade.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)
    log(f"wrote {os.path.join(shade_dir(), 'shade.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
