"""
Cross-implementation checks for the heat layer, on one tile at the park's
south edge, next to the grown towers.

    .venv-heat/Scripts/python tools/heat_crosscheck.py

  V3   solweig (Rust, CPU) against UMEP's reference Python SOLWEIG (the pip
       package `umep`, 2025a generation, which the solweig repo's own parity
       tests use as their reference). Fully independent on the UMEP side:
       its own wall heights, its own sky view (pure Python shadowing), its
       own runner. Same tile, same weather (written as a UMEP met file),
       same parameter file, same location. Compared on ground cells (not
       roofs) at three daytime steps of the December day. Pass: the today
       minus 2017 Tmrt change agrees within 1 C RMSE.
  V3b  the same tile through solweig on the GPU and on the CPU. Reports the
       largest absolute Tmrt difference, per scenario and for the change.

The tile is 300 x 300 cells (600 m) cut from the full inputs. Casters outside
it are missing, which does not matter here: both sides of each comparison see
the same tile.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pipeline.heat_inputs import HEAT_DIR                  # noqa: E402
from pipeline.heat_inputs import INPUTS_DIR as H_INPUTS    # noqa: E402
from pipeline.verify_heat import save_gate, line           # noqa: E402

TILE = 300
V3_RMSE_MAX = 1.0


V3_STEPS = ("10:00", "12:00", "14:00")      # file times; sun at 09:30, 11:30, 13:30 EST


def write_umep_met(path, weather):
    """The solweig weather series as a UMEP met file (hour-ending, same clock)."""
    cols = ("%iy id it imin qn qh qe qs qf U RH Tair pres rain kdown snow ldown fcld "
            "wuh xsmd lai kdiff kdir wdir")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(cols + "\n")
        for w in weather:
            t = w.datetime
            v = [t.year, t.timetuple().tm_yday, t.hour, t.minute, -999, -999, -999, -999, -999,
                 w.ws, w.rh, w.ta, w.pressure / 10.0, -999, w.global_rad, -999, -999, -999,
                 -999, -999, -999, w.measured_diffuse_rad, w.measured_direct_rad, -999]
            f.write(" ".join(f"{x:g}" if isinstance(x, float) else str(x) for x in v) + "\n")


def _numpy2_shim():
    """
    umep 0.0.1b47 stores the sun position's 1-element arrays into scalar
    slots, which NumPy 2.4 refuses (older NumPy only warned). Hand it plain
    floats instead. Same numbers, no change to UMEP's maths.
    """
    from umep import class_configs
    sp = class_configs.sp
    if getattr(sp.sun_position, "_scalar_shim", False):
        return
    raw = sp.sun_position

    def scalar_sun(time, location):
        out = raw(time, location)
        for k, v in list(out.items()):
            a = np.asarray(v)
            if a.size == 1:
                out[k] = float(a.reshape(()))
        return out

    scalar_sun._scalar_shim = True
    sp.sun_position = scalar_sun


def v3():
    """UMEP reference Python SOLWEIG against solweig on the tower tile."""
    import glob
    import rasterio
    from importlib.metadata import version
    try:
        from umep import skyviewfactor_algorithm, wall_heightaspect_algorithm
        from umep.functions.SOLWEIGpython.solweig_runner_core import SolweigRunCore
    except ImportError as e:
        out = {"pass": False, "ran": False,
               "blocker": f"pip package umep not importable ({e}); "
                          "pip install --pre umep==0.0.1b47 into .venv-heat"}
        line(False, f"V3 not run: {out['blocker']}")
        save_gate("V3", out)
        return False
    import solweig
    import pipeline.run_heat as RH
    _numpy2_shim()
    solweig.disable_gpu()
    r0, c0, tower = tower_crop(TILE)
    tile_dir = os.path.join(HEAT_DIR, "cache", "v3", "inputs")
    stamp = os.path.join(tile_dir, "crop.json")
    want = {"row": r0, "col": c0, "size": TILE,
            "inputs_mtime": max(os.path.getmtime(os.path.join(H_INPUTS, f)) for f in os.listdir(H_INPUTS))}
    have = None
    if os.path.exists(stamp):
        import json as _json
        with open(stamp, encoding="utf-8") as f:
            have = _json.load(f)
    if have != want:          # re-cut only when the tile or the inputs changed
        RH.crop_inputs(r0, c0, TILE, tile_dir)
        import json as _json
        with open(stamp, "w", encoding="utf-8", newline="\n") as f:
            _json.dump(want, f)
    work = os.path.join(HEAT_DIR, "verify", "v3")
    os.makedirs(work, exist_ok=True)
    wl = RH.weather_list("dec")
    # UMEP's Perez sky divides by the diffuse light whenever the sun is up,
    # and the weather file has zero light in the hour the sun rises. Give both
    # models 1 W/m2 of diffuse light in those hours (identical input on both
    # sides, physically nothing) instead of letting the reference crash.
    from datetime import timedelta, timezone
    from pipeline.solar import sun_position
    with open(os.path.join(H_INPUTS, "domain.json"), encoding="utf-8") as f:
        import json as _json
        site = _json.load(f)["site"]
    patched = []
    for w in wl:
        st = w.datetime - timedelta(minutes=30) + timedelta(hours=5)
        if sun_position(st.replace(tzinfo=timezone.utc), site["lat"], site["lon"]).altitude > 0 \
                and (w.measured_diffuse_rad or 0) < 1.0:
            w.measured_diffuse_rad = 1.0
            w.global_rad = max(w.global_rad, 1.0)
            patched.append(w.datetime.strftime("%m-%d %H:%M"))
    if patched:
        print(f"V3: 1 W/m2 diffuse floor at sunrise/sunset hours {patched}", flush=True)
    met = os.path.join(work, "met_dec.txt")
    write_umep_met(met, wl)
    params = os.path.join(os.path.dirname(solweig.__file__), "parametersforsolweig.json")
    day = RH.days()["seasons"]["dec"]["chosen"]["date"]
    steps = [w.datetime for w in wl if w.datetime.date().isoformat() == day
             and w.datetime.strftime("%H:%M") in V3_STEPS]
    with rasterio.open(os.path.join(tile_dir, "landcover.tif")) as src:
        lc = src.read(1)
        b = src.bounds
    bbox = [b.left, b.bottom, b.right, b.top]
    ground = lc != 2
    tm = {}
    for sc in ("2017", "today"):
        dsm = os.path.join(tile_dir, f"dsm_{sc}.tif")
        d = os.path.join(work, sc)
        done = all(os.path.exists(os.path.join(d, *p)) and
                   os.path.getmtime(os.path.join(d, *p)) > os.path.getmtime(dsm)
                   for p in (("walls", "wall_hts.tif"), ("svf", "svfs.zip"), ("svf", "shadowmats.npz")))
        if done:
            print(f"V3 {sc}: reusing UMEP walls and sky view made from this tile", flush=True)
        else:
            print(f"V3 {sc}: UMEP walls and sky view (pure Python)", flush=True)
            wall_heightaspect_algorithm.generate_wall_hts(dsm, bbox, os.path.join(d, "walls"))
            skyviewfactor_algorithm.generate_svf(dsm, bbox, os.path.join(d, "svf"),
                                                 dem_path=os.path.join(tile_dir, "dem.tif"),
                                                 cdsm_path=os.path.join(tile_dir, "cdsm.tif"))
        out_dir = os.path.join(d, "out")
        ini = os.path.join(d, "config.ini")
        cfg = {"output_dir": out_dir, "working_dir": os.path.join(d, "work"), "dsm_path": dsm,
               "svf_path": os.path.join(d, "svf", "svfs.zip"),
               "wh_path": os.path.join(d, "walls", "wall_hts.tif"),
               "wa_path": os.path.join(d, "walls", "wall_aspects.tif"),
               "use_epw_file": 0, "met_path": met,
               "cdsm_path": os.path.join(tile_dir, "cdsm.tif"),
               "dem_path": os.path.join(tile_dir, "dem.tif"),
               "lc_path": os.path.join(tile_dir, "landcover.tif"),
               "aniso_path": os.path.join(d, "svf", "shadowmats.npz"),
               "only_global": 0, "use_veg_dem": 1, "conifer": 0, "person_cylinder": 1,
               "utc": -5, "use_landcover": 1, "use_dem_for_buildings": 0, "use_aniso": 1,
               "use_wall_scheme": 0, "wall_type": "Brick", "output_tmrt": 1, "output_kup": 0,
               "output_kdown": 0, "output_lup": 0, "output_ldown": 0, "output_sh": 1,
               "save_buildings": 0, "output_kdiff": 0, "output_tree_planter": 0, "wall_netcdf": 0}
        with open(ini, "w", encoding="utf-8", newline="\n") as f:
            f.write("[DEFAULT]\n" + "".join(f"{k}={v}\n" for k, v in cfg.items()))
        print(f"V3 {sc}: UMEP SOLWEIG, {len(wl)} steps", flush=True)
        runner = SolweigRunCore(ini, params)
        runner.run()
        loc = runner.location
        print(f"V3 {sc}: solweig on the CPU at {loc['latitude']:.5f}, {loc['longitude']:.5f}, "
              f"{loc['altitude']:.1f} m", flush=True)
        surf, _ = RH.prepare(sc, in_dir=tile_dir, cache_dir=os.path.join(HEAT_DIR, "cache", "v3", sc))
        sdir = os.path.join(d, "solweig")
        orig = RH.location
        RH.location = lambda meta=None: solweig.Location(latitude=loc["latitude"],
                                                         longitude=loc["longitude"],
                                                         altitude=loc["altitude"], utc_offset=-5)
        try:
            RH.run(sc, "dec", surface=surf, out_dir=sdir, gpu=False, weather=list(wl),
                   prepared_with="cpu", log=lambda s: print(s, flush=True))
        finally:
            RH.location = orig
        for ts in steps:
            code = f"{ts.year}_{ts.timetuple().tm_yday}_{ts.hour:02d}{ts.minute:02d}"
            hit = glob.glob(os.path.join(out_dir, f"Tmrt_{code}*.tif"))
            if not hit:
                raise SystemExit(f"UMEP output for {code} not found in {out_dir}")
            with rasterio.open(hit[0]) as src:
                tm[("umep", sc, ts)] = src.read(1)
            with rasterio.open(os.path.join(sdir, "tmrt", f"tmrt_{ts:%Y%m%d_%H%M}.tif")) as src:
                tm[("solweig", sc, ts)] = src.read(1)
    out = {"tile": {"row": r0, "col": c0, "size": TILE}, "umep_version": version("umep"),
           "diffuse_floor_hours": patched,
           "solweig_version": solweig.__version__, "backend": "cpu", "steps": [],
           "compared_cells": int(ground.sum()), "params": params}
    worst = 0.0
    for ts in steps:
        rec = {"file_time": ts.strftime("%H:%M"), "sun_time": f"{ts.hour - 1:02d}:30"}
        for sc in ("2017", "today"):
            dd = (tm[("solweig", sc, ts)] - tm[("umep", sc, ts)])[ground]
            dd = dd[np.isfinite(dd)]
            rec[f"bias_{sc}"] = float(dd.mean())
            rec[f"rmse_{sc}"] = float(np.sqrt((dd ** 2).mean()))
        ds = (tm[("solweig", "today", ts)] - tm[("solweig", "2017", ts)])[ground]
        du = (tm[("umep", "today", ts)] - tm[("umep", "2017", ts)])[ground]
        ok = np.isfinite(ds) & np.isfinite(du)
        e = ds[ok] - du[ok]
        rec["change_bias"] = float(e.mean())
        rec["change_rmse"] = float(np.sqrt((e ** 2).mean()))
        rec["cells_change_le_minus2_solweig"] = int((ds[ok] <= -2).sum())
        rec["cells_change_le_minus2_umep"] = int((du[ok] <= -2).sum())
        worst = max(worst, rec["change_rmse"])
        out["steps"].append(rec)
        print(f"V3 {rec['sun_time']} EST: Tmrt solweig minus UMEP, bias/RMSE 2017 "
              f"{rec['bias_2017']:+.2f}/{rec['rmse_2017']:.2f} C, today {rec['bias_today']:+.2f}/"
              f"{rec['rmse_today']:.2f} C; change today - 2017 bias {rec['change_bias']:+.2f}, RMSE "
              f"{rec['change_rmse']:.2f} C; cells >= 2 C colder: solweig "
              f"{rec['cells_change_le_minus2_solweig']:,}, UMEP {rec['cells_change_le_minus2_umep']:,}",
              flush=True)
    out["worst_change_rmse"] = worst
    out["pass"] = bool(worst <= V3_RMSE_MAX)
    line(out["pass"], f"V3 change agrees within {worst:.2f} C RMSE (limit {V3_RMSE_MAX:.0f} C)")
    save_gate("V3", out)
    return out["pass"]


def tower_crop(size):
    """
    Top-left (row, col) of a tile holding the tallest grown building (Central
    Park Tower) a quarter of the way up from its south side, so most of the
    tile is the park its shadow falls on.
    """
    import json
    from pyproj import Transformer
    from pipeline.heat_inputs import INPUTS_DIR
    with open(os.path.join(INPUTS_DIR, "domain.json"), encoding="utf-8") as f:
        meta = json.load(f)
    lon, lat = meta["grown_buildings"][0]["lonlat"]
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32118", always_xy=True).transform(lon, lat)
    a, b, x0, d, e, y0 = meta["grid"]["transform_m"]
    row, col = int((y - y0) / e), int((x - x0) / a)
    r0 = max(0, min(meta["grid"]["rows"] - size, row - 3 * size // 4))
    c0 = max(0, min(meta["grid"]["cols"] - size, col - size // 2))
    return r0, c0, meta["grown_buildings"][0]


def v3b():
    from pipeline.run_heat import crop_inputs, prepare, run, weather_list, days
    import rasterio
    r0, c0, tower = tower_crop(TILE)
    print(f"V3b tile {TILE} x {TILE} at row {r0}, col {c0}, holding the {tower['height_m']} m grown "
          f"tower at {tower['lonlat'][1]:.5f}, {tower['lonlat'][0]:.5f}", flush=True)
    tile_dir = crop_inputs(r0, c0, TILE, os.path.join(HEAT_DIR, "cache", "v3b", "inputs"))
    with rasterio.open(os.path.join(tile_dir, "dsm_2017.tif")) as a,             rasterio.open(os.path.join(tile_dir, "dsm_today.tif")) as b:
        n_changed = int((a.read(1) != b.read(1)).sum())
    print(f"V3b {n_changed:,} cells of the tile differ between 2017 and today", flush=True)
    if n_changed == 0:
        raise SystemExit("V3b tile holds no grown building")
    d = days()["seasons"]["dec"]
    day = d["chosen"]["date"]
    want = {"10:00", "12:00", "14:00"}
    wl = [w for w in weather_list("dec") if w.datetime.date().isoformat() == day
          and w.datetime.strftime("%H:%M") in want]
    import shutil
    for sc in ("2017", "today"):                 # caches are per tile; never reuse
        shutil.rmtree(os.path.join(HEAT_DIR, "cache", "v3b", sc), ignore_errors=True)
    surfaces = {sc: prepare(sc, in_dir=tile_dir, cache_dir=os.path.join(HEAT_DIR, "cache", "v3b", sc))[0]
                for sc in ("2017", "today")}
    tm = {}
    for backend in ("gpu", "cpu"):
        for sc in ("2017", "today"):
            od = os.path.join(HEAT_DIR, "verify", "v3b", f"{sc}_{backend}")
            m = run(sc, "dec", surface=surfaces[sc], out_dir=od, gpu=(backend == "gpu"),
                    weather=list(wl), log=lambda s: print(s, flush=True))
            if backend == "gpu" and not m["backend"].startswith("gpu"):
                raise SystemExit(f"asked for the GPU, ran on {m['backend']}")
            if backend == "cpu" and m["backend"] != "cpu":
                raise SystemExit(f"asked for the CPU, ran on {m['backend']}")
            for w in wl:
                p = os.path.join(od, "tmrt", f"tmrt_{w.datetime:%Y%m%d_%H%M}.tif")
                with rasterio.open(p) as src:
                    tm[(backend, sc, w.datetime)] = src.read(1)
    out = {"tile": {"row": r0, "col": c0, "size": TILE, "cells_changed": n_changed}, "steps": [],
           "pass": None}
    worst = 0.0
    for w in wl:
        rec = {"file_time": w.datetime.strftime("%H:%M")}
        for sc in ("2017", "today"):
            dd = np.abs(tm[("gpu", sc, w.datetime)] - tm[("cpu", sc, w.datetime)])
            rec[f"max_abs_dtmrt_{sc}"] = float(np.nanmax(dd))
            rec[f"mean_abs_dtmrt_{sc}"] = float(np.nanmean(dd))
            rec[f"share_cells_over_1c_{sc}"] = float((dd > 1.0).mean())
            worst = max(worst, rec[f"max_abs_dtmrt_{sc}"])
        dg = tm[("gpu", "today", w.datetime)] - tm[("gpu", "2017", w.datetime)]
        dc = tm[("cpu", "today", w.datetime)] - tm[("cpu", "2017", w.datetime)]
        rec["max_abs_diff_of_change"] = float(np.nanmax(np.abs(dg - dc)))
        rec["rmse_of_change"] = float(np.sqrt(np.nanmean((dg - dc) ** 2)))
        rec["change_cells_gpu_le_minus2"] = int((dg <= -2).sum())
        rec["change_cells_cpu_le_minus2"] = int((dc <= -2).sum())
        out["steps"].append(rec)
        print(f"V3b {rec['file_time']} (sun {int(rec['file_time'][:2]) - 1}:30 EST): max |GPU - CPU| Tmrt "
              f"2017 {rec['max_abs_dtmrt_2017']:.4f} C, today {rec['max_abs_dtmrt_today']:.4f} C; "
              f"change today - 2017 differs by at most {rec['max_abs_diff_of_change']:.4f} C "
              f"(RMSE {rec['rmse_of_change']:.4f}); cells with Tmrt change <= -2 C: GPU "
              f"{rec['change_cells_gpu_le_minus2']:,}, CPU {rec['change_cells_cpu_le_minus2']:,}; "
              f"cells differing > 1 C {rec['share_cells_over_1c_2017']*100:.2f}%", flush=True)
    out["max_abs_dtmrt"] = worst
    out["note"] = "reported only; the brief sets no threshold for V3b"
    save_gate("V3b", out)
    return worst


def main():
    wanted = [a.lower() for a in sys.argv[1:]] or ["v3", "v3b"]
    if "v3" in wanted:
        v3()
    if "v3b" in wanted:
        v3b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
