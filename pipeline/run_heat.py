"""
Run SOLWEIG on the heat inputs, for 2017 and today, December and June.

    .venv-heat/Scripts/python -m pipeline.run_heat --scenario 2017 --season dec
    .venv-heat/Scripts/python -m pipeline.run_heat --all
    .venv-heat/Scripts/python -m pipeline.run_heat --time-subset

Engine: the solweig package from UMEP-dev (Rust core). One SurfaceData.prepare
per scenario, cached in data/heat/cache/<scenario>/, so walls and sky view are
computed once and shared by both seasons. Per-timestep tmrt, utci and shadow
land in data/heat/runs/<scenario>_<season>/, with run_metadata.json.

Settings that matter, all asserted here:

  max_shadow_distance_m 3500   The package default of 1000 m cuts the shadows
                               of the tallest towers at low December sun.
  conifer False                Deciduous: leaf-on day 97 to 300, shortwave
                               transmissivity 0.03 leaf-on, 0.5 leaf-off.
  cdsm_relative True           Canopy is height above ground; the DSM is
                               absolute and the DEM is always passed.

Everything else is the package default: standing person, anisotropic sky, no
experimental ground scheme.

Time: weather rows are hour-ending EST. A file stamped 13:00 is the hour from
12:00 to 13:00, with the sun at 12:30. See heat_weather.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime

from .heat_inputs import HEAT_DIR, INPUTS_DIR

MAX_SHADOW_M = 3500.0
CONIFER = False
OUTPUTS = ["tmrt", "utci", "shadow"]
# UTCI is defined for 10 m wind from 0.5 m/s. Calm hours in the weather file
# would come back as NaN UTCI over the whole grid, so they are raised to it.
# Tmrt does not depend on wind.
MIN_WIND = 0.5
SCENARIOS = ("2017", "today")
SEASONS = ("dec", "jun")
DAYS_JSON = os.path.join(HEAT_DIR, "weather", "days.json")


def days():
    with open(DAYS_JSON, encoding="utf-8") as f:
        return json.load(f)


def epw_path(d=None):
    d = d or days()
    return os.path.join(HEAT_DIR, "weather", d["epw"])


def weather_list(season, timesteps=None):
    """solweig.Weather for the chosen day of a season, with a day of spin-up."""
    import solweig
    from . import heat_weather as HW
    d = days()
    _, rows = HW.read_epw(epw_path(d))
    ser = HW.series(d["seasons"][season]["chosen"]["date"], rows)
    if timesteps is not None:
        ser = [s for s in ser if s["datetime"] in timesteps]
    out = []
    for s in ser:
        out.append(solweig.Weather(
            datetime=s["datetime"], ta=s["ta"], rh=s["rh"], global_rad=s["ghi"],
            ws=max(s["ws"], MIN_WIND), pressure=s["pressure_hpa"] or 1013.25, timestep_minutes=60.0,
            measured_direct_rad=s["dni"], measured_diffuse_rad=s["dhi"]))
    return out


def location(meta=None):
    import solweig
    if meta is None:
        with open(os.path.join(INPUTS_DIR, "domain.json"), encoding="utf-8") as f:
            meta = json.load(f)
    d = days()
    return solweig.Location(latitude=meta["site"]["lat"], longitude=meta["site"]["lon"],
                            altitude=float(d["header"]["elev_m"]), utc_offset=meta["site"]["utc_offset"])


def inputs(scenario, in_dir=INPUTS_DIR):
    return {"dsm": os.path.join(in_dir, f"dsm_{scenario}.tif"),
            "dem": os.path.join(in_dir, "dem.tif"),
            "cdsm": os.path.join(in_dir, "cdsm.tif"),
            "land_cover": os.path.join(in_dir, "landcover.tif")}


def prepare(scenario, in_dir=INPUTS_DIR, cache_dir=None, log=print):
    import solweig
    paths = inputs(scenario, in_dir)
    wd = cache_dir or os.path.join(HEAT_DIR, "cache", scenario)
    t = time.time()
    surface = solweig.SurfaceData.prepare(
        dsm=paths["dsm"], working_dir=wd, cdsm=paths["cdsm"], dem=paths["dem"],
        land_cover=paths["land_cover"], dsm_relative=False, cdsm_relative=True)
    log(f"  prepared {scenario}: {surface.dsm.shape[1]} x {surface.dsm.shape[0]} px at "
        f"{surface.pixel_size} m in {time.time() - t:.0f} s (cache {wd})")
    if abs(surface.pixel_size - 2.0) > 1e-6:
        raise SystemExit(f"solweig read a {surface.pixel_size} m pixel, expected 2.0")
    return surface, time.time() - t


def effective_settings(weather):
    """What solweig will actually use, read back from its own resolver."""
    import solweig
    from solweig.components.shadows import compute_transmissivity
    from solweig.models.settings import Settings
    s = Settings.resolve(config=None, use_anisotropic_sky=None, conifer=CONIFER,
                         wall_material=None, max_shadow_distance_m=MAX_SHADOW_M,
                         human=None, physics=None, materials=None).with_loaded_defaults()
    doy = weather[-1].datetime.timetuple().tm_yday
    psi = compute_transmissivity(doy, s.physics, s.conifer)
    return {"max_shadow_distance_m": s.max_shadow_distance_m, "conifer": s.conifer,
            "use_anisotropic_sky": s.use_anisotropic_sky,
            "vegetation_transmissivity": psi, "day_of_year": doy,
            "human": {"posture": s.human.posture, "abs_k": s.human.abs_k, "abs_l": s.human.abs_l},
            "solweig_version": solweig.__version__}


def run(scenario, season, surface=None, out_dir=None, gpu=True, log=print, weather=None,
        prepared_with=None):
    import solweig
    if MAX_SHADOW_M != 3500.0 or CONIFER is not False:
        raise SystemExit("max_shadow_distance_m must be 3500 and conifer False")
    weather = weather or weather_list(season)
    eff = effective_settings(weather)
    assert eff["max_shadow_distance_m"] == 3500.0, eff
    assert eff["conifer"] is False, eff
    out_dir = out_dir or os.path.join(HEAT_DIR, "runs", f"{scenario}_{season}")
    if surface is None:
        surface, _ = prepare(scenario, log=log)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)          # stale timesteps must not survive a rerun
    os.makedirs(out_dir)
    loc = location()

    def go():
        solweig.reset_gpu_metrics()
        t = time.time()
        solweig.calculate(surface=surface, weather=weather, location=loc, output_dir=out_dir,
                          max_shadow_distance_m=MAX_SHADOW_M, conifer=CONIFER, outputs=OUTPUTS)
        return time.time() - t

    backend, err = None, None
    if gpu and solweig.is_gpu_available():
        try:
            wall = go()
            backend = "gpu" if solweig.gpu_dispatch_count() > 0 else "cpu"
            if solweig.gpu_fallback_count() > 0:
                backend += f" ({solweig.gpu_fallback_count()} GPU fallbacks to CPU)"
        except Exception as e:           # GPU first; on any error, CPU
            err = f"{type(e).__name__}: {e}"
            log(f"  GPU run failed ({err}); disabling the GPU and rerunning on the CPU")
            solweig.disable_gpu()
    if backend is None:
        solweig.disable_gpu()
        wall = go()
        backend = "cpu" if solweig.gpu_dispatch_count() == 0 else "gpu (disable_gpu did not take)"
    d = days()
    meta = {
        "scenario": scenario, "season": season, "solweig_version": solweig.__version__,
        "backend": backend, "gpu_error": err,
        "gpu_limits": solweig.get_gpu_limits(),
        "epw": d["epw"], "epw_sha256": d["epw_sha256"], "epw_source_url": d["source_url"],
        "chosen_date": d["seasons"][season]["chosen"]["date"],
        "timesteps": len(weather),
        "first": weather[0].datetime.isoformat(), "last": weather[-1].datetime.isoformat(),
        "report_timestamps": d["seasons"][season]["report_timestamps"],
        "report_sun_times": d["seasons"][season]["report_sun_times"],
        "clock": d["clock"],
        "location": {"lat": loc.latitude, "lon": loc.longitude, "altitude": loc.altitude,
                     "utc_offset": loc.utc_offset},
        "settings": eff, "outputs": OUTPUTS, "wall_time_s": round(wall, 1),
        "prepared_with": prepared_with,
        "wind_floor_m_s": MIN_WIND,
        "wind_floored_hours": [w.datetime.isoformat() for w in weather if w.ws <= MIN_WIND],
        "inputs": inputs(scenario),
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1, default=str)
    log(f"  {scenario} {season}: {len(weather)} steps on {backend} in {wall:.0f} s, "
        f"psi {eff['vegetation_transmissivity']}, shadow reach {eff['max_shadow_distance_m']:.0f} m")
    return meta


def crop_inputs(r0, c0, size, out_dir, scenarios=SCENARIOS):
    """Write a size x size crop of the heat inputs, same CRS and pixel grid."""
    import rasterio
    from rasterio.windows import Window
    os.makedirs(out_dir, exist_ok=True)
    names = ["dem", "cdsm", "landcover"] + [f"dsm_{s}" for s in scenarios]
    win = Window(c0, r0, size, size)
    for name in names:
        with rasterio.open(os.path.join(INPUTS_DIR, f"{name}.tif")) as src:
            prof = src.profile.copy()
            prof.update(height=size, width=size, transform=src.window_transform(win))
            with rasterio.open(os.path.join(out_dir, f"{name}.tif"), "w", **prof) as dst:
                dst.write(src.read(1, window=win), 1)
    return out_dir


def south_edge_crop(size):
    """Top-left (row, col) of a crop straddling the window's south edge."""
    import numpy as np
    with open(os.path.join(INPUTS_DIR, "domain.json"), encoding="utf-8") as f:
        meta = json.load(f)
    z = np.load(os.path.join(INPUTS_DIR, "masks.npz"))
    rows, cols = np.nonzero(z["window"])
    r_c = int(rows.max()) - size // 4
    c_c = int(np.median(cols[rows > rows.max() - 50]))
    r0 = max(0, min(meta["grid"]["rows"] - size, r_c - size // 2))
    c0 = max(0, min(meta["grid"]["cols"] - size, c_c - size // 2))
    return r0, c0, meta


def time_subset(log=print, size=500, steps=3):
    """Prepare and run a size x size crop for a few daytime steps, and time it."""
    r0, c0, meta = south_edge_crop(size)
    sub = crop_inputs(r0, c0, size, os.path.join(HEAT_DIR, "cache", "timing", "inputs"), ("today",))
    wd = os.path.join(HEAT_DIR, "cache", "timing", "work")
    if os.path.isdir(wd):
        shutil.rmtree(wd)
    surface, t_prep = prepare("today", in_dir=sub, cache_dir=wd, log=log)
    d = days()
    wl = weather_list("dec")
    wanted = set(d["seasons"]["dec"]["report_timestamps"][:steps])
    wl = [w for w in wl if w.datetime.strftime("%H:%M") in wanted
          and w.datetime.date().isoformat() == d["seasons"]["dec"]["chosen"]["date"]]
    out = os.path.join(HEAT_DIR, "cache", "timing", "run")
    m = run("today", "dec", surface=surface, out_dir=out, log=log, weather=wl)
    res = {"crop": [r0, c0, size], "prepare_s": round(t_prep, 1),
           "steps": len(wl), "run_s": m["wall_time_s"], "backend": m["backend"]}
    full = meta["grid"]["rows"] * meta["grid"]["cols"]
    k = full / float(size * size)
    res["cells_ratio"] = round(k, 2)
    res["extrapolated_prepare_s_per_scenario"] = round(t_prep * k)
    res["extrapolated_day_step_s"] = round(m["wall_time_s"] / max(1, len(wl)) * k, 1)
    log(f"timing: {json.dumps(res)}")
    with open(os.path.join(HEAT_DIR, "logs", "timing.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=SCENARIOS)
    ap.add_argument("--season", choices=SEASONS)
    ap.add_argument("--all", action="store_true", help="both scenarios, both seasons")
    ap.add_argument("--cpu", action="store_true",
                    help="everything on the CPU, prepare included, caches in <scenario>_cpu")
    ap.add_argument("--time-subset", action="store_true",
                    help="time a 500 x 500 crop for 3 daytime steps and extrapolate")
    args = ap.parse_args()
    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    if args.time_subset:
        time_subset(log=log)
        return 0
    if args.all:
        jobs = [(sc, se) for sc in SCENARIOS for se in SEASONS]
    else:
        if not (args.scenario and args.season):
            raise SystemExit("give --scenario and --season, or --all")
        jobs = [(args.scenario, args.season)]
    if args.cpu:
        # Everything on the CPU, sky view included. solweig 0.1.0b96's GPU path
        # disagrees with its CPU path on vegetation (shadow and sky view); see
        # tools/heat_crosscheck.py. The CPU caches are kept apart.
        import solweig
        solweig.disable_gpu()
    surfaces = {}
    for sc, se in jobs:
        if sc not in surfaces:
            cache = os.path.join(HEAT_DIR, "cache", f"{sc}_cpu") if args.cpu else None
            surfaces[sc], _ = prepare(sc, cache_dir=cache, log=log)
        run(sc, se, surface=surfaces[sc], gpu=not args.cpu, log=log,
            prepared_with="cpu" if args.cpu else "gpu")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
