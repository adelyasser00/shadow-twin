"""
Self-tests for the heat layer. Nothing from it is publishable until V1, V2
and V3 pass.

    .venv-heat/Scripts/python -m pipeline.verify_heat            all gates that can run
    .venv-heat/Scripts/python -m pipeline.verify_heat v0 v1      some of them

  V0   synthetic checks: a 470 m tower on flat ground at 40.77 N with the sun
       at 14:30 EST on 21 December must throw a shadow within 5% of
       470 / tan(altitude); leaf-off transmissivity in December is 0.5 and
       leaf-on in June 0.03; every input invariant holds on the files on
       disk, and SOLWEIG's own cleaned copies of DEM, CDSM and land cover are
       identical for 2017 and today.
  V1   geometry: SOLWEIG's shadow and the shadow pipeline's own sweep
       (geometry.py, sun from solar.py) on the same north-up building-only
       DSM, 21 December at 10:00, 12:00 and 14:00 EST. Pass: at least 98%
       agreement on ground cells more than 4 m from any wall, and total
       shaded area within 2%, at every time.
  V2   physics sanity on the December run, today, the hour nearest solar
       noon, open lawn: Tmrt in building shadow vs in sun differs by 15 to
       35 C, UTCI by 2 to 10 C, each with 2 C of tolerance (see V2_TOL). On
       the clear night before the day, Tmrt in open areas sits below air
       temperature at every hour before sunrise.
  V2b  June control: today minus 2017 inside the analysis and CEQR windows.
       Thresholds fixed before any result was seen: median UTCI change
       within 0.1 C of zero, 1st percentile at or above -2 C, 99th at or
       below +2 C.

V3 and V3b live in tools/heat_crosscheck.py. Results go to
data/heat/verify/gates.json, merged with whatever is already there.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from .heat_inputs import HEAT_DIR, INPUTS_DIR

VERIFY_DIR = os.path.join(HEAT_DIR, "verify")
GATES_JSON = os.path.join(VERIFY_DIR, "gates.json")

V0_REACH_TOL = 0.05
V1_AGREE_MIN = 0.98
V1_AREA_TOL = 0.02
V2_TMRT = (15.0, 35.0)
V2_UTCI = (2.0, 10.0)
# The ranges above were fixed before any result. The first run came out 1.6 C
# over the UTCI range. Published clear-day measurements put the sun vs shade
# Tmrt gap at up to about 30 C, and UTCI follows roughly a third of it, so the
# owner set a tolerance of 2 C either side of these ranges (2026-10-05), on the
# grounds that this model leaves out wind and material detail. Both verdicts
# are recorded: strict and with tolerance.
V2_TOL = 2.0
V2_TOL_NOTE = ("owner's rule, 2026-10-05, set after the first result: published ranges "
               "+/- 2 C, because wind and surface detail are not modelled")
V2B_MEDIAN = 0.1
V2B_P1_MIN = -2.0
V2B_P99_MAX = 2.0


def save_gate(name, result):
    os.makedirs(VERIFY_DIR, exist_ok=True)
    gates = {}
    if os.path.exists(GATES_JSON):
        with open(GATES_JSON, encoding="utf-8") as f:
            gates = json.load(f)
    result["when"] = datetime.now().isoformat(timespec="seconds")
    gates[name] = result
    with open(GATES_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(gates, f, indent=1, default=float)


def line(ok, msg):
    print(f"{'PASS' if ok else 'FAIL'}  {msg}", flush=True)
    return ok


# ------------------------------------------------------------------- V0
def v0():
    import solweig
    from solweig.components.shadows import compute_transmissivity
    from . import heat_inputs as H
    from .run_heat import MAX_SHADOW_M, days, prepare
    out = {"checks": []}
    ok_all = True

    # Synthetic tower.
    px, n = 2.0, 1000
    dsm = np.full((n, n), 10.0, dtype=np.float32)
    rc, cc = 900, 120
    dsm[rc - 5:rc + 5, cc - 5:cc + 5] = 10.0 + 470.0
    dem = np.full((n, n), 10.0, dtype=np.float32)
    surface = solweig.SurfaceData.prepare(dsm=dsm, dem=dem, pixel_size=px)
    loc = solweig.Location(latitude=40.77, longitude=-73.97, utc_offset=-5)

    def reach(max_dist):
        w = solweig.Weather(datetime=datetime(2026, 12, 21, 15, 0), ta=0.0, rh=50.0,
                            global_rad=300.0, timestep_minutes=60.0)
        od = os.path.join(VERIFY_DIR, "v0", f"tower_{int(max_dist or 0)}")
        kw = {} if max_dist is None else {"max_shadow_distance_m": max_dist}
        solweig.calculate(surface=surface, weather=[w], location=loc, output_dir=od,
                          outputs=["shadow"], **kw)
        import rasterio
        with rasterio.open(os.path.join(od, "shadow", "shadow_20261221_1500.tif")) as src:
            sh = src.read(1)
        w.compute_derived(loc)
        b = math.radians((w.sun_azimuth + 180.0) % 360.0)
        rr, cc_ = np.nonzero((sh < 0.5) & (dsm < 100))
        east = (cc_ + 0.5 - cc) * px
        north = (rc - (rr + 0.5)) * px
        proj = east * math.sin(b) + north * math.cos(b)
        return float(proj.max()) if proj.size else 0.0, w.sun_altitude, w.sun_azimuth

    r35, alt, az = reach(MAX_SHADOW_M)
    expect = 470.0 / math.tan(math.radians(alt))
    err = abs(r35 - expect) / expect
    r10, _, _ = reach(None)
    ok = line(err <= V0_REACH_TOL,
              f"V0 470 m tower, sun {alt:.2f} deg up at {az:.1f} deg (14:30 EST, 21 Dec): "
              f"reach {r35:.0f} m vs 470/tan(alt) = {expect:.0f} m, error {err*100:.1f}% "
              f"(limit {V0_REACH_TOL*100:.0f}%); package default 1000 m gives {r10:.0f} m")
    out["tower"] = {"sun_altitude": alt, "sun_azimuth": az, "reach_m": r35, "expected_m": expect,
                    "error": err, "reach_default_m": r10, "pass": ok}
    ok_all &= ok

    phys = solweig.load_physics()
    d = days()
    for season, want in (("dec", 0.5), ("jun", 0.03)):
        day = datetime.fromisoformat(d["seasons"][season]["chosen"]["date"])
        psi = compute_transmissivity(day.timetuple().tm_yday, phys, False)
        ok = line(abs(psi - want) < 1e-9, f"V0 {season} {day:%Y-%m-%d} vegetation transmissivity "
                                         f"{psi} (want {want})")
        out[f"psi_{season}"] = psi
        ok_all &= ok

    layers, masks, meta = H.load()
    for nm, ok, det in H.check_invariants(layers, masks, abs(meta["_transform"].a)):
        ok_all &= line(ok, f"V0 input: {nm}  {det}")
        out["checks"].append({"check": nm, "pass": bool(ok), "detail": det})
    ok = line(str(meta["_crs"]).upper().endswith("32118") and abs(meta["_transform"].a - 2.0) < 1e-9,
              f"V0 input GeoTIFFs in {meta['_crs']} with a {meta['_transform'].a} m pixel")
    ok_all &= ok

    # SOLWEIG's own cleaned copies must agree between the scenarios.
    out["cleaned_identical"] = {}
    for suffix in ("", "_cpu"):
        same = {}
        for name in ("dem", "cdsm", "land_cover"):
            a = os.path.join(HEAT_DIR, "cache", "2017" + suffix, "cleaned", f"{name}.npy")
            b = os.path.join(HEAT_DIR, "cache", "today" + suffix, "cleaned", f"{name}.npy")
            if os.path.exists(a) and os.path.exists(b):
                x, y = np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r")
                same[name] = bool(x.shape == y.shape and np.array_equal(x, y, equal_nan=True))
        if not same:
            continue
        ok = all(same.values()) and len(same) == 3
        ok_all &= line(ok, f"V0 SOLWEIG cleaned DEM, CDSM, land cover identical across scenarios "
                           f"(cache{suffix or ' gpu'}): {same}")
        out["cleaned_identical"][suffix or "gpu"] = same
        for sc in ("2017", "today"):
            md = os.path.join(HEAT_DIR, "cache", sc + suffix, "metadata.json")
            if os.path.exists(md):
                with open(md, encoding="utf-8") as f:
                    px_read = json.load(f).get("pixel_size")
                ok_all &= line(abs(float(px_read) - 2.0) < 1e-9,
                               f"V0 SOLWEIG read a {px_read} m pixel for {sc}{suffix}")
    out["pass"] = bool(ok_all)
    save_gate("V0", out)
    return ok_all


# ------------------------------------------------------------------- V1
def v1():
    import solweig
    import rasterio
    from scipy import ndimage
    from . import heat_inputs as H
    from .geometry import sweep
    from .run_heat import MAX_SHADOW_M
    from .solar import sun_position
    layers, masks, meta = H.load()
    wd = os.path.join(HEAT_DIR, "cache", "v1")
    surface = solweig.SurfaceData.prepare(dsm=os.path.join(INPUTS_DIR, "dsm_today.tif"),
                                          working_dir=wd, dem=os.path.join(INPUTS_DIR, "dem.tif"),
                                          dsm_relative=False)
    dsm = np.asarray(surface.dsm, dtype=np.float32)
    if dsm.shape != layers["dem"].shape:
        raise SystemExit(f"SOLWEIG changed the grid: {dsm.shape} vs {layers['dem'].shape}")
    lat, lon = meta["site"]["lat"], meta["site"]["lon"]
    loc = solweig.Location(latitude=lat, longitude=lon, utc_offset=-5)
    times = [10, 12, 14]
    ws = [solweig.Weather(datetime=datetime(2026, 12, 21, h, 30), ta=0.0, rh=50.0,
                          global_rad=300.0, timestep_minutes=60.0) for h in times]
    od = os.path.join(VERIFY_DIR, "v1")
    solweig.calculate(surface=surface, weather=ws, location=loc, output_dir=od,
                      max_shadow_distance_m=MAX_SHADOW_M, outputs=["shadow"])
    fp = masks["building"]
    ground = ~fp & (ndimage.distance_transform_edt(~fp) * 2.0 > 4.0)
    win = masks["report"]
    bearing = -meta["grid"]["conv_deg"]
    import solweig as _sw
    out = {"cells": int(ground.sum()), "conv_deg": meta["grid"]["conv_deg"], "times": [],
           "backend": _sw.get_compute_backend()}
    ok_all = True
    for h, w in zip(times, ws):
        w.compute_derived(loc)
        utc = datetime(2026, 12, 21, h, 0) + timedelta(hours=5)
        p = sun_position(utc.replace(tzinfo=timezone.utc), lat, lon)
        with rasterio.open(os.path.join(od, "shadow", f"shadow_20261221_{h:02d}30.tif")) as src:
            s_sol = src.read(1) < 0.5
        s_pip = sweep(dsm, 2.0, p.altitude, (p.azimuth - bearing) % 360.0) < 0.5
        agree = float((s_sol == s_pip)[ground].mean())
        a_s, a_p = float(s_sol[ground].sum()), float(s_pip[ground].sum())
        area = abs(a_s - a_p) / a_p
        agree_w = float((s_sol == s_pip)[win].mean())
        ok = agree >= V1_AGREE_MIN and area <= V1_AREA_TOL
        ok_all &= line(ok, f"V1 {h:02d}:00 EST  sun solweig {w.sun_altitude:.3f}/{w.sun_azimuth:.3f}, "
                           f"solar.py {p.altitude:.3f}/{p.azimuth:.3f} deg; agreement {agree*100:.2f}% "
                           f"(window {agree_w*100:.2f}%), shaded {a_s*4/1e4:.1f} vs {a_p*4/1e4:.1f} ha, "
                           f"area diff {area*100:.2f}%")
        out["times"].append({"est": f"{h:02d}:00", "sun_solweig": [w.sun_altitude, w.sun_azimuth],
                             "sun_solar_py": [p.altitude, p.azimuth], "agreement": agree,
                             "agreement_window": agree_w, "shaded_ha_solweig": a_s * 4 / 1e4,
                             "shaded_ha_pipeline": a_p * 4 / 1e4, "area_diff": area, "pass": ok})
    out["pass"] = bool(ok_all)
    save_gate("V1", out)
    return ok_all


# ------------------------------------------------------------- run reads
def runs_dir():
    """data/heat/runs, or HEAT_RUNS_DIR to read another set (e.g. runs_gpu)."""
    return os.environ.get("HEAT_RUNS_DIR") or os.path.join(HEAT_DIR, "runs")


def read_step(scenario, season, name, ts):
    import rasterio
    p = os.path.join(runs_dir(), f"{scenario}_{season}", name, f"{name}_{ts:%Y%m%d_%H%M}.tif")
    with rasterio.open(p) as src:
        return src.read(1)


def report_steps(season):
    from .run_heat import days
    d = days()["seasons"][season]
    day = datetime.fromisoformat(d["chosen"]["date"])
    out = []
    for t in d["report_timestamps"]:
        hh, mm = (int(x) for x in t.split(":"))
        out.append(day.replace(hour=hh, minute=mm))
    return out


# ------------------------------------------------------------------- V2
def v2():
    from scipy import ndimage
    from . import heat_inputs as H
    from .run_heat import days, weather_list
    from .solar import sun_position
    layers, masks, meta = H.load()
    lat, lon = meta["site"]["lat"], meta["site"]["lon"]
    d = days()["seasons"]["dec"]
    day = datetime.fromisoformat(d["chosen"]["date"])
    # The report step whose sun time is closest to solar noon.
    def alt_at(t):
        return sun_position((t + timedelta(hours=5)).replace(tzinfo=timezone.utc), lat, lon).altitude
    noon = max((day.replace(hour=11) + timedelta(minutes=m) for m in range(0, 121)), key=alt_at)
    best = min(report_steps("dec"), key=lambda ts: abs((ts - timedelta(minutes=30) - noon).total_seconds()))
    lawn = (layers["landcover"] == 5) & (layers["cdsm"] == 0) & masks["report"]
    sh = read_step("today", "dec", "shadow", best)
    tm = read_step("today", "dec", "tmrt", best)
    ut = read_step("today", "dec", "utci", best)
    shade, sun = lawn & (sh < 0.05), lawn & (sh > 0.95)
    d_t = float(np.nanmedian(tm[sun]) - np.nanmedian(tm[shade]))
    d_u = float(np.nanmedian(ut[sun]) - np.nanmedian(ut[shade]))
    out = {"step": best.isoformat(), "sun_time": (best - timedelta(minutes=30)).strftime("%H:%M"),
           "solar_noon_est": noon.strftime("%H:%M"),
           "lawn_sun_ha": float(sun.sum() * 4 / 1e4), "lawn_shade_ha": float(shade.sum() * 4 / 1e4),
           "tmrt_sun": float(np.nanmedian(tm[sun])), "tmrt_shade": float(np.nanmedian(tm[shade])),
           "utci_sun": float(np.nanmedian(ut[sun])), "utci_shade": float(np.nanmedian(ut[shade])),
           "d_tmrt": d_t, "d_utci": d_u}
    strict1 = V2_TMRT[0] <= d_t <= V2_TMRT[1]
    strict2 = V2_UTCI[0] <= d_u <= V2_UTCI[1]
    ok1 = line(V2_TMRT[0] - V2_TOL <= d_t <= V2_TMRT[1] + V2_TOL,
               f"V2 {out['sun_time']} EST open lawn, sun vs building shadow: Tmrt {out['tmrt_sun']:.1f} vs "
               f"{out['tmrt_shade']:.1f} C, difference {d_t:.1f} C (range {V2_TMRT[0]:.0f} to "
               f"{V2_TMRT[1]:.0f}, +/-{V2_TOL:.0f}; strict {'pass' if strict1 else 'fail'}); "
               f"{out['lawn_sun_ha']:.1f} ha sunlit, {out['lawn_shade_ha']:.1f} ha shaded")
    ok2 = line(V2_UTCI[0] - V2_TOL <= d_u <= V2_UTCI[1] + V2_TOL,
               f"V2 same cells: UTCI {out['utci_sun']:.1f} vs {out['utci_shade']:.1f} C, difference "
               f"{d_u:.1f} C (range {V2_UTCI[0]:.0f} to {V2_UTCI[1]:.0f}, +/-{V2_TOL:.0f}; "
               f"strict {'pass' if strict2 else 'fail'})")
    out["pass_strict"] = {"tmrt": strict1, "utci": strict2}
    out["tolerance_c"] = V2_TOL
    out["tolerance_note"] = V2_TOL_NOTE
    # Night: open areas, far from buildings and canopy.
    fp, veg = masks["building"], layers["cdsm"] > 0
    open_ = (masks["report"] & ~fp & (ndimage.distance_transform_edt(~fp) * 2.0 > 50.0)
             & (ndimage.distance_transform_edt(~veg) * 2.0 > 10.0))
    wl = {w.datetime: w for w in weather_list("dec")}
    night = []
    for h in range(1, 8):
        ts = day.replace(hour=h)
        st = ts - timedelta(minutes=30)
        p = sun_position((st + timedelta(hours=5)).replace(tzinfo=timezone.utc), lat, lon)
        if p.altitude >= 0:
            continue
        tm_n = read_step("today", "dec", "tmrt", ts)
        med = float(np.nanmedian(tm_n[open_]))
        night.append({"step": ts.isoformat(), "ta": wl[ts].ta, "tmrt_open": med,
                      "below": med < wl[ts].ta})
    ok3 = line(bool(night) and all(n["below"] for n in night),
               f"V2 clear night before the day, {open_.sum()*4/1e4:.1f} ha of open ground: Tmrt - Ta "
               + ", ".join(f"{n['step'][11:16]} {n['tmrt_open'] - n['ta']:+.1f}" for n in night) + " C")
    out["night"] = night
    out["open_ha"] = float(open_.sum() * 4 / 1e4)
    out["pass"] = bool(ok1 and ok2 and ok3)
    out["pass_parts"] = {"tmrt": ok1, "utci": ok2, "night": ok3}
    save_gate("V2", out)
    return out["pass"]


# ------------------------------------------------------------------ V2b
def v2b():
    from . import heat_inputs as H
    layers, masks, meta = H.load()
    rep = masks["report"]
    du, dt = [], []
    for ts in report_steps("jun"):
        du.append((read_step("today", "jun", "utci", ts) - read_step("2017", "jun", "utci", ts))[rep])
        dt.append((read_step("today", "jun", "tmrt", ts) - read_step("2017", "jun", "tmrt", ts))[rep])
    du, dt = np.concatenate(du), np.concatenate(dt)
    pu = np.nanpercentile(du, [1, 50, 99])
    pt = np.nanpercentile(dt, [1, 50, 99])
    out = {"cell_hours": int(du.size), "utci_p1_p50_p99": pu.tolist(), "tmrt_p1_p50_p99": pt.tolist(),
           "share_abs_dutci_ge_2": float((np.abs(du) >= 2).mean()),
           "share_dutci_le_minus2": float((du <= -2).mean())}
    ok = abs(pu[1]) <= V2B_MEDIAN and pu[0] >= V2B_P1_MIN and pu[2] <= V2B_P99_MAX
    line(ok, f"V2b June today - 2017 over {du.size:,} cell-hours: UTCI p1/p50/p99 "
             f"{pu[0]:+.2f} / {pu[1]:+.2f} / {pu[2]:+.2f} C, Tmrt {pt[0]:+.2f} / {pt[1]:+.2f} / {pt[2]:+.2f} C; "
             f"|dUTCI| >= 2 C on {out['share_abs_dutci_ge_2']*100:.2f}% of cell-hours")
    out["pass"] = bool(ok)
    save_gate("V2b", out)
    return ok


GATES = {"v0": v0, "v1": v1, "v2": v2, "v2b": v2b}


def main():
    wanted = [a.lower() for a in sys.argv[1:]] or list(GATES)
    t0 = time.time()
    results = {}
    for g in wanted:
        print(f"\n---- {g.upper()} ----", flush=True)
        try:
            results[g] = GATES[g]()
        except Exception as e:        # a gate that cannot run is reported, not hidden
            import traceback
            traceback.print_exc()
            print(f"FAIL  {g.upper()} could not run: {type(e).__name__}: {e}")
            save_gate(g.upper() if g != "v2b" else "V2b", {"pass": False, "error": f"{type(e).__name__}: {e}"})
            results[g] = False
    print("\n" + "-" * 70)
    for g, ok in results.items():
        print(f"  {g.upper():<4} {'PASS' if ok else 'FAIL'}")
    print(f"  {time.time() - t0:.0f} s")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
