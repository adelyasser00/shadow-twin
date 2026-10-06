"""
Felt temperature with the wind each spot actually gets.

    .venv-heat/Scripts/python -m pipeline.wind_heat

The heat layer gave every cell of the park the weather station's wind. Here
each 2 m cell gets the station wind times its simulated ratio to the open lawn
(pipeline/wind_export.py writes that ratio, k_heatgrid.npy, next to each run),
and UTCI is recomputed with SOLWEIG's own UTCI routine from the same mean
radiant temperature, air temperature and humidity. Nothing else changes, so
any difference from the heat layer is the wind's doing.

Each hour uses the simulated direction nearest to the one the weather file
reports (the sector method of pedestrian wind studies, NEN 8100). UTCI takes
wind at 10 m between 0.5 and 17 m/s; the local wind is held inside that range.
The ratio is taken at the same height for every cell, the first open cell
above the ground, so it is used directly as a ratio of 10 m winds.

Check before anything is written (W0): with the station wind everywhere, the
recomputed UTCI must reproduce SOLWEIG's saved UTCI.

Writes viewer/wind/felt_*.png, viewer/wind/windheat.json and
data/wind/wind_numbers.csv.
"""

from __future__ import annotations

import base64
import csv
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from .wind_domain import WIND_DIR
from .wind_export import OUT_DIR, runs_available

UTCI_WIND_MIN, UTCI_WIND_MAX = 0.5, 17.0
CELL_HA = 4.0 / 1e4
NUMBERS_CSV = os.path.join(WIND_DIR, "wind_numbers.csv")


def epw_wind(ts):
    """Wind direction and speed of the weather file's row for an hour-ending timestamp."""
    from .heat_weather import row_for
    from .run_heat import epw_path
    global _EPW
    if "_EPW" not in globals():
        rows = {}
        with open(epw_path(), encoding="utf-8") as f:
            for ln in f.read().splitlines()[8:]:
                p = ln.split(",")
                if len(p) > 21:
                    rows[(int(p[1]), int(p[2]), int(p[3]))] = {"wd": float(p[20]), "ws": float(p[21])}
        _EPW = rows
    r = row_for(ts, _EPW)
    return r["wd"], r["ws"]


def nearest_run(runs, scenario, season, wd):
    cands = [m for m in runs if m["scenario"] == scenario and m["season"] == season
             and m["res_m"] == 8.0 and "small" not in m["_name"]]
    if not cands:
        return None
    return min(cands, key=lambda m: abs((m["theta_from_deg"] - wd + 180.0) % 360.0 - 180.0))


def utci(ta, rh, tmrt, wind):
    from solweig.rustalgos import utci as U
    return np.asarray(U.utci_grid(float(ta), float(rh), np.ascontiguousarray(tmrt, np.float32),
                                  np.ascontiguousarray(wind, np.float32)))


def main():
    from . import export
    from .heat_export import (CHANGE_BREAKS, FELT_BREAKS, change_palette, columns, felt_palette,
                              heat_grid, png_change, section_lonlat, section_means)
    from .heat_inputs import load
    from .run_heat import weather_list
    from .solar import sun_position
    from .verify_heat import read_step, report_steps
    from pyproj import Transformer

    layers, masks, meta = load()
    report = masks["report"]
    grid = heat_grid(meta, masks)
    fsl = grid.frame_slice
    idx, bounds = grid.northup_index()
    painter = export.Painter(idx, bounds)
    to_ll = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True).transform
    bmask = masks["building"][fsl]
    felt_cols, chg_cols = felt_palette(), change_palette()
    runs = runs_available()
    version = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        if f.startswith("felt_") and f.endswith(".png"):
            os.remove(os.path.join(OUT_DIR, f))

    probe = section_means(np.zeros(report.shape, np.float32), report, grid)
    sec_i, sec_j = np.nonzero(np.isfinite(probe))

    def felt_ints(arr):
        m = section_means(arr, report, grid)[sec_i, sec_j]
        return [int(round(float(v) * 2)) for v in np.nan_to_num(m, nan=0.0)]

    def png(name, uri):
        with open(os.path.join(OUT_DIR, name), "wb") as f:
            f.write(base64.b64decode(uri.split(",", 1)[1]))
        return f"wind/{name}?v={version}"

    lat, lon = meta["site"]["lat"], meta["site"]["lon"]
    w0 = []
    numbers = []
    seasons = []
    for season in ("dec", "jun"):
        wl = {w.datetime: w for w in weather_list(season)}
        steps = []
        any5_w = np.zeros(report.sum(), bool)
        any5_u = np.zeros(report.sum(), bool)
        for ts in report_steps(season):
            st = ts - timedelta(minutes=30)
            hhmm = st.strftime("%H%M")
            w = wl[ts]
            wd, ws_raw = epw_wind(ts)
            out = {"time": st.strftime("%H:%M"), "wind_dir": wd, "wind_m_s": ws_raw, "air_c": w.ta}
            per = {}
            for sc in ("2017", "today"):
                m = nearest_run(runs, sc, season, wd)
                tmrt = read_step(sc, season, "tmrt", ts)
                saved = read_step(sc, season, "utci", ts)
                uni = utci(w.ta, w.rh, tmrt, np.full(tmrt.shape, max(ws_raw, UTCI_WIND_MIN), np.float32))
                ok = np.isfinite(saved) & np.isfinite(uni) & report
                w0.append(float(np.median(np.abs(uni[ok] - saved[ok]))))
                if m is None:
                    per[sc] = None
                    continue
                k = np.load(os.path.join(m["_dir"], "k_heatgrid.npy"))
                wind = np.clip(np.nan_to_num(k, nan=1.0) * ws_raw, UTCI_WIND_MIN, UTCI_WIND_MAX)
                per[sc] = {"run": m["_name"], "dir": m["theta_from_deg"], "k": k, "wind": wind,
                           "felt": utci(w.ta, w.rh, tmrt, wind), "uniform": saved}
            if per["2017"] is None or per["today"] is None:
                print(f"{season} {out['time']}: no wind run near {wd:.0f} degrees for both years, skipped")
                continue
            a, b = per["2017"], per["today"]
            d_wind = b["felt"] - a["felt"]
            d_uni = b["uniform"] - a["uniform"]
            r = report
            img = {
                "today": png(f"felt_{season}_{hhmm}_today.png",
                             painter.png(b["felt"][fsl], FELT_BREAKS, felt_cols, alpha=215, mask=bmask)),
                "y2017": png(f"felt_{season}_{hhmm}_2017.png",
                             painter.png(a["felt"][fsl], FELT_BREAKS, felt_cols, alpha=215, mask=bmask)),
                "change": png(f"felt_{season}_{hhmm}_change.png",
                              png_change(painter, d_wind[fsl], bmask, chg_cols)),
            }
            any5_w |= d_wind[r] <= -5
            any5_u |= d_uni[r] <= -5
            p = sun_position((st + timedelta(hours=5)).replace(tzinfo=timezone.utc), lat, lon)
            out.update({
                "sun": {"alt": round(p.altitude, 2), "az": round(p.azimuth, 2)},
                "img": img, "run_dir": {"2017": a["dir"], "today": b["dir"]},
                "park_wind_today_m_s": round(float(np.nanmedian(b["wind"][r])), 2),
                "park_wind_2017_m_s": round(float(np.nanmedian(a["wind"][r])), 2),
                "park_felt_today_c": round(float(np.nanmedian(b["felt"][r])), 1),
                "park_felt_2017_c": round(float(np.nanmedian(a["felt"][r])), 1),
                "park_felt_today_uniform_c": round(float(np.nanmedian(b["uniform"][r])), 1),
                "ha_colder_2": round(float((d_wind[r] <= -2).sum() * CELL_HA), 2),
                "ha_colder_5": round(float((d_wind[r] <= -5).sum() * CELL_HA), 2),
                "ha_colder_2_uniform": round(float((d_uni[r] <= -2).sum() * CELL_HA), 2),
                "ha_colder_5_uniform": round(float((d_uni[r] <= -5).sum() * CELL_HA), 2),
                "biggest_drop_c": round(float(-np.nanmin(d_wind[r])), 1),
                "wind_part_median_c": round(float(np.nanmedian((d_wind - d_uni)[r])), 2),
                "wind_part_p05_c": round(float(np.nanpercentile((d_wind - d_uni)[r], 5)), 2),
                "wind_part_p95_c": round(float(np.nanpercentile((d_wind - d_uni)[r], 95)), 2),
                "local_vs_station_today_median_c": round(float(np.nanmedian((b["felt"] - b["uniform"])[r])), 2),
                "cols": columns(d_wind, report, grid, to_ll),
                "felt2": {"today": felt_ints(b["felt"]), "y2017": felt_ints(a["felt"])},
            })
            steps.append(out)
            print(f"{season} {out['time']}: wind {wd:.0f} deg {ws_raw:.1f} m/s (runs {a['dir']:.1f}); "
                  f"park wind {out['park_wind_2017_m_s']} -> {out['park_wind_today_m_s']} m/s; "
                  f"5 C+ colder {out['ha_colder_5_uniform']} ha with one wind, {out['ha_colder_5']} ha "
                  f"with local wind; wind part median {out['wind_part_median_c']:+.2f} C", flush=True)
        if steps:
            seasons.append({"id": season, "steps": steps,
                            "ha_any_hour_le_-5": round(float(any5_w.sum() * CELL_HA), 2),
                            "ha_any_hour_le_-5_uniform": round(float(any5_u.sum() * CELL_HA), 2)})
            numbers += [(f"{season}_ha_any_hour_dutci_le_-5_local_wind", seasons[-1]["ha_any_hour_le_-5"], "ha"),
                        (f"{season}_ha_any_hour_dutci_le_-5_station_wind", seasons[-1]["ha_any_hour_le_-5_uniform"], "ha")]
    gate = {"W0_median_abs_diff_c": round(float(np.max(w0)) if w0 else -1, 4),
            "pass": bool(w0) and float(np.max(w0)) < 0.05}
    print(f"W0 (recomputed UTCI with the station wind reproduces SOLWEIG): worst median "
          f"{gate['W0_median_abs_diff_c']} C, {'pass' if gate['pass'] else 'FAIL'}")
    bundle = {"generated": datetime.now().isoformat(timespec="seconds"), "gate_w0": gate,
              "seasons": seasons}
    with open(os.path.join(OUT_DIR, "windheat.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(bundle, f, separators=(",", ":"))
    with open(NUMBERS_CSV, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["key", "value", "unit"])
        wr.writerow(["gate_w0_median_abs_diff_c", gate["W0_median_abs_diff_c"], "C"])
        for s in seasons:
            for stp in s["steps"]:
                for k in ("park_wind_today_m_s", "park_wind_2017_m_s", "park_felt_today_c",
                          "park_felt_today_uniform_c", "ha_colder_5", "ha_colder_5_uniform",
                          "wind_part_median_c", "local_vs_station_today_median_c"):
                    wr.writerow([f"{s['id']}_{stp['time'].replace(':', '')}_{k}", stp[k], ""])
        for row in numbers:
            wr.writerow(row)
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
