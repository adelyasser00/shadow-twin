"""
Today minus 2017, for the heat layer, the way compare_runs.py does it for sun.

    .venv-heat/Scripts/python tools/compare_heat.py

Reads the SOLWEIG runs in data/heat/runs/, keeps only the reported cells
(Central Park within 1.5 km of 59th Street, no water, no buildings, away from
the domain's west, north and east edges) and only the hours inside the CEQR
window, and answers, for the December day and for the June control:

  how many hectares feel at least 2, 5 or 10 C colder (UTCI) for at least
  one hour, and how many hectare-hours that adds up to
  the same in 500 m bands north of 59th Street
  how many hectare-hours moved into a colder UTCI stress category
  which hour is worst

A cell "changes" when its UTCI moves by 0.1 C or more in some hour (the
precision UTCI is reported at). Percentiles of the change are over the
cell-hours that moved that much.

Writes per-hour change rasters to data/heat/compare/, maps to
data/heat/figures/, and every number to data/heat/heat_numbers.csv, each one
marked publishable only if gates V1, V2 and V3 all passed.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pipeline import heat_inputs as H                     # noqa: E402
from pipeline.run_heat import days                         # noqa: E402
from pipeline.verify_heat import GATES_JSON, read_step, report_steps, runs_dir  # noqa: E402

# HEAT_RUNS_DIR=data/heat/runs_gpu reads another set of runs; its tables go
# to compare_runs_gpu/ and heat_numbers_runs_gpu.csv, and no figures are made.
_ALT = os.path.basename(os.path.normpath(runs_dir())) if os.environ.get("HEAT_RUNS_DIR") else None
COMPARE_DIR = os.path.join(H.HEAT_DIR, "compare" + (f"_{_ALT}" if _ALT else ""))
FIG_DIR = os.path.join(H.HEAT_DIR, "figures")
NUMBERS_CSV = os.path.join(H.HEAT_DIR, "heat_numbers" + (f"_{_ALT}" if _ALT else "") + ".csv")
THRESHOLDS = (-2.0, -5.0, -10.0)
BANDS = ((0, 500), (500, 1000), (1000, 1500))
CHANGE_MIN = 0.1
CELL_HA = 4.0 / 1e4

# Standard UTCI assessment scale (Blazejczyk et al. 2013), lower bounds, C.
UTCI_EDGES = [-40.0, -27.0, -13.0, 0.0, 9.0, 26.0, 32.0, 38.0, 46.0]
UTCI_NAMES = ["extreme cold stress", "very strong cold stress", "strong cold stress",
              "moderate cold stress", "slight cold stress", "no thermal stress",
              "moderate heat stress", "strong heat stress", "very strong heat stress",
              "extreme heat stress"]


def category(u):
    return np.digitize(u, UTCI_EDGES)          # 0 extreme cold ... 9 extreme heat


def season_tables(season, masks, rows, log=print):
    rep = masks["report"]
    dist = masks["dist59"]
    steps = report_steps(season)
    label = {ts: (ts - timedelta(minutes=30)).strftime("%H:%M") for ts in steps}
    du_all, dt_all, cats = [], [], []
    per_hour = []
    out_dir = os.path.join(COMPARE_DIR, season)
    os.makedirs(out_dir, exist_ok=True)
    layers_meta = H.load()[2]
    tf, crs = layers_meta["_transform"], layers_meta["_crs"]
    import rasterio
    u17_by, unow_by = {}, {}
    for ts in steps:
        u17 = read_step("2017", season, "utci", ts)
        unow = read_step("today", season, "utci", ts)
        t17 = read_step("2017", season, "tmrt", ts)
        tnow = read_step("today", season, "tmrt", ts)
        du, dt = unow - u17, tnow - t17
        for nm, arr in (("dutci", du), ("dtmrt", dt)):
            p = os.path.join(out_dir, f"{nm}_{label[ts].replace(':', '')}.tif")
            with rasterio.open(p, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                               count=1, dtype="float32", crs=crs, transform=tf,
                               compress="deflate", tiled=True) as dst:
                dst.write(arr.astype(np.float32), 1)
        du_all.append(du[rep])
        dt_all.append(dt[rep])
        colder = category(unow[rep]) < category(u17[rep])
        cats.append(colder)
        per_hour.append({"sun_time": label[ts],
                         "ha_le_minus2": float((du[rep] <= -2).sum() * CELL_HA),
                         "ha_colder_category": float(colder.sum() * CELL_HA),
                         "median_dutci": float(np.nanmedian(du[rep])),
                         "min_dutci": float(np.nanmin(du[rep]))})
        u17_by[ts], unow_by[ts] = u17, unow
    DU = np.stack(du_all)          # hours x cells
    DT = np.stack(dt_all)
    CAT = np.stack(cats)
    res = {"season": season, "date": days()["seasons"][season]["chosen"]["date"],
           "hours": [label[t] for t in steps], "report_ha": float(rep.sum() * CELL_HA),
           "per_hour": per_hour}
    for thr in THRESHOLDS:
        hit = DU <= thr
        res[f"ha_any_hour_le_{int(thr)}"] = float(hit.any(axis=0).sum() * CELL_HA)
        res[f"ha_hours_le_{int(thr)}"] = float(hit.sum() * CELL_HA)
    moved = np.abs(DU) >= CHANGE_MIN
    res["ha_changed_any_hour"] = float(moved.any(axis=0).sum() * CELL_HA)
    vals = DU[moved]
    res["dutci_pct_1_5_50_95_99_changed"] = (np.percentile(vals, [1, 5, 50, 95, 99]).tolist()
                                             if vals.size else None)
    vt = DT[np.abs(DT) >= CHANGE_MIN]
    res["dtmrt_pct_1_5_50_95_99_changed"] = (np.percentile(vt, [1, 5, 50, 95, 99]).tolist()
                                             if vt.size else None)
    res["ha_hours_colder_category"] = float(CAT.sum() * CELL_HA)
    worst = max(per_hour, key=lambda h: h["ha_le_minus2"])
    res["worst_hour"] = worst["sun_time"]
    res["worst_hour_ha_le_minus2"] = worst["ha_le_minus2"]
    # Bands north of 59th Street.
    dcells = dist[rep]
    res["bands"] = []
    for lo, hi in BANDS:
        sel = (dcells >= lo) & (dcells < hi)
        b = {"from_m": lo, "to_m": hi, "area_ha": float(sel.sum() * CELL_HA)}
        for thr in THRESHOLDS:
            hit = DU[:, sel] <= thr
            b[f"ha_any_hour_le_{int(thr)}"] = float(hit.any(axis=0).sum() * CELL_HA)
            b[f"ha_hours_le_{int(thr)}"] = float(hit.sum() * CELL_HA)
        b["ha_hours_colder_category"] = float(CAT[:, sel].sum() * CELL_HA)
        b["mean_dutci"] = float(np.nanmean(DU[:, sel]))
        res["bands"].append(b)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)

    log(f"\n{season}: {res['date']}, reported {res['report_ha']:.1f} ha, hours (sun time EST) "
        + ", ".join(res["hours"]))
    log(f"  UTCI today minus 2017, hectares colder for at least one hour / hectare-hours:")
    for thr in THRESHOLDS:
        log(f"    <= {thr:+.0f} C   {res[f'ha_any_hour_le_{int(thr)}']:7.2f} ha   "
            f"{res[f'ha_hours_le_{int(thr)}']:8.2f} ha-h")
    p = res["dutci_pct_1_5_50_95_99_changed"]
    if p:
        log(f"  {res['ha_changed_any_hour']:.1f} ha move by {CHANGE_MIN} C or more in some hour; "
            f"change p1/p5/p50/p95/p99 over those cell-hours {p[0]:+.1f} / {p[1]:+.1f} / "
            f"{p[2]:+.1f} / {p[3]:+.1f} / {p[4]:+.1f} C")
    log(f"  {res['ha_hours_colder_category']:.2f} ha-h moved into a colder UTCI stress category")
    log(f"  worst hour {res['worst_hour']} EST: {res['worst_hour_ha_le_minus2']:.2f} ha at -2 C or colder")
    log("  band, m      area     <= -2 C (ha, ha-h)     <= -5 C (ha, ha-h)    <= -10 C (ha, ha-h)   colder cat ha-h")
    for b in res["bands"]:
        log(f"  {b['from_m']:>4}-{b['to_m']:<5} {b['area_ha']:6.1f} ha   "
            f"{b['ha_any_hour_le_-2']:6.2f} {b['ha_hours_le_-2']:7.2f}        "
            f"{b['ha_any_hour_le_-5']:6.2f} {b['ha_hours_le_-5']:7.2f}       "
            f"{b['ha_any_hour_le_-10']:6.2f} {b['ha_hours_le_-10']:7.2f}      "
            f"{b['ha_hours_colder_category']:7.2f}")
    return res, u17_by, unow_by


# ---------------------------------------------------------------- figures
def figures(res, u17_by, unow_by, masks, layers, meta, season, log=print):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    tf = meta["_transform"]
    win = masks["window"]
    rr, cc = np.nonzero(win)
    pad = 150                                     # 300 m around the window
    r0, r1 = max(0, rr.min() - pad), min(win.shape[0], rr.max() + pad)
    c0, c1 = max(0, cc.min() - pad), min(win.shape[1], cc.max() + pad)
    sl = (slice(r0, r1), slice(c0, c1))
    x0, y0 = tf.c + c0 * tf.a, tf.f + r0 * tf.e
    extent = [x0, x0 + (c1 - c0) * tf.a, y0 + (r1 - r0) * tf.e, y0]
    bld = masks["building"][sl]
    steps = {(ts - timedelta(minutes=30)).strftime("%H:%M"): ts for ts in u17_by}
    worst = steps[res["worst_hour"]]
    day_label = {"dec": "21 Dec-like clear day", "jun": "21 Jun-like clear day"}[season]

    def overlays(ax):
        xs = np.linspace(extent[0], extent[1], c1 - c0)
        ys = np.linspace(extent[3], extent[2], r1 - r0)
        ax.contour(xs, ys, masks["park"][sl].astype(float), levels=[0.5], colors="k", linewidths=0.8)
        d = masks["dist59"][sl]
        ax.contour(xs, ys, d, levels=[0.0], colors="k", linewidths=1.2)
        ax.contour(xs, ys, d, levels=[500, 1000, 1500], colors="0.4", linewidths=0.5, linestyles="--")
        iy, ix = np.unravel_index(np.argmin(np.abs(d) + (~masks["park"][sl]) * 1e9), d.shape)
        ax.annotate("59th St", (xs[ix], ys[iy]), xytext=(4, -12), textcoords="offset points", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    def show(ax, arr, cmap, vmin, vmax):
        a = np.ma.masked_where(bld | ~np.isfinite(arr), arr)
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("0.85")
        return ax.imshow(a, extent=extent, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")

    paths = []
    du_w = (unow_by[worst] - u17_by[worst])[sl]
    fig, ax = plt.subplots(figsize=(6, 7))
    im = show(ax, du_w, "RdBu_r", -12, 12)
    overlays(ax)
    ax.text(0.02, 0.02, f"{day_label}, {res['worst_hour']} EST", transform=ax.transAxes, fontsize=7,
            bbox=dict(fc="w", ec="none", alpha=0.8))
    fig.colorbar(im, ax=ax, shrink=0.7, label="UTCI today minus 2017, C")
    p = os.path.join(FIG_DIR, f"{season}_utci_change_worst_hour.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    mean = np.mean([(unow_by[t] - u17_by[t])[sl] for t in u17_by], axis=0)
    fig, ax = plt.subplots(figsize=(6, 7))
    im = show(ax, mean, "RdBu_r", -6, 6)
    overlays(ax)
    ax.text(0.02, 0.02, f"{day_label}, mean of {res['hours'][0]} to {res['hours'][-1]} EST",
            transform=ax.transAxes, fontsize=7, bbox=dict(fc="w", ec="none", alpha=0.8))
    fig.colorbar(im, ax=ax, shrink=0.7, label="UTCI today minus 2017, mean over the CEQR window, C")
    p = os.path.join(FIG_DIR, f"{season}_utci_change_window_mean.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    a17, anow = u17_by[worst][sl], unow_by[worst][sl]
    lo = float(np.nanpercentile(np.concatenate([a17[~bld], anow[~bld]]), 1))
    hi = float(np.nanpercentile(np.concatenate([a17[~bld], anow[~bld]]), 99))
    fig, axs = plt.subplots(1, 2, figsize=(10, 7))
    for ax, arr, nm in ((axs[0], a17, "2017"), (axs[1], anow, "today")):
        im = show(ax, arr, "viridis", lo, hi)
        overlays(ax)
        ax.text(0.02, 0.02, f"{nm}, {day_label}, {res['worst_hour']} EST", transform=ax.transAxes,
                fontsize=7, bbox=dict(fc="w", ec="none", alpha=0.8))
    fig.colorbar(im, ax=axs, shrink=0.6, label="UTCI, C (model estimate; absolute values run warm)")
    p = os.path.join(FIG_DIR, f"{season}_utci_2017_vs_today_worst_hour.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    for p in paths:
        log(f"  figure {p}")
    return paths


# ---------------------------------------------------------------- numbers
def publishable():
    if not os.path.exists(GATES_JSON):
        return False, {}
    with open(GATES_JSON, encoding="utf-8") as f:
        g = json.load(f)
    ok = all(g.get(k, {}).get("pass") is True for k in ("V1", "V2", "V3"))
    return ok, g


def write_numbers(results, extra_rows):
    ok, gates = publishable()
    status = "publishable" if ok else "internal"
    rows = []

    def add(key, value, unit, scope, note="", st=None):
        rows.append({"key": key, "value": value if not isinstance(value, float) else round(value, 4),
                     "unit": unit, "scope": scope, "status": st or status, "note": note})

    for res in results:
        s = res["season"]
        scope = f"{s} {res['date']}, report cells, CEQR window {res['hours'][0]}-{res['hours'][-1]} EST"
        add(f"{s}_report_area", res["report_ha"], "ha", scope)
        for thr in THRESHOLDS:
            add(f"{s}_ha_any_hour_dutci_le_{int(thr)}", res[f"ha_any_hour_le_{int(thr)}"], "ha", scope)
            add(f"{s}_ha_hours_dutci_le_{int(thr)}", res[f"ha_hours_le_{int(thr)}"], "ha-h", scope)
        add(f"{s}_ha_changed_any_hour", res["ha_changed_any_hour"], "ha", scope, f"|dUTCI| >= {CHANGE_MIN} C")
        for nm in ("dutci", "dtmrt"):
            p = res[f"{nm}_pct_1_5_50_95_99_changed"]
            if p:
                for q, v in zip((1, 5, 50, 95, 99), p):
                    add(f"{s}_{nm}_p{q}_changed", v, "C", scope, "over cell-hours that moved >= 0.1 C")
        add(f"{s}_ha_hours_colder_category", res["ha_hours_colder_category"], "ha-h", scope)
        add(f"{s}_worst_hour", res["worst_hour"], "EST sun time", scope)
        add(f"{s}_worst_hour_ha_dutci_le_-2", res["worst_hour_ha_le_minus2"], "ha", scope)
        for b in res["bands"]:
            bs = f"{scope}, {b['from_m']}-{b['to_m']} m north of 59th St"
            add(f"{s}_band{b['from_m']}_area", b["area_ha"], "ha", bs)
            for thr in THRESHOLDS:
                add(f"{s}_band{b['from_m']}_ha_any_hour_dutci_le_{int(thr)}", b[f"ha_any_hour_le_{int(thr)}"], "ha", bs)
                add(f"{s}_band{b['from_m']}_ha_hours_dutci_le_{int(thr)}", b[f"ha_hours_le_{int(thr)}"], "ha-h", bs)
            add(f"{s}_band{b['from_m']}_ha_hours_colder_category", b["ha_hours_colder_category"], "ha-h", bs)
            add(f"{s}_band{b['from_m']}_mean_dutci", b["mean_dutci"], "C", bs)
    for r in extra_rows:
        add(*r, st="internal")
    with open(NUMBERS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "value", "unit", "scope", "status", "note"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return status, len(rows)


def context_rows():
    """Inputs, weather, gates and timing: internal numbers MORNING.md quotes."""
    out = []
    with open(os.path.join(H.INPUTS_DIR, "domain.json"), encoding="utf-8") as f:
        m = json.load(f)
    g = m["grid"]
    out += [("domain_rows", g["rows"], "cells", "inputs"), ("domain_cols", g["cols"], "cells", "inputs"),
            ("window_area", m["areas_ha"]["window"], "ha", "inputs"),
            ("casters_near", m["footprints_near"], "footprints", "inputs", "within 500 m of the window"),
            ("casters_far", len(m["casters_far"]), "footprints", "inputs", "100 m+, reach the window in Dec"),
            ("buildings_grown", m["scenario"]["buildings_grown"], "buildings", "inputs"),
            ("cells_raised", m["scenario"]["cells_raised"], "cells", "inputs"),
            ("cells_changed", m["scenario"]["cells_changed"], "cells", "inputs"),
            ("prism_cells_held_equal", m["scenario"]["prism_cells_held_equal"], "cells", "inputs"),
            ("grown_suspect", sum(1 for b in m["grown_buildings"] if b.get("suspect")), "buildings", "inputs"),
            ("grown_suspect_cells", sum(b["cells_raised"] for b in m["grown_buildings"] if b.get("suspect")),
             "cells", "inputs"),
            ("canopy_cells", m["canopy"]["cells"], "cells", "inputs"),
            ("canopy_edge_rule_removed", m["canopy"]["edge_rule_removed"], "cells", "inputs"),
            ("canopy_capped_40m", m["canopy"]["capped"], "cells", "inputs")]
    for q, v in zip((5, 25, 50, 75, 95, 99, 100), m["canopy"]["percentiles_5_25_50_75_95_99_max"]):
        out.append((f"canopy_height_p{q}", v, "m", "inputs"))
    d = days()
    for s in ("dec", "jun"):
        c = d["seasons"][s]["chosen"]
        out += [(f"{s}_clear_day", c["date"], "date", "weather"),
                (f"{s}_clear_ratio", c["ratio"], "ratio", "weather", "GHI over Haurwitz clear sky")]
    if os.path.exists(GATES_JSON):
        with open(GATES_JSON, encoding="utf-8") as f:
            gates = json.load(f)
        if "V0" in gates:
            t = gates["V0"]["tower"]
            out += [("V0_reach_m", t["reach_m"], "m", "gate"), ("V0_expected_m", t["expected_m"], "m", "gate"),
                    ("V0_error", t["error"], "fraction", "gate")]
        for t in gates.get("V1", {}).get("times", []):
            out += [(f"V1_{t['est']}_agreement", t["agreement"], "fraction", "gate"),
                    (f"V1_{t['est']}_area_diff", t["area_diff"], "fraction", "gate")]
        if "V2" in gates and "d_tmrt" in gates["V2"]:
            v = gates["V2"]
            out += [("V2_d_tmrt_sun_vs_shade", v["d_tmrt"], "C", "gate"),
                    ("V2_d_utci_sun_vs_shade", v["d_utci"], "C", "gate")]
            for n in v["night"]:
                out.append((f"V2_night_{n['step'][11:16]}_tmrt_minus_ta", n["tmrt_open"] - n["ta"], "C", "gate"))
        if "V2b" in gates and "utci_p1_p50_p99" in gates["V2b"]:
            v = gates["V2b"]
            for q, x in zip((1, 50, 99), v["utci_p1_p50_p99"]):
                out.append((f"V2b_dutci_p{q}", x, "C", "gate"))
            for q, x in zip((1, 50, 99), v["tmrt_p1_p50_p99"]):
                out.append((f"V2b_dtmrt_p{q}", x, "C", "gate"))
        if "V3b" in gates and "max_abs_dtmrt" in gates["V3b"]:
            out.append(("V3b_max_abs_dtmrt_gpu_cpu", gates["V3b"]["max_abs_dtmrt"], "C", "gate"))
            for s in gates["V3b"].get("steps", []):
                out.append((f"V3b_{s['file_time']}_rmse_of_change_gpu_cpu", s["rmse_of_change"], "C", "gate"))
        if "V3" in gates and "steps" in gates["V3"]:
            for s in gates["V3"]["steps"]:
                t = s["sun_time"]
                out += [(f"V3_{t}_tmrt_bias_2017", s["bias_2017"], "C", "gate", "solweig minus UMEP"),
                        (f"V3_{t}_tmrt_rmse_2017", s["rmse_2017"], "C", "gate"),
                        (f"V3_{t}_tmrt_bias_today", s["bias_today"], "C", "gate"),
                        (f"V3_{t}_tmrt_rmse_today", s["rmse_today"], "C", "gate"),
                        (f"V3_{t}_change_rmse", s["change_rmse"], "C", "gate"),
                        (f"V3_{t}_cells_colder2_solweig", s["cells_change_le_minus2_solweig"], "cells", "gate"),
                        (f"V3_{t}_cells_colder2_umep", s["cells_change_le_minus2_umep"], "cells", "gate")]
    sens = os.path.join(H.HEAT_DIR, "sensitivity", "summary.json")
    if os.path.exists(sens):
        with open(sens, encoding="utf-8") as f:
            sv = json.load(f)
        for thr in (-2, -5, -10):
            out.append((f"dec_sens_without_suspects_ha_any_hour_dutci_le_{thr}",
                        sv["without_suspects"][f"ha_any_hour_le_{thr}"], "ha", "sensitivity, GPU runs",
                        "4 suspect grown buildings reverted to 2017"))
    if os.path.exists(TOWERS_JSON):
        with open(TOWERS_JSON, encoding="utf-8") as f:
            tw = json.load(f)
        for i, t in enumerate(tw[:10], start=1):
            out.append((f"tower{i}_dec_ha_any_hour_5", t["dec_ha_any_hour_5"], "ha", "per building, Dec",
                        f"{t['lidar_p99_m']} -> {t['height_m']} m at {t['lonlat'][1]}, {t['lonlat'][0]}"))
    return out


TOWERS_JSON = os.path.join(COMPARE_DIR, "towers.json")


def towers(log=print):
    """
    Which changed building makes which part of the park colder, December.

    Each changed building (a connected group of cells whose height differs
    between the years) is added on its own to the 2017 surface, and the shadow
    study's own sweep (geometry.py, checked against SOLWEIG in V1) finds the
    ground it newly shades at each reported hour. A park cell that feels at
    least 5 C (or 2 C) colder at that hour and lies in that new shadow is
    credited to the building. Where two new buildings shade the same cell,
    both are credited, so the per-building figures can add up to more than
    the park total.
    """
    from scipy import ndimage
    from pipeline.geometry import sweep
    from pipeline.solar import sun_position
    from datetime import timezone
    from pyproj import Transformer
    layers, masks, meta = H.load()
    d17, dnow, dem = layers["dsm_2017"], layers["dsm_today"], layers["dem"]
    rep, changed = masks["report"], masks["changed"]
    conv = meta["grid"]["conv_deg"]
    lat, lon = meta["site"]["lat"], meta["site"]["lon"]
    rr, cc = np.nonzero(rep)
    win = (int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1)
    wsl = (slice(win[0], win[1]), slice(win[2], win[3]))
    repw = rep[wsl]
    steps = report_steps("dec")
    suns, base, du = {}, {}, {}
    for ts in steps:
        st = ts - timedelta(minutes=30) + timedelta(hours=5)
        p = sun_position(st.replace(tzinfo=timezone.utc), lat, lon)
        suns[ts] = p
        base[ts] = sweep(d17, 2.0, p.altitude, (p.azimuth + conv) % 360.0, window=win) < 0.5
        du[ts] = (read_step("today", "dec", "utci", ts) - read_step("2017", "dec", "utci", ts))[wsl]
    min_alt = min(p.altitude for p in suns.values())
    lab, n = ndimage.label(changed, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    a, b, x0, d, e, y0 = meta["grid"]["transform_m"]
    to_ll = Transformer.from_crs("EPSG:32118", "EPSG:4326", always_xy=True).transform
    out = []
    log(f"{n} changed buildings; window {win}")
    for k in range(1, n + 1):
        sl = objs[k - 1]
        cells = lab[sl] == k
        rise = float((dnow[sl] - d17[sl])[cells].max())
        h_now = float((dnow[sl] - dem[sl])[cells].max())
        h_17 = float(np.percentile((d17[sl] - dem[sl])[cells], 99))
        # Distance to the reported park, to skip buildings whose new height
        # cannot throw a shadow that far at the lowest reported sun.
        r_c = (sl[0].start + sl[0].stop) / 2.0
        c_c = (sl[1].start + sl[1].stop) / 2.0
        dr = max(win[0] - r_c, 0, r_c - win[1])
        dc = max(win[2] - c_c, 0, c_c - win[3])
        if rise < 3.0 or h_now / math.tan(math.radians(min_alt)) < math.hypot(dr, dc) * 2.0:
            continue
        dsm_k = d17.copy()
        dsm_k[sl][cells] = dnow[sl][cells]
        any5 = np.zeros(repw.shape, bool)
        any2 = np.zeros(repw.shape, bool)
        h5 = h2 = 0
        for ts in steps:
            p = suns[ts]
            new = (sweep(dsm_k, 2.0, p.altitude, (p.azimuth + conv) % 360.0, window=win) < 0.5) & ~base[ts]
            hit5 = new & repw & (du[ts] <= -5)
            hit2 = new & repw & (du[ts] <= -2)
            any5 |= hit5
            any2 |= hit2
            h5 += int(hit5.sum())
            h2 += int(hit2.sum())
        if not any2.any():
            continue
        # A cell inside the building, nearest its middle, so a click on the
        # footprint in the viewer finds it.
        ii, jj = np.nonzero(cells)
        m = np.argmin((ii - ii.mean()) ** 2 + (jj - jj.mean()) ** 2)
        r_cell, c_cell = sl[0].start + ii[m] + 0.5, sl[1].start + jj[m] + 0.5
        lo, la = to_ll(x0 + c_cell * a, y0 + r_cell * e)
        out.append({"lonlat": [round(lo, 6), round(la, 6)], "height_m": round(h_now, 1),
                    "lidar_p99_m": round(h_17, 1), "cells": int(cells.sum()),
                    "dec_ha_any_hour_5": round(float(any5.sum() * CELL_HA), 2),
                    "dec_ha_hours_5": round(float(h5 * CELL_HA), 2),
                    "dec_ha_any_hour_2": round(float(any2.sum() * CELL_HA), 2),
                    "dec_ha_hours_2": round(float(h2 * CELL_HA), 2)})
        log(f"  {h_17:6.1f} -> {h_now:6.1f} m at {la:.5f}, {lo:.5f}: "
            f"{out[-1]['dec_ha_any_hour_5']:.2f} ha 5 C+ colder at some hour")
    out.sort(key=lambda t: -t["dec_ha_any_hour_5"])
    os.makedirs(COMPARE_DIR, exist_ok=True)
    with open(TOWERS_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {TOWERS_JSON}: {len(out)} buildings make part of the park colder")
    return out


def main():
    if "--towers" in sys.argv:
        towers()
        return 0
    layers, masks, meta = H.load()
    results = []
    for season in ("dec", "jun"):
        res, u17, unow = season_tables(season, masks, None)
        res["runs_dir"] = runs_dir()
        res["figures"] = [] if _ALT else figures(res, u17, unow, masks, layers, meta, season)
        results.append(res)
    status, n = write_numbers(results, context_rows())
    print(f"\nwrote {n} numbers to {NUMBERS_CSV}, all marked {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
