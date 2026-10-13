"""
Clocks and analysis days for any city.

The New York study follows the CEQR Technical Manual: four analysis days,
from 1.5 hours after sunrise to 1.5 hours before sunset, on standard time all
year. CEQR prints those windows for New York; for any other city they are
computed here the same way, sunrise and sunset to the minute from the NOAA
sun (pipeline.solar), with the usual -0.833 degree horizon (refraction and
the sun's radius). For New York, run_nyc.CEQR_DAYS stays the source.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

from .solar import sun_position

PAD_H = 1.5
CEQR = [
    {"id": "dec", "label": "21 December", "short": "Dec 21", "date": "2026-12-21"},
    {"id": "mar", "label": "21 March / 21 September", "short": "Mar 21", "date": "2026-03-21"},
    {"id": "may", "label": "6 May / 6 August", "short": "May 6", "date": "2026-05-06"},
    {"id": "jun", "label": "21 June", "short": "Jun 21", "date": "2026-06-21"},
]


def sun_at(local_naive, lat, lon, utc_offset, refraction=None):
    utc = (local_naive - timedelta(hours=utc_offset)).replace(tzinfo=timezone.utc)
    return sun_position(utc, lat, lon, refraction=refraction)


def crossings(day, lat, lon, utc_offset, alt=-0.833):
    """Sunrise and sunset in local standard time, to the minute."""
    if isinstance(day, str):
        day = date.fromisoformat(day)
    base = datetime(day.year, day.month, day.day)
    # Geometric sun: -0.833 deg already holds the horizon's refraction.
    prev = sun_at(base, lat, lon, utc_offset, refraction=False).altitude
    rise = set_ = None
    for m in range(1, 24 * 60 + 1):
        t = base + timedelta(minutes=m)
        a = sun_at(t, lat, lon, utc_offset, refraction=False).altitude
        if prev < alt <= a and rise is None:
            rise = t
        if prev >= alt > a:
            set_ = t
        prev = a
    return rise, set_


def day_window(day, lat, lon, utc_offset, pad_h=PAD_H):
    """(start, end, sunrise, sunset): 1.5 h after sunrise to 1.5 h before sunset."""
    rise, set_ = crossings(day, lat, lon, utc_offset)
    return rise + timedelta(hours=pad_h), set_ - timedelta(hours=pad_h), rise, set_


def analysis_days(cfg, lat, lon):
    """The city's analysis days with their windows, in the shape of run_nyc.CEQR_DAYS."""
    if cfg["id"] == "nyc":
        from .run_nyc import CEQR_DAYS
        return [dict(d) for d in CEQR_DAYS]
    specs = {d["id"]: dict(d) for d in CEQR}
    for k, v in cfg.get("analysis", {}).get("extra_days", {}).items():
        specs[k] = {"id": k, **{kk: vv for kk, vv in v.items() if kk in ("label", "short", "date")}}
    out = []
    for k in cfg["analysis"]["days"]:
        d = specs[k]
        w0, w1, _, _ = day_window(d["date"], lat, lon, float(cfg["utc_offset"]))
        d["window"] = (w0.strftime("%H:%M"), w1.strftime("%H:%M"))
        out.append(d)
    return out


def frames_for(spec, lat, lon, utc_offset, step_min=30):
    """Sun positions every step_min minutes inside a day's window: run_nyc.frames_for for any clock."""
    y, m, d = (int(x) for x in spec["date"].split("-"))
    h0, m0 = (int(x) for x in spec["window"][0].split(":"))
    h1, m1 = (int(x) for x in spec["window"][1].split(":"))
    start, end = h0 * 60 + m0, h1 * 60 + m1
    t = int(math.ceil(start / step_min) * step_min)
    out = []
    while t <= end:
        local = datetime(y, m, d) + timedelta(minutes=t)
        p = sun_at(local, lat, lon, utc_offset)
        if p.above_horizon:
            out.append((p, f"{t // 60:02d}:{t % 60:02d}"))
        t += step_min
    return out
