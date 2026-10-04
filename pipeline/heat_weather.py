"""
Weather for the heat layer: the Central Park typical year, and the clearest
day near each solstice.

    .venv-heat/Scripts/python -m pipeline.heat_weather data/heat/weather/<file>.epw

Source: TMYx for New York Central Park (Belvedere Castle), WMO 725053, from
climate.onebuilding.org. A typical year stitches each month from a different
real year, which the file's COMMENTS line lists.

Clearest day: within 12 days of 21 December (and of 21 June for the control),
the day with the highest ratio of measured global horizontal irradiance to
clear-sky irradiance, summed over the day. Clear sky is Haurwitz (1945), sun
from pipeline/solar.py at the middle of each hour.

Clocks
------
EPW rows are hour-ending, in local standard time: hour 13 holds 12:00 to 13:00
EST. SOLWEIG reads Weather.datetime as the end of the interval and puts the sun
at datetime minus half a timestep. So the row for hour 13 becomes a Weather at
13:00, SOLWEIG evaluates the sun at 12:30, and every output file stamped 13:00
shows 12:30 EST. Everything downstream labels results with that sun time.

Years: the shadow study runs its CEQR days in 2026, so December and June dates
are put in 2026 and January dates in 2027. Only month, day and hour come from
the file, so the sun is computed for the same calendar as the shadow study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import date, datetime, timedelta, timezone

from .solar import sun_position

EST = -5.0
SEARCH_DAYS = 12
SOLSTICES = {"dec": (12, 21), "jun": (6, 21)}
CEQR_PAD_H = 1.5

# EPW missing-value markers, as solweig.io_epw reads them.
_MISSING_EXACT = {"ta": 99.9, "rh": 999.0, "ws": 999.0}
_MISSING_ABOVE = {"pressure_pa": 999999.0, "ghi": 9999.0, "dni": 9999.0, "dhi": 9999.0}


def read_epw(path):
    """Header and hourly rows of an EPW file, keyed by (month, day, hour 1-24)."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()          # tolerant of \r\n
    loc = lines[0].split(",")
    header = {"city": loc[1], "state": loc[2], "country": loc[3], "source": loc[4],
              "wmo": loc[5], "lat": float(loc[6]), "lon": float(loc[7]),
              "tz": float(loc[8]), "elev_m": float(loc[9]),
              "comments": [ln for ln in lines[:8] if ln.startswith("COMMENTS")]}
    rows = {}
    for ln in lines[8:]:
        p = ln.split(",")
        if len(p) < 22:
            continue
        r = {"year": int(p[0]), "month": int(p[1]), "day": int(p[2]), "hour": int(p[3]),
             "ta": float(p[6]), "rh": float(p[8]), "pressure_pa": float(p[9]),
             "ghi": float(p[13]), "dni": float(p[14]), "dhi": float(p[15]),
             "ws": float(p[21])}
        for k, v in _MISSING_EXACT.items():
            if r[k] == v:
                r[k] = None
        for k, v in _MISSING_ABOVE.items():
            if r[k] >= v:
                r[k] = None
        rows[(r["month"], r["day"], r["hour"])] = r
    if len(rows) != 8760:
        raise SystemExit(f"{path}: expected 8760 hourly rows, got {len(rows)}")
    return header, rows


def canonical(month, day):
    """The date the shadow study's calendar puts this month and day on."""
    return date(2027 if month == 1 else 2026, month, day)


def row_for(ts, rows):
    """EPW row for an hour-ending timestamp: 00:00 is hour 24 of the day before."""
    if ts.hour == 0:
        d = ts - timedelta(days=1)
        return rows[(d.month, d.day, 24)]
    return rows[(ts.month, ts.day, ts.hour)]


def sun_at(local_naive, lat, lon):
    utc = (local_naive - timedelta(hours=EST)).replace(tzinfo=timezone.utc)
    return sun_position(utc, lat, lon)


def haurwitz(alt_deg):
    cz = math.sin(math.radians(alt_deg))
    return 1098.0 * cz * math.exp(-0.057 / cz) if cz > 0 else 0.0


def clearness(day_date, rows, lat, lon):
    """Daily sum of measured over clear-sky global horizontal irradiance."""
    meas = clear = 0.0
    for h in range(1, 25):
        ts = datetime(day_date.year, day_date.month, day_date.day) + timedelta(hours=h)
        alt = sun_at(ts - timedelta(minutes=30), lat, lon).altitude
        cs = haurwitz(alt)
        if cs <= 0:
            continue
        r = row_for(ts, rows)
        meas += r["ghi"] or 0.0
        clear += cs
    return meas / clear if clear > 0 else 0.0, meas, clear


def sun_crossings(day_date, lat, lon, alt=-0.833):
    """Sunrise and sunset, EST, to the minute."""
    base = datetime(day_date.year, day_date.month, day_date.day)
    prev = sun_at(base, lat, lon).altitude
    rise = set_ = None
    for m in range(1, 24 * 60 + 1):
        t = base + timedelta(minutes=m)
        a = sun_at(t, lat, lon).altitude
        if prev < alt <= a and rise is None:
            rise = t
        if prev >= alt > a:
            set_ = t
        prev = a
    return rise, set_


def ceqr_window(day_date, lat, lon):
    rise, set_ = sun_crossings(day_date, lat, lon)
    return (rise + timedelta(hours=CEQR_PAD_H), set_ - timedelta(hours=CEQR_PAD_H), rise, set_)


def choose(rows, lat, lon, season):
    m, d = SOLSTICES[season]
    centre = canonical(m, d)
    cands = []
    for k in range(-SEARCH_DAYS, SEARCH_DAYS + 1):
        day = centre + timedelta(days=k)
        day = canonical(day.month, day.day)
        ratio, meas, clear = clearness(day, rows, lat, lon)
        cands.append({"date": day.isoformat(), "offset_days": k, "ratio": round(ratio, 4),
                      "ghi_wh_m2": round(meas), "clear_wh_m2": round(clear)})
    best = max(cands, key=lambda c: c["ratio"])
    return best, sorted(cands, key=lambda c: -c["ratio"])


def series(day_iso, rows):
    """
    Hourly weather from 00:00 the day before (spin-up) through 23:00 of the
    day, as hour-ending timestamps. Each entry carries the sun time it stands
    for, half an hour earlier.
    """
    d = date.fromisoformat(day_iso)
    start = datetime(d.year, d.month, d.day) - timedelta(days=1)
    out = []
    for k in range(48):
        ts = start + timedelta(hours=k)
        r = row_for(ts, rows)
        out.append({"datetime": ts, "sun_time": ts - timedelta(minutes=30),
                    "epw_source": f"{r['year']}-{r['month']:02d}-{r['day']:02d} hour {r['hour']}",
                    "ta": r["ta"], "rh": r["rh"], "ghi": r["ghi"], "dni": r["dni"],
                    "dhi": r["dhi"], "ws": r["ws"],
                    "pressure_hpa": r["pressure_pa"] / 100.0 if r["pressure_pa"] else None})
    return out


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("epw")
    ap.add_argument("--source-url", default=None)
    ap.add_argument("--out", default=os.path.join("data", "heat", "weather", "days.json"))
    args = ap.parse_args()

    header, rows = read_epw(args.epw)
    lat, lon = header["lat"], header["lon"]
    print(f"{header['city']}, WMO {header['wmo']}, {lat}, {lon}, UTC{header['tz']:+g}")
    for c in header["comments"][:1]:
        print(f"  {c[:160]}")
    if header["tz"] != EST:
        raise SystemExit("this file is not on EST; the CEQR clock needs EST")
    out = {"epw": os.path.basename(args.epw), "epw_sha256": file_sha256(args.epw),
           "source_url": args.source_url, "header": header,
           "clear_sky": "Haurwitz (1945), sun at mid-hour from pipeline/solar.py",
           "clock": ("EPW hour-ending rows in EST; solweig Weather.datetime is the end of "
                     "the hour and the sun is evaluated 30 min earlier; outputs are labelled "
                     "with that sun time"),
           "seasons": {}}
    for season in ("dec", "jun"):
        best, ranked = choose(rows, lat, lon, season)
        d = date.fromisoformat(best["date"])
        w0, w1, rise, set_ = ceqr_window(d, lat, lon)
        ser = series(best["date"], rows)
        report = [s["datetime"].strftime("%H:%M") for s in ser
                  if s["datetime"].date() == d and w0 <= s["sun_time"] <= w1]
        tas = [s["ta"] for s in ser[24:]]
        out["seasons"][season] = {
            "chosen": best, "ranked": ranked[:6],
            "sunrise": rise.strftime("%H:%M"), "sunset": set_.strftime("%H:%M"),
            "ceqr_window": [w0.strftime("%H:%M"), w1.strftime("%H:%M")],
            "report_timestamps": report,
            "report_sun_times": [(datetime.strptime(t, "%H:%M") - timedelta(minutes=30)).strftime("%H:%M")
                                 for t in report],
            "spinup_from": ser[0]["datetime"].isoformat(), "through": ser[-1]["datetime"].isoformat(),
            "timesteps": len(ser), "epw_rows": [ser[0]["epw_source"], ser[-1]["epw_source"]],
            "ta_range_c": [min(tas), max(tas)],
            "ws_range_m_s": [min(s["ws"] for s in ser[24:]), max(s["ws"] for s in ser[24:])],
        }
        print(f"\n{season}: clearest day within {SEARCH_DAYS} days of the solstice is "
              f"{best['date']} (ratio {best['ratio']:.3f}, {best['ghi_wh_m2']} of "
              f"{best['clear_wh_m2']} Wh/m2 clear sky)")
        for c in ranked[:5]:
            print(f"    {c['date']}  {c['ratio']:.3f}")
        print(f"  sunrise {rise:%H:%M}, sunset {set_:%H:%M} EST; CEQR window "
              f"{w0:%H:%M} to {w1:%H:%M}")
        print(f"  reported steps (file time / sun time): "
              + ", ".join(f"{a}/{b}" for a, b in zip(report, out['seasons'][season]['report_sun_times'])))
        print(f"  air {min(tas):.1f} to {max(tas):.1f} C on the day")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
