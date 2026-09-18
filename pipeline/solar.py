"""
Solar position.

NOAA solar position algorithm (Meeus, Astronomical Algorithms, low-precision
form as published by NOAA ESRL). Accurate to roughly 0.01 degrees for dates
between 1800 and 2100, which is far better than we need: a 0.01 degree error
in solar altitude moves a shadow edge by about 1 cm per 60 m of shadow length.

This is the only place in the pipeline where astronomy happens. Everything
downstream takes (altitude, azimuth) and does pure geometry.

Convention used everywhere in this project:
    altitude  degrees above the horizon, 0 = horizon, 90 = overhead
    azimuth   degrees clockwise from true north, 0 = N, 90 = E, 180 = S, 270 = W
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass


@dataclass(frozen=True)
class SunPosition:
    altitude: float          # degrees above horizon
    azimuth: float           # degrees clockwise from north
    declination: float       # degrees
    when: datetime           # the UTC instant this was computed for

    @property
    def above_horizon(self) -> bool:
        return self.altitude > 0.0


def _julian_day(dt_utc: datetime) -> float:
    """Julian Day number, including fractional day, from a UTC datetime."""
    y, m = dt_utc.year, dt_utc.month
    d = (
        dt_utc.day
        + dt_utc.hour / 24.0
        + dt_utc.minute / 1440.0
        + dt_utc.second / 86400.0
    )
    if m <= 2:
        y -= 1
        m += 12
    a = math.floor(y / 100.0)
    b = 2 - a + math.floor(a / 4.0)
    return (
        math.floor(365.25 * (y + 4716))
        + math.floor(30.6001 * (m + 1))
        + d
        + b
        - 1524.5
    )


def sun_position(dt_utc: datetime, lat: float, lon: float) -> SunPosition:
    """
    Solar altitude and azimuth for a UTC instant at a geographic point.

    lat  degrees north, positive north
    lon  degrees east, positive east
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_utc = dt_utc.astimezone(timezone.utc)

    jd = _julian_day(dt_utc)
    # Julian centuries since J2000.0
    t = (jd - 2451545.0) / 36525.0

    # Geometric mean longitude and anomaly of the sun, degrees
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    m_rad = math.radians(m)

    # Equation of centre
    c = (
        math.sin(m_rad) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * m_rad) * (0.019993 - 0.000101 * t)
        + math.sin(3 * m_rad) * 0.000289
    )
    true_long = l0 + c

    # Apparent longitude, corrected for nutation and aberration
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    # Obliquity of the ecliptic, with correction
    seconds = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))
    eps0 = 23.0 + (26.0 + seconds / 60.0) / 60.0
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))

    # Declination and right ascension
    decl = math.degrees(
        math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(app_long)))
    )

    # Equation of time, minutes
    y = math.tan(math.radians(eps / 2.0)) ** 2
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    l0_rad = math.radians(l0)
    eq_time = 4.0 * math.degrees(
        y * math.sin(2 * l0_rad)
        - 2.0 * e * math.sin(m_rad)
        + 4.0 * e * y * math.sin(m_rad) * math.cos(2 * l0_rad)
        - 0.5 * y * y * math.sin(4 * l0_rad)
        - 1.25 * e * e * math.sin(2 * m_rad)
    )

    # True solar time, minutes past local midnight
    minutes_utc = dt_utc.hour * 60.0 + dt_utc.minute + dt_utc.second / 60.0
    true_solar_time = (minutes_utc + eq_time + 4.0 * lon) % 1440.0

    # Hour angle, degrees. Negative before solar noon.
    ha = true_solar_time / 4.0 - 180.0
    if ha < -180.0:
        ha += 360.0

    lat_rad = math.radians(lat)
    decl_rad = math.radians(decl)
    ha_rad = math.radians(ha)

    cos_zenith = math.sin(lat_rad) * math.sin(decl_rad) + math.cos(lat_rad) * math.cos(
        decl_rad
    ) * math.cos(ha_rad)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.degrees(math.acos(cos_zenith))
    altitude = 90.0 - zenith

    # Azimuth measured clockwise from north
    sin_zenith = math.sin(math.radians(zenith))
    if abs(sin_zenith) < 1e-9:
        azimuth = 180.0
    else:
        cos_az = (
            math.sin(lat_rad) * cos_zenith - math.sin(decl_rad)
        ) / (math.cos(lat_rad) * sin_zenith)
        cos_az = max(-1.0, min(1.0, cos_az))
        a = math.degrees(math.acos(cos_az))
        # NOAA branch form. acos returns an angle measured from south; the
        # branch on hour angle resolves the morning/afternoon ambiguity and
        # rebases onto clockwise-from-north.
        if ha > 0:
            azimuth = (a + 180.0) % 360.0
        else:
            azimuth = (540.0 - a) % 360.0

    return SunPosition(
        altitude=altitude, azimuth=azimuth, declination=decl, when=dt_utc
    )


def local_day_positions(
    date_str: str,
    lat: float,
    lon: float,
    utc_offset_hours: float,
    hours: tuple[int, ...] = tuple(range(5, 20)),
) -> list[SunPosition]:
    """
    Sun positions at each given local clock hour on one local calendar day.

    Egypt is UTC+2 in winter and UTC+3 under daylight saving. Pass the offset
    that was in force on the date you are modelling; do not guess it.
    """
    y, m, d = (int(x) for x in date_str.split("-"))
    out = []
    for h in hours:
        local_naive = datetime(y, m, d, h, 0, 0)
        dt_utc = (local_naive - timedelta(hours=utc_offset_hours)).replace(
            tzinfo=timezone.utc
        )
        out.append(sun_position(dt_utc, lat, lon))
    return out
