"""
Self-tests for solar.py against values you can derive by hand.

Run:  python -m pipeline.verify_solar
Every check prints the expected value, the computed value and the error.
If any line says FAIL, nothing downstream can be trusted.
"""

from datetime import datetime, timedelta, timezone

from .solar import sun_position

ALEX_LAT, ALEX_LON = 31.2001, 29.9187
OBLIQUITY = 23.44


def _peak_of_day(y, m, d, lat, lon):
    """Scan the UTC day at one-minute steps and return the highest sun."""
    best = None
    start = datetime(y, m, d, 0, 0, tzinfo=timezone.utc)
    for i in range(1440):
        p = sun_position(start + timedelta(minutes=i), lat, lon)
        if best is None or p.altitude > best.altitude:
            best = p
    return best


def check(label, expected, got, tol, unit="deg"):
    err = abs(expected - got)
    ok = err <= tol
    print(
        f"{'PASS' if ok else 'FAIL'}  {label:<44} "
        f"expected {expected:8.3f}  got {got:8.3f}  err {err:6.3f} {unit}"
    )
    return ok


def main():
    ok = True
    print("solar geometry self-test\n" + "-" * 84)

    # 1. Declination at the solstices and equinoxes.
    for (y, m, d), exp, tol, name in [
        ((2026, 6, 21), +OBLIQUITY, 0.20, "declination, June solstice"),
        ((2026, 12, 21), -OBLIQUITY, 0.20, "declination, December solstice"),
        ((2026, 3, 20), 0.0, 0.60, "declination, March equinox"),
        ((2026, 9, 22), 0.0, 0.60, "declination, September equinox"),
    ]:
        p = sun_position(datetime(y, m, d, 12, 0, tzinfo=timezone.utc), 0.0, 0.0)
        ok &= check(name, exp, p.declination, tol)

    print()

    # 2. Peak solar altitude over Alexandria.
    #    Analytical: 90 - |latitude - declination|
    for (y, m, d), decl, name in [
        ((2026, 6, 21), +OBLIQUITY, "peak altitude Alexandria, June solstice"),
        ((2026, 12, 21), -OBLIQUITY, "peak altitude Alexandria, Dec solstice"),
        ((2026, 7, 15), None, "peak altitude Alexandria, 15 July"),
    ]:
        p = _peak_of_day(y, m, d, ALEX_LAT, ALEX_LON)
        expected = 90.0 - abs(ALEX_LAT - (decl if decl is not None else p.declination))
        ok &= check(name, expected, p.altitude, 0.35)
        ok &= check(f"  azimuth at that moment (due south)", 180.0, p.azimuth, 1.0)

    print()

    # 3. Equator at equinox, local solar noon: sun essentially overhead.
    p = _peak_of_day(2026, 3, 20, 0.0, 0.0)
    ok &= check("peak altitude equator, March equinox", 90.0, p.altitude, 0.6)

    print()

    # 4. Azimuth sweeps west through the afternoon and the sun descends.
    day = [
        sun_position(datetime(2026, 7, 15, h, 0, tzinfo=timezone.utc), ALEX_LAT, ALEX_LON)
        for h in range(4, 19)
    ]
    lit = [p for p in day if p.above_horizon]
    monotonic_az = all(
        lit[i + 1].azimuth > lit[i].azimuth for i in range(len(lit) - 1)
    )
    print(
        f"{'PASS' if monotonic_az else 'FAIL'}  "
        f"azimuth increases N->E->S->W through the day"
    )
    ok &= monotonic_az

    morning = sun_position(
        datetime(2026, 7, 15, 5, 0, tzinfo=timezone.utc), ALEX_LAT, ALEX_LON
    )
    evening = sun_position(
        datetime(2026, 7, 15, 16, 0, tzinfo=timezone.utc), ALEX_LAT, ALEX_LON
    )
    east_ok = 45 < morning.azimuth < 110
    west_ok = 250 < evening.azimuth < 315
    print(
        f"{'PASS' if east_ok else 'FAIL'}  "
        f"sun is in the east at 05:00 UTC        azimuth {morning.azimuth:6.1f}"
    )
    print(
        f"{'PASS' if west_ok else 'FAIL'}  "
        f"sun is in the west at 16:00 UTC        azimuth {evening.azimuth:6.1f}"
    )
    ok &= east_ok and west_ok

    print("-" * 84)
    print("ALL CHECKS PASSED" if ok else "SOMETHING FAILED, STOP HERE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
