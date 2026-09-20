"""
How much sun each building's walls actually get.

The ground shadow map answers a question about the pavement. It says nothing
about the building you are looking at, which is why a 3D view of it reads as
decoration. This module puts the physics on the walls.

For every building, points are sampled up its facade. For each hour, each
sample point is tested twice:

  Orientation. A wall only sees the sun if it faces it. The dot product of the
  outward wall normal and the direction to the sun has to be positive. A north
  wall in New York never sees a winter sun at all, whatever is in front of it.

  Occlusion. If it faces the sun, a ray is marched from that point toward the
  sun through the surface model. Anything poking above the ray blocks it.

Two things come out:

  Sun hours. How many of the modelled hours that point sees direct sun.

  Solar gain. Energy landing on a square metre of that wall across the day, in
  watt-hours, from DNI * cos(angle of incidence) integrated over the hours it
  is lit. This is the standard plane-of-array direct term used in facade PV and
  cooling-load work.

Irradiance comes from a clear-sky model, not from weather data. It is an upper
bound on a cloudless day, and the number to quote is the relative one: this
wall gets four times what that wall gets. Say clear-sky every time.
"""

from __future__ import annotations

import math

import numpy as np

SOLAR_CONSTANT = 1367.0  # W/m2 at the top of the atmosphere


def clear_sky_dni(altitude_deg: float) -> float:
    """
    Direct normal irradiance under a cloudless sky, W/m2.

    Meinel and Meinel's air-mass attenuation: DNI = 1367 * 0.7 ^ (AM ^ 0.678),
    where air mass AM is roughly 1 / sin(altitude). Simple, widely used for
    first-order work, and it needs no atmospheric measurements at all, which is
    the point: everything here stays derivable from geometry.

    Kasten and Young's correction keeps the air mass finite near the horizon,
    where 1 / sin blows up.
    """
    if altitude_deg <= 0.5:
        return 0.0
    a = math.radians(altitude_deg)
    am = 1.0 / (math.sin(a) + 0.50572 * (altitude_deg + 6.07995) ** -1.6364)
    am = min(am, 38.0)
    return SOLAR_CONSTANT * (0.7 ** (am**0.678))


def sun_vector(altitude_deg: float, azimuth_deg: float):
    """
    Unit vector pointing from the ground toward the sun.

    Returns (east, north, up) so it lines up with the raster convention used
    everywhere else: east is +column, north is -row.
    """
    a = math.radians(altitude_deg)
    z = math.radians(azimuth_deg)
    return (
        math.cos(a) * math.sin(z),   # east
        math.cos(a) * math.cos(z),   # north
        math.sin(a),                 # up
    )


def wall_samples(built: np.ndarray, heights: np.ndarray, labels: np.ndarray,
                 cellsize: float, levels: int = 4, log=print):
    """
    Find points on building walls, with an outward normal for each.

    A wall cell is a building cell with at least one non-building neighbour.
    The outward normal comes from the gradient of the building mask: it points
    from solid toward open, which is the direction the wall faces.

    Each wall cell contributes `levels` sample points spaced up its height, so
    a tower reports its base and its crown separately. That split is the whole
    point, because in a dense city they get completely different amounts of sun.
    """
    from scipy import ndimage

    inner = ndimage.binary_erosion(built, structure=np.ones((3, 3)))
    wall = built & ~inner
    rows, cols = np.nonzero(wall)
    if rows.size == 0:
        return None
    log(f"  {rows.size:,} wall cells, {levels} sample heights each")

    # Outward normal from the mask gradient. Smoothing first keeps the normal
    # stable along a straight facade instead of flipping cell to cell.
    m = ndimage.gaussian_filter(built.astype(np.float32), sigma=1.2)
    gy, gx = np.gradient(m)
    # Gradient climbs toward solid, so flip it to face outward.
    ne = -gx[rows, cols]
    nn = gy[rows, cols]          # +row is south, so -grad_row is north
    norm = np.hypot(ne, nn)
    ok = norm > 1e-6
    rows, cols, ne, nn, norm = rows[ok], cols[ok], ne[ok], nn[ok], norm[ok]
    ne, nn = ne / norm, nn / norm

    h = heights[rows, cols]
    base = h - h  # zeros, kept explicit: sampling is relative to local ground
    fracs = [(i + 0.5) / levels for i in range(levels)]

    pts = {
        "row": np.concatenate([rows] * levels),
        "col": np.concatenate([cols] * levels),
        "ne": np.concatenate([ne] * levels),
        "nn": np.concatenate([nn] * levels),
        "label": np.concatenate([labels[rows, cols]] * levels),
        "height_frac": np.concatenate([np.full(rows.size, f) for f in fracs]),
        "wall_height": np.concatenate([h] * levels),
    }
    pts["z_above_ground"] = pts["wall_height"] * pts["height_frac"]
    return pts


def solve(pts, surface: np.ndarray, ground: np.ndarray, cellsize: float,
          positions, labels_hours, max_march_m: float = 1500.0, log=print):
    """
    Sun hours and clear-sky solar gain for every wall sample point.

    Returns a dict of arrays, one entry per sample point, plus the per-hour
    lit mask so the viewer can animate the facades.
    """
    n_pts = pts["row"].size
    n_hours = len(positions)
    lit = np.zeros((n_hours, n_pts), dtype=bool)
    gain = np.zeros(n_pts, dtype=np.float32)

    h, w = surface.shape
    r0 = pts["row"]
    c0 = pts["col"]
    # Absolute height of each sample point above the vertical datum.
    z0 = ground[r0, c0] + pts["z_above_ground"]

    for k, p in enumerate(positions):
        if p.altitude <= 0.5:
            continue
        se, sn, su = sun_vector(p.altitude, p.azimuth)

        # Orientation test. cos of the angle between the wall normal and the
        # direction to the sun. Vertical wall, so only the horizontal parts of
        # the sun vector project onto the normal.
        cos_aoi = pts["ne"] * se + pts["nn"] * sn
        faces = cos_aoi > 0.0

        # Occlusion test, only for the points that face the sun at all.
        idx = np.nonzero(faces)[0]
        if idx.size == 0:
            continue
        blocked = np.zeros(idx.size, dtype=bool)
        tan_alt = math.tan(math.radians(p.altitude))
        steps = int(min(max_march_m, 1500.0) / cellsize)
        rr0, cc0, zz0 = r0[idx], c0[idx], z0[idx]

        for n in range(1, steps + 1):
            dcol = int(round(n * se / max(abs(se), abs(sn), 1e-9)))
            drow = int(round(-n * sn / max(abs(se), abs(sn), 1e-9)))
            dist = math.hypot(drow, dcol) * cellsize
            if dist > max_march_m:
                break
            rr = rr0 + drow
            cc = cc0 + dcol
            inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            if not inside.any():
                break
            ray_z = zz0 + dist * tan_alt
            hit = np.zeros(idx.size, dtype=bool)
            hit[inside] = surface[rr[inside], cc[inside]] > ray_z[inside]
            blocked |= hit
            if blocked.all():
                break

        visible = np.zeros(n_pts, dtype=bool)
        visible[idx[~blocked]] = True
        lit[k] = visible

        dni = clear_sky_dni(p.altitude)
        if dni > 0:
            # Plane-of-array direct term for a vertical surface, one hour.
            gain[visible] += (dni * cos_aoi[visible]).astype(np.float32)

        if log:
            log(f"    {labels_hours[k]}  {visible.mean()*100:5.1f}% of wall "
                f"samples in sun, DNI {dni:4.0f} W/m2")

    return {"lit": lit, "sun_hours": lit.sum(axis=0).astype(np.float32),
            "gain_wh_m2": gain}


def per_building(pts, res, n_labels: int):
    """
    Roll wall samples up to one record per building part.

    Reported separately for the lower and upper half of each wall, because in
    a dense city that difference is the story: the crown is in sun all day
    while the base never leaves the shade.
    """
    lab = pts["label"]
    frac = pts["height_frac"]
    sun = res["sun_hours"]
    gain = res["gain_wh_m2"]

    out = {}
    lower = frac < 0.5
    upper = ~lower
    for name, sel in (("all", np.ones_like(lower)), ("low", lower), ("high", upper)):
        s = np.zeros(n_labels + 1, dtype=np.float32)
        g = np.zeros(n_labels + 1, dtype=np.float32)
        cnt = np.zeros(n_labels + 1, dtype=np.float32)
        np.add.at(s, lab[sel], sun[sel])
        np.add.at(g, lab[sel], gain[sel])
        np.add.at(cnt, lab[sel], 1.0)
        cnt[cnt == 0] = 1.0
        out[name] = {"sun_hours": s / cnt, "gain_wh_m2": g / cnt}
    return out
