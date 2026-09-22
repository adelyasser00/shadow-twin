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
          positions, labels_hours, max_march_m: float = 1500.0, log=print,
          labels_grid: np.ndarray | None = None):
    """
    Direct sun hours and clear-sky solar gain for every wall sample point.

    Self-shadowing, and why the march starts outside the wall
    ---------------------------------------------------------
    Manhattan's walls run at about 29 degrees to the raster grid, so on the grid
    every wall is a staircase of cells. A ray marched toward the sun from a
    staircase cell, in whole-cell steps, often lands on the next step of the same
    wall and reports the building as shading itself. On an isolated tower with
    nothing around it, that bug cut the sun on every sun-facing wall by roughly
    half. The fix has three parts:

      1. The ray starts one cell outside the wall, along its outward normal.
      2. It moves in continuous steps along the true sun direction and is only
         rounded to a cell when sampling, so it does not zig-zag.
      3. Hits on the building's own cells within the first 6 m are ignored.
         Beyond that they count, so a genuine L-shaped building still shades
         its own inner corner.

    verify_facade.py checks this on an isolated rotated tower, where every wall
    that faces the sun must be lit for every hour it faces it.

    Only the direct beam is modelled. Diffuse sky light and light reflected off
    other buildings also reach a wall, including a north wall that never sees the
    sun directly. Those are not zero in reality and this does not claim they are.
    """
    n_pts = pts["row"].size
    n_hours = len(positions)
    lit = np.zeros((n_hours, n_pts), dtype=bool)
    gain = np.zeros(n_pts, dtype=np.float32)

    h, w = surface.shape
    r0 = pts["row"].astype(np.float64)
    c0 = pts["col"].astype(np.float64)
    own = pts["label"]
    z0 = ground[pts["row"], pts["col"]] + pts["z_above_ground"]
    # One cell outward along the wall normal. East is +col, north is -row.
    rs = r0 - pts["nn"] * 1.0
    cs = c0 + pts["ne"] * 1.0
    self_skip = max(2, int(round(6.0 / cellsize)))

    for k, p in enumerate(positions):
        if p.altitude <= 0.5:
            continue
        se, sn, su = sun_vector(p.altitude, p.azimuth)
        cos_aoi = pts["ne"] * se + pts["nn"] * sn
        faces = cos_aoi > 0.0
        idx = np.nonzero(faces)[0]
        if idx.size == 0:
            continue

        hz = math.hypot(se, sn)
        de, dn = se / hz, sn / hz               # unit horizontal sun direction
        tan_alt = math.tan(math.radians(p.altitude))
        steps = int(max_march_m / cellsize)

        blocked = np.zeros(idx.size, dtype=bool)
        alive = np.ones(idx.size, dtype=bool)
        rr0, cc0, zz0, oo = rs[idx], cs[idx], z0[idx], own[idx]

        for n in range(0, steps + 1):
            fr = rr0 - dn * n
            fc = cc0 + de * n
            rr = np.rint(fr).astype(np.int64)
            cc = np.rint(fc).astype(np.int64)
            inside = alive & (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            if not inside.any():
                break
            ray_z = zz0 + (n + 1.0) * cellsize * tan_alt
            sel = np.nonzero(inside)[0]
            hit = surface[rr[sel], cc[sel]] > ray_z[sel]
            if labels_grid is not None and n < self_skip:
                hit &= labels_grid[rr[sel], cc[sel]] != oo[sel]
            blocked[sel[hit]] = True
            alive &= ~blocked
            # Points whose ray has climbed above everything can stop early.
            alive[sel[ray_z[sel] > surface.max()]] = False
            if not alive.any():
                break

        visible = np.zeros(n_pts, dtype=bool)
        visible[idx[~blocked]] = True
        lit[k] = visible
        dni = clear_sky_dni(p.altitude)
        if dni > 0:
            gain[visible] += (dni * cos_aoi[visible]).astype(np.float32)
        if log:
            log(f"    {labels_hours[k]}  {visible.mean()*100:5.1f}% of wall "
                f"samples in sun, DNI {dni:4.0f} W/m2")

    return {"lit": lit, "sun_hours": lit.sum(axis=0).astype(np.float32),
            "gain_wh_m2": gain}


ORIENTS = [("N", 0.0), ("E", 90.0), ("S", 180.0), ("W", 270.0)]


def per_building(pts, res, n_labels: int):
    """
    Roll wall samples up to one record per building part.

    Averaging every wall of a building into one number is the mistake that
    makes all of this useless. A building's north wall gets almost nothing and
    its south wall gets everything; the mean of the two is a number that
    describes no wall on the building and is roughly the same for every
    building in the city. Every value below is therefore reported by compass
    orientation, and by lower and upper half of the wall.

    The headline number for colouring is the SOUTH-facing wall in winter and
    the worst wall in summer, because those are the two questions people
    actually ask: where does the heat get in, and where does it never arrive.
    """
    lab = pts["label"]
    frac = pts["height_frac"]
    sun = res["sun_hours"]
    gain = res["gain_wh_m2"]

    # Compass bearing each wall faces, from its outward normal.
    bearing = (np.degrees(np.arctan2(pts["ne"], pts["nn"])) + 360.0) % 360.0

    def roll(sel):
        s = np.zeros(n_labels + 1, dtype=np.float32)
        g = np.zeros(n_labels + 1, dtype=np.float32)
        c = np.zeros(n_labels + 1, dtype=np.float32)
        if sel.any():
            np.add.at(s, lab[sel], sun[sel])
            np.add.at(g, lab[sel], gain[sel])
            np.add.at(c, lab[sel], 1.0)
        c[c == 0] = 1.0
        return {"sun_hours": s / c, "gain_wh_m2": g / c}

    out = {"all": roll(np.ones(lab.shape, dtype=bool)),
           "low": roll(frac < 0.5),
           "high": roll(frac >= 0.5)}

    # Each wall belongs to the compass quarter it faces.
    for name, centre in ORIENTS:
        d = np.abs(((bearing - centre + 180.0) % 360.0) - 180.0)
        out[name] = roll(d <= 45.0)

    # The metric that actually separates buildings: how much energy the whole
    # envelope takes on the sunniest side. A tall tower with open sky on its
    # south face scores high; the same tower buried in a canyon does not.
    best_gain = np.zeros(n_labels + 1, dtype=np.float32)
    best_sun = np.zeros(n_labels + 1, dtype=np.float32)
    for name, _ in ORIENTS:
        np.maximum(best_gain, out[name]["gain_wh_m2"], out=best_gain)
        np.maximum(best_sun, out[name]["sun_hours"], out=best_sun)
    out["best"] = {"sun_hours": best_sun, "gain_wh_m2": best_gain}
    return out
