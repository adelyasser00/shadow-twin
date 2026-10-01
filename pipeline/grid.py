"""
The solver grid: where it sits, which way it faces, and how far it reaches.

Two rectangles, one inside the other.

  Frame   the area being studied. Every statistic, overlay and scored wall
          comes from inside it.

  Domain  the frame plus a buffer on every side. Buildings in the buffer cast
          shadows into the frame, but nothing in the buffer is reported.

Without a buffer, a tower one block outside the frame casts nothing, and the
frame edge reads brighter than the real street. In Midtown on 21 December the
sun sits about 13 degrees up at the start of the CEQR window, so a 426 m tower
throws a shadow close to 2 km long. The buffer is how those towers get in.

Rotation
--------
The grid can turn to follow the street grid. Manhattan's avenues run about 29
degrees east of true north, and so does Central Park. A frame aligned with the
park covers it with a tight rectangle instead of a north-up box that is mostly
somewhere else, and building walls land along cell edges instead of as
staircases.

Everything downstream works in grid space, rows and columns. The only places
that care about the rotation are the sun direction (turned into grid terms
here), the compass bearing of walls (turned back), and the overlay images,
which are resampled to north-up before they reach the browser.

Conventions
-----------
bearing_deg  true compass bearing of the grid's "up" direction, degrees
             clockwise from true north. 0 is north-up.
row          increases toward bearing + 180 (grid "down")
col          increases toward bearing + 90  (grid "right")

CRS units are whatever the LiDAR uses. For NYC that is EPSG:2263, US survey
feet, which is why every distance here is converted through unit_m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

M_PER_DEG_LAT = 111_320.0


def convergence_deg(crs, lon: float, lat: float) -> float:
    """
    Angle from the CRS grid's north to true north, degrees, at a point.

    Positive means true north points east of grid north. For EPSG:2263 in
    Midtown it is about -0.02 degrees, which moves a 2 km shadow tip by under a
    metre. It is applied anyway, because it costs nothing.
    """
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = tr.transform(lon, lat)
    x1, y1 = tr.transform(lon, lat + 0.01)
    return math.degrees(math.atan2(x1 - x0, y1 - y0))


@dataclass
class Grid:
    crs: object            # the LiDAR CRS
    unit_m: float          # metres per CRS unit
    res_m: float           # cell size, metres
    bearing_deg: float     # true bearing of grid "up"
    conv_deg: float        # CRS grid north to true north, see convergence_deg
    x0: float              # CRS x of the domain's top-left corner
    y0: float              # CRS y of the domain's top-left corner
    rows: int              # domain shape
    cols: int
    frame: tuple           # (r0, r1, c0, c1), frame window inside the domain
    centre_lat: float      # frame centre
    centre_lon: float

    # ------------------------------------------------------------ geometry
    @property
    def res_u(self) -> float:
        return self.res_m / self.unit_m

    @property
    def theta_crs(self) -> float:
        """Rotation of grid "up" from CRS grid north, radians."""
        return math.radians(self.bearing_deg + self.conv_deg)

    @property
    def right(self):
        t = self.theta_crs
        return math.cos(t), -math.sin(t)

    @property
    def down(self):
        t = self.theta_crs
        return -math.sin(t), -math.cos(t)

    @property
    def transform(self):
        """rasterio Affine of the whole domain, CRS units."""
        from rasterio.transform import Affine
        (rx, ry), (dx, dy) = self.right, self.down
        r = self.res_u
        return Affine(r * rx, r * dx, self.x0, r * ry, r * dy, self.y0)

    @property
    def frame_shape(self):
        r0, r1, c0, c1 = self.frame
        return (r1 - r0, c1 - c0)

    @property
    def frame_slice(self):
        r0, r1, c0, c1 = self.frame
        return (slice(r0, r1), slice(c0, c1))

    @property
    def frame_transform(self):
        from rasterio.transform import Affine
        r0, _, c0, _ = self.frame
        return self.transform * Affine.translation(c0, r0)

    @property
    def buffer_m(self) -> float:
        r0, r1, c0, c1 = self.frame
        return min(r0, c0, self.rows - r1, self.cols - c1) * self.res_m

    def azimuth_to_grid(self, azimuth_true_deg: float) -> float:
        """A true compass azimuth, expressed relative to grid "up"."""
        return (azimuth_true_deg - self.bearing_deg) % 360.0

    # ------------------------------------------------------- conversions
    def crs_to_rowcol(self, x, y):
        """Fractional (row, col) in the domain for CRS coordinates."""
        (rx, ry), (dx, dy) = self.right, self.down
        ex = np.asarray(x, dtype=np.float64) - self.x0
        ey = np.asarray(y, dtype=np.float64) - self.y0
        col = (ex * rx + ey * ry) / self.res_u
        row = (ex * dx + ey * dy) / self.res_u
        return row, col

    def rowcol_to_crs(self, row, col):
        """CRS coordinates of fractional (row, col). Cell centres are +0.5."""
        (rx, ry), (dx, dy) = self.right, self.down
        row = np.asarray(row, dtype=np.float64)
        col = np.asarray(col, dtype=np.float64)
        x = self.x0 + (col * rx + row * dx) * self.res_u
        y = self.y0 + (col * ry + row * dy) * self.res_u
        return x, y

    def to_crs(self):
        from pyproj import Transformer
        return Transformer.from_crs("EPSG:4326", self.crs, always_xy=True).transform

    def to_lonlat(self):
        from pyproj import Transformer
        return Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True).transform

    def corners_crs(self, window=None):
        """Four corners of a window (default the frame), CRS units, clockwise."""
        r0, r1, c0, c1 = window if window is not None else self.frame
        xs, ys = self.rowcol_to_crs([r0, r0, r1, r1], [c0, c1, c1, c0])
        return list(zip(xs.tolist(), ys.tolist()))

    def corners_lonlat(self, window=None):
        tr = self.to_lonlat()
        return [tr(x, y) for x, y in self.corners_crs(window)]

    def bounds_lonlat(self, window=None, pad_deg: float = 0.0):
        c = self.corners_lonlat(window)
        lons = [p[0] for p in c]
        lats = [p[1] for p in c]
        return (min(lons) - pad_deg, min(lats) - pad_deg,
                max(lons) + pad_deg, max(lats) + pad_deg)

    def frame_polygon_crs(self):
        from shapely.geometry import Polygon
        return Polygon(self.corners_crs())

    def domain_polygon_crs(self):
        from shapely.geometry import Polygon
        return Polygon(self.corners_crs((0, self.rows, 0, self.cols)))

    def frame_size_m(self):
        h, w = self.frame_shape
        return h * self.res_m, w * self.res_m

    def cell_centres_crs(self, window=None):
        """CRS x, y of every cell centre in a window, as 2D arrays."""
        r0, r1, c0, c1 = window if window is not None else self.frame
        rr, cc = np.mgrid[r0:r1, c0:c1]
        return self.rowcol_to_crs(rr + 0.5, cc + 0.5)

    # --------------------------------------------------------- overlays
    def northup_index(self, pixel_m: float | None = None):
        """
        Map a north-up lon/lat image onto the frame.

        The browser drapes each overlay over a lon/lat rectangle. A rotated
        frame cannot be draped directly, so every overlay is resampled once:
        each output pixel looks up the frame cell under its centre, nearest
        neighbour, so class bins are never blended. Pixels outside the frame
        come back as -1 and are drawn transparent.

        Returns (index, bounds) where index is an int32 (H, W) array of flat
        frame indices and bounds is (west, south, east, north).
        """
        px = pixel_m or self.res_m
        w, s, e, n = self.bounds_lonlat()
        lat_c = 0.5 * (s + n)
        dlat = px / M_PER_DEG_LAT
        dlon = px / (M_PER_DEG_LAT * math.cos(math.radians(lat_c)))
        W = int(math.ceil((e - w) / dlon))
        H = int(math.ceil((n - s) / dlat))
        e = w + W * dlon
        s = n - H * dlat
        lons = w + (np.arange(W) + 0.5) * dlon
        lats = n - (np.arange(H) + 0.5) * dlat
        LON, LAT = np.meshgrid(lons, lats)
        x, y = self.to_crs()(LON.ravel(), LAT.ravel())
        row, col = self.crs_to_rowcol(np.asarray(x), np.asarray(y))
        r0, r1, c0, c1 = self.frame
        fr = np.floor(row).astype(np.int64) - r0
        fc = np.floor(col).astype(np.int64) - c0
        fh, fw = r1 - r0, c1 - c0
        ok = (fr >= 0) & (fr < fh) & (fc >= 0) & (fc < fw)
        idx = np.where(ok, fr * fw + fc, -1).astype(np.int32).reshape(H, W)
        return idx, (w, s, e, n)

    # ------------------------------------------------------ description
    def describe(self) -> dict:
        h, w = self.frame_size_m()
        return {
            "bearing_deg": round(self.bearing_deg, 2),
            "length_m": round(h), "width_m": round(w),
            "buffer_m": round(self.buffer_m),
            "corners": [[round(a, 6), round(b, 6)] for a, b in self.corners_lonlat()],
        }


# ---------------------------------------------------------------- builders
def _make(crs, unit_m, res_m, bearing_deg, cx, cy, length_m, width_m,
          buffer_m, centre_lat, centre_lon, conv):
    """
    A frame of length_m along the bearing and width_m across it, centred on
    CRS point (cx, cy), inside a domain buffer_m bigger on every side.
    """
    fh = int(round(length_m / res_m))
    fw = int(round(width_m / res_m))
    b = int(round(buffer_m / res_m))
    rows, cols = fh + 2 * b, fw + 2 * b
    g = Grid(crs=crs, unit_m=unit_m, res_m=res_m, bearing_deg=bearing_deg,
             conv_deg=conv, x0=0.0, y0=0.0, rows=rows, cols=cols,
             frame=(b, b + fh, b, b + fw),
             centre_lat=centre_lat, centre_lon=centre_lon)
    # Put the domain centre on (cx, cy).
    (rx, ry), (dx, dy) = g.right, g.down
    half_c = cols / 2.0 * g.res_u
    half_r = rows / 2.0 * g.res_u
    g.x0 = cx - half_c * rx - half_r * dx
    g.y0 = cy - half_c * ry - half_r * dy
    return g


def rotated(crs, unit_m, lat, lon, length_m, width_m, bearing_deg,
            res_m, buffer_m=0.0):
    from pyproj import Transformer
    cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    conv = convergence_deg(crs, lon, lat)
    return _make(crs, unit_m, res_m, bearing_deg, cx, cy, length_m, width_m,
                 buffer_m, lat, lon, conv)


def square(crs, unit_m, lat, lon, span_m, res_m, buffer_m=0.0):
    """
    The original north-up square, on the CRS grid.

    bearing is set so that grid up is CRS north exactly, which reproduces the
    earlier runs cell for cell.
    """
    from pyproj import Transformer
    cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    conv = convergence_deg(crs, lon, lat)
    return _make(crs, unit_m, res_m, -conv, cx, cy, span_m, span_m,
                 buffer_m, lat, lon, conv)


def around(crs, unit_m, geoms_lonlat, bearing_deg, res_m, margin_m, buffer_m):
    """
    The tightest frame along `bearing_deg` that holds every geometry, plus
    `margin_m` on every side.
    """
    from pyproj import Transformer
    from shapely.ops import transform as shp_tf, unary_union

    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    u = unary_union([shp_tf(tr, g) for g in geoms_lonlat])
    hull = u.convex_hull
    xs, ys = hull.exterior.xy if hull.geom_type == "Polygon" else hull.xy
    xs, ys = np.asarray(xs), np.asarray(ys)
    cxl, cyl = float(xs.mean()), float(ys.mean())
    lon0, lat0 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(cxl, cyl)
    conv = convergence_deg(crs, lon0, lat0)
    t = math.radians(bearing_deg + conv)
    up = (math.sin(t), math.cos(t))
    rt = (math.cos(t), -math.sin(t))
    a = (xs - cxl) * up[0] + (ys - cyl) * up[1]      # along the bearing
    c = (xs - cxl) * rt[0] + (ys - cyl) * rt[1]      # across it
    m_u = margin_m / unit_m
    a_lo, a_hi = a.min() - m_u, a.max() + m_u
    c_lo, c_hi = c.min() - m_u, c.max() + m_u
    ca, cc = 0.5 * (a_lo + a_hi), 0.5 * (c_lo + c_hi)
    cx = cxl + ca * up[0] + cc * rt[0]
    cy = cyl + ca * up[1] + cc * rt[1]
    length_m = (a_hi - a_lo) * unit_m
    width_m = (c_hi - c_lo) * unit_m
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(cx, cy)
    return _make(crs, unit_m, res_m, bearing_deg, cx, cy, length_m, width_m,
                 buffer_m, lat, lon, conv)


def build(crs, unit_m, res_m, *, lat=None, lon=None, span_m=700.0, length_m=None,
          width_m=None, bearing_deg=0.0, buffer_m=0.0, fit_geoms=None,
          margin_m=250.0):
    """
    The one place a run's grid is decided, so tile_info checks exactly the
    grid run_nyc will use.

    fit_geoms given     frame fitted around them along the bearing, see around()
    length/width/bearing a rectangle centred on lat, lon
    otherwise           the original north-up square of span_m
    """
    if fit_geoms is not None:
        return around(crs, unit_m, fit_geoms, bearing_deg, res_m, margin_m, buffer_m)
    if length_m or width_m or bearing_deg:
        return rotated(crs, unit_m, lat, lon, length_m or span_m, width_m or span_m,
                       bearing_deg, res_m, buffer_m)
    return square(crs, unit_m, lat, lon, span_m, res_m, buffer_m)
