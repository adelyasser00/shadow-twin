"""
Turn solver rasters into map overlays the browser can draw.

Design choices worth defending out loud, because someone will ask:

Fixed breaks, not a per-scene stretch.
    Every layer is binned on fixed, stated thresholds. Stretching each
    layer to its own min and max makes every map look dramatic and makes
    two days impossible to compare. Fixed breaks mean the colour in the
    legend has a constant meaning.

Discrete bins, not a smooth ramp.
    A continuous gradient implies continuous precision. These layers are
    computed on a 2 m grid from heights with metre-scale error. Eight bins
    tells the viewer honestly how much resolution the numbers deserve, and
    it compresses to a far smaller PNG.

Nearest-neighbour magnification in the viewer.
    The pixels stay visibly square when you zoom in. That is deliberate.
    Smoothing a 2 m grid into a soft blur is a claim about detail that is
    not in the data.

One image per layer, not a tile pyramid.
    A district at 2 m is a couple of thousand pixels a side. A tile
    pyramid would add thousands of files and a server for nothing.

Palette PNGs, not full colour.
    Every layer has at most eight colours plus transparent, so each pixel
    is stored as one palette index instead of four bytes of RGBA. Same
    image in the browser, a fraction of the size, which is what keeps a
    whole-park build loadable on a phone.

North-up images from a rotated grid.
    The browser drapes each image over a lon/lat rectangle. When the grid
    is turned to follow the streets, a Painter resamples every layer onto a
    north-up lon/lat raster first, nearest neighbour, so bins never blend.
    Statistics are always taken on the solver grid itself, never on the
    resampled picture.
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
from PIL import Image

# Shaded and sunlit. Deliberately not red and green.
SHADOW_COLORS = ["#1b3a6b", "#f5c451"]
SHADOW_LABELS = ["in shadow", "in direct sun"]

# Hours of direct sun across the modelled day.
SUNHOUR_COLORS = [
    "#10203f", "#1d3f6e", "#2e6a8e", "#49939a",
    "#84b78c", "#d3c96b", "#e99a4a", "#c9482f",
]

# Sky View Factor, 0 fully enclosed to 1 fully open.
SVF_COLORS = [
    "#2a1a3d", "#43285c", "#5b4080", "#6c619c",
    "#7d86ac", "#94a9bb", "#b6c9cd", "#e2ecea",
]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def quantize(
    values: np.ndarray,
    breaks: list[float],
    colors: list[str],
    alpha: int = 215,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Bin a float raster into an RGBA image plus a legend description.

    breaks  ascending bin edges, length = len(colors) + 1
    mask    True where the pixel should be fully transparent
    """
    assert len(breaks) == len(colors) + 1, "breaks must be one longer than colors"
    h, w = values.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    legend = []

    for i, col in enumerate(colors):
        lo, hi = breaks[i], breaks[i + 1]
        last = i == len(colors) - 1
        sel = (values >= lo) & (values <= hi) if last else (values >= lo) & (values < hi)
        r, g, b = _hex_to_rgb(col)
        rgba[sel] = (r, g, b, alpha)
        legend.append(
            {
                "color": col,
                "min": float(lo),
                "max": float(hi),
                "count": int(sel.sum()),
            }
        )

    if mask is not None:
        rgba[mask] = (0, 0, 0, 0)

    return rgba, legend


def encode_indexed(index: np.ndarray, colors: list[str], alphas: list[int]) -> str:
    """
    Palette PNG data URI. index 0 is transparent, index i is colors[i-1].
    """
    pal = [0, 0, 0]
    for c in colors:
        pal.extend(_hex_to_rgb(c))
    img = Image.fromarray(index.astype(np.uint8), mode="P")
    img.putpalette(pal + [0, 0, 0] * (256 - len(pal) // 3))
    trns = bytes([0] + [int(a) for a in alphas])
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, transparency=trns)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class Painter:
    """
    Turns grid rasters into north-up palette PNGs.

    index   int32 (H, W) flat frame index per output pixel, -1 outside the
            frame, from Grid.northup_index
    bounds  (west, south, east, north) of that image
    """

    def __init__(self, index: np.ndarray, bounds):
        self.index = index
        self.bounds = bounds
        self.inside = index >= 0
        self._flat = np.where(self.inside, index, 0)

    def bin_index(self, values, breaks, mask=None):
        """Palette index per output pixel: 0 transparent, 1..k the bins."""
        v = values.ravel()[self._flat]
        out = np.zeros(v.shape, dtype=np.uint8)
        k = len(breaks) - 1
        for i in range(k):
            lo, hi = breaks[i], breaks[i + 1]
            last = i == k - 1
            sel = (v >= lo) & (v <= hi) if last else (v >= lo) & (v < hi)
            out[sel] = i + 1
        drop = ~self.inside
        if mask is not None:
            drop = drop | mask.ravel()[self._flat]
        out[drop] = 0
        return out

    def png(self, values, breaks, colors, alpha=215, mask=None):
        idx = self.bin_index(values, breaks, mask)
        return encode_indexed(idx, colors, [alpha] * len(colors))


def to_data_uri(rgba: np.ndarray, downsample: int = 1) -> str:
    """PNG data URI. Downsampling uses nearest so bins are never blended."""
    img = Image.fromarray(rgba, mode="RGBA")
    if downsample > 1:
        img = img.resize(
            (img.width // downsample, img.height // downsample), Image.NEAREST
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def layer(
    name: str,
    title: str,
    units: str,
    values: np.ndarray,
    breaks: list[float],
    colors: list[str],
    mask: np.ndarray | None = None,
    labels: list[str] | None = None,
    alpha: int = 215,
    downsample: int = 1,
    stats_mask: np.ndarray | None = None,
    painter: "Painter | None" = None,
) -> dict:
    """
    mask        pixels drawn fully transparent in the image
    stats_mask  pixels excluded from the reported statistics. Usually wider
                than mask: the image can show rooftops while the numbers are
                about the ground people stand on. Defaults to mask.
    painter     when given, the image is a north-up palette PNG from it
    """
    rgba, legend = quantize(values, breaks, colors, alpha=alpha, mask=mask)
    if labels:
        for entry, lab in zip(legend, labels):
            entry["label"] = lab
    sm = stats_mask if stats_mask is not None else mask
    valid = values if sm is None else values[~sm]
    png = (painter.png(values, breaks, colors, alpha=alpha, mask=mask)
           if painter is not None else to_data_uri(rgba, downsample))
    return {
        "name": name,
        "title": title,
        "units": units,
        "png": png,
        "legend": legend,
        "stats": {
            "min": float(np.min(valid)) if valid.size else None,
            "max": float(np.max(valid)) if valid.size else None,
            "mean": float(np.mean(valid)) if valid.size else None,
            "median": float(np.median(valid)) if valid.size else None,
        },
    }


def write_bundle(path: str, bundle: dict) -> int:
    """Write the viewer payload and return its size in bytes."""
    text = json.dumps(bundle, separators=(",", ":"))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return len(text.encode("utf-8"))
