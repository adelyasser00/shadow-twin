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
    A district at 2 m is well under a thousand pixels a side. A tile
    pyramid would add thousands of files and a server for nothing.
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
) -> dict:
    """
    mask        pixels drawn fully transparent in the image
    stats_mask  pixels excluded from the reported statistics. Usually wider
                than mask: the image can show rooftops while the numbers are
                about the ground people stand on. Defaults to mask.
    """
    rgba, legend = quantize(values, breaks, colors, alpha=alpha, mask=mask)
    if labels:
        for entry, lab in zip(legend, labels):
            entry["label"] = lab
    sm = stats_mask if stats_mask is not None else mask
    valid = values if sm is None else values[~sm]
    return {
        "name": name,
        "title": title,
        "units": units,
        "png": to_data_uri(rgba, downsample),
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
    with open(path, "w") as f:
        f.write(text)
    return len(text.encode("utf-8"))
