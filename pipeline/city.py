"""
Which city the pipeline works on, and where that city's files live.

    TWIN_CITY=rotterdam .venv-heat/Scripts/python -m pipeline.city_inputs

One JSON file per city in cities/ says everything that differs between
cities: the projection, the clock, the weather file, the analysis days, the
study area, the reference spot for wind, the scenarios and where the input
data comes from. The engines (sun, shadow sweep, SOLWEIG, the wind solver)
read it through this module and never name a city.

TWIN_CITY picks the city; without it everything is New York, with the paths
and constants the published runs were made with, so those runs reproduce.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITIES_DIR = os.path.join(HERE, "cities")
DEFAULT = "nyc"


def city_id() -> str:
    return (os.environ.get("TWIN_CITY") or DEFAULT).strip().lower()


@lru_cache(maxsize=None)
def config(cid: str | None = None) -> dict:
    cid = cid or city_id()
    path = os.path.join(CITIES_DIR, f"{cid}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no city config {path}; the cities are "
                         + ", ".join(sorted(f[:-5] for f in os.listdir(CITIES_DIR) if f.endswith(".json"))))
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_path"] = path
    return cfg


def path(rel: str) -> str:
    """A path from a config, relative to the repository root."""
    return rel if os.path.isabs(rel) else os.path.join(HERE, rel)


def heat_dir(cid: str | None = None) -> str:
    return path(config(cid)["paths"]["heat"])


def wind_dir(cid: str | None = None) -> str:
    return path(config(cid)["paths"]["wind"])


def data_dir(cid: str | None = None) -> str:
    return path(config(cid)["paths"]["data"])


def viewer_dir(cid: str | None = None) -> str:
    return path(config(cid)["paths"]["viewer"])


def scenarios(cid: str | None = None) -> tuple:
    """(before, today): the older survey first."""
    return tuple(config(cid)["scenarios"])


def before(cid: str | None = None) -> str:
    return scenarios(cid)[0]


def utc_offset(cid: str | None = None) -> float:
    """Standard time all year, as the CEQR method does: no daylight saving."""
    return float(config(cid)["utc_offset"])


def clock_name(cid: str | None = None) -> str:
    return config(cid).get("clock_name", f"UTC{utc_offset(cid):+g}")


def get(key: str, default=None, cid: str | None = None):
    """A dotted key from the config, e.g. get('wind.reference_spot')."""
    v = config(cid)
    for k in key.split("."):
        if not isinstance(v, dict) or k not in v:
            return default
        v = v[k]
    return v
