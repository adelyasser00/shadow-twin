"""
Download the open data a city's config names, into data/cities/<id>/.

    .venv-heat/Scripts/python -I tools/get_city_data.py rotterdam
    .venv-heat/Scripts/python -I tools/get_city_data.py dubai      (needs duckdb, see below)

Rotterdam
  3DBAG v2025.09.03 GeoPackage tiles over the study box, checked against the
  SHA-256 in 3DBAG's tile index. AHN3 0.5 m DSM and DTM sheets 37HN1 and
  37FZ1 from the GeoTiles mirror of the official AHN files (AHN5 is read
  remotely, window by window, by the builder). Rotterdam The Hague Airport
  TMYx. CBS neighbourhoods 2024 and the BGT's water and vegetation, both
  through PDOK's OGC APIs (current objects only).

Dubai
  Overture 2026-09-23.1 buildings and building parts in a box around
  Downtown, through DuckDB's S3 reader (pip install duckdb into any Python;
  it is not in .venv-heat). Meta / WRI canopy height v2 tiles, Copernicus
  GLO-30 tile N25 E055, Dubai International Airport TMYx.

Everything downloaded is data: it is never executed, and data/ is not in git.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "shadow-twin data download (github.com/adelyasser00)"}


def fetch(url, dst, timeout=1800):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, dst)
    print(f"  {os.path.basename(dst)}  {os.path.getsize(dst) / 1e6:.1f} MB", flush=True)
    return dst


def ogc_items(base, collection, bbox, out, keep=None, current_only=False):
    """All features of a PDOK OGC API collection in a lon/lat box, paged, as GeoJSON."""
    if os.path.exists(out):
        return out
    url = f"{base}/collections/{collection}/items?bbox={','.join(map(str, bbox))}&f=json&limit=1000"
    feats = []
    while url:
        for k in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                    d = json.load(r)
                break
            except OSError:
                time.sleep(5)
        else:
            raise SystemExit(f"failed: {url}")
        for ft in d["features"]:
            p = ft["properties"]
            if current_only and p.get("eind_registratie") is not None:
                continue
            feats.append({"type": "Feature", "geometry": ft["geometry"],
                          "properties": {k: p.get(k) for k in keep} if keep else p})
        url = next((l["href"] for l in d["links"] if l["rel"] == "next"), None)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    print(f"  {collection}: {len(feats):,} features -> {out}", flush=True)
    return out


def rotterdam():
    d = os.path.join(HERE, "data", "cities", "rotterdam")
    print("3DBAG tiles")
    idx = fetch("https://data.3dbag.nl/v20250903/tile_index.fgb", os.path.join(d, "3dbag", "tile_index.fgb"))
    import geopandas as gpd
    from shapely.geometry import box
    t = gpd.read_file(idx)
    t = t[t.intersects(box(89000, 432500, 96000, 439500))]

    def tile(row):
        name = row.gpkg_download.rsplit("/", 1)[1].replace(".gz", "")
        dst = os.path.join(d, "3dbag", "tiles", name)
        if os.path.exists(dst):
            return name, "cached"
        with urllib.request.urlopen(urllib.request.Request(row.gpkg_download, headers=UA), timeout=600) as r:
            data = r.read()
        raw = gzip.decompress(data)
        ok = row.gpkg_sha256 in (hashlib.sha256(raw).hexdigest(), hashlib.sha256(data).hexdigest())
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(raw)
        return name, "ok" if ok else "SHA-256 MISMATCH"

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(tile, [r for _, r in t.iterrows()]))
    bad = [n for n, s in res if s.startswith("SHA")]
    print(f"  {len(res)} tiles, {len(bad)} checksum mismatches")
    print("AHN3 0.5 m DSM and DTM")
    for sheet in ("37HN1", "37FZ1"):
        for kind, folder in (("R", "DSM_50cm"), ("M", "DTM_50cm")):
            z = fetch(f"https://basisdata.nl/hwh-ahn/AHN3/{folder}/{kind}_{sheet}.zip",
                      os.path.join(d, "ahn3", f"{kind}_{sheet}.zip"))
            tif = os.path.join(d, "ahn3", f"{kind}_{sheet}.TIF")
            if not os.path.exists(tif):
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(os.path.join(d, "ahn3"))
    print("weather")
    z = fetch("https://climate.onebuilding.org/WMO_Region_6_Europe/NLD_Netherlands/ZH_Zuid-Holland/"
              "NLD_ZH_Rotterdam.The.Hague.AP.063440_TMYx.2011-2025.zip", os.path.join(d, "weather", "rtm.zip"))
    with zipfile.ZipFile(z) as zf:
        zf.extractall(os.path.join(d, "weather"))
    print("CBS neighbourhoods and BGT")
    ogc_items("https://api.pdok.nl/cbs/wijken-en-buurten-2024/ogc/v1", "buurten",
              (4.465, 51.905, 4.495, 51.925), os.path.join(d, "cbs", "buurten.json"))
    keep = ("fysiek_voorkomen", "plus_fysiek_voorkomen", "type", "plus_type", "lokaal_id",
            "relatieve_hoogteligging")
    box_ll = (4.452, 51.895, 4.508, 51.932)
    ogc_items("https://api.pdok.nl/lv/bgt/ogc/v1", "waterdeel", box_ll,
              os.path.join(d, "bgt", "water.geojson"), keep, current_only=True)
    ogc_items("https://api.pdok.nl/lv/bgt/ogc/v1", "begroeidterreindeel", box_ll,
              os.path.join(d, "bgt", "begroeid.geojson"), keep, current_only=True)


def dubai():
    d = os.path.join(HERE, "data", "cities", "dubai")
    print("Meta / WRI canopy height v2")
    for q in ("1230231303", "1230231321"):
        fetch(f"https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3/chm/{q}.tif",
              os.path.join(d, "canopy", f"{q}.tif"))
    print("Copernicus GLO-30")
    n = "Copernicus_DSM_COG_10_N25_00_E055_00_DEM"
    fetch(f"https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif", os.path.join(d, "dem", f"{n}.tif"))
    print("weather")
    z = fetch("https://climate.onebuilding.org/WMO_Region_2_Asia/ARE_United_Arab_Emirates/"
              "ARE_DU_Dubai.Intl.AP.411940_TMYx.2011-2025.zip", os.path.join(d, "weather", "dxb.zip"))
    with zipfile.ZipFile(z) as zf:
        zf.extractall(os.path.join(d, "weather"))
    print("Overture buildings and parts (DuckDB)")
    try:
        import duckdb
    except ImportError:
        raise SystemExit("pip install duckdb, then run this again for the Overture step")
    out = os.path.join(d, "overture")
    os.makedirs(out, exist_ok=True)
    x0, y0, x1, y1 = 55.225, 25.150, 55.325, 25.245
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2';")
    for typ in ("building", "building_part"):
        dst = os.path.join(out, f"{typ}.gpkg")
        if os.path.exists(dst) and os.path.getsize(dst) > 100_000:
            continue
        src = f"s3://overturemaps-us-west-2/release/2026-09-23.1/theme=buildings/type={typ}/*"
        extra = "building_id," if typ == "building_part" else "has_parts,"
        con.execute(f"""COPY (
            SELECT id, {extra} names.primary AS name, height, min_height, num_floors, min_floor,
                   roof_height, class, subtype,
                   list_filter(sources, s -> s.dataset = 'OpenStreetMap')[1].record_id AS osm_id,
                   sources[1].dataset AS source, geometry
            FROM read_parquet('{src}', hive_partitioning=1)
            WHERE bbox.xmin > {x0} AND bbox.xmax < {x1} AND bbox.ymin > {y0} AND bbox.ymax < {y1})
            TO '{dst.replace(os.sep, "/")}' WITH (FORMAT GDAL, DRIVER 'GPKG', SRS 'EPSG:4326')""")
        print(f"  {typ}: {dst}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    {"rotterdam": rotterdam, "dubai": dubai}.get(which, lambda: sys.exit(__doc__))()
