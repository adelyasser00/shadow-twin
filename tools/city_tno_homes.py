"""
The Dutch TNO sunlight test, homes only: which buildings with homes lost
their 2 hours, and how many homes are in them.

    TWIN_CITY=rotterdam .venv-heat/Scripts/python -I tools/city_tno_homes.py

Homes are BAG verblijfsobjecten (addressable units) with "woonfunctie" among
their uses and in use (status "Verblijfsobject in gebruik"), from PDOK's BAG
OGC API v2, inside the study window. A unit is placed in the 3DBAG outline
that holds its address point. The point is one per entrance, not per flat
(all flats of a block share it), so the test stays per building: the
building's sunniest lower wall on the test day, from city_shade --tno
(shade/tno_feb_buildings.npz), against 2 hours.

Writes shade/tno_feb_homes.json and caches the units in
<data>/bag/homes.geojson (data/, not committed).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pipeline import city  # noqa: E402

API = "https://api.pdok.nl/kadaster/bag/ogc/v2/collections/verblijfsobject/items"
UA = {"User-Agent": "shadow-twin research (github.com/adelyasser00)"}
IN_USE = ("Verblijfsobject in gebruik", "Verblijfsobject in gebruik (niet ingemeten)")


def fetch_homes(bbox_ll, path, log=print):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    url = f"{API}?bbox={','.join(f'{v:.6f}' for v in bbox_ll)}&f=json&limit=1000"
    feats, seen, n = [], set(), 0
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
            n += 1
            if "woonfunctie" not in str(p.get("gebruiksdoel")) or p.get("status") not in IN_USE:
                continue
            if p["identificatie"] in seen:
                continue
            seen.add(p["identificatie"])
            feats.append({"type": "Feature", "geometry": ft["geometry"],
                          "properties": {"id": p["identificatie"], "use": p.get("gebruiksdoel"),
                                         "area_m2": p.get("oppervlakte"),
                                         "street": p.get("openbare_ruimte_naam"), "number": p.get("huisnummer")}})
        url = next((l["href"] for l in d["links"] if l["rel"] == "next"), None)
        log(f"  {n:,} units read, {len(feats):,} homes in use")
    out = {"type": "FeatureCollection", "features": feats,
           "source": "BAG verblijfsobject, PDOK OGC API v2", "bbox": bbox_ll,
           "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def main(day="feb"):
    import geopandas as gpd
    from pipeline.city_shade import shade_dir
    from pipeline.heat_inputs import INPUTS_DIR, load
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    layers, masks, meta = load()
    bld = gpd.read_file(os.path.join(INPUTS_DIR, "buildings.gpkg"))
    win = masks["window"]
    labels = masks["labels"]
    in_win = np.unique(labels[win])
    sel = bld[bld["label"].isin(in_win)]
    bbox = tuple(sel.to_crs("EPSG:4326").total_bounds)
    homes = fetch_homes(bbox, os.path.join(city.data_dir(), "bag", "homes.geojson"), log=log)
    pts = gpd.GeoDataFrame.from_features(homes["features"], crs="EPSG:4326").to_crs(bld.crs)
    joined = gpd.sjoin(pts, bld[["label", "identificatie", "geometry"]], predicate="within", how="left")
    per = joined.dropna(subset=["label"]).groupby("label").size()
    log(f"  {len(pts):,} homes in use in the box; {int(per.sum()):,} fall inside a 3DBAG outline, "
        f"in {len(per):,} buildings")

    res = {}
    for tag in (f"tno_{day}", f"tno_{day}_focus_only"):
        p = os.path.join(shade_dir(), f"{tag}_buildings.npz")
        if not os.path.exists(p):
            log(f"  {tag}: no per-building file yet (run city_shade --tno / --tno-focus)")
            continue
        z = np.load(p)
        lab, b0, b1, need = z["label"], z["before_h"], z["today_h"], float(z["need_h"])
        homes_n = per.reindex(lab).fillna(0).to_numpy().astype(int)
        has = homes_n > 0
        lost = has & (b0 >= need) & (b1 < need)
        r = {"buildings_with_homes": int(has.sum()), "homes": int(homes_n[has].sum()),
             "pass_before": int((has & (b0 >= need)).sum()), "pass_today": int((has & (b1 >= need)).sum()),
             "homes_in_buildings_passing_before": int(homes_n[has & (b0 >= need)].sum()),
             "homes_in_buildings_passing_today": int(homes_n[has & (b1 >= need)].sum()),
             "lost_the_norm": int(lost.sum()), "homes_in_buildings_that_lost_it": int(homes_n[lost].sum()),
             "lost": [{"label": int(lab[i]), "id": str(bld.loc[bld["label"] == lab[i], "identificatie"].iloc[0]),
                       "homes": int(homes_n[i]), "before_h": round(float(b0[i]), 2),
                       "today_h": round(float(b1[i]), 2)} for i in np.nonzero(lost)[0]]}
        res[tag] = r
        log(f"  {tag}: {r['buildings_with_homes']:,} buildings with homes ({r['homes']:,} homes); sunniest lower "
            f"wall at {need:.0f} h or more: {r['pass_before']:,} before, {r['pass_today']:,} today; "
            f"{r['lost_the_norm']} lost it, holding {r['homes_in_buildings_that_lost_it']:,} homes")
    out = {"day": day, "source": homes["source"], "homes_fetched": homes.get("fetched"),
           "homes_in_box": len(pts), "results": res}
    path = os.path.join(shade_dir(), f"tno_{day}_homes.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
