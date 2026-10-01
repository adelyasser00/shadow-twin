# Getting the Manhattan data

This is the only step you do by hand. Everything after it is a few commands.

Three things: LiDAR tiles, building footprints, and the park polygon.

---

## LiDAR

**The downloader gives you `.laz` point clouds named by tile number, like
`990217.laz`.** That is the raw classified LiDAR, not a derived raster. The
pipeline reads it directly. You choose the output resolution, you never build a
huge intermediate raster, and the surface is made in front of you instead of by
someone else.

**How the tiles are named.** Each tile is a 2,500 ft square on the New York
state plane grid (EPSG:2263), named by its lower-left corner in thousands of
feet. `990217` starts at x 990,000 ft, y 217,500 ft. `992217` is the next tile
east, `990220` the next one north. Lower first number is further west, lower
second number is further south. The squares are not rotated; Manhattan's
streets are.

**1. Open the downloader.** `https://finder.nyc.gov/orthoimagery`, the NYC
Imagery and LiDAR Downloader.

**2. Download the tiles.** Search a coordinate, click the tile, press Download
LiDAR. For the whole of Central Park:

```
987215  987217  987220
990217  990220  990222  990225  990215
992217  992220  992222  992225  992227  992230
995220  995222  995225  995227  995230
997225  997227  997230
```

`990215` holds the corner at 57th to 59th Street between Sixth and Madison, and
432 Park Avenue. `995220` and `997225` are slivers of Park Avenue at the frame's
east edge. For a measured south buffer instead of footprint prisms, `992215` and
`990212` are the next most useful.

**Do not guess, measure.** After each download:

```
python -m pipeline.tile_info --laz data/nyc --preset central-park --resource data/nyc/park.geojson
```

It reads only the file headers, so it is instant even on a 400 MB tile. It
prints a map of the tiles you have, checks the exact frame and buffer the run
will use, cell by cell against each tile, and names every tile still missing.
Add `--scan` once to read the points too and see how much of each tile really
has returns; water shows up as empty corners there.

**3. One folder.**

```
data/
  nyc/
    987215.laz
    ...
    footprints.geojson
    park.geojson
```

The loader takes every `.laz` and `.las` in the folder, skipping any tile that
does not touch the run without reading a point.

---

## Building footprints

NYC Open Data, "Building Footprints" (id `5zhs-2jue`), export as GeoJSON. The
full city is about a gigabyte. Crop it once to the area around your tiles:

```
python -m pipeline.nyc_footprints crop data/nyc/footprints.geojson data/nyc/footprints_crop.geojson --laz data/nyc
```

Every run after that reads the crop in seconds instead of loading the whole
city. The run uses `height_roof` for heights and the construction year to keep
towers finished after the survey out of the 2017 buffer.

## Park polygon

`data/nyc/park.geojson`. If the file holds more than one park, the run lists
them by name. Keep only the one you mean with
`--resource-where "Central Park"`, or the other parks inside the frame are
counted as Central Park.

---

## Run it

Check the regression first. The original 700 m frame, with the new code, should
print the published numbers again, 46.9% and 55.3% of the park never in sun on
21 December:

```
python -m pipeline.run_nyc --laz data/nyc --footprints data/nyc/footprints_crop.geojson --resource data/nyc/park.geojson --resource-name "Central Park" --lat 40.7668 --lon -73.9790 --span 700 --skip-svf --out data/runs/check_2017.json
python -m pipeline.run_nyc --laz data/nyc --footprints data/nyc/footprints_crop.geojson --resource data/nyc/park.geojson --resource-name "Central Park" --lat 40.7668 --lon -73.9790 --span 700 --skip-svf --burn-footprints --burn-mode all --out data/runs/check_today.json
```

Then the whole park. The first run grids the LiDAR and caches it in
`data/cache/`; the second reuses it:

```
python -m pipeline.run_nyc --preset central-park --laz data/nyc --footprints data/nyc/footprints_crop.geojson --resource data/nyc/park.geojson --save-rasters data/runs/cp_2017.npz
python -m pipeline.build_viewer
python -m pipeline.run_nyc --preset central-park --laz data/nyc --footprints data/nyc/footprints_crop.geojson --resource data/nyc/park.geojson --burn-footprints --save-rasters data/runs/cp_today.npz
python -m pipeline.build_viewer
python tools/compare_runs.py data/runs/cp_2017.npz data/runs/cp_today.npz --csv data/runs/cp_profile.csv
```

**One command per line, no backslashes.** PowerShell uses a backtick for line
continuation, not a backslash, so a bash-style multi-line command silently
breaks apart and only the first fragment runs. If `run_nyc` errors and you then
run `build_viewer` anyway, it rebuilds whatever `data.json` was already there
and you get an old result that looks like a new one. Check the site name and
scenario printed by `build_viewer` match what you just asked for.

---

## What a correct run prints

```
CRS EPSG:2263  1 unit = 0.304801 m
frame 4,606 m by 1,358 m at 2.0 m, long axis 28.9 deg
domain adds 2,000 m of buffer
... tile(s) touch the domain
tallest object in the frame, LiDAR as surveyed, ... m above street
tallest object in the frame after the scenario 472.4 m (1550 ft)
```

The second tallest line only appears in the today run.

**Four things to check, in order:**

1. **Units.** `1 unit = 0.304801 m` means feet were detected. If it says 1.0 the
   file is in metres and that is unexpected for NYC.
2. **Tallest object.** In the today run it should be Central Park Tower, 472 m.
   If it prints 1,500, feet are being read as metres and every shadow is 3.28
   times too long.
3. **The park.** The run lists which park features it kept and their area.
   Central Park is about 341 ha.
4. **Coverage.** The run reports the share of the frame outside the tiles. That
   ground is blanked and left out of every number; tile_info names the tiles
   that fill it.

---

## When it goes wrong

**"None of these tiles touches the requested area"**
Wrong folder, or a coordinate outside the tiles. Run tile_info.

**"No resource features matched"**
The `--resource-where` text is not in the park file. The error lists the names
that are there.

**"reading 1,009 MB of GeoJSON"**
You pointed the run at the full city footprints. It works, slowly and with a lot
of memory. Crop it once.

**It is slow**
Gridding the tiles takes a few minutes the first time and is cached after that.
The shadows take a couple of minutes per run on a laptop CPU, the walls a few
more. `--skip-facades` for a quick look, or the GPU path on Kaggle.
