# Getting the Manhattan LiDAR

This is the only step you do by hand. Everything after it is a few commands.

**The downloader gives you `.laz` point clouds named by tile number, like
`990217.laz`.** That is the raw classified LiDAR, not a derived raster. The
pipeline reads it directly.

Point clouds are better anyway. You choose the output resolution, you never
build a huge intermediate raster, and the surface is made in front of you
instead of by someone else.

---

## Step by step

**1. Open the downloader.**

`https://finder.nyc.gov/orthoimagery`, the NYC Imagery and LiDAR Downloader.

**2. Search a coordinate.**

Paste **`40.7668, -73.9790`** into the search box.

That is the centre of the published study, just south of the park. Aiming here
rather than at the park centre puts the tower cluster along 57th Street in frame
together with the park's southern lobe, which is the area this study covers.
Aiming at the park centre gives you mostly grass.

**3. Click the tile, press Download LiDAR.**

You get `<tile>.laz`. Repeat for the neighbouring tiles. The published run uses
tiles `987215`, `987217`, `990217` and `992217`.

**Which tiles: do not guess, measure.** The grid is rotated to the Manhattan
street grid and the numbering does not run the way it looks on screen. After
each download, run:

```bash
python -m pipeline.tile_info --laz data/nyc --lat 40.7668 --lon -73.9790 --span 700
```

That reads only the file headers, so it is instant even on a 400 MB tile. It
prints the real lat/lon box of every tile you have, whether your centre point is
inside them, which compass directions are short and by how many metres, and the
largest square that does fit what you already hold.

Work the loop: download a tile, run tile_info, read which direction is missing,
go get that one. Stop when the square fits, these files are hundreds of
megabytes.

**4. Put them all in one folder.**

```
data/
  nyc/
    987215.laz
    987217.laz
    990217.laz
    992217.laz
```

Filenames do not matter. The loader takes every `.laz` and `.las` in the folder
and grids them together.

**5. Run it.**

```
pip install laspy lazrs
python -m pipeline.tile_info --laz data/nyc --lat 40.7668 --lon -73.9790 --span 700
python -m pipeline.run_nyc --laz data/nyc --lat 40.7668 --lon -73.9790 --span 700 --res 3.0 --skip-svf
python -m pipeline.build_viewer
```

Small and fast first. Confirm it looks right, then go to 2 m. See
`python -m pipeline.run_nyc --help` for the footprint and scenario options.

**One command per line, no backslashes.** PowerShell uses a backtick for line
continuation, not a backslash, so a bash-style multi-line command silently
breaks apart and only the first fragment runs. If `run_nyc` errors and you then
run `build_viewer` anyway, it rebuilds whatever `data.json` was already there
and you get an old result that looks like a new one. Check the site name printed
by `build_viewer` matches what you just asked for.

---

## What a correct run prints

```
CRS EPSG:2263  1 unit = 0.304801 m
target grid ... at 3.0 m
reading 990217.laz  ... points
... points gridded, ... of them ground class
surface coverage ...% of cells got a return
tallest object 472.4 m above street (1550 ft)
```

**Three things to check, in order:**

1. **Units.** `1 unit = 0.304801 m` means feet were detected. If it says 1.0 the
   file is in metres and that is unexpected for NYC.
2. **Tallest object.** Midtown supertalls run 300 to 470 m. If it prints 1500,
   feet are being read as metres and every shadow is 3.28 times too long. If it
   prints 30, you are looking at the park.
3. **Coverage.** Under 55% triggers a warning and means you are missing
   neighbouring tiles. Download the gap and re-run.

---

## When it goes wrong

**"No points from these tiles landed inside the square"**
The tiles do not cover the coordinate you asked for. Check the footprints on the
downloader map.

**"warning: large gaps. You are probably missing neighbouring tiles."**
Add tiles to the same folder and re-run. Nothing else changes.

**"warning: almost no ground-classified points"**
The tile has few ASPRS class 2 returns, so street level is estimated from block
minima instead. Usable, less accurate.

**"This point cloud carries no CRS"**
Unexpected for NYC. Open an issue with the filename.

**It is slow**
Tens of millions of points take a minute or two to grid. After that the solver
is seconds. If the solve itself is slow, raise `--res` or use the GPU path.

---

## Before you pick a span

On 21 December in New York the sun peaks at about 26 degrees and sits near
13 degrees at the start of the CEQR window. A 472 m tower then throws a shadow
about 2 km long. Towers outside your square cast nothing here, so crop
generously or the edge of the frame will look brighter than the real street is.
