# Shadow twin

Computes where the shadows fall across a city block, hour by hour, from raw
LiDAR. Runs in a browser, from one self-contained HTML file.

For any patch of ground, at any hour, it answers three questions:

- Is this spot in direct sun, or is something in the way?
- How many hours of direct sun does it get across the whole day?
- How much of the open sky can it see?

All from measured geometry and calculated sun position. No weather data, no
temperature, no wind.

**It is not a heat model.** It computes no temperature of any kind, no wind, and
no thermal comfort index. Shade is one input to how a place feels. What this does
compute, it computes correctly, and it stops where the data stops.

**Current target:** Central Park South, Manhattan, on NYC 2017 LiDAR. The winter
solstice against the summer solstice, side by side.

| | |
|---|---|
| Sun position accuracy | 0.002 degrees on altitude |
| Solver | shadow sweep plus horizon scan, pure array shifts |
| Speed | seconds on a CPU for a 600 x 600 grid |
| Input | classified LAS/LAZ point clouds |
| Output | one HTML file, no server, no build step |
| Tests | analytical, against values derived by hand |

---

## Try it before the download finishes

```bash
python -m pipeline.make_nyc_test_tifs --out /tmp/nyc_test --span-m 700
python -m pipeline.run_nyc --dsm /tmp/nyc_test --dem /tmp/nyc_test --span 595 --res 2.0
python -m pipeline.build_viewer
```

That writes GeoTIFFs in the exact NYC format, feet and EPSG:2263 and all, with
towers of known height. It proves the reader, the unit conversion, the solver,
the export and the viewer all work. The geometry is synthetic and the page says
so in red. It is a smoke test, not a demo.

---

## The one thing you have to do by hand

Download the LiDAR. See **`tools/GET_THE_DATA.md`**. Ten minutes.

Everything after that is two commands.

---

## Run it

```bash
python -m pipeline.verify_solar          # solar geometry vs hand values
python -m pipeline.verify_geometry       # shadows and SVF vs hand values

python -m pipeline.run_nyc --dsm data/nyc --dem data/nyc --span 2400 --res 2.0
python -m pipeline.build_viewer
```

Open `viewer/shadow-twin.html`.

Requirements: `pip install numpy pillow rasterio shapely scipy matplotlib`

**Start small.** Run `--span 1600 --res 3.0 --skip-svf` first to confirm the
tiles are right. It takes seconds. Only then go to full resolution.

**Heavy renders go to Kaggle**, not your laptop. See
`notebooks/kaggle_render.py`. Free T4, 16 GB, ~30 h a week. Pass `--gpu` there.

**No internet?** See `tools/vendor_cesium.md`. One npm install and the viewer
never needs the network again.

---

## What's in the viewer

- **Winter / Summer** toggle. Same geometry, two days of the year.
- **Time slider** and **Play the day**. Watch the shadows sweep.
- **Shade now / Sun hours / Sky view** layers.
- **Click a building** for its height, storeys, and current shadow length.
- **Clean view for recording** (button, or press `R`). Hides every panel and
  leaves a single caption line. This is what you record for the post.
- Keys: `R` clean view, `space` play, arrows step through hours, `Esc` close.

---

## How to evaluate it

Do not trust it because it renders. Every claim below is checkable.

**Solar geometry.** `verify_solar.py` checks declination at both solstices and
both equinoxes, peak altitude against the analytical `90 - |lat - declination|`,
that the sun is due south at its highest, that it is overhead at the equator at
equinox, and that azimuth sweeps east to west. Altitude is accurate to **0.002
degrees**. Worst azimuth error is 0.46 degrees, at the moment the sun is nearly
overhead and azimuth is least well defined.

For New York this shows up directly: 21 December peaks at 25.8 degrees against
the analytical 25.79, and 21 June at 72.7 against 72.67.

**Shadow casting.** `verify_geometry.py` builds a tower of known height, puts
the sun at a known altitude, measures the shadow, and compares it to
`H / tan(altitude)`. Three heights, three sun angles. Plus direction tests (sun
east throws the shadow west), an overhead sun shading nothing, and a
below-horizon sun shading everything.

**Sky view factor.** Flat ground returns exactly 1.0. A point against an
infinitely tall wall returns 0.531 where the analytical answer is 0.5, the gap
being finite azimuth sampling. SVF falls monotonically as canyon walls rise and
never leaves 0 to 1.

**Units.** The single most dangerous bug in this project is reading US survey
feet as metres, which makes every shadow 3.28 times too long. `pipeline/nyc.py`
detects the unit from the CRS and refuses to guess. The run prints the tallest
object in both metres and feet. **Check it.** Midtown supertalls are 300 to
470 m. If it prints 1500, stop.

**GPU path.** `verify_torch.py` runs both backends on the same input and
compares cell by cell. The CPU path is the reference because it is the one with
analytical tests.

---

## How to explain it

**One breath.** It computes where the shadows fall across Midtown, hour by hour,
from the city's own LiDAR survey. Scrub the day and watch them move. Switch
between December and June and watch half the street lose the sun.

**To an engineer.** Take a digital surface model, terrain plus everything built
on it. For each hour compute the sun's position with the NOAA algorithm, then
sweep every pixel outward toward the sun in one-cell steps. If any surface along
that ray rises above the sun ray from that pixel, the pixel is shaded. Accumulate
over the day for sun hours. Scan the horizon in 16 directions and take a
cosine-squared weighted mean for sky view factor. It is all array shifts, which
is why it runs in seconds on a CPU and in well under a second on a GPU.

**To an architect.** It is a shadow study, the same deliverable you would produce
in Revit or Ladybug for a planning submission. The difference is that this one
covers every building in a square kilometre at once, from public survey data,
and opens in a browser tab instead of a licensed desktop package.

**If someone asks whether this is a heat model.** It is not, and say so before
they do. Shade is one input to how a place feels. Thermal comfort also needs mean
radiant temperature, air temperature, humidity and wind, and this computes none
of them. What it does compute, it computes correctly, and it stops where the
data stops.

---

## What it does not do

- No temperature of any kind, no wind, no comfort index.
- No cloud. These are clear-sky geometric sun hours, so the real figure on any
  given day is lower.
- Vegetation in the surface model is treated as fully opaque. Real canopy
  transmits some light, so tree shade is slightly overstated.
- Max pooling on downsample keeps roof heights but grows each building by up to
  one cell, erring toward more shadow rather than less.
- Whole-cell ray stepping puts each shadow edge within about one cell.
- Buildings outside the frame cast nothing. Winter shadows in Manhattan run over
  a kilometre, so crop generously.
- The survey is a snapshot. Anything built after the capture date is not in it.

---

## Layout

```
pipeline/
  solar.py              NOAA sun position. The only astronomy here.
  geometry.py           Shadow sweep and sky view factor. CPU reference.
  geometry_torch.py     Same maths on a GPU. For Kaggle and Modal.
  laz.py                LAS/LAZ point cloud to DSM and DEM. The NYC download path.
  tile_info.py          Reads LiDAR headers and says which tiles you still need.
  nyc.py                Reader for derived GeoTIFF rasters, if you ever get them.
  footprints.py         Building mask to 3D footprints with median heights.
  export.py             Rasters to binned PNG overlays plus legend and stats.
  run_nyc.py            End to end on NYC tiles. The one you run.
  build_viewer.py       Inlines data.json into the template.
  make_nyc_test_tifs.py Writes NYC-format test rasters in feet, so the whole
                        pipeline can be exercised before any download finishes.
  verify_solar.py       Solar self-tests.
  verify_geometry.py    Geometry self-tests.
  verify_torch.py       GPU against CPU.
tools/
  GET_THE_DATA.md       Step by step LiDAR download.
  vendor_cesium.md      Making the viewer work offline.
  OMNIVERSE.md          Whether Omniverse belongs in this. Short answer: no.
notebooks/
  kaggle_render.py      Cell by cell script for the full resolution render.
viewer/
  template.html         The viewer, with a data placeholder.
  data.json             Current solver output.
  shadow-twin.html      Built, self-contained. Open this one.
```

---

## Next, in order

1. Download the tiles. `tools/GET_THE_DATA.md`.
2. Small run. Check the tallest-object figure is sane.
3. Full run, on Kaggle if it is slow.
4. Vendor Cesium so the page works without a connection.
5. Press `R`, play the December day, record 20 seconds.

---

## Publishing a build

`viewer/shadow-twin.html` is deliberately gitignored. It is generated output and
it can reach several megabytes, which does not belong in git history.

To share a result, drag the file onto Cloudflare Pages or any static host and
link it from here. Do not upload `viewer/vendor/` with it; the CDN serves Cesium
faster than a static host will.

---

## Licence and credit

MIT, see `LICENSE`. Data sources, methods and the papers behind them are in
`CREDITS.md`. No LiDAR data is redistributed in this repository.

Contributing rules and how credit is recorded: `CONTRIBUTING.md` and
`AUTHORS.md`.
