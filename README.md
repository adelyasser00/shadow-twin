# Shadow twin

A shadow study of Central Park, Manhattan, computed from the city's own LiDAR
survey and run with New York City's official shadow assessment method.
It runs in a browser, on desktop and on phones.

Built by Adel Yasser.

**Try it:** https://shadow-twin-adelyasser00.pages.dev/
**2017 only:** https://shadow-twin-adelyasser00.pages.dev/2017

For any patch of ground, at any time on the four CEQR analysis days, it answers:

- Is this spot in direct sun, or is something in the way?
- How many hours of direct sun does it get across the day?
- How much direct sun does each wall of each building get?

**It is not a heat model.** It computes no temperature, no wind and no thermal
comfort index. Shade is one input to how a place feels. What this computes, it
computes from measured geometry and calculated sun position, and it stops where
the data stops.

---

## The first result

Frame: 700 m across, 2 m cells, centred on the southern end of Central Park.
This is the run published in September 2026.
"Today" means the 2017 LiDAR surface raised to current recorded roof heights
from NYC Building Footprints.

Share of the park inside the frame that never gets direct sun during the CEQR
window, as first published, and corrected:

| CEQR day | 2017 survey | Today, as published | 2017, corrected | Today, corrected |
|---|---|---|---|---|
| 21 December | 46.9% | 55.3% | 46.6% | 48.8% |
| 21 March / 21 September | 10.5% | 12.0% | 10.6% | 10.9% |
| 6 May / 6 August | 2.8% | 3.1% | 2.9% | 3.0% |
| 21 June | 1.3% | 1.4% | 1.4% | 1.4% |

**The published "today" figures overstated the change.** The first version
built "today" by stamping every footprint at its recorded roof height over the
whole lot. A footprint has one height, its tallest roof, so every setback,
terrace and lower wing of a building that had not changed since 2017 became a
solid slab. In this frame 88% of the cells it raised belonged to buildings that
had not changed. The corrected method only raises buildings whose recorded
roof stands more than 6 m above the tallest thing the 2017 survey measured
inside them: seven buildings in this frame. Run with only that one setting
changed, 21 December comes out at 48.7% instead of 55.3%.

The corrected columns also count Central Park only, not every park polygon in
the frame, and let towers up to 2 km away cast in. Those two changes move the
numbers by under a point.

The new towers do add shade. About a fifth as much as first published.

The difference is almost all in winter. The sun peaks at 25.8 degrees on
21 December and 72.7 degrees on 21 June, so tall buildings cast their longest
shadows exactly when the park has the least sun to lose.

Sanity check: the today run reports a tallest object of 472.4 m (1,550 ft),
Central Park Tower's height. It was finished after the 2017 survey, so this
confirms the footprint heights are being applied.

**Read these numbers with their limits:** a 700 m frame, not the whole park;
"today" is a raised 2017 surface, not a new survey; direct sun under clear sky,
geometry only. And in that run nothing outside the frame cast a shadow, which
left out towers like 432 Park Avenue, a few blocks south-east of the frame. On
21 December it throws a shadow well into the park.

---

## Whole park

The pipeline now covers the whole of Central Park, 59th to 110th Street:

- **Frame:** fitted to the park polygon, turned 28.9 degrees to line up with
  the avenues, with 250 m of city on every side. About 4.6 by 1.4 km.
- **Buffer:** 2 km around the frame. Buildings there cast shadows in but are
  never counted. 2 km covers a 426 m tower at the lowest sun in the December
  window. Inside the LiDAR tiles the buffer is the measured surface; beyond
  them each footprint stands as a prism at its recorded roof height, and the
  2017 run only stands buildings built before 2017.
- **Same ground, old and new:** the run reports the published 700 m frame's
  numbers again, with the buffer, so the correction is visible.
- **Profile:** share of the park never in sun, in bands north from 59th
  Street. If the change comes from the towers, it fades with distance.
- `tools/compare_runs.py` puts the 2017 and today runs side by side, cell by
  cell: hectares newly without direct sun, hectares that lost an hour or more.

Central Park, 339.8 ha, on 21 December:

| | |
|---|---|
| Never in direct sun, 2017 | 27.3% |
| Never in direct sun, today | 27.5% |
| Lost some direct sun | 26.2 ha |
| Lost an hour or more | 4.8 ha |
| Lost all of it | 0.7 ha |
| Lost sun-hours within 1.5 km of 59th Street | 97% |

North of about 1.5 km from 59th Street nothing changes. The new towers throw
long fingers of shadow up the southern end, and the effect fades with
distance, which is what you would expect if they are the cause.

The absolute share never in sun is mostly trees: the LiDAR treats canopy as
solid, and it was flown with leaves on. The change between the two runs is the
result, not the absolute figure.

---

## Method

- **Timing:** NYC CEQR Technical Manual, Chapter 8. Four analysis days
  (21 December, 21 March, 6 May, 21 June), from 1.5 hours after sunrise to
  1.5 hours before sunset, Eastern Standard Time all year, sampled every
  30 minutes. CEQR assesses shadows on sunlight-sensitive resources such as
  parks. Street and wall figures are reported here because they matter for
  people on the ground, not because CEQR asks for them.
- **Sun position:** NOAA solar position algorithm.
- **Surface:** highest LiDAR return per 2 m cell, ground from the lowest ground
  return. Cells more than 6 m from any return are one of two things. Water
  inside the survey (the Reservoir, the Lake, the Pond) returns nothing to the
  laser; it is kept as open surface at the level of the ground around it,
  because it gets sun like any lawn. Ground outside the tiles is marked no-data
  and left blank, never filled.
- **Buildings:** NYC Building Footprints, one polygon per real building. In the
  2017 run a building's height is the median LiDAR height inside its footprint;
  in the today run it is the recorded `height_roof`.
- **Today:** the 2017 surface, raised only where a building stands more than
  6 m above the tallest thing the survey measured inside it, meaning it was
  built or topped out since. `--burn-mode all` reproduces the first,
  overstated method.
- **Frame:** turned to the street grid when that fits the study better, and
  resampled to north-up only for the images the browser drapes. Every number is
  computed on the solver's own grid.
- **Shadows:** shift-and-subtract shadow sweep (Ratti and Richens).
- **Walls:** sample points up each wall. For every step, is the wall facing the
  sun, and is the ray to the sun blocked? Reported by compass quarter (N, E, S,
  W) and by lower and upper half of the wall. Nothing in the first 4 m in front
  of a wall counts as a blocker: the LiDAR keeps the highest return per cell,
  so every roof spills about one cell past its walls, and read literally that
  sliver would shade the wall it belongs to.
- **Solar gain on walls:** clear-sky direct normal irradiance (Meinel, with
  Kasten and Young air mass) times the cosine of incidence. Direct beam only,
  no diffuse light, no reflections. An upper bound.

---

## What's in the viewer

- The four CEQR days, a time slider and **Play the day**.
- **Shade now** and **Sun hours** layers.
- Buildings coloured by **Wall sun** or **Solar gain**, or plain.
- Click a building for its height and the sun on each wall, in two seasons.
- **Compare 2017 vs today:** split view, one camera driving both sides, shared
  colour scale, draggable divider.
- The city around the frame in plain grey: every building within 200 m, and
  towers of 100 m or more out to 1.5 km, the ones whose shadows reach in. They
  cast shadows in the calculation but are not scored.
- A compass with the sun on it, and a 3D sun ray.
- A phone layout, in portrait and landscape.
- **Clean view for recording** (`R`).

---

## How to evaluate it

Do not trust it because it renders. Every claim below is checkable.

**Solar geometry.** `verify_solar.py` checks declination at both solstices and
both equinoxes, peak altitude against the analytical `90 - |lat - declination|`,
that the sun is due south at its highest, and that azimuth sweeps east to west.
For New York, 21 December peaks at 25.8 degrees against the analytical 25.79,
and 21 June at 72.7 against 72.67.

**Shadow casting.** `verify_geometry.py` builds a tower of known height, puts
the sun at a known altitude, measures the shadow and compares it to
`H / tan(altitude)`. Three heights, three sun angles, plus direction tests, an
overhead sun shading nothing and a below-horizon sun shading everything.

**Walls.** `verify_facade.py` checks an isolated tower whose walls must be lit
for every hour they face the sun, on a north-up grid and again on a grid turned
to the tower.

**Grid.** `verify_grid.py` checks cell positions round trip through EPSG:2263,
cells are 2 m on the ground, grid up points 28.9 degrees true, a block drawn in
state plane feet rasterises to the right cells, and the north-up images map
back to the right cells.

**Regression.** The rewritten shadow sweep gives the same answer as the
original, cell for cell, on every test angle. The original 700 m command still
runs, and should print the published numbers again.

**Units.** The most dangerous bug in this project is reading US survey feet as
metres, which makes every shadow 3.28 times too long. The run prints the tallest
object in both metres and feet. Midtown supertalls are 300 to 470 m. If it
prints 1,500 m, stop.

**GPU path.** `verify_torch.py` runs the CPU and GPU backends on the same input
and compares them cell by cell. The CPU path is the reference.

---

## Run it

```bash
python -m pipeline.verify_solar
python -m pipeline.verify_geometry
python -m pipeline.verify_facade
python -m pipeline.verify_grid

python -m pipeline.tile_info --laz data/nyc --preset central-park --resource data/nyc/park.geojson
python -m pipeline.run_nyc --preset central-park --laz data/nyc --footprints data/nyc/footprints_crop.geojson --resource data/nyc/park.geojson --save-rasters data/runs/cp_2017.npz
python -m pipeline.build_viewer
python -m pipeline.run_nyc --preset central-park --laz data/nyc --footprints data/nyc/footprints_crop.geojson --resource data/nyc/park.geojson --burn-footprints --save-rasters data/runs/cp_today.npz
python -m pipeline.build_viewer
python tools/compare_runs.py data/runs/cp_2017.npz data/runs/cp_today.npz
python serve.py
```

`build_viewer` names each build after its scenario, `shadow-twin-2017.html` or
`shadow-twin-today.html`. A whole-park build keeps its images in a folder next
to the page, `frames-2017/` or `frames-today/`, loaded as they are shown, so the
page itself stays small.

The viewer must be served over http. Opening the HTML file from disk breaks
Cesium's asset loading and the comparison view.

Data: see `tools/GET_THE_DATA.md`. No LiDAR data is redistributed in this
repository.

---

## What it does not do

- No temperature, no wind, no comfort index.
- No cloud. These are clear-sky geometric sun hours, so the real figure on any
  given day is lower.
- Trees in the LiDAR surface are treated as solid. Real canopy lets some light
  through, and in December the trees are bare, so tree shade is overstated,
  most of all in winter.
- Nothing beyond the 2 km buffer casts. Beyond the LiDAR tiles, buildings are
  flat-roofed prisms from their footprints, and trees there cast nothing.
- The 2017 survey is a snapshot. Newer buildings enter only through their
  footprint roof heights, and "today" only raises the 2017 surface, it never
  lowers it: a building demolished since 2017 still casts.

---

## Planned

- Mean radiant temperature, to move from "is it in the sun" to "how does it
  feel".

---

## Data

- NYC 2017 topobathymetric LiDAR, via NYC's orthoimagery finder.
- NYC Building Footprints, NYC Open Data.
- Central Park boundary polygon.

Methods, data sources and papers are credited in `CREDITS.md`.

## Licence

MIT, see `LICENSE`.
