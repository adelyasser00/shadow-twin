# Shadow twin

A shadow study of Central Park South, Manhattan, computed from the city's own
LiDAR survey and run with New York City's official shadow assessment method.
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

## The result

Frame: 700 m across, 2 m cells, centred on the southern end of Central Park.
"Today" means the 2017 LiDAR surface raised to current recorded roof heights
from NYC Building Footprints.

Share of the park inside the frame that never gets direct sun during the CEQR
window:

| CEQR day | 2017 survey | Today |
|---|---|---|
| 21 December | 46.9% | 55.3% |
| 21 March / 21 September | 10.5% | 12.0% |
| 6 May / 6 August | 2.8% | 3.1% |
| 21 June | 1.3% | 1.4% |

The difference is almost all in winter. The sun peaks at 25.8 degrees on
21 December and 72.7 degrees on 21 June, so tall buildings cast their longest
shadows exactly when the park has the least sun to lose.

Sanity check: the today run reports a tallest object of 472.4 m (1,550 ft),
Central Park Tower's height. It was finished after the 2017 survey, so this
confirms the footprint heights are being applied.

**Read these numbers with their limits:** a 700 m frame, not the whole park;
"today" is a raised 2017 surface, not a new survey; direct sun under clear sky,
geometry only.

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
  return. Cells more than 6 m from any return are marked no-data and left blank,
  never filled.
- **Buildings:** NYC Building Footprints, one polygon per real building, roof
  height from `height_roof`.
- **Shadows:** shift-and-subtract shadow sweep (Ratti and Richens).
- **Walls:** sample points up each wall. For every step, is the wall facing the
  sun, and is the ray to the sun blocked? Reported by compass quarter (N, E, S,
  W) and by lower and upper half of the wall.
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

**Walls.** `verify_facade.py` checks wall orientation and blocking against
hand-derived cases.

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

python -m pipeline.run_nyc --help
python -m pipeline.build_viewer
python serve.py
```

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
  through, so tree shade is overstated.
- Buildings outside the frame cast nothing. Winter shadows from the tallest
  towers here can reach about 2 km.
- The 2017 survey is a snapshot. Newer buildings enter only through their
  footprint roof heights.

---

## Planned

- A larger frame, with shadows cast from buildings outside it.
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
