# Data, methods and libraries

Built by Adel Yasser.

## Data

**NYC 2017 Topobathymetric LiDAR.** Collected for New York City in 2017 using a
Leica ALS80 sensor at 8 or more pulses per square metre. Published by the City
of New York and distributed through the NYC Imagery and LiDAR Downloader
(`finder.nyc.gov/orthoimagery`) and NYS ITS Geospatial Services.

This project reads the classified point clouds directly. No derived product from
any third party is redistributed here, and no LiDAR data is included in this
repository.

**NYC Building Footprints.** NYC Open Data. Used for one polygon per building
and the current recorded roof height (`height_roof`).

**Central Park boundary.** Polygon used to report park figures separately from
streets.

**Basemap tiles.** OpenStreetMap contributors, via the standard tile server.
Rendered in the viewer under the OpenStreetMap licence.

## Methods

**Analysis days and window.** NYC CEQR Technical Manual, Chapter 8 (Shadows):
21 December, 21 March, 6 May and 21 June, from 1.5 hours after sunrise to
1.5 hours before sunset, standard time.

**Solar position.** NOAA solar calculator formulation, following the low
precision algorithms in Jean Meeus, *Astronomical Algorithms*. Implemented in
`pipeline/solar.py` and checked against solstice and equinox geometry that can
be derived by hand.

**Shadow casting.** Shift-and-subtract sweep in the manner of Ratti and Richens,
*Urban texture analysis with image processing techniques* (1999). The same
approach underlies the shadow modelling in UMEP and SOLWEIG.

**Sky view factor.** Horizon scanning with the discretised cosine-squared
weighting of Steyn, *The calculation of view factors from fisheye lens
photographs* (1980).

**Solar gain on walls.** Clear-sky direct normal irradiance after Meinel, with
the Kasten and Young air mass formula, times the cosine of incidence. Direct
beam only.

This project implements the geometric stage only. It is not SOLWEIG, it does not
compute a radiation budget, and it produces no mean radiant temperature or
thermal comfort index. If you need those, use UMEP.

## Libraries

CesiumJS for the 3D viewer. laspy and lazrs for point cloud reading. rasterio,
shapely, scipy, numpy and Pillow for the pipeline. PyTorch for the optional GPU
backend.
