"""
Kaggle cell script for the full resolution render.

Kaggle gives you a free T4 with 16 GB and about 30 hours a week. That is far
more than this needs and far more than your laptop has.

HOW TO USE
----------
1. kaggle.com  ->  Create  ->  New Notebook
2. Settings panel on the right:
       Accelerator  ->  GPU T4 x2   (one is enough, it takes what it needs)
       Internet     ->  On          (only to pip install, turn it off after)
3. Upload your data:  Add Data -> Upload -> New Dataset
       put the .laz tiles, footprints.geojson and park.geojson in
       it lands at /kaggle/input/<your-dataset-name>/
4. Upload this repo the same way, or clone it if it is on GitHub.
5. Paste the cells below, in order, one per Kaggle cell.
6. When it finishes, download shadow-twin-build.zip from the output panel.

Each block below is marked as its own cell. Keep them separate so a tweak to
the plot does not re-run the solver.
"""

# ===================== CELL 1 : setup =====================
CELL_1 = r'''
!pip install -q rasterio shapely laspy lazrs
import os, sys
# Adjust if you named things differently.
REPO = "/kaggle/input/shadow-twin-repo/heat-twin"
DATA = "/kaggle/input/nyc-lidar-central-park"   # folder holding your .laz tiles
WORK = "/kaggle/working/heat-twin"

!cp -r {REPO} /kaggle/working/ 2>/dev/null || echo "copy the repo manually"
sys.path.insert(0, WORK)
os.chdir(WORK)

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("tiles:", os.listdir(DATA)[:10])
'''

# ===================== CELL 2 : check the backends agree =====================
# Do this once. If CPU and GPU disagree, stop and use the CPU path.
CELL_2 = r'''
!python -m pipeline.verify_solar | tail -2
!python -m pipeline.verify_geometry | tail -2
!python -m pipeline.verify_torch
'''

# ===================== CELL 3 : check coverage and the regression =====================
# Prove the tiles and the park file are right before spending time on a big
# render. The 700 m regression should print the published 46.9% for December.
CELL_3 = r'''
!python -m pipeline.nyc_footprints crop {DATA}/footprints.geojson /kaggle/working/footprints_crop.geojson --laz {DATA}
!python -m pipeline.tile_info --laz {DATA} --preset central-park --resource {DATA}/park.geojson
!python -m pipeline.run_nyc --laz {DATA} --footprints /kaggle/working/footprints_crop.geojson --resource {DATA}/park.geojson --resource-name "Central Park" --lat 40.7668 --lon -73.9790 --span 700 --skip-svf --skip-facades --out /kaggle/working/check.json
'''

# ===================== CELL 4 : the real render =====================
# Only after cell 3 reported a sensible tallest object and the regression matched.
# The first run grids the LiDAR into data/cache, the second reuses it.
CELL_4 = r'''
FP = "/kaggle/working/footprints_crop.geojson"
!python -m pipeline.run_nyc --preset central-park --laz {DATA} --footprints {FP} --resource {DATA}/park.geojson --gpu --save-rasters data/runs/cp_2017.npz
!python -m pipeline.build_viewer
!python -m pipeline.run_nyc --preset central-park --laz {DATA} --footprints {FP} --resource {DATA}/park.geojson --gpu --burn-footprints --save-rasters data/runs/cp_today.npz
!python -m pipeline.build_viewer
!python tools/compare_runs.py data/runs/cp_2017.npz data/runs/cp_today.npz
'''

# ===================== CELL 5 : sanity picture =====================
# A quick look without needing Cesium, so you know the render is good
# before you download 8 MB of HTML.
CELL_5 = r'''
import json, base64, io
from PIL import Image
import matplotlib.pyplot as plt

d = json.load(open("viewer/data.json"))
dec = [x for x in d["days"] if x["id"] == "dec"][0]
frames = [1, len(dec["hours"]) // 2, len(dec["hours"]) - 2]

fig, ax = plt.subplots(1, len(frames) + 1, figsize=(5 * (len(frames) + 1), 5))
for i, f in enumerate(frames):
    h = dec["hours"][f]
    img = Image.open(io.BytesIO(base64.b64decode(h["png"].split(",")[1])))
    ax[i].imshow(img); ax[i].set_title(f"{dec['label']} {h['hour']}\\nsun {h['altitude']:.0f} deg")
    ax[i].axis("off")
img = Image.open(io.BytesIO(base64.b64decode(dec["sunhours"]["png"].split(",")[1])))
ax[-1].imshow(img); ax[-1].set_title("hours of direct sun"); ax[-1].axis("off")
plt.tight_layout(); plt.show()

for day in d["days"]:
    s = day["stats"]
    print(f"{day['label']:<14} median {s['median_sun_hours']:.1f} h   "
          f"{s['frac_street_no_sun']*100:.1f}% of street never sees the sun")
print("tallest in frame:", d["site"]["tallest_m"], "m")
'''

# ===================== CELL 6 : get the files out =====================
# A whole-park build is a small page plus a folder of images per scenario.
CELL_6 = r'''
import shutil
shutil.make_archive("/kaggle/working/shadow-twin-build", "zip", "viewer",
                    ".")
print("download shadow-twin-build.zip from the Output panel, unzip it into viewer/")
'''

if __name__ == "__main__":
    for i, c in enumerate(
        [CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6], start=1
    ):
        print(f"\n{'='*70}\nCELL {i}\n{'='*70}{c}")
