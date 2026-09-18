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
3. Upload your tiles:  Add Data -> Upload -> New Dataset
       put the hh_*.tif and be_*.tif files in
       it lands at /kaggle/input/<your-dataset-name>/
4. Upload this repo the same way, or clone it if it is on GitHub.
5. Paste the cells below, in order, one per Kaggle cell.
6. When it finishes, download viewer/shadow-twin.html from the output panel.

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

# ===================== CELL 3 : small run first =====================
# Prove the tiles are right before spending time on a big render.
CELL_3 = r'''
!python -m pipeline.tile_info --laz {DATA} --lat 40.7655 --lon -73.9800 --span 1600
!python -m pipeline.run_nyc --laz {DATA} --lat 40.7655 --lon -73.9800 --span 1600 --res 3.0 --skip-svf
'''

# ===================== CELL 4 : the real render =====================
# Only after cell 3 reported a sensible "tallest object" figure.
CELL_4 = r'''
!python -m pipeline.run_nyc --laz {DATA} --lat 40.7655 --lon -73.9800 --span 2400 --res 1.0 --gpu --site-name "Central Park South, Manhattan"
!python -m pipeline.build_viewer
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

# ===================== CELL 6 : get the file out =====================
CELL_6 = r'''
import shutil, os
shutil.copy("viewer/shadow-twin.html", "/kaggle/working/shadow-twin.html")
print("size:", os.path.getsize("/kaggle/working/shadow-twin.html") / 1e6, "MB")
# It now appears in the Output panel on the right. Download it from there.
'''

if __name__ == "__main__":
    for i, c in enumerate(
        [CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6], start=1
    ):
        print(f"\n{'='*70}\nCELL {i}\n{'='*70}{c}")
