"""
Solve shadows over Manhattan from NYC LiDAR and build the viewer payload.

    python -m pipeline.run_nyc --dsm data/nyc --span 2400 --res 2.0
    python -m pipeline.build_viewer

Defaults are aimed at Central Park South, because the long winter shadows the
towers on 57th Street throw across the park are the reason anyone outside New
York has ever heard of a shadow study.

Two days are solved by default, 21 December and 21 June, so the viewer can put
them side by side. That contrast is the story. A shadow map of one day is a
picture; the difference between the shortest and longest day of the year is an
argument.

Heavy runs belong on Kaggle, not your laptop. See notebooks/kaggle_render.py.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from . import export, nyc
from .geometry import run_day, sky_view_factor
from .solar import local_day_positions

# Central Park South, between the park and Billionaires' Row.
DEFAULT_LAT = 40.7670
DEFAULT_LON = -73.9770

# New York runs on EST (UTC-5) in December and EDT (UTC-4) in June.
DAYS = [
    {
        "id": "dec",
        "label": "21 December",
        "short": "Winter",
        "date": "2026-12-21",
        "utc_offset": -5.0,
        "hours": (8, 16),
    },
    {
        "id": "jun",
        "label": "21 June",
        "short": "Summer",
        "date": "2026-06-21",
        "utc_offset": -4.0,
        "hours": (6, 19),
    },
]

EDGE_MARGIN_M = 80.0
SVF_AZIMUTHS = 16
SVF_RADIUS_M = 250.0


def build_bundle(prep, days_solved, svf, args, log):
    """Assemble the viewer payload from solved days."""
    cellsize = prep["cellsize"]
    built = prep["built"]
    n_rows, n_cols = prep["surface"].shape

    m = max(1, int(EDGE_MARGIN_M / cellsize))
    edge = np.ones((n_rows, n_cols), dtype=bool)
    edge[m:-m, m:-m] = False
    street_mask = built | edge
    open_street = ~street_mask
    log(f"reporting over {int(open_street.sum()):,} open street cells")

    days_out = []
    for spec, res in days_solved:
        hour_frames = []
        for i, lab in enumerate(res.hours):
            hour_frames.append(
                {
                    "hour": lab,
                    "altitude": round(res.altitudes[i], 2),
                    "azimuth": round(res.azimuths[i], 2),
                    "sunlit_fraction": float(res.sunlit[i][open_street].mean()),
                    "png": export.to_data_uri(
                        export.quantize(
                            res.sunlit[i],
                            [0.0, 0.5, 1.01],
                            export.SHADOW_COLORS,
                            alpha=205,
                            mask=edge,
                        )[0]
                    ),
                }
            )

        n_h = float(len(res.hours))
        step = n_h / 8.0
        breaks = [round(i * step, 3) for i in range(8)] + [n_h + 0.01]
        sun_layer = export.layer(
            f"sunhours_{spec['id']}",
            "Hours of direct sun",
            f"hours on {spec['label']}",
            res.sun_hours,
            breaks,
            export.SUNHOUR_COLORS,
            mask=edge,
            stats_mask=street_mask,
        )

        sh = res.sun_hours[open_street]
        days_out.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "short": spec["short"],
                "date": spec["date"],
                "utc_offset": spec["utc_offset"],
                "daylight_hours_modelled": len(res.hours),
                "peak_altitude": round(max(res.altitudes), 2),
                "hours": hour_frames,
                "sunhours": sun_layer,
                "stats": {
                    "median_sun_hours": float(np.median(sh)),
                    "mean_sun_hours": float(sh.mean()),
                    "frac_street_no_sun": float((sh < 0.5).mean()),
                    "frac_street_under_2h": float((sh < 2).mean()),
                    "frac_street_over_half_day": float((sh > n_h / 2).mean()),
                },
            }
        )
        log(
            f"  {spec['label']}: median {np.median(sh):.1f} h of sun, "
            f"{(sh < 0.5).mean()*100:.1f}% of street never sees the sun"
        )

    svf_layer = None
    if svf is not None:
        svf_layer = export.layer(
            "svf",
            "Sky view factor",
            "fraction of sky visible",
            svf,
            [0.0, 0.2, 0.32, 0.44, 0.56, 0.68, 0.8, 0.92, 1.0001],
            export.SVF_COLORS,
            mask=edge,
            stats_mask=street_mask,
        )

    heights = prep["heights"]
    b = prep["bounds"]
    return {
        "mode": "REAL",
        "warning": "",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "site": {
            "name": args.site_name,
            "centre_lat": args.lat,
            "centre_lon": args.lon,
            "bounds": {"west": b[0], "south": b[1], "east": b[2], "north": b[3]},
            "span_m": round(max(n_rows, n_cols) * cellsize),
            "cellsize_m": round(cellsize, 3),
            "grid": [n_rows, n_cols],
            "edge_margin_m": EDGE_MARGIN_M,
            "tallest_m": round(float(heights.max()), 1),
            "tallest_ft": round(float(heights.max()) * 3.28084),
        },
        "model": {
            "solar_algorithm": "NOAA / Meeus low precision",
            "shadow_method": "Ratti and Richens shift-and-subtract sweep, whole-cell stepping",
            "svf_method": (
                f"horizon scan, {SVF_AZIMUTHS} azimuths, {SVF_RADIUS_M:.0f} m radius, "
                "Steyn cos^2 weighting"
            )
            if svf is not None
            else None,
            "surface_source": prep.get(
                "source_desc",
                "NYC 2017 airborne LiDAR, 1 ft highest-hit digital surface model, "
                "Leica ALS80 at 8+ pulses per square metre, captured 3 to 17 May 2017",
            ),
            "ground_source": prep["ground_source"],
            "native_resolution_m": round(prep["native_res_m"], 3),
            "resample": (
                f"max pooling by {prep['resample_factor']} to {cellsize:.2f} m"
                if prep["resample_factor"] > 1
                else "none, solved at native resolution"
            ),
            "computes": [
                "direct sun or shade",
                "hours of direct sun",
            ]
            + (["sky view factor"] if svf is not None else []),
            "does_not_compute": [
                "air temperature",
                "surface temperature",
                "mean radiant temperature",
                "wind",
                "thermal comfort indices such as UTCI or PET",
                "cloud",
            ],
        },
        "days": days_out,
        "svf": svf_layer,
        "buildings": [],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--laz", default=None,
                    help="folder of .laz or .las point cloud tiles, which is what "
                         "the NYC downloader actually gives you. Use this or --dsm.")
    ap.add_argument("--dsm", default=None,
                    help="folder of hh_*.tif surface rasters, if you have them")
    ap.add_argument("--dem", default=None,
                    help="optional folder of be_*.tif bare earth tiles")
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--span", type=float, default=2400.0, help="metres a side")
    ap.add_argument("--res", type=float, default=2.0, help="target metres per cell")
    ap.add_argument("--site-name", default="Central Park South, Manhattan")
    ap.add_argument("--days", default="dec,jun",
                    help="comma separated day ids from dec,jun")
    ap.add_argument("--skip-svf", action="store_true",
                    help="skip sky view factor, which is the slow part and is "
                         "not what this post is about")
    ap.add_argument("--max-buildings", type=int, default=2500)
    ap.add_argument("--gpu", action="store_true",
                    help="use the torch backend. Only worth it on Kaggle or Modal; "
                         "your laptop GPU does not have the memory for this.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    try:
        import rasterio  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"pip install rasterio scipy shapely\n({e})")

    if not args.laz and not args.dsm:
        raise SystemExit("give me --laz (point clouds) or --dsm (rasters)")

    if args.laz:
        from . import laz
        log("rasterising point cloud tiles")
        prep = laz.rasterise(
            args.laz, args.lat, args.lon, args.span, args.res, log=log
        )
    else:
        log("preparing NYC rasters")
        prep = nyc.prepare(
            args.dsm, args.lat, args.lon, args.span, args.res,
            dem_path=args.dem, log=log,
        )

    wanted = [d for d in DAYS if d["id"] in args.days.split(",")]
    if not wanted:
        raise SystemExit(f"no known day ids in {args.days!r}, pick from dec,jun")

    use_gpu = False
    if args.gpu:
        from .geometry_torch import available, device_name, run_day_gpu
        use_gpu = available()
        log(f"gpu requested: {device_name()}"
            + ("" if use_gpu else "  falling back to CPU"))

    solved = []
    for spec in wanted:
        h0, h1 = spec["hours"]
        positions = local_day_positions(
            spec["date"], args.lat, args.lon, spec["utc_offset"],
            tuple(range(h0, h1 + 1)),
        )
        labels = [f"{h:02d}:00" for h in range(h0, h1 + 1)]
        lit = [(p, l) for p, l in zip(positions, labels) if p.above_horizon]
        log(f"{spec['label']}: {len(lit)} daylight hours, "
            f"peak sun {max(p.altitude for p, _ in lit):.1f} degrees")
        if use_gpu:
            res = run_day_gpu(
                prep["surface"], prep["cellsize"],
                [p for p, _ in lit], [l for _, l in lit],
                skip_svf=True, progress=None,
            )
        else:
            res = run_day(
                prep["surface"], prep["cellsize"],
                [p for p, _ in lit], [l for _, l in lit],
                n_azimuths=SVF_AZIMUTHS, svf_radius_m=SVF_RADIUS_M,
                progress=None,
            )
        solved.append((spec, res))
        log(f"{spec['label']} solved")

    svf = None
    if not args.skip_svf:
        log(f"sky view factor, {SVF_AZIMUTHS} azimuths, {SVF_RADIUS_M:.0f} m")
        if use_gpu:
            import torch
            from .geometry_torch import sky_view_factor_gpu
            t = torch.from_numpy(prep["surface"]).to("cuda")
            svf = sky_view_factor_gpu(
                t, prep["cellsize"], SVF_AZIMUTHS, SVF_RADIUS_M).cpu().numpy()
            del t; torch.cuda.empty_cache()
        else:
            svf = sky_view_factor(
                prep["surface"], prep["cellsize"], SVF_AZIMUTHS, SVF_RADIUS_M
            )

    log("assembling bundle")
    bundle = build_bundle(prep, solved, svf, args, log)

    log("vectorising footprints for the 3D view")
    from .footprints import from_mask
    bundle["buildings"] = from_mask(
        prep["built"], prep["heights"], prep["transform"], prep["crs"],
        prep["resample_factor"], max_features=args.max_buildings, log=log,
    )
    log(f"{len(bundle['buildings'])} footprints")

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = args.out or os.path.join(here, "viewer", "data.json")
    size = export.write_bundle(path, bundle)
    log(f"wrote {path}  {size/1024/1024:.1f} MB")

    print()
    for d in bundle["days"]:
        s = d["stats"]
        print(f"  {d['label']:<14} median {s['median_sun_hours']:.1f} h of sun   "
              f"{s['frac_street_no_sun']*100:5.1f}% of street never sees the sun")
    print(f"\n  tallest object in frame: {bundle['site']['tallest_m']:.0f} m "
          f"({bundle['site']['tallest_ft']:.0f} ft)")
    print("\n  next: python -m pipeline.build_viewer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
