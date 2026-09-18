# Making the viewer work with no internet

The page loads the Cesium map library. By default it pulls it from a CDN, which
means no connection, no map. Do this once and it never needs the network again.

**With a connection, once:**

```bash
cd heat-twin
npm install cesium@1.145.0
```

**Then copy the built library next to the viewer:**

Windows:
```
xcopy /E /I node_modules\cesium\Build\Cesium viewer\vendor\Cesium
```

Mac or Linux:
```bash
mkdir -p viewer/vendor && cp -r node_modules/cesium/Build/Cesium viewer/vendor/Cesium
```

You should end up with `viewer/vendor/Cesium/Cesium.js` and about 23 MB in that
folder. `python -m pipeline.build_viewer` will confirm it found it.

The page tries the local copy first and silently falls back to the CDN, so it
works either way. Open the browser console and it prints which one it used.

---

## Two things that still want the network

**Map tiles.** The OpenStreetMap basemap under the buildings streams from the
internet. With no connection you get the dark globe, your buildings and your
shadow overlay, which is enough to check a render and honestly looks cleaner for
recording anyway.

**Nothing else.** The rasters are embedded in the HTML as data URIs. The solver
never touches the network at all.

---

## Before you publish

Do not upload `viewer/vendor/` to Cloudflare Pages. It is 23 MB of library that
the CDN serves faster. Either delete the vendor folder before deploying, or add
it to `.gitignore`, which it already is.
