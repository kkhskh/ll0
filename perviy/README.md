# HEPHAELION EARTH

The `perviy` app is the Hephaelion Earth foundation: a branded 3D globe you can rotate, search, and restyle. It is not a forecast, and it does not call the wildfire, flood, hail, typhoon, tsunami, or volcanic packages in this repository.

## Run

From `perviy/`:

```bash
npm install
npm run dev
```

Open http://localhost:3000. `npm run dev` and `npm run build` copy Cesium Workers, Assets, Widgets, and ThirdParty into `public/cesium` first. That folder is generated. Do not commit it.

```bash
npm run build
npm start
```

## Environment

Copy `.env.example` to `.env.local` if you have a Cesium ion token:

```bash
NEXT_PUBLIC_CESIUM_ION_TOKEN=
```

Leave it empty to open the globe immediately. Without a token the terrain is the WGS84 ellipsoid, imagery comes from Esri's public map servers, and the sidebar says terrain detail is limited. With a token, satellite imagery and world terrain are requested from Cesium ion, and search uses the Ion geocoder. The token is a `NEXT_PUBLIC_` value, so it is visible in the browser. Do not commit a real token.

Search without a token goes through `/api/geocode`, which calls Nominatim with the user agent `HephaelionEarth/0.1` and a one-request-per-second pause. The search box waits 420ms after typing stops.

## What this foundation includes

- One CesiumJS globe in a custom shell. Cesium's toolbar, animation, timeline, and base-layer picker are off.
- Streamed imagery styles on that same globe: satellite, geographic, dark analytical, and terrain (shaded relief and contours).
- Optional Cesium World Terrain when a token is set, with a vertical exaggeration control that is marked whenever it is not 1×.
- Camera: drag to rotate or pan, scroll to zoom, Alt-drag or Ctrl-drag to tilt, double-click to fly toward the picked point, and a reset control. Zoom speed tightens as the camera gets closer to the surface. Modes that work here are Free (the default tilted view), Globe, Orbit around the last focus, Top-down, Fly-to from search, and a 2D overhead toggle.
- Click the surface to read longitude, latitude, and sampled elevation, or a clear unavailable state.
- Both sidebars collapse. Tile, terrain, and geocoder failures stay in the status line. If WebGL is missing, the canvas shows a readable message.
- Analytical display mode turns atmosphere and fog off. Cinematic keeps Cesium's standard atmosphere.

The React shell, the scene controller, the Cesium adapter, and the coordinate formatters are separate modules.

## Intentionally not built

Water shaders, buildings, 3D Tiles cities, measurement, drawing, saved workspaces, time animation, follow mode, and cinematic camera paths. Those should attach to the scene controller and `cesium/adapter.ts` later, not to a second app.

## Imagery and terrain licenses

Usage of these services is not automatically free for a commercial product. Read the provider terms before shipping this beyond local development.

- **CesiumJS** is Apache-2.0. **Cesium ion** World Imagery (asset 2) and World Terrain are available only with an ion account. The default evaluation token inside the Cesium library is not used. Ion assets have their own terms, including attribution that this app keeps in the credit line.
- **Esri** World Imagery, World Street Map, World Dark Gray Canvas (base and reference), and World Topo Map are requested from `server.arcgisonline.com` without an API key. Esri's terms restrict how those services can be used in a product. They are the no-key fallback here, not a grant to embed them in a commercial service. World Hillshade is not used: it is a near-white overlay and disappears on this dark globe. World Physical Map stops at level 8, so the terrain style uses World Topo Map, which keeps shaded relief when the camera moves from orbit down to a city.
- **CARTO** basemaps now return an API-key placeholder, so they are not wired up. A future dark or light style that needs a CARTO key should fall back to the Esri canvas styles.
- **OpenStreetMap** Nominatim is used only as the no-token gazetteer, under the [Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/) (identify the application, at most one request per second). The public OSM tile servers are not used for imagery; they are a volunteer service with a strict tile policy.
- **HK Grotesk** is used under the SIL Open Font License. See `public/fonts/OFL.txt`.
