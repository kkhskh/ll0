"use client";

import { formatExaggeration } from "@/geo/coordinates";
import type { SceneController } from "@/scene/controller";
import type { ImageryStyle } from "@/scene/types";
import { useScene } from "@/scene/use-scene";

const STYLES: Array<{ id: ImageryStyle; label: string; detail: string }> = [
  { id: "satellite", label: "Satellite", detail: "Aerial imagery" },
  { id: "geographic", label: "Geographic", detail: "Roads and place names" },
  { id: "dark", label: "Dark", detail: "Analytical basemap" },
  { id: "terrain", label: "Terrain", detail: "Shaded relief and contours" },
];

export function WorldLayers({ controller }: { controller: SceneController }) {
  const state = useScene(controller);

  return (
    <aside className={state.layersCollapsed ? "earth-side is-collapsed" : "earth-side"}>
      <div className="earth-side-inner">
        <div className="earth-panel-head">
          {state.layersCollapsed ? null : <h2>World layers</h2>}
          <button
            type="button"
            className="earth-icon-button"
            aria-label={state.layersCollapsed ? "Show world layers" : "Collapse world layers"}
            onClick={() => controller.toggleLayers()}
          >
            {state.layersCollapsed ? "›" : "‹"}
          </button>
        </div>
        {state.layersCollapsed ? null : (
          <>
            <section className="earth-section" role="radiogroup" aria-label="Imagery style">
              {STYLES.map((style) => (
                <button
                  key={style.id}
                  type="button"
                  className="earth-radio"
                  role="radio"
                  aria-checked={state.imageryStyle === style.id}
                  onClick={() => controller.setImageryStyle(style.id)}
                >
                  <span className="earth-mark" />
                  <span>
                    <b>{style.label}</b>
                    <small>{style.detail}</small>
                  </span>
                </button>
              ))}
            </section>
            <section className="earth-section">
              <button
                type="button"
                className="earth-toggle"
                aria-pressed={state.terrainEnabled}
                onClick={() => controller.setTerrainEnabled(!state.terrainEnabled)}
              >
                <span className="earth-switch" />
                <span>
                  <b>Terrain</b>
                  <small>{state.terrainEnabled ? "Elevation on" : "Ellipsoid only"}</small>
                </span>
              </button>
              <div className="earth-slider">
                <label htmlFor="terrain-exaggeration">
                  <span>Exaggeration</span>
                  <span>{formatExaggeration(state.terrainExaggeration)}</span>
                </label>
                <input
                  id="terrain-exaggeration"
                  type="range"
                  min={1}
                  max={3}
                  step={0.25}
                  value={state.terrainExaggeration}
                  onChange={(event) =>
                    controller.setTerrainExaggeration(Number(event.target.value))
                  }
                />
              </div>
              {state.terrainExaggeration !== 1 ? (
                <div className="earth-badge">
                  Terrain exaggeration {formatExaggeration(state.terrainExaggeration)}. Heights
                  on the globe are scaled for viewing.
                </div>
              ) : null}
              {state.terrainDetailLimited ? (
                <p className="earth-note">
                  Terrain detail is limited until a Cesium ion token is configured.
                </p>
              ) : null}
            </section>
            <p className="earth-footnote">Imagery and terrain only. No hazard models in this view.</p>
          </>
        )}
      </div>
    </aside>
  );
}
