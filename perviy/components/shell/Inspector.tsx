"use client";

import { formatElevation, formatExaggeration, formatLatitude, formatLongitude } from "@/geo/coordinates";
import type { SceneController } from "@/scene/controller";
import type { Selection } from "@/scene/types";
import { useScene } from "@/scene/use-scene";

export function Inspector({ controller }: { controller: SceneController }) {
  const state = useScene(controller);
  const selection = state.selection;
  if (!selection) return null;

  return (
    <aside className={state.inspectorCollapsed ? "earth-inspector is-collapsed" : "earth-inspector"}>
      <div className="earth-inspector-inner">
        <div className="earth-panel-head">
          {state.inspectorCollapsed ? null : <h2>Inspector</h2>}
          <span>
            <button
              type="button"
              className="earth-icon-button"
              aria-label={state.inspectorCollapsed ? "Show inspector" : "Collapse inspector"}
              onClick={() => controller.toggleInspector()}
            >
              {state.inspectorCollapsed ? "‹" : "›"}
            </button>
            {state.inspectorCollapsed ? null : (
              <button
                type="button"
                className="earth-icon-button"
                aria-label="Close inspector"
                onClick={() => controller.setSelection(null)}
              >
                ×
              </button>
            )}
          </span>
        </div>
        {state.inspectorCollapsed ? null : (
          <dl className="earth-facts">
            <div>
              <dt>Longitude</dt>
              <dd>{formatLongitude(selection.longitude)}</dd>
            </div>
            <div>
              <dt>Latitude</dt>
              <dd>{formatLatitude(selection.latitude)}</dd>
            </div>
            <div>
              <dt>Elevation</dt>
              <dd>
                <Elevation selection={selection} exaggeration={state.terrainExaggeration} />
              </dd>
            </div>
          </dl>
        )}
      </div>
    </aside>
  );
}

function Elevation({
  selection,
  exaggeration,
}: {
  selection: Selection;
  exaggeration: number;
}) {
  if (selection.elevationState === "pending") return <>Sampling terrain…</>;
  if (selection.elevationState === "sampled" && selection.elevationMeters !== null) {
    return (
      <>
        {formatElevation(selection.elevationMeters)}
        {exaggeration !== 1 ? (
          <small>
            Sampled height, before the {formatExaggeration(exaggeration)} visual exaggeration.
          </small>
        ) : null}
      </>
    );
  }
  if (selection.elevationState === "disabled") {
    return (
      <>
        Unavailable
        <small>Turn terrain on to sample elevation.</small>
      </>
    );
  }
  if (selection.elevationState === "ellipsoid") {
    return (
      <>
        Unavailable
        <small>Terrain detail is limited until a Cesium ion token is configured.</small>
      </>
    );
  }
  return (
    <>
      Unavailable
      <small>The terrain provider did not return a height for this point.</small>
    </>
  );
}
