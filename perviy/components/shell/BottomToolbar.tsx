"use client";

import { formatAltitude, formatExaggeration, formatLatitude, formatLongitude } from "@/geo/coordinates";
import type { SceneController } from "@/scene/controller";
import type { CameraMode } from "@/scene/types";
import { useCamera, useScene } from "@/scene/use-scene";

const MODES: Array<{ id: CameraMode; label: string }> = [
  { id: "free", label: "Free" },
  { id: "globe", label: "Globe" },
  { id: "orbit", label: "Orbit" },
  { id: "top-down", label: "Top-down" },
];

export function BottomToolbar({ controller }: { controller: SceneController }) {
  const state = useScene(controller);
  const camera = useCamera(controller);

  return (
    <footer className="earth-bottom">
      <div className="earth-modes" role="toolbar" aria-label="Camera mode">
        {MODES.map((mode) => (
          <button
            key={mode.id}
            type="button"
            className="earth-mode"
            aria-pressed={state.cameraMode === mode.id}
            onClick={() => controller.setCameraMode(mode.id)}
          >
            {mode.label}
          </button>
        ))}
        <button
          type="button"
          className="earth-mode"
          aria-pressed={state.cameraMode === "fly-to"}
          onClick={() => controller.repeatFlyTo()}
        >
          Fly-to
        </button>
        <button
          type="button"
          className="earth-mode"
          aria-pressed={state.scene2d}
          onClick={() => controller.setScene2d(!state.scene2d)}
        >
          2D
        </button>
      </div>
      <div className="earth-readout">
        {camera ? (
          <>
            <span>{formatLongitude(camera.longitude)}</span>
            <span>{formatLatitude(camera.latitude)}</span>
            <span>{formatAltitude(camera.height)}</span>
          </>
        ) : (
          <span>Camera settling</span>
        )}
        {state.terrainExaggeration !== 1 ? (
          <span className="earth-exag">
            {formatExaggeration(state.terrainExaggeration)} terrain
          </span>
        ) : null}
      </div>
      <button type="button" className="earth-reset" onClick={() => controller.resetCamera()}>
        Reset
      </button>
      <div className="earth-status" role="status">
        <span className={`earth-dot is-${state.statusLevel}`} />
        <span>{state.statusText}</span>
      </div>
    </footer>
  );
}
