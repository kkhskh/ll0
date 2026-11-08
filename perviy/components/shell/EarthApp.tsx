"use client";

import { useState } from "react";
import { EarthCanvas } from "@/components/viewport/EarthCanvas";
import { BottomToolbar } from "@/components/shell/BottomToolbar";
import { Inspector } from "@/components/shell/Inspector";
import { TopNav } from "@/components/shell/TopNav";
import { WorldLayers } from "@/components/shell/WorldLayers";
import { formatExaggeration } from "@/geo/coordinates";
import { createSceneController, type SceneController } from "@/scene/controller";
import { readIonToken } from "@/scene/ion-token";
import { useScene } from "@/scene/use-scene";

export function EarthApp() {
  const [controller] = useState<SceneController>(() =>
    createSceneController({ ionConfigured: Boolean(readIonToken()) }),
  );
  const state = useScene(controller);
  const layersWidth = state.layersCollapsed ? "44px" : "272px";
  const inspectorWidth =
    state.selection === null ? "0px" : state.inspectorCollapsed ? "44px" : "300px";

  return (
    <div
      className="earth-shell"
      style={{ gridTemplateColumns: `${layersWidth} minmax(0, 1fr) ${inspectorWidth}` }}
    >
      <TopNav controller={controller} />
      <WorldLayers controller={controller} />
      <div className="earth-stage">
        <EarthCanvas controller={controller} />
        {state.terrainExaggeration !== 1 ? (
          <div className="earth-chip">
            Terrain {formatExaggeration(state.terrainExaggeration)}
          </div>
        ) : null}
      </div>
      <Inspector controller={controller} />
      <BottomToolbar controller={controller} />
    </div>
  );
}
