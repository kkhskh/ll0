import {
  EllipsoidTerrainProvider,
  Ion,
  Terrain,
  type TerrainProvider,
  type Viewer,
} from "cesium";
import { messageFromUnknown } from "@/cesium/events";
import type { SceneController } from "@/scene/controller";

export function terrainIsEllipsoid(provider: TerrainProvider): boolean {
  return provider instanceof EllipsoidTerrainProvider;
}

export function applyTerrain(options: {
  viewer: Viewer;
  enabled: boolean;
  ionToken: string | undefined;
  controller: SceneController;
  replaceSubscriptions: (unsubscribers: Array<() => void>) => void;
}): void {
  const { viewer, enabled, ionToken, controller, replaceSubscriptions } = options;
  const scene = viewer.scene;

  const showEllipsoid = () => {
    replaceSubscriptions([]);
    scene.setTerrain(new Terrain(Promise.resolve(new EllipsoidTerrainProvider())));
    scene.globe.depthTestAgainstTerrain = false;
  };

  if (!enabled) {
    showEllipsoid();
    return;
  }

  if (!ionToken) {
    showEllipsoid();
    controller.publish({ terrainDetailLimited: true });
    return;
  }

  Ion.defaultAccessToken = ionToken;
  const terrain = Terrain.fromWorldTerrain({ requestVertexNormals: true });
  const unsubscribers = [
    terrain.readyEvent.addEventListener(() => {
      scene.globe.depthTestAgainstTerrain = true;
      controller.publish({ terrainDetailLimited: false });
      controller.setStatus("Terrain ready", "ready");
    }),
    terrain.errorEvent.addEventListener((error: Error) => {
      showEllipsoid();
      controller.publish({ terrainDetailLimited: true });
      controller.setStatus(
        `Terrain failed to load. Showing the ellipsoid. ${messageFromUnknown(error)}`,
        "error",
      );
    }),
  ];
  replaceSubscriptions(unsubscribers);
  scene.setTerrain(terrain);
}
