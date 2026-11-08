import {
  buildModuleUrl,
  Cartesian2,
  Cartographic,
  EllipsoidTerrainProvider,
  ImageryLayer,
  Math as CesiumMath,
  sampleTerrainMostDetailed,
  SceneMode,
  ScreenSpaceEventType,
  Viewer,
  type ImageryProvider,
  type ScreenSpaceEventHandler,
} from "cesium";
import { applyDisplayMode } from "@/cesium/display-mode";
import { messageFromUnknown, onEvent } from "@/cesium/events";
import { createImageryStack, createPublicImagery } from "@/cesium/imagery";
import { registerIonSearch } from "@/cesium/ion-search";
import {
  configureCameraControls,
  enterOrbit,
  exitOrbit,
  flyGlobe,
  flyHome,
  flyToPlace,
  flyTopDown,
  flyToward,
  focusUnderCamera,
  holdOverhead,
  pickGlobe,
  readCamera,
  showHome,
  syncNavigationSpeed,
  syncTiltEnabled,
} from "@/cesium/navigation";
import { applyTerrain, terrainIsEllipsoid } from "@/cesium/terrain";
import type { SceneController } from "@/scene/controller";
import { readIonToken } from "@/scene/ion-token";
import type { CameraMode, ImageryStyle, SceneCommand } from "@/scene/types";

export interface EarthSession {
  destroy: () => void;
}

type ClickEvent = ScreenSpaceEventHandler.PositionedEvent;

/**
 * Cesium rendering adapter.
 *
 * React never imports Cesium. It changes the scene controller, and this
 * module applies those commands to one Viewer. Cities, water, measurement,
 * drawing, saved workspaces, and time should attach here later.
 */
export function mountEarthViewer(options: {
  container: HTMLElement;
  creditContainer: HTMLElement;
  creditViewport: HTMLElement;
  controller: SceneController;
}): EarthSession {
  const { container, creditContainer, creditViewport, controller } = options;
  // The published buildModuleUrl type is only the URL function. The runtime
  // function also has setBaseUrl, which points workers at public/cesium.
  const moduleUrl = buildModuleUrl as typeof buildModuleUrl & {
    setBaseUrl?: (value: string) => void;
  };
  moduleUrl.setBaseUrl?.("/cesium/");
  window.CESIUM_BASE_URL = "/cesium/";

  const ionToken = readIonToken();
  const initial = controller.getState();
  let destroyed = false;
  let imageryGeneration = 0;
  let pickGeneration = 0;
  let imageryUnsubs: Array<() => void> = [];
  let terrainUnsubs: Array<() => void> = [];
  let pendingAfterMorph: (() => void) | null = null;
  let tileTimer = 0;
  let queuedTiles = 0;
  let sawImageryError = false;

  const openingProviders = createPublicImagery(initial.imageryStyle);
  const viewer = new Viewer(container, {
    animation: false,
    timeline: false,
    baseLayerPicker: false,
    geocoder: false,
    homeButton: false,
    sceneModePicker: false,
    navigationHelpButton: false,
    fullscreenButton: false,
    infoBox: false,
    selectionIndicator: false,
    vrButton: false,
    projectionPicker: false,
    shouldAnimate: false,
    showRenderLoopErrors: false,
    creditContainer,
    creditViewport,
    baseLayer: new ImageryLayer(openingProviders[0]),
    terrainProvider: new EllipsoidTerrainProvider(),
  });

  configureCameraControls(viewer);
  applyDisplayMode(viewer, initial.displayMode);
  viewer.scene.verticalExaggeration = initial.terrainExaggeration;
  viewer.scene.verticalExaggerationRelativeHeight = 0;
  viewer.camera.percentageChanged = 0.01;
  showHome(viewer);

  const unsubscribers: Array<() => void> = [];

  const replaceTerrainSubs = (next: Array<() => void>) => {
    for (const remove of terrainUnsubs) remove();
    terrainUnsubs = next;
  };

  const clearImagerySubs = () => {
    for (const remove of imageryUnsubs) remove();
    imageryUnsubs = [];
  };

  const noteImageryError = (error: unknown) => {
    if (sawImageryError) return;
    sawImageryError = true;
    controller.setStatus(
      `Imagery tile failed. The globe stays open. ${messageFromUnknown(error)}`,
      "error",
    );
  };

  const watchProvider = (provider: ImageryProvider) => {
    imageryUnsubs.push(
      onEvent(provider.errorEvent, (error) => {
        noteImageryError(error);
      }),
    );
  };

  const installProviders = (providers: ImageryProvider[]) => {
    clearImagerySubs();
    const layers = viewer.imageryLayers;
    layers.removeAll();
    for (const provider of providers) {
      layers.addImageryProvider(provider);
      watchProvider(provider);
    }
  };

  const publishCamera = () => {
    const point = readCamera(viewer);
    if (point) controller.publishCamera(point);
    syncNavigationSpeed(viewer);
  };

  const tiltAllowed = () => {
    const state = controller.getState();
    return !state.scene2d && state.cameraMode !== "top-down" && state.cameraMode !== "orbit";
  };

  const finishFlight = (mode: CameraMode) => {
    if (destroyed) return;
    const current = controller.getState().cameraMode;
    if (current === mode) {
      const keepError = controller.getState().statusLevel === "error";
      controller.publish({
        cameraMode: mode === "fly-to" ? "free" : mode,
        ...(keepError ? {} : { statusText: "Ready", statusLevel: "ready" as const }),
      });
    }
    syncTiltEnabled(viewer, tiltAllowed());
    publishCamera();
  };

  const ensure3d = (then: () => void) => {
    if (viewer.scene.mode === SceneMode.SCENE3D) {
      then();
      return;
    }
    pendingAfterMorph = then;
    viewer.scene.morphTo3D(0.75);
    controller.publish({ scene2d: false });
  };

  const runCameraMode = (mode: CameraMode) => {
    if (mode === "free") {
      exitOrbit(viewer);
      syncTiltEnabled(viewer, !controller.getState().scene2d);
      return;
    }
    if (mode === "globe") {
      ensure3d(() => {
        flyGlobe(viewer, controller.getState().focus, () => finishFlight("globe"));
      });
      return;
    }
    if (mode === "top-down") {
      ensure3d(() => {
        exitOrbit(viewer);
        syncTiltEnabled(viewer, false);
        flyTopDown(viewer, () => finishFlight("top-down"));
      });
      return;
    }
    if (mode === "orbit") {
      ensure3d(() => {
        const focus = controller.getState().focus ?? focusUnderCamera(viewer);
        controller.setFocus(focus);
        enterOrbit(viewer, focus);
        syncTiltEnabled(viewer, false);
        controller.setStatus("Orbiting the focus point. Drag to move around it.", "info");
      });
      return;
    }
    const place = controller.getState().lastPlace;
    if (!place) {
      controller.setStatus("Search for a place to fly there.", "info");
      return;
    }
    flyToPlace(viewer, place, () => finishFlight("fly-to"));
  };

  const selectAt = (position: Cartesian2) => {
    const generation = ++pickGeneration;
    const point = pickGlobe(viewer, position);
    if (!point) {
      controller.setStatus("No surface under the cursor.", "info");
      return;
    }
    controller.setFocus(point);
    const state = controller.getState();
    if (!state.terrainEnabled) {
      controller.setSelection({
        longitude: point.longitude,
        latitude: point.latitude,
        elevationMeters: null,
        elevationState: "disabled",
      });
      return;
    }
    if (terrainIsEllipsoid(viewer.terrainProvider) || state.terrainDetailLimited) {
      controller.setSelection({
        longitude: point.longitude,
        latitude: point.latitude,
        elevationMeters: null,
        elevationState: "ellipsoid",
      });
      return;
    }
    controller.setSelection({
      longitude: point.longitude,
      latitude: point.latitude,
      elevationMeters: null,
      elevationState: "pending",
    });
    const sample = Cartographic.fromDegrees(point.longitude, point.latitude);
    void sampleTerrainMostDetailed(viewer.terrainProvider, [sample])
      .then((samples) => {
        if (destroyed || generation !== pickGeneration) return;
        const height = samples[0]?.height;
        if (typeof height !== "number" || !Number.isFinite(height)) {
          controller.setSelection({
            longitude: point.longitude,
            latitude: point.latitude,
            elevationMeters: null,
            elevationState: "unavailable",
          });
          return;
        }
        controller.setSelection({
          longitude: point.longitude,
          latitude: point.latitude,
          elevationMeters: height,
          elevationState: "sampled",
        });
      })
      .catch(() => {
        if (destroyed || generation !== pickGeneration) return;
        controller.setSelection({
          longitude: point.longitude,
          latitude: point.latitude,
          elevationMeters: null,
          elevationState: "unavailable",
        });
      });
  };

  const onClick: ScreenSpaceEventHandler.PositionedEventCallback = (event: ClickEvent) => {
    selectAt(event.position);
  };

  const onDoubleClick: ScreenSpaceEventHandler.PositionedEventCallback = (event: ClickEvent) => {
    const point = pickGlobe(viewer, event.position);
    if (!point) {
      controller.setStatus("No surface under the cursor.", "info");
      return;
    }
    controller.setFocus(point);
    const state = controller.getState();
    const currentHeight = viewer.camera.positionCartographic.height;
    const nextHeight = Math.min(Math.max(currentHeight * 0.36, 350), 2_200_000);
    const pitch =
      state.scene2d || state.cameraMode === "top-down"
        ? CesiumMath.toRadians(-90)
        : CesiumMath.clamp(
            viewer.camera.pitch,
            CesiumMath.toRadians(-75),
            CesiumMath.toRadians(-30),
          );
    const returnToOrbit = state.cameraMode === "orbit";
    flyToward(viewer, point, nextHeight, pitch, () => {
      if (destroyed) return;
      if (returnToOrbit && controller.getState().cameraMode === "orbit") {
        enterOrbit(viewer, point);
      }
      publishCamera();
    });
  };

  viewer.screenSpaceEventHandler.setInputAction(onClick, ScreenSpaceEventType.LEFT_CLICK);
  viewer.screenSpaceEventHandler.setInputAction(
    onDoubleClick,
    ScreenSpaceEventType.LEFT_DOUBLE_CLICK,
  );

  const onCameraChanged = () => {
    publishCamera();
  };
  viewer.camera.changed.addEventListener(onCameraChanged);
  viewer.camera.moveEnd.addEventListener(onCameraChanged);

  unsubscribers.push(
    onEvent(viewer.scene.morphComplete, () => {
      const is2d = viewer.scene.mode === SceneMode.SCENE2D;
      controller.publish({ scene2d: is2d });
      const pending = pendingAfterMorph;
      pendingAfterMorph = null;
      pending?.();
      syncTiltEnabled(viewer, tiltAllowed());
      publishCamera();
    }),
  );

  const flushTiles = () => {
    tileTimer = 0;
    if (destroyed || controller.getState().statusLevel === "error") return;
    if (queuedTiles > 0) {
      controller.setStatus(`Loading tiles · ${queuedTiles}`, "loading");
    } else if (controller.getState().statusLevel === "loading") {
      controller.setStatus("Ready", "ready");
    }
  };

  unsubscribers.push(
    onEvent(viewer.scene.globe.tileLoadProgressEvent, (remaining) => {
      queuedTiles = typeof remaining === "number" ? remaining : 0;
      if (tileTimer) return;
      tileTimer = window.setTimeout(flushTiles, 200);
    }),
  );

  let renderFailures = 0;
  unsubscribers.push(
    onEvent(viewer.scene.renderError, (_scene, error) => {
      renderFailures += 1;
      controller.setStatus(
        `The globe hit a render error. ${messageFromUnknown(error)}`,
        "error",
      );
      if (renderFailures < 3) viewer.useDefaultRenderLoop = true;
    }),
  );

  const onContextLost = (event: Event) => {
    event.preventDefault();
    controller.setStatus(
      "WebGL context was lost. Reload the page to restore the globe.",
      "error",
    );
  };
  viewer.scene.canvas.addEventListener("webglcontextlost", onContextLost);

  const setImagery = (style: ImageryStyle) => {
    const generation = ++imageryGeneration;
    sawImageryError = false;
    void createImageryStack(style, ionToken)
      .then((result) => {
        if (destroyed || generation !== imageryGeneration) return;
        installProviders(result.providers);
        if (result.fallbackNote) controller.setStatus(result.fallbackNote, "info");
      })
      .catch((error: unknown) => {
        if (destroyed || generation !== imageryGeneration) return;
        installProviders(createPublicImagery(style));
        controller.setStatus(
          `Imagery failed to load. Showing the public fallback. ${messageFromUnknown(error)}`,
          "error",
        );
      });
  };

  unsubscribers.push(
    controller.subscribeCommands((command: SceneCommand) => {
      if (destroyed) return;
      try {
      handleCommand(command);
      } catch (error: unknown) {
        controller.setStatus(
          `That control could not be applied. ${messageFromUnknown(error)}`,
          "error",
        );
      }
    }),
  );

  const handleCommand = (command: SceneCommand) => {
    if (destroyed) return;
      if (command.type === "camera-mode") runCameraMode(command.mode);
      if (command.type === "scene-2d") {
        exitOrbit(viewer);
        const focus = controller.getState().focus ?? focusUnderCamera(viewer);
        const height = viewer.camera.positionCartographic?.height;
        const range = Number.isFinite(height) ? height : 4_000_000;
        if (command.enabled) {
          pendingAfterMorph = () => {
            holdOverhead(viewer, focus.longitude, focus.latitude, range);
            publishCamera();
          };
          viewer.scene.morphTo2D(0.75);
        } else {
          pendingAfterMorph = () => {
            flyToward(
              viewer,
              focus,
              Math.min(Math.max(range, 1_500), 20_000_000),
              CesiumMath.toRadians(-55),
              () => publishCamera(),
            );
          };
          viewer.scene.morphTo3D(0.75);
        }
        syncTiltEnabled(
          viewer,
          !command.enabled && controller.getState().cameraMode !== "top-down",
        );
      }
      if (command.type === "imagery") setImagery(command.style);
      if (command.type === "terrain") {
        applyTerrain({
          viewer,
          enabled: command.enabled,
          ionToken,
          controller,
          replaceSubscriptions: replaceTerrainSubs,
        });
      }
      if (command.type === "exaggeration") {
        viewer.scene.verticalExaggeration = command.value;
      }
      if (command.type === "display-mode") applyDisplayMode(viewer, command.mode);
      if (command.type === "fly-to") {
        controller.setStatus(`Flying to ${command.place.label}`, "info");
        flyToPlace(viewer, command.place, () => finishFlight("fly-to"));
      }
      if (command.type === "reset-camera") {
        syncTiltEnabled(viewer, true);
        if (viewer.scene.mode === SceneMode.SCENE3D) {
          flyHome(viewer, () => finishFlight("free"));
        } else {
          pendingAfterMorph = () => flyHome(viewer, () => finishFlight("free"));
          viewer.scene.morphTo3D(0.6);
        }
      }
  };

  if (openingProviders.length > 1) {
    installProviders(openingProviders);
  } else {
    const baseProvider = viewer.imageryLayers.get(0)?.imageryProvider;
    if (baseProvider) watchProvider(baseProvider);
  }

  if (initial.imageryStyle === "satellite" && ionToken) setImagery("satellite");
  applyTerrain({
    viewer,
    enabled: initial.terrainEnabled,
    ionToken,
    controller,
    replaceSubscriptions: replaceTerrainSubs,
  });
  if (ionToken) registerIonSearch(viewer, ionToken, controller);
  publishCamera();

  const limited = !ionToken;
  controller.publish({
    viewerReady: true,
    terrainDetailLimited: limited,
    statusText: limited
      ? "Terrain detail is limited until a Cesium ion token is configured."
      : "Ready",
    statusLevel: limited ? "info" : "ready",
  });

  return {
    destroy() {
      if (destroyed) return;
      destroyed = true;
      window.clearTimeout(tileTimer);
      pendingAfterMorph = null;
      for (const remove of unsubscribers) remove();
      clearImagerySubs();
      replaceTerrainSubs([]);
      viewer.camera.changed.removeEventListener(onCameraChanged);
      viewer.camera.moveEnd.removeEventListener(onCameraChanged);
      viewer.scene.canvas.removeEventListener("webglcontextlost", onContextLost);
      if (!viewer.isDestroyed()) viewer.destroy();
    },
  };
}
