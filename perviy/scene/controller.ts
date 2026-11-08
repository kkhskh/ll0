import { searchPublicGazetteer } from "@/geo/search-client";
import type {
  CameraMode,
  DisplayMode,
  GeoPoint,
  ImageryStyle,
  PlaceResult,
  SceneCommand,
  SceneState,
  SearchOutcome,
  Selection,
  StatusLevel,
} from "@/scene/types";

export type PlaceSearcher = (
  query: string,
  signal: AbortSignal,
) => Promise<SearchOutcome>;

const EXAGGERATION_MIN = 1;
const EXAGGERATION_MAX = 3;

function clampExaggeration(value: number): number {
  if (!Number.isFinite(value)) return 1;
  const clamped = Math.min(EXAGGERATION_MAX, Math.max(EXAGGERATION_MIN, value));
  return Math.round(clamped * 100) / 100;
}

function createInitialState(ionConfigured: boolean): SceneState {
  return {
    cameraMode: "free",
    scene2d: false,
    imageryStyle: "satellite",
    terrainEnabled: true,
    terrainExaggeration: 1,
    displayMode: "cinematic",
    selection: null,
    focus: null,
    lastPlace: null,
    layersCollapsed: false,
    inspectorCollapsed: false,
    settingsOpen: false,
    sceneMenuOpen: false,
    statusText: "Opening globe",
    statusLevel: "loading",
    terrainDetailLimited: !ionConfigured,
    ionConfigured,
    viewerReady: false,
    webglAvailable: null,
  };
}

export interface SceneController {
  getState: () => SceneState;
  subscribe: (listener: () => void) => () => void;
  getCamera: () => GeoPoint | null;
  subscribeCamera: (listener: () => void) => () => void;
  publishCamera: (point: GeoPoint) => void;
  publish: (patch: Partial<SceneState>) => void;
  setStatus: (text: string, level: StatusLevel) => void;
  setCameraMode: (mode: CameraMode) => void;
  setScene2d: (enabled: boolean) => void;
  setImageryStyle: (style: ImageryStyle) => void;
  setTerrainEnabled: (enabled: boolean) => void;
  setTerrainExaggeration: (value: number) => void;
  setDisplayMode: (mode: DisplayMode) => void;
  setSelection: (selection: Selection | null) => void;
  setFocus: (focus: GeoPoint) => void;
  flyToPlace: (place: PlaceResult) => void;
  repeatFlyTo: () => void;
  resetCamera: () => void;
  toggleLayers: () => void;
  toggleInspector: () => void;
  setSettingsOpen: (open: boolean) => void;
  setSceneMenuOpen: (open: boolean) => void;
  subscribeCommands: (listener: (command: SceneCommand) => void) => () => void;
  registerSearcher: (searcher: PlaceSearcher) => void;
  search: (query: string, signal: AbortSignal) => Promise<SearchOutcome>;
}

export function createSceneController(options: {
  ionConfigured: boolean;
}): SceneController {
  let state = createInitialState(options.ionConfigured);
  let camera: GeoPoint | null = null;
  const listeners = new Set<() => void>();
  const cameraListeners = new Set<() => void>();
  const commandListeners = new Set<(command: SceneCommand) => void>();
  let searcher: PlaceSearcher = searchPublicGazetteer;

  const emitState = () => {
    for (const listener of listeners) listener();
  };

  const emitCommand = (command: SceneCommand) => {
    for (const listener of commandListeners) listener(command);
  };

  const commit = (patch: Partial<SceneState>) => {
    let changed = false;
    for (const key of Object.keys(patch) as (keyof SceneState)[]) {
      if (!Object.is(state[key], patch[key])) changed = true;
    }
    if (!changed) return;
    state = { ...state, ...patch };
    emitState();
  };

  return {
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    getCamera: () => camera,
    subscribeCamera(listener) {
      cameraListeners.add(listener);
      return () => {
        cameraListeners.delete(listener);
      };
    },
    publishCamera(point) {
      const previous = camera;
      if (
        previous &&
        Math.abs(previous.longitude - point.longitude) < 0.00001 &&
        Math.abs(previous.latitude - point.latitude) < 0.00001 &&
        Math.abs(previous.height - point.height) < 1
      ) {
        return;
      }
      camera = point;
      for (const listener of cameraListeners) listener();
    },
    publish: commit,
    setStatus(text, level) {
      if (level === "loading" && state.statusLevel === "error") return;
      commit({ statusText: text, statusLevel: level });
    },
    setCameraMode(mode) {
      commit({
        cameraMode: mode,
        sceneMenuOpen: false,
        scene2d: mode === "orbit" ? false : state.scene2d,
      });
      emitCommand({ type: "camera-mode", mode });
    },
    setScene2d(enabled) {
      commit({
        scene2d: enabled,
        sceneMenuOpen: false,
        cameraMode: enabled && state.cameraMode === "orbit" ? "free" : state.cameraMode,
      });
      emitCommand({ type: "scene-2d", enabled });
    },
    setImageryStyle(style) {
      commit({
        imageryStyle: style,
        statusText: "Switching imagery",
        statusLevel: "loading",
      });
      emitCommand({ type: "imagery", style });
    },
    setTerrainEnabled(enabled) {
      commit({ terrainEnabled: enabled });
      emitCommand({ type: "terrain", enabled });
    },
    setTerrainExaggeration(value) {
      const next = clampExaggeration(value);
      commit({ terrainExaggeration: next });
      emitCommand({ type: "exaggeration", value: next });
    },
    setDisplayMode(mode) {
      commit({ displayMode: mode });
      emitCommand({ type: "display-mode", mode });
    },
    setSelection(selection) {
      commit({
        selection,
        inspectorCollapsed: selection ? state.inspectorCollapsed : false,
      });
    },
    setFocus(focus) {
      commit({ focus });
    },
    flyToPlace(place) {
      commit({
        lastPlace: place,
        focus: {
          longitude: place.longitude,
          latitude: place.latitude,
          height: 0,
        },
        cameraMode: "fly-to",
        sceneMenuOpen: false,
      });
      emitCommand({ type: "fly-to", place });
    },
    repeatFlyTo() {
      if (!state.lastPlace) {
        commit({
          statusText: "Search for a place to fly there.",
          statusLevel: "info",
        });
        return;
      }
      commit({ cameraMode: "fly-to", sceneMenuOpen: false });
      emitCommand({ type: "fly-to", place: state.lastPlace });
    },
    resetCamera() {
      commit({
        cameraMode: "free",
        scene2d: false,
        sceneMenuOpen: false,
      });
      emitCommand({ type: "reset-camera" });
    },
    toggleLayers() {
      commit({ layersCollapsed: !state.layersCollapsed });
    },
    toggleInspector() {
      commit({ inspectorCollapsed: !state.inspectorCollapsed });
    },
    setSettingsOpen(open) {
      commit({
        settingsOpen: open,
        sceneMenuOpen: open ? false : state.sceneMenuOpen,
      });
    },
    setSceneMenuOpen(open) {
      commit({
        sceneMenuOpen: open,
        settingsOpen: open ? false : state.settingsOpen,
      });
    },
    subscribeCommands(listener) {
      commandListeners.add(listener);
      return () => {
        commandListeners.delete(listener);
      };
    },
    registerSearcher(next) {
      searcher = next;
    },
    search(query, signal) {
      return searcher(query, signal);
    },
  };
}
