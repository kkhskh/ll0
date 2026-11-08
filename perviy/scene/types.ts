/**
 * Scene state for the Earth shell.
 *
 * The controller owns mode, imagery, terrain, selection, and display mode.
 * The Cesium adapter applies those commands. Later pieces — 3D Tiles cities,
 * water, measurement, drawing, saved workspaces, and time — should extend
 * this state and the adapter, not the React shell.
 */

export type CameraMode = "free" | "globe" | "orbit" | "top-down" | "fly-to";

export type ImageryStyle = "satellite" | "geographic" | "dark" | "terrain";

export type DisplayMode = "analytical" | "cinematic";

export type StatusLevel = "ready" | "loading" | "error" | "info";

export type ElevationState =
  | "pending"
  | "sampled"
  | "ellipsoid"
  | "disabled"
  | "unavailable";

export interface GeoPoint {
  longitude: number;
  latitude: number;
  height: number;
}

export interface PlaceResult {
  id: string;
  label: string;
  longitude: number;
  latitude: number;
  west?: number;
  south?: number;
  east?: number;
  north?: number;
}

export interface Selection {
  longitude: number;
  latitude: number;
  elevationMeters: number | null;
  elevationState: ElevationState;
}

export interface SceneState {
  cameraMode: CameraMode;
  scene2d: boolean;
  imageryStyle: ImageryStyle;
  terrainEnabled: boolean;
  terrainExaggeration: number;
  displayMode: DisplayMode;
  selection: Selection | null;
  focus: GeoPoint | null;
  lastPlace: PlaceResult | null;
  layersCollapsed: boolean;
  inspectorCollapsed: boolean;
  settingsOpen: boolean;
  sceneMenuOpen: boolean;
  statusText: string;
  statusLevel: StatusLevel;
  terrainDetailLimited: boolean;
  ionConfigured: boolean;
  viewerReady: boolean;
  webglAvailable: boolean | null;
}

export interface SearchOutcome {
  results: PlaceResult[];
  error?: string;
  aborted?: boolean;
}

export type SceneCommand =
  | { type: "camera-mode"; mode: CameraMode }
  | { type: "scene-2d"; enabled: boolean }
  | { type: "imagery"; style: ImageryStyle }
  | { type: "terrain"; enabled: boolean }
  | { type: "exaggeration"; value: number }
  | { type: "display-mode"; mode: DisplayMode }
  | { type: "fly-to"; place: PlaceResult }
  | { type: "reset-camera" };
