import { Color, type Viewer } from "cesium";
import type { DisplayMode } from "@/scene/types";

/**
 * Analytical turns off the atmosphere and fog Cesium exposes on the scene.
 * Cinematic keeps those effects at Cesium's own defaults. It does not add
 * water shaders, lighting-driven day/night, or any other cinematic pass.
 */
export function applyDisplayMode(viewer: Viewer, mode: DisplayMode): void {
  const scene = viewer.scene;
  const analytical = mode === "analytical";
  scene.fog.enabled = !analytical;
  scene.fog.renderable = !analytical;
  scene.globe.showGroundAtmosphere = !analytical;
  scene.globe.enableLighting = false;
  scene.highDynamicRange = false;
  scene.backgroundColor = Color.fromCssColorString("#050505");
  scene.globe.baseColor = Color.fromCssColorString("#141414");
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = !analytical;
  if (scene.skyBox) scene.skyBox.show = !analytical;
  if (scene.sun) scene.sun.show = !analytical;
  if (scene.moon) scene.moon.show = !analytical;
}
