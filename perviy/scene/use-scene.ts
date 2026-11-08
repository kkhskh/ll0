"use client";

import { useSyncExternalStore } from "react";
import type { SceneController } from "@/scene/controller";
import type { GeoPoint, SceneState } from "@/scene/types";

export function useScene(controller: SceneController): SceneState {
  return useSyncExternalStore(
    controller.subscribe,
    controller.getState,
    controller.getState,
  );
}

export function useCamera(controller: SceneController): GeoPoint | null {
  return useSyncExternalStore(
    controller.subscribeCamera,
    controller.getCamera,
    controller.getCamera,
  );
}
