import type { Event } from "cesium";

type CesiumEventListener = Parameters<Event["addEventListener"]>[0];

export function onEvent(event: Event, listener: CesiumEventListener): () => void {
  return event.addEventListener(listener);
}

export function messageFromUnknown(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string" && error.trim()) return error;
  return "Unexpected error";
}
