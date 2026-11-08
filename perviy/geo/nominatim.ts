import type { PlaceResult } from "@/scene/types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function finite(value: unknown): number | undefined {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : undefined;
}

export function parseNominatim(payload: unknown): PlaceResult[] {
  if (!Array.isArray(payload)) return [];
  const results: PlaceResult[] = [];

  for (const [index, entry] of payload.entries()) {
    if (!isRecord(entry)) continue;
    const latitude = finite(entry.lat);
    const longitude = finite(entry.lon);
    const label = typeof entry.display_name === "string" ? entry.display_name.trim() : "";
    if (
      latitude === undefined ||
      longitude === undefined ||
      label.length === 0 ||
      latitude < -90 ||
      latitude > 90 ||
      longitude < -180 ||
      longitude > 180
    ) {
      continue;
    }

    const place: PlaceResult = {
      id: `nominatim-${typeof entry.place_id === "number" ? entry.place_id : index}`,
      label,
      longitude,
      latitude,
    };

    if (Array.isArray(entry.boundingbox) && entry.boundingbox.length >= 4) {
      const south = finite(entry.boundingbox[0]);
      const north = finite(entry.boundingbox[1]);
      const west = finite(entry.boundingbox[2]);
      const east = finite(entry.boundingbox[3]);
      if (
        south !== undefined &&
        north !== undefined &&
        west !== undefined &&
        east !== undefined
      ) {
        place.south = south;
        place.north = north;
        place.west = west;
        place.east = east;
      }
    }

    if (results.some((existing) => existing.label === label)) continue;
    results.push(place);
  }

  return results;
}
