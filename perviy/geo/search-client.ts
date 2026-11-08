import type { SearchOutcome } from "@/scene/types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readOutcome(payload: unknown): SearchOutcome {
  if (!isRecord(payload) || !Array.isArray(payload.results)) {
    return {
      results: [],
      error: "Search is unavailable. The globe is unaffected.",
    };
  }
  const error = typeof payload.error === "string" ? payload.error : undefined;
  const results = payload.results.flatMap((entry) => {
    if (!isRecord(entry)) return [];
    const label = typeof entry.label === "string" ? entry.label : "";
    const longitude = typeof entry.longitude === "number" ? entry.longitude : Number.NaN;
    const latitude = typeof entry.latitude === "number" ? entry.latitude : Number.NaN;
    const id = typeof entry.id === "string" ? entry.id : "";
    if (!label || !id || !Number.isFinite(longitude) || !Number.isFinite(latitude)) {
      return [];
    }
    return [
      {
        id,
        label,
        longitude,
        latitude,
        west: typeof entry.west === "number" ? entry.west : undefined,
        south: typeof entry.south === "number" ? entry.south : undefined,
        east: typeof entry.east === "number" ? entry.east : undefined,
        north: typeof entry.north === "number" ? entry.north : undefined,
      },
    ];
  });
  return { results, error };
}

export async function searchPublicGazetteer(
  query: string,
  signal: AbortSignal,
): Promise<SearchOutcome> {
  try {
    const response = await fetch(`/api/geocode?q=${encodeURIComponent(query)}`, {
      signal,
      cache: "no-store",
    });
    const payload: unknown = await response.json().catch(() => null);
    const outcome = readOutcome(payload);
    if (!response.ok && !outcome.error) {
      return {
        results: [],
        error: "Search is unavailable. The globe is unaffected.",
      };
    }
    return outcome;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return { results: [], aborted: true };
    }
    return {
      results: [],
      error: "Search is unavailable. The globe is unaffected.",
    };
  }
}
