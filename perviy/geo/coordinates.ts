export function formatLongitude(degrees: number): string {
  if (!Number.isFinite(degrees)) return "—";
  const hemisphere = degrees >= 0 ? "E" : "W";
  return `${Math.abs(degrees).toFixed(5)}° ${hemisphere}`;
}

export function formatLatitude(degrees: number): string {
  if (!Number.isFinite(degrees)) return "—";
  const hemisphere = degrees >= 0 ? "N" : "S";
  return `${Math.abs(degrees).toFixed(5)}° ${hemisphere}`;
}

export function formatAltitude(meters: number): string {
  if (!Number.isFinite(meters)) return "—";
  const sign = meters < 0 ? "−" : "";
  const absolute = Math.abs(meters);
  if (absolute >= 100_000) {
    return `${sign}${(absolute / 1000).toLocaleString("en-US", {
      maximumFractionDigits: 0,
    })} km`;
  }
  if (absolute >= 10_000) {
    return `${sign}${(absolute / 1000).toLocaleString("en-US", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    })} km`;
  }
  return `${sign}${absolute.toLocaleString("en-US", {
    maximumFractionDigits: 0,
  })} m`;
}

export function formatElevation(meters: number): string {
  if (!Number.isFinite(meters)) return "—";
  const digits = Math.abs(meters) >= 100 ? 0 : 1;
  return `${meters.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })} m`;
}

export function formatExaggeration(value: number): string {
  if (!Number.isFinite(value)) return "1×";
  const text = Number.isInteger(value)
    ? value.toFixed(0)
    : value.toFixed(2).replace(/\.?0+$/, "");
  return `${text}×`;
}
