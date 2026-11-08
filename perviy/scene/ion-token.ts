export function readIonToken(): string | undefined {
  const value = process.env.NEXT_PUBLIC_CESIUM_ION_TOKEN?.trim();
  return value ? value : undefined;
}
