import {
  Cartesian3,
  Cartographic,
  IonGeocoderService,
  Math as CesiumMath,
  Rectangle,
  type GeocoderService,
  type Viewer,
} from "cesium";
import { searchPublicGazetteer } from "@/geo/search-client";
import { messageFromUnknown } from "@/cesium/events";
import type { SceneController } from "@/scene/controller";
import type { PlaceResult, SearchOutcome } from "@/scene/types";

function finitePoint(
  longitude: number,
  latitude: number,
): { longitude: number; latitude: number } | undefined {
  if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) return undefined;
  if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) return undefined;
  return { longitude, latitude };
}

function placeFromResult(
  result: GeocoderService.Result,
  index: number,
): PlaceResult | undefined {
  const label = result.displayName?.trim();
  if (!label) return undefined;
  const destination = result.destination;

  if (destination instanceof Rectangle) {
    const center = Rectangle.center(destination);
    const point = finitePoint(
      CesiumMath.toDegrees(center.longitude),
      CesiumMath.toDegrees(center.latitude),
    );
    if (!point) return undefined;
    return {
      id: `ion-${index}`,
      label,
      longitude: point.longitude,
      latitude: point.latitude,
      west: CesiumMath.toDegrees(destination.west),
      south: CesiumMath.toDegrees(destination.south),
      east: CesiumMath.toDegrees(destination.east),
      north: CesiumMath.toDegrees(destination.north),
    };
  }

  const cartographic = Cartographic.fromCartesian(destination as Cartesian3);
  const point = finitePoint(
    CesiumMath.toDegrees(cartographic.longitude),
    CesiumMath.toDegrees(cartographic.latitude),
  );
  if (!point) return undefined;
  return {
    id: `ion-${index}`,
    label,
    longitude: point.longitude,
    latitude: point.latitude,
  };
}

export function registerIonSearch(
  viewer: Viewer,
  token: string,
  controller: SceneController,
): void {
  const geocoder = new IonGeocoderService({
    scene: viewer.scene,
    accessToken: token,
  });

  controller.registerSearcher(async (query, signal): Promise<SearchOutcome> => {
    if (signal.aborted) return { results: [], aborted: true };
    try {
      const results = await geocoder.geocode(query);
      if (signal.aborted) return { results: [], aborted: true };
      const places = results.flatMap((result, index) => {
        const place = placeFromResult(result, index);
        return place ? [place] : [];
      });
      if (places.length > 0) return { results: places };
    } catch (error) {
      if (signal.aborted) return { results: [], aborted: true };
      controller.setStatus(
        `Ion search failed. Trying the public gazetteer. ${messageFromUnknown(error)}`,
        "info",
      );
    }
    return searchPublicGazetteer(query, signal);
  });
}
