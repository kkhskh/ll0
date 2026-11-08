import {
  Ion,
  IonImageryProvider,
  UrlTemplateImageryProvider,
  type ImageryProvider,
} from "cesium";
import type { ImageryStyle } from "@/scene/types";

const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services";

export interface ImageryStack {
  providers: ImageryProvider[];
  /** Set when a keyed style could not be used and a public provider replaced it. */
  fallbackNote?: string;
}

function esri(path: string, credit: string, maximumLevel: number): ImageryProvider {
  return new UrlTemplateImageryProvider({
    url: `${ESRI}/${path}/MapServer/tile/{z}/{y}/{x}`,
    credit,
    maximumLevel,
  });
}

const SATELLITE_CREDIT =
  "Esri, Maxar, Earthstar Geographics, and the GIS User Community";
const STREET_CREDIT =
  "Esri, HERE, Garmin, FAO, NOAA, USGS, © OpenStreetMap contributors, and the GIS User Community";
const CANVAS_CREDIT =
  "Esri, HERE, Garmin, © OpenStreetMap contributors, and the GIS User Community";
const RELIEF_CREDIT =
  "Esri, HERE, Garmin, Intermap, INCREMENT P, GEBCO, USGS, FAO, NPS, NRCAN, GeoBase, IGN, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), © OpenStreetMap contributors, and the GIS User Community";

/**
 * Imagery styles share one viewer. CARTO basemaps now answer with an
 * "API key required" tile, so the no-key styles use Esri's public map
 * servers. Those endpoints are not a license for commercial product use.
 */
export function createPublicImagery(style: ImageryStyle): ImageryProvider[] {
  if (style === "geographic") {
    return [esri("World_Street_Map", STREET_CREDIT, 19)];
  }
  if (style === "dark") {
    return [
      esri("Canvas/World_Dark_Gray_Base", CANVAS_CREDIT, 16),
      esri("Canvas/World_Dark_Gray_Reference", CANVAS_CREDIT, 16),
    ];
  }
  if (style === "terrain") {
    // World Hillshade is a near-white overlay and World Physical Map stops
    // at level 8, so neither stays readable from orbit down to a city.
    // World Topo Map is shaded relief with contours through the high levels.
    return [esri("World_Topo_Map", RELIEF_CREDIT, 16)];
  }
  return [esri("World_Imagery", SATELLITE_CREDIT, 19)];
}

export async function createImageryStack(
  style: ImageryStyle,
  ionToken: string | undefined,
): Promise<ImageryStack> {
  if (style === "satellite" && ionToken) {
    try {
      Ion.defaultAccessToken = ionToken;
      const provider = await IonImageryProvider.fromAssetId(2);
      return { providers: [provider] };
    } catch (error) {
      const reason = error instanceof Error ? error.message : "Ion imagery failed";
      return {
        providers: createPublicImagery("satellite"),
        fallbackNote: `Cesium ion imagery was unavailable (${reason}). Showing Esri World Imagery.`,
      };
    }
  }
  return { providers: createPublicImagery(style) };
}
