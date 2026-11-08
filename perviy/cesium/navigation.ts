import {
  BoundingSphere,
  CameraEventType,
  Cartesian2,
  Cartesian3,
  Cartographic,
  HeadingPitchRange,
  KeyboardEventModifier,
  Math as CesiumMath,
  Matrix4,
  Rectangle,
  SceneMode,
  type Viewer,
} from "cesium";
import type { GeoPoint, PlaceResult } from "@/scene/types";

const HOME = {
  longitude: 15,
  latitude: 18,
  /** Distance from the framed surface point. Far enough to hold the whole disk. */
  range: 21_000_000,
  /** A few degrees off nadir so the default reads as a tilted globe, not a map. */
  pitch: CesiumMath.toRadians(-80),
};

export function configureCameraControls(viewer: Viewer): void {
  const controller = viewer.scene.screenSpaceCameraController;
  controller.zoomEventTypes = [CameraEventType.WHEEL, CameraEventType.PINCH];
  controller.rotateEventTypes = CameraEventType.LEFT_DRAG;
  controller.translateEventTypes = CameraEventType.LEFT_DRAG;
  controller.tiltEventTypes = [
    CameraEventType.MIDDLE_DRAG,
    CameraEventType.PINCH,
    {
      eventType: CameraEventType.LEFT_DRAG,
      modifier: KeyboardEventModifier.CTRL,
    },
    {
      eventType: CameraEventType.LEFT_DRAG,
      modifier: KeyboardEventModifier.ALT,
    },
    {
      eventType: CameraEventType.RIGHT_DRAG,
      modifier: KeyboardEventModifier.CTRL,
    },
    {
      eventType: CameraEventType.RIGHT_DRAG,
      modifier: KeyboardEventModifier.ALT,
    },
  ];
  controller.minimumZoomDistance = 40;
  controller.maximumZoomDistance = 42_000_000;
  controller.enableCollisionDetection = true;
  controller.inertiaSpin = 0.8;
  controller.inertiaTranslate = 0.8;
  controller.inertiaZoom = 0.75;
}

/**
 * Cesium already scales wheel zoom by camera height. The factor is tightened
 * near the surface so street-level scrolling stays short, and left at the
 * library default once the camera is back out in orbit.
 */
export function syncNavigationSpeed(viewer: Viewer): void {
  const height = viewer.camera.positionCartographic.height;
  const controller = viewer.scene.screenSpaceCameraController;
  if (!Number.isFinite(height)) return;
  if (height < 2_000) controller.zoomFactor = 2;
  else if (height < 40_000) controller.zoomFactor = 3.2;
  else if (height < 800_000) controller.zoomFactor = 4.2;
  else controller.zoomFactor = 5;
}

export function syncTiltEnabled(viewer: Viewer, allowTilt: boolean): void {
  const controller = viewer.scene.screenSpaceCameraController;
  controller.enableTilt = allowTilt;
  controller.enableLook = allowTilt;
}

export function exitOrbit(viewer: Viewer): void {
  if (!Matrix4.equals(viewer.camera.transform, Matrix4.IDENTITY)) {
    viewer.camera.lookAtTransform(Matrix4.IDENTITY);
  }
}

export function enterOrbit(viewer: Viewer, focus: GeoPoint): void {
  exitOrbit(viewer);
  const target = Cartesian3.fromDegrees(
    focus.longitude,
    focus.latitude,
    Math.max(0, focus.height),
  );
  const camera = viewer.camera;
  const range = Math.min(
    Math.max(Cartesian3.distance(camera.positionWC, target), 800),
    30_000_000,
  );
  let pitch = camera.pitch;
  if (!Number.isFinite(pitch) || pitch > CesiumMath.toRadians(-20)) {
    pitch = CesiumMath.toRadians(-35);
  }
  // Straight down makes the local look-at frame singular.
  if (pitch < CesiumMath.toRadians(-80)) pitch = CesiumMath.toRadians(-55);
  const heading = Number.isFinite(camera.heading) ? camera.heading : 0;
  camera.lookAt(target, new HeadingPitchRange(heading, pitch, range));
}

export function readCamera(viewer: Viewer): GeoPoint | null {
  const cartographic = viewer.camera.positionCartographic;
  if (!cartographic || !Number.isFinite(cartographic.height)) return null;
  return {
    longitude: CesiumMath.toDegrees(cartographic.longitude),
    latitude: CesiumMath.toDegrees(cartographic.latitude),
    height: cartographic.height,
  };
}

export function focusUnderCamera(viewer: Viewer): GeoPoint {
  const canvas = viewer.scene.canvas;
  const center = new Cartesian2(canvas.clientWidth / 2, canvas.clientHeight / 2);
  const picked = pickGlobe(viewer, center);
  if (picked) return picked;
  const camera = readCamera(viewer);
  return {
    longitude: camera?.longitude ?? HOME.longitude,
    latitude: camera?.latitude ?? HOME.latitude,
    height: 0,
  };
}

export function pickGlobe(viewer: Viewer, windowPosition: Cartesian2): GeoPoint | null {
  const ray = viewer.camera.getPickRay(windowPosition);
  if (!ray) return null;
  const hit = viewer.scene.globe.pick(ray, viewer.scene);
  const cartesian =
    hit ?? viewer.camera.pickEllipsoid(windowPosition, viewer.scene.globe.ellipsoid);
  if (!cartesian) return null;
  return pointFromCartesian(cartesian);
}

function pointFromCartesian(cartesian: Cartesian3): GeoPoint | null {
  const cartographic = Cartographic.fromCartesian(cartesian);
  if (!Number.isFinite(cartographic.longitude) || !Number.isFinite(cartographic.latitude)) {
    return null;
  }
  return {
    longitude: CesiumMath.toDegrees(cartographic.longitude),
    latitude: CesiumMath.toDegrees(cartographic.latitude),
    height: cartographic.height,
  };
}

/**
 * Move so `target` sits in the middle of the view. `range` is the distance
 * from that point and `pitch` is the tilt (negative looks down). The flight
 * ends in the world frame so drag, tilt, and orbit keep working afterwards.
 */
function flyToTarget(
  viewer: Viewer,
  longitude: number,
  latitude: number,
  height: number,
  range: number,
  pitch: number,
  duration: number,
  onComplete?: () => void,
): void {
  const start = () => {
    if (
      !Number.isFinite(longitude) ||
      !Number.isFinite(latitude) ||
      !Number.isFinite(range) ||
      !Number.isFinite(pitch)
    ) {
      onComplete?.();
      return;
    }
    try {
    exitOrbit(viewer);
    viewer.camera.cancelFlight();
    const heading = Number.isFinite(viewer.camera.heading) ? viewer.camera.heading : 0;
    // Nadir is a singularity for flyToBoundingSphere's look-at frame.
    if (pitch <= CesiumMath.toRadians(-89)) {
      viewer.camera.flyTo({
        destination: Cartesian3.fromDegrees(longitude, latitude, Math.max(range, 40)),
        orientation: { heading, pitch: CesiumMath.toRadians(-90), roll: 0 },
        duration,
        complete: onComplete,
      });
      return;
    }
    const target = Cartesian3.fromDegrees(longitude, latitude, height);
    viewer.camera.flyToBoundingSphere(new BoundingSphere(target, 1), {
      duration,
      offset: new HeadingPitchRange(heading, pitch, Math.max(range, 40)),
      endTransform: Matrix4.IDENTITY,
      complete: () => {
        if (!viewer.isDestroyed()) viewer.camera.lookAtTransform(Matrix4.IDENTITY);
        onComplete?.();
      },
    });
    } catch {
      onComplete?.();
    }
  };

  if (viewer.scene.mode === SceneMode.MORPHING) {
    const remove = viewer.scene.morphComplete.addEventListener(() => {
      remove();
      start();
    });
    return;
  }
  start();
}

function showTarget(
  viewer: Viewer,
  longitude: number,
  latitude: number,
  height: number,
  range: number,
  pitch: number,
): void {
  exitOrbit(viewer);
  viewer.camera.cancelFlight();
  const target = Cartesian3.fromDegrees(longitude, latitude, height);
  viewer.camera.viewBoundingSphere(
    new BoundingSphere(target, 1),
    new HeadingPitchRange(0, pitch, range),
  );
  viewer.camera.lookAtTransform(Matrix4.IDENTITY);
}

export function flyHome(viewer: Viewer, onComplete?: () => void): void {
  flyToTarget(
    viewer,
    HOME.longitude,
    HOME.latitude,
    0,
    HOME.range,
    HOME.pitch,
    1.4,
    onComplete,
  );
}

export function showHome(viewer: Viewer): void {
  showTarget(viewer, HOME.longitude, HOME.latitude, 0, HOME.range, HOME.pitch);
}

export function flyGlobe(
  viewer: Viewer,
  focus: GeoPoint | null,
  onComplete?: () => void,
): void {
  flyToTarget(
    viewer,
    focus?.longitude ?? HOME.longitude,
    focus?.latitude ?? HOME.latitude,
    0,
    18_000_000,
    CesiumMath.toRadians(-68),
    1.5,
    onComplete,
  );
}

/** Keep the current place framed when the scene drops into the 2D map. */
export function holdOverhead(
  viewer: Viewer,
  longitude: number,
  latitude: number,
  range: number,
): void {
  if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) return;
  const halfLat = CesiumMath.clamp(
    (Number.isFinite(range) ? range : 4_000_000) / 111_000,
    0.04,
    60,
  );
  const halfLon = halfLat / Math.max(Math.cos(CesiumMath.toRadians(latitude)), 0.25);
  let south = latitude - halfLat;
  let north = latitude + halfLat;
  if (south < -85) south = -85;
  if (north > 85) north = 85;
  if (north - south < 0.08) {
    south = latitude - 0.04;
    north = latitude + 0.04;
  }
  let west = longitude - halfLon;
  let east = longitude + halfLon;
  if (east - west > 340) {
    west = -180;
    east = 180;
  }
  exitOrbit(viewer);
  viewer.camera.cancelFlight();
  viewer.camera.setView({
    destination: Rectangle.fromDegrees(west, south, east, north),
  });
}

export function flyTopDown(viewer: Viewer, onComplete?: () => void): void {
  const focus = focusUnderCamera(viewer);
  const height = viewer.camera.positionCartographic?.height;
  const range = Math.min(Math.max(Number.isFinite(height) ? height : 8_000, 1_500), 20_000_000);
  flyToTarget(
    viewer,
    focus.longitude,
    focus.latitude,
    0,
    range,
    CesiumMath.toRadians(-90),
    1.05,
    onComplete,
  );
}

export function flyToPlace(
  viewer: Viewer,
  place: PlaceResult,
  onComplete?: () => void,
): void {
  const span = placeSpan(place);
  const range =
    span === null
      ? 16_000
      : Math.min(Math.max(span * 111_000 * 1.8, 2_000), 14_000_000);
  const pitch =
    span !== null && span > 12 ? CesiumMath.toRadians(-70) : CesiumMath.toRadians(-42);
  flyToTarget(viewer, place.longitude, place.latitude, 0, range, pitch, 1.8, onComplete);
}

export function flyToward(
  viewer: Viewer,
  point: GeoPoint,
  range: number,
  pitch: number,
  onComplete?: () => void,
): void {
  flyToTarget(viewer, point.longitude, point.latitude, 0, range, pitch, 0.9, onComplete);
}

function placeSpan(place: PlaceResult): number | null {
  if (
    place.west === undefined ||
    place.east === undefined ||
    place.south === undefined ||
    place.north === undefined
  ) {
    return null;
  }
  const width = Math.abs(place.east - place.west);
  const height = Math.abs(place.north - place.south);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width === 0 || height === 0) {
    return null;
  }
  return Math.max(width, height);
}
