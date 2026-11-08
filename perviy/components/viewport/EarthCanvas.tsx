"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import type { EarthSession } from "@/cesium/adapter";
import type { SceneController } from "@/scene/controller";
import { useScene } from "@/scene/use-scene";

let webglCache: boolean | undefined;

function readWebgl(): boolean {
  if (webglCache !== undefined) return webglCache;
  try {
    const probe = document.createElement("canvas");
    const context = probe.getContext("webgl2") ?? probe.getContext("webgl");
    webglCache = Boolean(context);
    context?.getExtension("WEBGL_lose_context")?.loseContext();
  } catch {
    webglCache = false;
  }
  return webglCache;
}

function ensureCesiumStylesheet(): void {
  if (document.getElementById("cesium-widgets-css")) return;
  const link = document.createElement("link");
  link.id = "cesium-widgets-css";
  link.rel = "stylesheet";
  link.href = "/cesium/Widgets/widgets.css";
  document.head.appendChild(link);
}

export function EarthCanvas({ controller }: { controller: SceneController }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const creditRef = useRef<HTMLDivElement>(null);
  const state = useScene(controller);
  const webgl = useSyncExternalStore(
    () => () => {},
    readWebgl,
    () => true,
  );
  const [failure, setFailure] = useState<string | null>(null);

  useEffect(() => {
    controller.publish({ webglAvailable: webgl });
    if (!webgl) {
      controller.setStatus("WebGL is unavailable in this browser.", "error");
      return;
    }
    ensureCesiumStylesheet();

    const host = hostRef.current;
    const credits = creditRef.current;
    if (!host || !credits) return;

    const session: { current: EarthSession | null; cancelled: boolean } = {
      current: null,
      cancelled: false,
    };

    void import("@/cesium/adapter")
      .then(({ mountEarthViewer }) => {
        if (session.cancelled) return;
        session.current = mountEarthViewer({
          container: host,
          creditContainer: credits,
          creditViewport: host,
          controller,
        });
      })
      .catch((error: unknown) => {
        if (session.cancelled) return;
        const message = error instanceof Error ? error.message : "The globe failed to start.";
        setFailure(message);
        controller.setStatus(message, "error");
      });

    return () => {
      session.cancelled = true;
      session.current?.destroy();
    };
  }, [controller, webgl]);

  return (
    <>
      <div ref={hostRef} className="earth-host" />
      <div ref={creditRef} className="earth-credits" />
      {webgl ? null : (
        <div className="earth-overlay">
          <div>
            <div className="earth-kicker">HEPHAELION EARTH</div>
            <h1>This browser cannot open the 3D globe.</h1>
            <p>
              WebGL is unavailable or blocked. Earth needs a browser with WebGL enabled.
              Hardware acceleration, a current GPU driver, and allowing WebGL for this site
              are the usual fixes.
            </p>
          </div>
        </div>
      )}
      {failure ? (
        <div className="earth-overlay">
          <div>
            <div className="earth-kicker">HEPHAELION EARTH</div>
            <h1>The globe did not start.</h1>
            <p>{failure}</p>
          </div>
        </div>
      ) : null}
      {webgl && !failure && !state.viewerReady ? (
        <div className="earth-overlay">
          <div>
            <div className="earth-kicker">HEPHAELION EARTH</div>
            <h1>Opening globe</h1>
          </div>
        </div>
      ) : null}
    </>
  );
}
