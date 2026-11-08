"use client";

import { useEffect, useRef, useState } from "react";
import { formatExaggeration } from "@/geo/coordinates";
import type { SceneController } from "@/scene/controller";
import type { CameraMode, PlaceResult } from "@/scene/types";
import { useScene } from "@/scene/use-scene";

const CAMERA_MODES: Array<{ id: CameraMode; label: string }> = [
  { id: "free", label: "Free" },
  { id: "globe", label: "Globe" },
  { id: "orbit", label: "Orbit" },
  { id: "top-down", label: "Top-down" },
];

export function TopNav({ controller }: { controller: SceneController }) {
  const state = useScene(controller);
  const actionsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (actionsRef.current?.contains(target)) return;
      controller.setSceneMenuOpen(false);
      controller.setSettingsOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [controller]);

  return (
    <header className="earth-nav">
      <div className="earth-wordmark">
        <strong>HEPHAELION</strong>
        <span>EARTH ENGINE</span>
      </div>
      <SearchBox controller={controller} />
      <div className="earth-nav-actions" ref={actionsRef}>
        <button
          type="button"
          className="earth-text-button"
          aria-expanded={state.sceneMenuOpen}
          onClick={() => controller.setSceneMenuOpen(!state.sceneMenuOpen)}
        >
          Scene
        </button>
        <button
          type="button"
          className="earth-text-button"
          aria-expanded={state.settingsOpen}
          onClick={() => controller.setSettingsOpen(!state.settingsOpen)}
        >
          Settings
        </button>
        {state.sceneMenuOpen ? (
          <div className="earth-popover" role="menu">
            <h2>Camera</h2>
            <div className="earth-choice-row">
              {CAMERA_MODES.map((mode) => (
                <button
                  key={mode.id}
                  type="button"
                  className="earth-mode"
                  aria-pressed={state.cameraMode === mode.id}
                  onClick={() => controller.setCameraMode(mode.id)}
                >
                  {mode.label}
                </button>
              ))}
              <button
                type="button"
                className="earth-mode"
                aria-pressed={state.cameraMode === "fly-to"}
                onClick={() => controller.repeatFlyTo()}
              >
                Fly-to
              </button>
            </div>
            <div className="earth-choice-row">
              <button
                type="button"
                className="earth-mode"
                aria-pressed={state.scene2d}
                onClick={() => controller.setScene2d(!state.scene2d)}
              >
                {state.scene2d ? "2D map on" : "2D map"}
              </button>
              <button type="button" className="earth-reset" onClick={() => controller.resetCamera()}>
                Reset
              </button>
            </div>
            <p className="earth-hint">
              Drag to rotate or pan. Scroll to zoom. Alt-drag or Ctrl-drag to tilt.
              Double-click a point to fly toward it. Orbit circles the last focus.
            </p>
          </div>
        ) : null}
        {state.settingsOpen ? (
          <div className="earth-popover">
            <h2>Display</h2>
            <div className="earth-choice-row">
              <button
                type="button"
                className="earth-mode"
                aria-pressed={state.displayMode === "analytical"}
                onClick={() => controller.setDisplayMode("analytical")}
              >
                Analytical
              </button>
              <button
                type="button"
                className="earth-mode"
                aria-pressed={state.displayMode === "cinematic"}
                onClick={() => controller.setDisplayMode("cinematic")}
              >
                Cinematic
              </button>
            </div>
            <p>
              Analytical turns off atmosphere and fog. Cinematic keeps Cesium&apos;s standard
              atmosphere. Water, buildings, and time are not in this foundation.
            </p>
            <h2 style={{ marginTop: 16 }}>Cesium ion</h2>
            <p>
              {state.ionConfigured
                ? "A token is configured. World terrain is requested from Cesium ion."
                : "No token is configured. Terrain detail is limited to the ellipsoid until NEXT_PUBLIC_CESIUM_ION_TOKEN is set. See .env.example. Do not commit a token."}
            </p>
            {state.terrainExaggeration !== 1 ? (
              <p>Terrain exaggeration is {formatExaggeration(state.terrainExaggeration)}.</p>
            ) : null}
          </div>
        ) : null}
      </div>
    </header>
  );
}

function SearchBox({ controller }: { controller: SceneController }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<PlaceResult[]>([]);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLFormElement>(null);
  const chosenQuery = useRef<string | null>(null);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) return;
    if (chosenQuery.current === trimmed) return;
    const ac = new AbortController();
    const timer = window.setTimeout(() => {
      setBusy(true);
      void controller.search(trimmed, ac.signal).then((outcome) => {
        if (outcome.aborted) return;
        setBusy(false);
        setResults(outcome.results);
        setError(outcome.error ?? null);
        setActive(0);
        setOpen(true);
      });
    }, 420);
    return () => {
      window.clearTimeout(timer);
      ac.abort();
    };
  }, [controller, query]);

  useEffect(() => {
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (target instanceof Node && rootRef.current?.contains(target)) return;
      setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, []);

  const choose = (index: number) => {
    const place = results[index];
    if (!place) return;
    controller.flyToPlace(place);
    chosenQuery.current = place.label;
    setQuery(place.label);
    setOpen(false);
  };

  return (
    <form
      className="earth-search"
      ref={rootRef}
      role="search"
      onSubmit={(event) => {
        event.preventDefault();
        choose(active);
      }}
    >
      <input
        value={query}
        role="combobox"
        aria-label="Search the world"
        aria-autocomplete="list"
        aria-expanded={open}
        aria-controls="earth-search-results"
        placeholder="Search the world"
        onChange={(event) => {
          const next = event.target.value;
          if (chosenQuery.current !== next.trim()) chosenQuery.current = null;
          setQuery(next);
          if (next.trim().length < 2) {
            setResults([]);
            setError(null);
            setBusy(false);
            setOpen(false);
            return;
          }
          setOpen(true);
        }}
        onFocus={() => {
          if (query.trim().length >= 2) setOpen(true);
        }}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            setOpen(false);
            return;
          }
          if (!open || results.length === 0) return;
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setActive((value) => (value + 1) % results.length);
          }
          if (event.key === "ArrowUp") {
            event.preventDefault();
            setActive((value) => (value - 1 + results.length) % results.length);
          }
        }}
      />
      {open ? (
        <ul className="earth-results" id="earth-search-results" role="listbox">
          {busy && results.length === 0 ? <li className="is-muted">Searching…</li> : null}
          {error ? <li className="is-muted">{error}</li> : null}
          {!busy && !error && results.length === 0 && query.trim().length >= 2 ? (
            <li className="is-muted">No matches.</li>
          ) : null}
          {results.map((place, index) => (
            <li key={place.id}>
              <button
                type="button"
                role="option"
                aria-selected={index === active}
                onMouseEnter={() => setActive(index)}
                onClick={() => choose(index)}
              >
                {place.label}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </form>
  );
}
