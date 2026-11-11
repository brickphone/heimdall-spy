"use client";

import "maplibre-gl/dist/maplibre-gl.css";
import type { GeoJSONSource, Map as MlMap, Marker } from "maplibre-gl";
import * as Toggle from "@radix-ui/react-toggle";
import { useEffect, useRef, useState } from "react";

import { IconMinus, IconPlus } from "@/components/Icons";
import type { Prediction } from "@/lib/api";
import { type LatLng, circle, formatCoords, metresPerPixel } from "@/lib/geo";
import { mapStyle, token } from "@/lib/mapStyle";
import { RING_ZOOM, ringRadiiPx } from "@/lib/rings";

const LINKOPING: [number, number] = [15.6214, 58.4108];
const START_ZOOM = 12;
const CALLOUT_ROOM_PX = 260; // Space the callout needs to the right of its point.

export interface MapItem {
  /** 0 = the prediction; 1..4 = other candidates (rank 2..5). */
  index: number;
  rank: number;
  at: LatLng;
  inliers: number | null;
  label: string;
}

interface Props {
  prediction: Prediction | null;
  items: MapItem[];
  selected: number;
  onSelect: (index: number) => void;
  coverage: boolean;
  onCoverage: (on: boolean) => void;
  compact?: boolean;
}

function reducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function ringFeatures(items: MapItem[], state: string): GeoJSON.FeatureCollection {
  const features: GeoJSON.Feature[] = [];
  for (const item of items) {
    const kind = item.index === 0 ? "prediction" : "candidate";
    const mpp = metresPerPixel(item.at.lat, RING_ZOOM);
    ringRadiiPx(item.inliers, kind, state).forEach((rPx, level) => {
      features.push({
        type: "Feature",
        properties: { kind, level },
        geometry: { type: "LineString", coordinates: circle(item.at, rPx * mpp) },
      });
    });
  }
  return { type: "FeatureCollection", features };
}

export default function MapView(props: Props) {
  const { prediction, items, selected, onSelect, coverage, onCoverage, compact } = props;
  const element = useRef<HTMLDivElement>(null);
  const map = useRef<MlMap | null>(null);
  const markers = useRef<Marker[]>([]);
  const lib = useRef<typeof import("maplibre-gl") | null>(null);
  const [ready, setReady] = useState(false);
  const select = useRef(onSelect);

  useEffect(() => {
    select.current = onSelect;
  }, [onSelect]);

  // Create the map once (MapLibre needs the browser).
  useEffect(() => {
    let cancelled = false;
    void import("maplibre-gl").then((ml) => {
      if (cancelled || !element.current || map.current) return;
      lib.current = ml;
      ml.setWorkerUrl("/maplibre/maplibre-gl-worker.mjs"); // Copied by scripts/prepare.mjs.
      const m = new ml.Map({
        container: element.current,
        style: mapStyle(),
        center: LINKOPING,
        zoom: START_ZOOM,
        attributionControl: false,
        dragRotate: false,
        pitchWithRotate: false,
      });
      m.touchZoomRotate.disableRotation();
      m.addControl(new ml.AttributionControl({ compact: false }), "bottom-right");
      m.addControl(new ml.ScaleControl({ unit: "metric", maxWidth: 96 }), "bottom-left");
      m.on("load", () => {
        m.addSource("coverage", { type: "geojson", data: "/data/coverage.geojson" });
        m.addLayer({
          id: "coverage",
          type: "line",
          source: "coverage",
          layout: { "line-cap": "round", "line-join": "round", visibility: "none" },
          paint: {
            "line-color": token("--color-pin"),
            "line-width": 2.2,
            "line-dasharray": [0, 3.2],
            "line-opacity": 0.45,
          },
        });
        m.addSource("rings", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
        m.addLayer({
          id: "rings-candidate",
          type: "line",
          source: "rings",
          filter: ["==", ["get", "kind"], "candidate"],
          paint: {
            "line-color": token("--color-ink-2"),
            "line-width": 1,
            "line-dasharray": [2, 3],
            "line-opacity": 0.7,
          },
        });
        m.addLayer({
          id: "rings-prediction",
          type: "line",
          source: "rings",
          filter: ["==", ["get", "kind"], "prediction"],
          paint: {
            "line-color": token("--color-pin"),
            "line-width": ["match", ["get", "level"], 0, 1.4, 1, 1.1, 0.9],
            "line-opacity": ["match", ["get", "level"], 0, 1, 1, 0.6, 0.35],
          },
        });
        map.current = m;
        setReady(true);
      });
    });
    return () => {
      cancelled = true;
      map.current?.remove();
      map.current = null;
    };
  }, []);

  useEffect(() => {
    if (!ready || !map.current) return;
    map.current.setLayoutProperty("coverage", "visibility", coverage ? "visible" : "none");
  }, [coverage, ready]);

  // Rings, markers and the callout for the current result.
  useEffect(() => {
    const m = map.current;
    const ml = lib.current;
    if (!ready || !m || !ml) return;
    (m.getSource("rings") as GeoJSONSource).setData(
      ringFeatures(items, prediction?.confidence_state ?? "ok"),
    );
    markers.current.forEach((mk) => mk.remove());
    markers.current = [];
    for (const item of [...items].reverse()) {
      const el = document.createElement("button");
      el.type = "button";
      el.dataset.index = String(item.index);
      el.addEventListener("click", () => select.current(item.index));
      if (item.index === 0) {
        el.className = "map-pin";
        el.setAttribute("aria-label", `Best match: ${item.label}`);
      } else {
        el.className = "map-square";
        el.textContent = String(item.rank);
        el.setAttribute("aria-label", `Candidate ${item.rank}: ${item.label}`);
      }
      markers.current.push(new ml.Marker({ element: el }).setLngLat([item.at.lng, item.at.lat]).addTo(m));
    }
    if (items.length) {
      const b = new ml.LngLatBounds();
      items.forEach((i) => b.extend([i.at.lng, i.at.lat]));
      m.fitBounds(b, {
        padding: compact ? 48 : { top: 140, bottom: 80, left: 80, right: 80 },
        maxZoom: 15,
        duration: reducedMotion() ? 0 : 600,
      });
    }
  }, [items, prediction, ready, compact]);

  // Selection: highlight, callout on the selected item, ease the map to it.
  useEffect(() => {
    const m = map.current;
    const ml = lib.current;
    if (!ready || !m || !ml) return;
    markers.current.forEach((mk) => {
      const el = mk.getElement();
      el.setAttribute("aria-pressed", String(Number(el.dataset.index) === selected));
    });
    const item = items.find((i) => i.index === selected);
    const old = document.querySelector(".map-callout-marker");
    old?.remove();
    if (!item) return;
    const callout = document.createElement("div");
    callout.className = "map-callout-marker";
    callout.setAttribute("aria-hidden", "true");
    const leader = document.createElement("span");
    leader.className = "map-callout-leader";
    const box = document.createElement("span");
    box.className = "map-callout";
    const name = document.createElement("span");
    name.className = "map-callout-name";
    name.textContent = item.label;
    const coords = document.createElement("span");
    coords.className = "map-callout-coords";
    coords.textContent = formatCoords(item.at);
    box.append(name, coords);
    callout.append(leader, box);
    const mk = new ml.Marker({ element: callout, anchor: "bottom-left" })
      .setLngLat([item.at.lng, item.at.lat])
      .addTo(m);
    // Keep the callout on the map: flip it to the left near the right edge.
    const place = () => {
      const x = m.project([item.at.lng, item.at.lat]).x;
      callout.classList.toggle("flip", x > m.getContainer().clientWidth - CALLOUT_ROOM_PX);
    };
    place();
    m.on("move", place);
    if (selected !== 0) {
      m.easeTo({ center: [item.at.lng, item.at.lat], duration: reducedMotion() ? 0 : 450 });
    }
    return () => {
      m.off("move", place);
      mk.remove();
    };
  }, [selected, items, ready]);

  const zoom = (delta: number) =>
    map.current?.easeTo({ zoom: map.current.getZoom() + delta, duration: reducedMotion() ? 0 : 250 });

  return (
    <div className="relative h-full w-full bg-paper">
      {/* MapLibre sets position: relative on this element, so it needs an explicit size. */}
      <div ref={element} className="absolute inset-0 h-full w-full" aria-label="Map of Linköping" role="region" />
      {!compact && (
        <div className="absolute left-4 top-4 hidden items-center gap-3 desk:flex">
          <span className="border border-rule bg-surface px-3 py-2 text-sm font-medium text-ink">
            Linköping
          </span>
          <Toggle.Root
            pressed={coverage}
            onPressedChange={onCoverage}
            className="rounded-control border border-rule-control bg-surface px-3 py-2 text-sm text-ink data-[state=on]:border-ink data-[state=on]:bg-ink data-[state=on]:text-surface"
          >
            Show reference coverage
          </Toggle.Root>
        </div>
      )}
      <div className="absolute right-4 top-4 flex flex-col border border-rule-control bg-surface">
        <button
          type="button"
          onClick={() => zoom(1)}
          aria-label="Zoom in"
          className="grid size-touch place-items-center text-ink hover:bg-paper"
        >
          <IconPlus />
        </button>
        <span className="h-px bg-rule-control" aria-hidden="true" />
        <button
          type="button"
          onClick={() => zoom(-1)}
          aria-label="Zoom out"
          className="grid size-touch place-items-center text-ink hover:bg-paper"
        >
          <IconMinus />
        </button>
      </div>
      <style>{`
        .map-pin {
          width: var(--pin-size); height: var(--pin-size); border-radius: var(--radius-dot);
          background: var(--color-pin); box-shadow: 0 0 0 var(--pin-halo) var(--color-surface);
          border: 0; padding: 0; cursor: pointer;
        }
        .map-pin[aria-pressed="true"] { outline: var(--border) solid var(--color-ink); outline-offset: var(--pin-halo); }
        .map-square {
          width: var(--square-size); height: var(--square-size); border-radius: var(--radius-square);
          background: var(--color-surface); border: var(--ring-2) solid var(--color-ink);
          color: var(--color-ink); font: 600 var(--text-map)/1 var(--font-sans);
          font-feature-settings: "tnum"; display: grid; place-items: center; padding: 0; cursor: pointer;
        }
        .map-square[aria-pressed="true"] { background: var(--color-selected); border-color: var(--color-pin); color: var(--color-pin); }
        .map-callout-marker { pointer-events: none; position: relative; width: 0; height: 0; }
        .map-callout-leader {
          position: absolute; left: 0; bottom: 0; width: 34px; height: var(--border);
          background: var(--color-ink); transform-origin: left bottom; transform: rotate(-38deg);
        }
        .map-callout {
          position: absolute; left: 26px; bottom: 20px; white-space: nowrap;
          background: var(--color-surface); border: var(--callout-border) solid var(--color-ink);
          padding: 6px 9px; display: flex; flex-direction: column; gap: 2px;
        }
        .map-callout-marker.flip .map-callout-leader { left: auto; right: 0; transform-origin: right bottom; transform: rotate(38deg); }
        .map-callout-marker.flip .map-callout { left: auto; right: 26px; }
        .map-callout-name { font: 600 var(--text-control)/1.25 var(--font-sans); color: var(--color-ink); }
        .map-callout-coords { font: 400 var(--text-xs)/1.25 var(--font-sans); color: var(--color-ink-2); font-feature-settings: "tnum"; }
      `}</style>
    </div>
  );
}
