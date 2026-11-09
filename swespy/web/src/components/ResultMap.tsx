"use client";

import "leaflet/dist/leaflet.css";
import type { LayerGroup, Map as LeafletMap, Marker } from "leaflet";
import { useEffect, useRef } from "react";

import type { LatLng, Prediction } from "@/lib/api";

const LINKOPING: [number, number] = [58.41, 15.62];

interface Props {
  prediction: Prediction;
  /** When set, clicking the map places (or moves) the true-location pin. */
  onPickTruth?: (point: LatLng) => void;
  truth: LatLng | null;
}

// Plain Leaflet (BSD-2). react-leaflet is avoided: its Hippocratic licence is not OSI.
export default function ResultMap({ prediction, onPickTruth, truth }: Props) {
  const element = useRef<HTMLDivElement>(null);
  const map = useRef<LeafletMap | null>(null);
  const pins = useRef<LayerGroup | null>(null);
  const truthPin = useRef<Marker | null>(null);
  const pick = useRef(onPickTruth);
  const leaflet = useRef<typeof import("leaflet") | null>(null);

  useEffect(() => {
    pick.current = onPickTruth;
  }, [onPickTruth]);

  // Create the map once. Leaflet touches `window`, so it is loaded in the browser only.
  useEffect(() => {
    let cancelled = false;
    void import("leaflet").then((L) => {
      if (cancelled || !element.current || map.current) return;
      leaflet.current = L;
      const m = L.map(element.current).setView(LINKOPING, 12);
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "© OpenStreetMap contributors",
      }).addTo(m);
      pins.current = L.layerGroup().addTo(m);
      m.on("click", (event) => pick.current?.(event.latlng));
      map.current = m;
      draw();
    });
    return () => {
      cancelled = true;
      map.current?.remove();
      map.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function draw() {
    const L = leaflet.current;
    const m = map.current;
    if (!L || !m || !pins.current) return;
    pins.current.clearLayers();
    const main: [number, number] = [prediction.predicted_lat, prediction.predicted_lng];
    const bounds: [number, number][] = [main];
    // Candidates 2-5 first, so the main pin is drawn on top.
    for (const match of prediction.top_matches.slice(1)) {
      L.circleMarker([match.lat, match.lng], {
        radius: 6,
        weight: 2,
        color: "#c2410c",
        fillColor: "#fb923c",
        fillOpacity: 0.6,
      })
        .bindTooltip(`Candidate ${match.rank} · ${match.inliers ?? "–"} inliers`)
        .addTo(pins.current);
      bounds.push([match.lat, match.lng]);
    }
    L.marker(main, { icon: pinIcon(L, "predicted"), zIndexOffset: 1000 })
      .bindTooltip("Predicted location", { permanent: true, direction: "top", offset: [0, -30] })
      .addTo(pins.current);
    m.fitBounds(bounds, { padding: [48, 48], maxZoom: 17 });
  }

  useEffect(draw, [prediction]);

  // True-location pin (draggable), only while the user is picking it.
  useEffect(() => {
    const L = leaflet.current;
    const m = map.current;
    if (!L || !m) return;
    if (!truth) {
      truthPin.current?.remove();
      truthPin.current = null;
      return;
    }
    if (truthPin.current) {
      truthPin.current.setLatLng(truth);
      return;
    }
    const marker = L.marker(truth, { icon: pinIcon(L, "truth"), draggable: true })
      .bindTooltip("True location", { direction: "top", offset: [0, -30] })
      .addTo(m);
    marker.on("dragend", () => pick.current?.(marker.getLatLng()));
    truthPin.current = marker;
  }, [truth]);

  // Leaflet adds its own classes to the map element, so React must never re-render its
  // className: anything that changes lives on the wrapper.
  return (
    <div className={onPickTruth ? "[&_.leaflet-container]:cursor-crosshair" : ""}>
      <div
        ref={element}
        className="h-[440px] w-full rounded-xl"
        aria-label="Map of the predicted location and candidates"
      />
    </div>
  );
}

// Inline SVG pins: Leaflet's default PNG icons break under bundlers.
function pinIcon(L: typeof import("leaflet"), kind: "predicted" | "truth") {
  const fill = kind === "predicted" ? "#2563eb" : "#16a34a";
  return L.divIcon({
    className: "",
    iconSize: [28, 40],
    iconAnchor: [14, 40],
    html: `<svg width="28" height="40" viewBox="0 0 28 40" xmlns="http://www.w3.org/2000/svg">
      <path d="M14 0C6.3 0 0 6.3 0 14c0 10.5 14 26 14 26s14-15.5 14-26C28 6.3 21.7 0 14 0z"
        fill="${fill}" stroke="white" stroke-width="2"/>
      <circle cx="14" cy="14" r="5" fill="white"/></svg>`,
  });
}
