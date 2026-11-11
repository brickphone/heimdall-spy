"use client";

import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useRef } from "react";

import { mapStyle, token } from "@/lib/mapStyle";

/** A small, non-interactive map of where the reference library has photos. About page only. */
export default function CoverageMap({ area }: { area: [number, number][] }) {
  const element = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let map: import("maplibre-gl").Map | null = null;
    let cancelled = false;
    void import("maplibre-gl").then((ml) => {
      if (cancelled || !element.current) return;
      ml.setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");
      const b = new ml.LngLatBounds();
      area.forEach((p) => b.extend(p));
      map = new ml.Map({
        container: element.current,
        style: mapStyle(),
        bounds: b,
        fitBoundsOptions: { padding: 16 },
        interactive: false,
        attributionControl: false,
      });
      map.addControl(new ml.AttributionControl({ compact: false }), "bottom-right");
      map.on("load", () => {
        map!.addSource("coverage", { type: "geojson", data: "/data/coverage.geojson" });
        map!.addLayer({
          id: "coverage",
          type: "line",
          source: "coverage",
          layout: { "line-cap": "round", "line-join": "round" },
          paint: { "line-color": token("--color-pin"), "line-width": 2, "line-dasharray": [0, 3.2], "line-opacity": 0.55 },
        });
        map!.addSource("area", {
          type: "geojson",
          data: { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [...area, area[0]] } },
        });
        map!.addLayer({ id: "area", type: "line", source: "area", paint: { "line-color": token("--color-ink-2"), "line-width": 1 } });
      });
    });
    return () => {
      cancelled = true;
      map?.remove();
    };
  }, [area]);
  return (
    <div className="relative aspect-[4/3] w-full border border-rule">
      <div ref={element} className="absolute inset-0 h-full w-full" role="img" aria-label="Map of Linköping with dotted lines along streets that have reference photos" />
    </div>
  );
}
