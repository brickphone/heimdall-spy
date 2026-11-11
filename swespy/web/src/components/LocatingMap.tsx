"use client";

import "maplibre-gl/dist/maplibre-gl.css";
import type { GeoJSONSource, Map as MlMap, Marker } from "maplibre-gl";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState, type ReactNode } from "react";

import { IconMinus, IconPlus } from "@/components/Icons";
import { type LatLng, circle, formatCoords, metresPerPixel } from "@/lib/geo";
import { mapStyle, token } from "@/lib/mapStyle";
import { ringRadiiPx } from "@/lib/rings";
import type { CandidatePoint } from "@/lib/stream";

export interface MapItem {
  /** 0 = the prediction; 1..4 = other candidates (rank 2..5). */
  index: number;
  rank: number;
  at: LatLng;
  inliers: number | null;
  label: string;
}

/** Imperative steps of the locating sequence, called as stream events arrive. */
export interface LocatingMapHandle {
  begin(): void;
  candidates(items: CandidatePoint[]): void;
  verifying(top: CandidatePoint[]): void;
  verified(id: number, inliers: number): void;
  result(items: MapItem[], state: string): void;
  cancel(): void;
  fail(): void;
  select(index: number): void;
  reset(): void;
}

interface Props {
  area: [number, number][]; // [lng, lat] polygon of the covered area.
  dimmed: boolean;
  status: ReactNode; // The centre label (progress or error), or null.
  onSelect: (index: number) => void;
  onDropFile: (file: File) => void;
  compact?: boolean;
}

const ROTATE_DEG_PER_S = 4;
const SEARCH_PITCH = 35;
const RESULT_PITCH = 45;
const RESULT_BEARING = -18;
const RESULT_ZOOM = 16;
const DIM_MS = 400;
const CALLOUT_ROOM_PX = 260;
const DIMMED_LABELS = ["district-names", "water-names", "street-names"];

type RingSpec = { at: LatLng; kind: "prediction" | "candidate"; radii: number[]; progress: number[] };

const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const easeInOut = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

/** Animate 0..1 over ``ms`` with rAF; instant under reduced motion. Returns a canceller. */
function tween(ms: number, step: (t: number) => void, done?: () => void): () => void {
  if (ms <= 0 || reducedMotion()) {
    step(1);
    done?.();
    return () => {};
  }
  let raf = 0;
  const start = performance.now();
  const frame = (now: number) => {
    const t = Math.min(1, (now - start) / ms);
    step(t);
    if (t < 1) raf = requestAnimationFrame(frame);
    else done?.();
  };
  raf = requestAnimationFrame(frame);
  return () => cancelAnimationFrame(raf);
}

function arc(at: LatLng, radiusM: number, progress: number): [number, number][] {
  const full = circle(at, radiusM, 72);
  return full.slice(0, Math.max(2, Math.round(full.length * progress)));
}

const LocatingMap = forwardRef<LocatingMapHandle, Props>(function LocatingMap(props, ref) {
  const { area, dimmed, status, onSelect, onDropFile, compact } = props;
  const element = useRef<HTMLDivElement>(null);
  const map = useRef<MlMap | null>(null);
  const lib = useRef<typeof import("maplibre-gl") | null>(null);
  const [ready, setReady] = useState(false);
  const [dragging, setDragging] = useState(false);
  const select = useRef(onSelect);
  const st = useRef({
    rings: [] as RingSpec[],
    dots: [] as { p: CandidatePoint; o: number }[],
    top: [] as CandidatePoint[],
    items: [] as MapItem[],
    markers: [] as Marker[],
    callout: null as Marker | null,
    saved: null as null | { center: [number, number]; zoom: number; pitch: number; bearing: number },
    rotating: 0,
    cancels: [] as (() => void)[],
  });

  useEffect(() => {
    select.current = onSelect;
  }, [onSelect]);

  // ------------------------------------------------------------ drawing helpers

  function drawRings() {
    const m = map.current;
    if (!m) return;
    const features: GeoJSON.Feature[] = [];
    for (const r of st.current.rings) {
      const mpp = metresPerPixel(r.at.lat, m.getZoom());
      r.radii.forEach((px, level) => {
        const progress = r.progress[level] ?? 1;
        if (progress <= 0) return;
        features.push({
          type: "Feature",
          properties: { kind: r.kind, level },
          geometry: { type: "LineString", coordinates: arc(r.at, px * mpp, progress) },
        });
      });
    }
    (m.getSource("rings") as GeoJSONSource | undefined)?.setData({ type: "FeatureCollection", features });
  }

  function drawDots() {
    const m = map.current;
    if (!m) return;
    (m.getSource("cands") as GeoJSONSource | undefined)?.setData({
      type: "FeatureCollection",
      features: st.current.dots.map(({ p, o }) => ({
        type: "Feature",
        properties: { o },
        geometry: { type: "Point", coordinates: [p.lng, p.lat] },
      })),
    });
  }

  function setDim(on: boolean) {
    const m = map.current;
    if (!m) return;
    m.setPaintProperty("dim", "fill-opacity", on ? parseFloat(token("--dim-opacity")) : 0);
    // Labels are barely readable under the dim; skipping their placement keeps the
    // tilted search camera (many new tiles at once) within the frame budget.
    for (const id of DIMMED_LABELS) m.setLayoutProperty(id, "visibility", on ? "none" : "visible");
  }

  function stopRotation() {
    cancelAnimationFrame(st.current.rotating);
    st.current.rotating = 0;
  }

  function startRotation() {
    if (reducedMotion()) return;
    let last = performance.now();
    const frame = (now: number) => {
      const m = map.current;
      if (!m) return;
      if (!m.isMoving()) m.setBearing(m.getBearing() + ((now - last) / 1000) * ROTATE_DEG_PER_S);
      last = now;
      st.current.rotating = requestAnimationFrame(frame);
    };
    st.current.rotating = requestAnimationFrame(frame);
  }

  function clearSequence() {
    st.current.cancels.forEach((c) => c());
    st.current.cancels = [];
    stopRotation();
    st.current.rings = [];
    st.current.dots = [];
    st.current.markers.forEach((mk) => mk.remove());
    st.current.markers = [];
    st.current.callout?.remove();
    st.current.callout = null;
    drawRings();
    drawDots();
  }

  function frameArea(opts: { pitch?: number; duration?: number } = {}) {
    const m = map.current;
    const ml = lib.current;
    if (!m || !ml) return;
    const b = new ml.LngLatBounds();
    area.forEach((p) => b.extend(p));
    const cam = m.cameraForBounds(b, { padding: compact ? 24 : 56 });
    if (!cam) return;
    const target = { ...cam, pitch: opts.pitch ?? 0, bearing: m.getBearing() };
    if (reducedMotion() || !opts.duration) m.jumpTo(target);
    else m.easeTo({ ...target, duration: opts.duration, easing: easeInOut });
  }

  function showCallout(item: MapItem) {
    const m = map.current;
    const ml = lib.current;
    if (!m || !ml) return;
    st.current.callout?.remove();
    const el = document.createElement("div");
    el.className = "map-callout-marker";
    el.setAttribute("aria-hidden", "true");
    el.innerHTML = '<span class="map-callout-leader"></span><span class="map-callout"><span class="map-callout-name"></span><span class="map-callout-coords"></span></span>';
    (el.querySelector(".map-callout-name") as HTMLElement).textContent = item.label;
    (el.querySelector(".map-callout-coords") as HTMLElement).textContent = formatCoords(item.at);
    const mk = new ml.Marker({ element: el, anchor: "bottom-left" }).setLngLat([item.at.lng, item.at.lat]).addTo(m);
    const place = () => {
      const x = m.project([item.at.lng, item.at.lat]).x;
      el.classList.toggle("flip", x > m.getContainer().clientWidth - CALLOUT_ROOM_PX);
    };
    place();
    m.on("move", place);
    requestAnimationFrame(() => el.classList.add("shown"));
    st.current.callout = mk;
    st.current.cancels.push(() => m.off("move", place));
  }

  function markSelected(index: number) {
    st.current.markers.forEach((mk) => {
      const el = mk.getElement();
      el.setAttribute("aria-pressed", String(Number(el.dataset.index) === index));
    });
  }

  // ------------------------------------------------------------ the sequence

  useImperativeHandle(ref, () => ({
    begin() {
      const m = map.current;
      if (!m) return;
      clearSequence();
      const c = m.getCenter();
      st.current.saved = { center: [c.lng, c.lat], zoom: m.getZoom(), pitch: m.getPitch(), bearing: m.getBearing() };
      setDim(true);
      frameArea({ pitch: SEARCH_PITCH, duration: 900 });
      if (!reducedMotion()) m.once("moveend", () => startRotation());
    },

    candidates(items) {
      // Strongest first, staggered over ~600 ms.
      const sorted = [...items].sort((a, b) => b.similarity - a.similarity);
      st.current.dots = sorted.map((p) => ({ p, o: 0 }));
      st.current.cancels.push(
        tween(600 + 250, (t) => {
          const n = st.current.dots.length;
          st.current.dots.forEach((d, i) => {
            const startAt = (i / Math.max(1, n - 1)) * (600 / 850);
            d.o = Math.max(0, Math.min(1, (t - startAt) / (250 / 850)));
          });
          drawDots();
        }),
      );
    },

    verifying(top) {
      const m = map.current;
      const ml = lib.current;
      if (!m || !ml) return;
      stopRotation();
      st.current.top = top;
      const keep = new Set(top.map((p) => p.id));
      // The other candidates fade out; the closest matches get wide dashed rings.
      st.current.cancels.push(
        tween(500, (t) => {
          st.current.dots.forEach((d) => {
            if (!keep.has(d.p.id)) d.o = 1 - t;
          });
          drawDots();
        }, () => {
          st.current.dots = st.current.dots.filter((d) => keep.has(d.p.id));
          drawDots();
        }),
      );
      st.current.rings = top.map((p) => ({
        at: { lat: p.lat, lng: p.lng },
        kind: "candidate",
        radii: ringRadiiPx(0, "candidate"),
        progress: [1, 1, 1],
      }));
      drawRings();
      const b = new ml.LngLatBounds();
      top.forEach((p) => b.extend([p.lng, p.lat]));
      const cam = m.cameraForBounds(b, { padding: compact ? 60 : 160, maxZoom: 15 });
      if (!cam) return;
      if (reducedMotion()) m.jumpTo({ ...cam, pitch: SEARCH_PITCH, bearing: m.getBearing() });
      else m.flyTo({ ...cam, pitch: SEARCH_PITCH, bearing: m.getBearing(), duration: 1200, easing: easeInOut, essential: false });
    },

    verified(id, inliers) {
      const i = st.current.top.findIndex((p) => p.id === id);
      const ring = st.current.rings[i];
      if (!ring) return;
      const from = [...ring.radii];
      const to = ringRadiiPx(inliers, "candidate");
      st.current.cancels.push(
        tween(500, (t) => {
          const e = easeInOut(t);
          ring.radii = from.map((r, k) => r + (to[k] - r) * e);
          drawRings();
        }),
      );
    },

    result(items, state) {
      const m = map.current;
      const ml = lib.current;
      if (!m || !ml || !items.length) return;
      stopRotation();
      setDim(false);
      st.current.dots = [];
      drawDots();
      st.current.items = items;
      const best = items[0];
      // Candidates keep their final dashed rings; the prediction's rings draw in.
      st.current.rings = [
        ...items.slice(1).map((it) => ({
          at: it.at, kind: "candidate" as const, radii: ringRadiiPx(it.inliers, "candidate"), progress: [1, 1, 1],
        })),
        { at: best.at, kind: "prediction" as const, radii: ringRadiiPx(best.inliers, "prediction", state), progress: [0, 0, 0] },
      ];
      drawRings();
      st.current.markers.forEach((mk) => mk.remove());
      st.current.markers = [...items].reverse().map((item) => {
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
        return new ml.Marker({ element: el }).setLngLat([item.at.lng, item.at.lat]).addTo(m);
      });
      markSelected(0);
      const drawIn = () => {
        const pred = st.current.rings[st.current.rings.length - 1];
        // Inner to outer, 300 ms each, staggered by 120 ms.
        st.current.cancels.push(
          tween(300 + 2 * 120, (t) => {
            pred.progress = [0, 1, 2].map((k) => Math.max(0, Math.min(1, (t * 540 - k * 120) / 300)));
            drawRings();
          }),
        );
        showCallout(best);
      };
      const target = { center: [best.at.lng, best.at.lat] as [number, number], zoom: RESULT_ZOOM, pitch: RESULT_PITCH, bearing: RESULT_BEARING };
      if (reducedMotion()) {
        m.jumpTo(target);
        drawIn();
      } else {
        m.once("moveend", drawIn);
        m.flyTo({ ...target, duration: 1400, easing: easeInOut });
      }
    },

    cancel() {
      const m = map.current;
      clearSequence();
      setDim(false);
      const saved = st.current.saved;
      if (m && saved) {
        if (reducedMotion()) m.jumpTo(saved);
        else m.easeTo({ ...saved, duration: 700, easing: easeInOut });
      }
    },

    fail() {
      clearSequence();
      setDim(false);
    },

    select(index) {
      const m = map.current;
      const item = st.current.items.find((i) => i.index === index);
      if (!m || !item) return;
      markSelected(index);
      showCallout(item);
      const target = { center: [item.at.lng, item.at.lat] as [number, number], pitch: RESULT_PITCH };
      if (reducedMotion()) m.jumpTo(target);
      else m.flyTo({ ...target, duration: 800, easing: easeInOut });
    },

    reset() {
      clearSequence();
      setDim(false);
      st.current.items = [];
      const m = map.current;
      if (m) m.jumpTo({ pitch: 0, bearing: 0 });
      frameArea();
    },
  }));

  // ------------------------------------------------------------ map setup

  useEffect(() => {
    let cancelled = false;
    void import("maplibre-gl").then((ml) => {
      if (cancelled || !element.current || map.current) return;
      lib.current = ml;
      ml.setWorkerUrl("/maplibre/maplibre-gl-worker.mjs"); // Copied by scripts/prepare.mjs.
      const b = new ml.LngLatBounds();
      area.forEach((p) => b.extend(p));
      const m = new ml.Map({
        container: element.current,
        style: mapStyle(),
        bounds: b,
        fitBoundsOptions: { padding: compact ? 24 : 56 },
        attributionControl: false,
        maxPitch: 60,
      });
      m.touchZoomRotate.disableRotation();
      m.addControl(new ml.AttributionControl({ compact: false }), "bottom-right");
      m.addControl(new ml.ScaleControl({ unit: "metric", maxWidth: 96 }), "bottom-left");
      m.on("load", () => {
        const world: GeoJSON.Feature = {
          type: "Feature",
          properties: {},
          geometry: { type: "Polygon", coordinates: [[[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]]] },
        };
        m.addSource("dim", { type: "geojson", data: world });
        m.addLayer({
          id: "dim",
          type: "fill",
          source: "dim",
          paint: { "fill-color": token("--color-paper"), "fill-opacity": 0, "fill-opacity-transition": { duration: DIM_MS } },
        });
        m.addSource("area", {
          type: "geojson",
          data: { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [...area, area[0]] } },
        });
        m.addLayer({ id: "area", type: "line", source: "area", paint: { "line-color": token("--color-ink-2"), "line-width": 1 } });
        m.addSource("cands", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
        m.addLayer({
          id: "cands",
          type: "circle",
          source: "cands",
          paint: { "circle-radius": 3, "circle-color": token("--color-ink-2"), "circle-opacity": ["get", "o"] },
        });
        m.addSource("rings", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
        m.addLayer({
          id: "rings-candidate",
          type: "line",
          source: "rings",
          filter: ["==", ["get", "kind"], "candidate"],
          paint: { "line-color": token("--color-ink-2"), "line-width": 1, "line-dasharray": [2, 3], "line-opacity": 0.7 },
        });
        m.addLayer({
          id: "rings-prediction",
          type: "line",
          source: "rings",
          filter: ["==", ["get", "kind"], "prediction"],
          layout: { "line-cap": "round" },
          paint: {
            "line-color": token("--color-pin"),
            "line-width": ["match", ["get", "level"], 0, 1.4, 1, 1.1, 0.9],
            "line-opacity": ["match", ["get", "level"], 0, 1, 1, 0.6, 0.35],
          },
        });
        // Rings keep their on-screen size as the camera zooms.
        m.on("zoom", drawRings);
        map.current = m;
        setReady(true);
      });
    });
    return () => {
      cancelled = true;
      stopRotation();
      map.current?.remove();
      map.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const zoom = (delta: number) =>
    map.current?.easeTo({ zoom: map.current.getZoom() + delta, duration: reducedMotion() ? 0 : 250 });

  return (
    <div
      className="relative h-full w-full bg-paper"
      data-ready={ready}
      onDragEnter={(e) => {
        if (e.dataTransfer.types.includes("Files")) setDragging(true);
      }}
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes("Files")) e.preventDefault();
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files[0];
        if (f) onDropFile(f);
      }}
    >
      {/* MapLibre sets position: relative on this element, so it needs an explicit size. */}
      <div ref={element} className="absolute inset-0 h-full w-full" aria-label="Map of Linköping" role="region" />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 transition-opacity duration-[400ms] motion-reduce:transition-none"
        style={{
          opacity: dimmed ? 1 : 0,
          background: "radial-gradient(ellipse at center, transparent 55%, var(--vignette) 100%)",
        }}
      />
      {!compact && (
        <span className="absolute left-4 top-4 hidden border border-rule bg-surface px-3 py-2 text-sm font-medium text-ink desk:block">
          Linköping
        </span>
      )}
      <div className="absolute right-4 top-4 flex flex-col border border-rule-control bg-surface">
        <button type="button" onClick={() => zoom(1)} aria-label="Zoom in" className="grid size-touch place-items-center text-ink hover:bg-paper">
          <IconPlus />
        </button>
        <span className="h-px bg-rule-control" aria-hidden="true" />
        <button type="button" onClick={() => zoom(-1)} aria-label="Zoom out" className="grid size-touch place-items-center text-ink hover:bg-paper">
          <IconMinus />
        </button>
      </div>
      {status && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center p-4">
          <div className="pointer-events-auto">{status}</div>
        </div>
      )}
      {dragging && (
        <div className="pointer-events-none absolute inset-3 grid place-items-center border border-dashed border-ink bg-[color-mix(in_srgb,var(--color-paper)_70%,transparent)]">
          <span className="border border-ink bg-surface px-4 py-2 text-body font-medium text-ink">Drop to add photo</span>
        </div>
      )}
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
        .map-callout-marker { pointer-events: none; position: relative; width: 0; height: 0; opacity: 0; transition: opacity 300ms; }
        .map-callout-marker.shown { opacity: 1; }
        @media (prefers-reduced-motion: reduce) { .map-callout-marker { transition: none; } }
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
});

export default LocatingMap;
