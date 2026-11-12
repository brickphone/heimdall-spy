"use client";

import "maplibre-gl/dist/maplibre-gl.css";
import type { GeoJSONSource, Map as MlMap, Marker } from "maplibre-gl";
import { type CSSProperties, forwardRef, useEffect, useImperativeHandle, useRef, useState, type ReactNode } from "react";

import { Icon3d, IconCompass, IconMinus, IconPlus } from "@/components/Icons";
import { type LatLng, circle, metresPerPixel } from "@/lib/geo";
import { layoutMarkers } from "@/lib/markers";
import { mapStyle, px, token } from "@/lib/mapStyle";
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

/** Screen space covered by floating panels; the camera keeps results out from under them. */
export interface MapInsets {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

interface Props {
  area: [number, number][]; // [lng, lat] polygon of the covered area.
  status: ReactNode; // The centre label (progress or error), or null.
  insets: MapInsets;
  onSelect: (index: number) => void;
  onCompare: (index: number) => void;
  onDropFile: (file: File) => void;
  compact?: boolean;
}

const ROTATE_DEG_PER_S = 4;
const SEARCH_PITCH = 35;
const RESULT_PITCH = 45;
const RESULT_BEARING = -18;
const RESULT_ZOOM = 16;
// Low confidence: an overview of all candidates, flat and below the 3D-building zoom (15).
const LOW_MAX_ZOOM = 14.5;
const PITCH_3D = 45;
const DIM_MS = 400;
const CALLOUT_ROOM_PX = 220;
const CALLOUT_CHROME_PX = 110; // Leader, padding and the Compare button.
const DIMMED_LABELS = ["district-names", "water-names", "street-names"];

// "candidate": dashed rings while matches are verified; "selected": the solid contour
// rings of the one selected result, cobalt ("sure") or amber ("unsure").
type RingSpec = { at: LatLng; kind: "selected" | "candidate"; tone?: "sure" | "unsure"; radii: number[]; progress: number[] };
type Placed = { item: MapItem; mk: Marker; el: HTMLElement };

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
  const { area, status, insets, onSelect, onCompare, onDropFile, compact } = props;
  const element = useRef<HTMLDivElement>(null);
  const map = useRef<MlMap | null>(null);
  const lib = useRef<typeof import("maplibre-gl") | null>(null);
  const [ready, setReady] = useState(false);
  const [dragging, setDragging] = useState(false);
  const select = useRef(onSelect);
  const compare = useRef(onCompare);
  const compass = useRef<HTMLSpanElement>(null);
  const [tilted, setTilted] = useState(false);
  const st = useRef({
    rings: [] as RingSpec[],
    dots: [] as { p: CandidatePoint; o: number }[],
    top: [] as CandidatePoint[],
    items: [] as MapItem[],
    low: false,
    state: "ok",
    selected: 0,
    markers: [] as Placed[],
    layoutFrame: 0,
    callout: null as Marker | null,
    saved: null as null | { center: [number, number]; zoom: number; pitch: number; bearing: number },
    rotating: 0,
    cancels: [] as (() => void)[],
  });

  useEffect(() => {
    select.current = onSelect;
    compare.current = onCompare;
  }, [onSelect, onCompare]);

  // Keep the camera's idea of the visible area in step with the floating panels.
  // setPadding would stop a running flight (the mobile sheet grows as a result
  // appears), so changes wait for the camera to settle and then ease in.
  useEffect(() => {
    const m = map.current;
    if (!m) return;
    const apply = () => m.easeTo({ padding: insets, duration: reducedMotion() ? 0 : 300 });
    if (m.isMoving()) {
      m.once("moveend", apply);
      return () => {
        m.off("moveend", apply);
      };
    }
    apply();
  }, [insets, ready]);

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
          properties: { kind: r.kind, tone: r.tone ?? "sure", level },
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
    element.current?.removeAttribute("data-result");
    st.current.cancels.forEach((c) => c());
    st.current.cancels = [];
    stopRotation();
    st.current.rings = [];
    st.current.dots = [];
    st.current.markers.forEach(({ mk }) => mk.remove());
    st.current.markers = [];
    st.current.callout?.remove();
    st.current.callout = null;
    drawRings();
    drawDots();
  }

  /** Offset markers that would overlap; a thin leader points back to the true spot. */
  function placeMarkers() {
    const m = map.current;
    if (!m || !st.current.markers.length) return;
    const size = px("--square-size");
    // The open callout is an obstacle too, so it never covers a candidate.
    const box = m.getContainer().getBoundingClientRect();
    const callout = st.current.callout?.getElement().querySelector(".map-callout")?.getBoundingClientRect();
    const blocked = callout
      ? [{ left: callout.left - box.left, top: callout.top - box.top, right: callout.right - box.left, bottom: callout.bottom - box.top }]
      : [];
    const layout = layoutMarkers(
      st.current.markers.map(({ item }) => {
        const pt = m.project([item.at.lng, item.at.lat]);
        return { id: item.index, x: pt.x, y: pt.y, fixed: item.index === 0 };
      }),
      size,
      blocked,
    );
    for (const { item, mk, el } of st.current.markers) {
      const [dx, dy] = layout.get(item.index) ?? [0, 0];
      mk.setOffset([dx, dy]);
      const leader = el.querySelector<HTMLElement>(".map-leader");
      const len = Math.hypot(dx, dy);
      if (leader) {
        leader.style.width = `${Math.max(0, len - size / 2)}px`;
        leader.style.transform = `rotate(${Math.atan2(-dy, -dx)}rad)`;
        leader.hidden = len === 0;
      }
    }
  }

  function schedulePlacement() {
    if (st.current.layoutFrame) return;
    st.current.layoutFrame = requestAnimationFrame(() => {
      st.current.layoutFrame = 0;
      placeMarkers();
    });
  }

  /** Solid contour rings around the selected result only. */
  function selectedRings(index: number, progress: number[]): RingSpec[] {
    const item = st.current.items.find((i) => i.index === index);
    if (!item) return [];
    const tone = st.current.low ? "unsure" : "sure";
    return [{ at: item.at, kind: "selected", tone, radii: ringRadiiPx(item.inliers, "prediction", st.current.state), progress }];
  }

  function frameArea(opts: { pitch?: number; duration?: number } = {}) {
    const m = map.current;
    const ml = lib.current;
    if (!m || !ml) return;
    const b = new ml.LngLatBounds();
    area.forEach((p) => b.extend(p));
    const cam = m.cameraForBounds(b, { padding: compact ? 24 : 40 });
    if (!cam) return;
    const target = { ...cam, pitch: opts.pitch ?? 0, bearing: m.getBearing() };
    if (reducedMotion() || !opts.duration) m.jumpTo(target);
    else m.easeTo({ ...target, duration: opts.duration, easing: easeInOut });
  }

  /** Place name and a Compare button; coordinates live in the panel. */
  function showCallout(item: MapItem) {
    const m = map.current;
    const ml = lib.current;
    if (!m || !ml) return;
    st.current.callout?.remove();
    const el = document.createElement("div");
    el.className = "map-callout-marker";
    el.innerHTML = '<span class="map-callout-leader" aria-hidden="true"></span><span class="map-callout"><span class="map-callout-name"></span><button type="button" class="map-callout-compare">Compare</button></span>';
    (el.querySelector(".map-callout-name") as HTMLElement).textContent = item.label;
    const button = el.querySelector(".map-callout-compare") as HTMLButtonElement;
    button.setAttribute("aria-label", `Compare with ${item.label}`);
    button.addEventListener("click", () => compare.current(item.index));
    const mk = new ml.Marker({ element: el, anchor: "bottom-left" }).setLngLat([item.at.lng, item.at.lat]).addTo(m);
    // Open towards the side with more room; truncate the name if neither side fits.
    const name = el.querySelector(".map-callout-name") as HTMLElement;
    const place = () => {
      const x = m.project([item.at.lng, item.at.lat]).x;
      const p = m.getPadding();
      const right = m.getContainer().clientWidth - (p.right ?? 0) - x;
      const left = x - (p.left ?? 0);
      const flip = right < CALLOUT_ROOM_PX && left > right;
      el.classList.toggle("flip", flip);
      name.style.maxWidth = `${Math.max(80, (flip ? left : right) - CALLOUT_CHROME_PX)}px`;
    };
    place();
    m.on("move", place);
    st.current.callout = mk;
    requestAnimationFrame(() => {
      el.classList.add("shown");
      placeMarkers();
    });
    st.current.cancels.push(() => m.off("move", place));
  }

  function markSelected(index: number) {
    st.current.selected = index;
    st.current.markers.forEach(({ el, item }) => el.setAttribute("aria-pressed", String(item.index === index)));
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
      const low = state !== "ok";
      st.current.low = low;
      st.current.state = state;
      // Only the selected result (the prediction, at first) gets contour rings.
      st.current.rings = selectedRings(0, [0, 0, 0]);
      drawRings();
      st.current.markers.forEach(({ mk }) => mk.remove());
      st.current.markers = [...items].reverse().map((item) => {
        const el = document.createElement("button");
        el.type = "button";
        el.dataset.index = String(item.index);
        el.addEventListener("click", () => select.current(item.index));
        if (item.index === 0) {
          el.className = low ? "map-pin unsure" : "map-pin";
          el.setAttribute("aria-label", `Best match: ${item.label}`);
        } else {
          el.className = "map-square";
          el.innerHTML = '<span class="map-leader" hidden></span><span class="map-square-label"></span>';
          (el.querySelector(".map-square-label") as HTMLElement).textContent = String(item.rank);
          el.setAttribute("aria-label", `Candidate ${item.rank}: ${item.label}`);
        }
        const mk = new ml.Marker({ element: el }).setLngLat([item.at.lng, item.at.lat]).addTo(m);
        return { item, mk, el };
      });
      markSelected(0);
      placeMarkers();
      const drawIn = () => {
        // The settled camera, for the end-to-end checks (scripts/e2e-regions.mjs).
        element.current?.setAttribute("data-result", `${low ? "low" : "confident"} pitch=${Math.round(m.getPitch())} zoom=${m.getZoom().toFixed(1)}`);
        const pred = st.current.rings[0];
        // Inner to outer, 300 ms each, staggered by 120 ms.
        if (pred) {
          st.current.cancels.push(
            tween(300 + 2 * 120, (t) => {
              pred.progress = [0, 1, 2].map((k) => Math.max(0, Math.min(1, (t * 540 - k * 120) / 300)));
              drawRings();
            }),
          );
        }
        showCallout(best);
      };
      let target: { center: [number, number]; zoom: number; pitch: number; bearing: number } = {
        center: [best.at.lng, best.at.lat], zoom: RESULT_ZOOM, pitch: RESULT_PITCH, bearing: RESULT_BEARING,
      };
      if (low) {
        // No close-up and no 3D: fit all candidates, flat and north-up.
        const b = new ml.LngLatBounds();
        items.forEach((it) => b.extend([it.at.lng, it.at.lat]));
        const cam = m.cameraForBounds(b, { padding: compact ? 48 : 96, maxZoom: LOW_MAX_ZOOM });
        const c = cam?.center ? ml.LngLat.convert(cam.center) : null;
        target = { center: c ? [c.lng, c.lat] : target.center, zoom: cam?.zoom ?? LOW_MAX_ZOOM, pitch: 0, bearing: 0 };
      }
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
      st.current.rings = selectedRings(index, [1, 1, 1]);
      drawRings();
      showCallout(item);
      const target = { center: [item.at.lng, item.at.lat] as [number, number], pitch: st.current.low ? 0 : RESULT_PITCH };
      if (reducedMotion()) m.jumpTo(target);
      else m.flyTo({ ...target, duration: 800, easing: easeInOut });
    },

    reset() {
      clearSequence();
      setDim(false);
      st.current.items = [];
      st.current.low = false;
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
        fitBoundsOptions: { padding: compact ? 24 : 40 },
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
          id: "rings-selected",
          type: "line",
          source: "rings",
          filter: ["==", ["get", "kind"], "selected"],
          layout: { "line-cap": "round" },
          paint: {
            "line-color": ["match", ["get", "tone"], "unsure", token("--color-unsure"), token("--color-pin")],
            "line-width": px("--ring-width"),
            "line-opacity": ["match", ["get", "level"], 0, 1, 1, 0.7, 0.45],
          },
        });
        // Rings keep their on-screen size as the camera zooms; markers re-flow as it moves.
        m.on("zoom", drawRings);
        m.on("move", schedulePlacement);
        const turn = () => {
          if (compass.current) compass.current.style.transform = `rotate(${-m.getBearing()}deg)`;
        };
        m.on("rotate", turn);
        m.on("pitchend", () => setTilted(m.getPitch() > 1));
        m.setPadding(insets);
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
  const north = () => map.current?.easeTo({ bearing: 0, duration: reducedMotion() ? 0 : 400 });
  const toggle3d = () => {
    const m = map.current;
    if (!m) return;
    const next = m.getPitch() > 1 ? 0 : PITCH_3D;
    setTilted(next > 0);
    m.easeTo({ pitch: next, duration: reducedMotion() ? 0 : 500 });
  };
  // The part of the map not covered by floating panels.
  const visible = { left: insets.left, right: insets.right, top: insets.top, bottom: insets.bottom };

  return (
    <div
      className="relative h-full w-full bg-paper"
      data-ready={ready}
      style={{ "--map-inset-left": `${insets.left}px`, "--map-inset-right": `${insets.right}px`, "--map-inset-bottom": `${insets.bottom}px` } as CSSProperties}
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
        className="map-controls absolute flex flex-col overflow-hidden rounded-panel border border-rule bg-surface"
        style={{ right: insets.right, bottom: insets.bottom + 28 }}
      >
        <MapButton label="Zoom in" onClick={() => zoom(1)}>
          <IconPlus />
        </MapButton>
        <MapButton label="Zoom out" onClick={() => zoom(-1)}>
          <IconMinus />
        </MapButton>
        <MapButton label="Reset to north" onClick={north}>
          <span ref={compass} className="grid place-items-center">
            <IconCompass />
          </span>
        </MapButton>
        <MapButton label="3D view" pressed={tilted} onClick={toggle3d}>
          <Icon3d />
        </MapButton>
      </div>
      {status && (
        <div className="pointer-events-none absolute grid place-items-center p-4" style={visible}>
          <div className="pointer-events-auto">{status}</div>
        </div>
      )}
      {dragging && (
        <div className="pointer-events-none absolute grid place-items-center rounded-panel border border-dashed border-ink" style={visible}>
          <span className="rounded-control border border-ink bg-surface px-3 py-1.5 text-sm font-medium text-ink">Drop to add photo</span>
        </div>
      )}
      <style>{`
        .maplibregl-ctrl-bottom-right { right: var(--map-inset-right); bottom: var(--map-inset-bottom); }
        .maplibregl-ctrl-bottom-left { left: var(--map-inset-left); bottom: var(--map-inset-bottom); }
        .map-pin {
          width: var(--pin-size); height: var(--pin-size); border-radius: var(--radius-dot);
          background: var(--color-pin); box-shadow: 0 0 0 var(--pin-halo) var(--color-surface);
          border: 0; padding: 0; cursor: pointer;
        }
        .map-pin.unsure { background: var(--color-unsure); }
        .map-pin[aria-pressed="true"] { outline: var(--border) solid var(--color-ink); outline-offset: var(--pin-halo); }
        .map-square {
          width: var(--square-size); height: var(--square-size); border-radius: var(--radius-square);
          background: var(--color-surface); border: var(--square-border) solid var(--color-ink);
          color: var(--color-ink); font: 600 var(--text-map)/1 var(--font-sans);
          font-feature-settings: "tnum"; display: grid; place-items: center; padding: 0; cursor: pointer;
        }
        .map-square[aria-pressed="true"] { border-color: var(--color-pin); color: var(--color-pin); box-shadow: 0 0 0 var(--border) var(--color-pin); }
        .map-leader {
          position: absolute; left: 50%; top: 50%; height: var(--leader-width); background: var(--color-ink);
          transform-origin: 0 50%; pointer-events: none;
        }
        .map-square-label { position: relative; }
        .map-callout-marker { pointer-events: none; position: relative; width: 0; height: 0; opacity: 0; transition: opacity 300ms; }
        .map-callout-marker.shown { opacity: 1; }
        @media (prefers-reduced-motion: reduce) { .map-callout-marker { transition: none; } }
        .map-callout-leader {
          position: absolute; left: 0; bottom: 0; width: 28px; height: var(--border);
          background: var(--color-ink); transform-origin: left bottom; transform: rotate(-38deg);
        }
        .map-callout {
          position: absolute; left: 22px; bottom: 17px; white-space: nowrap; pointer-events: auto;
          background: var(--color-surface); border: var(--border) solid var(--color-ink); border-radius: var(--radius-control);
          padding: 4px 4px 4px 8px; display: flex; align-items: center; gap: 8px;
        }
        .map-callout-marker.flip .map-callout-leader { left: auto; right: 0; transform-origin: right bottom; transform: rotate(38deg); }
        .map-callout-marker.flip .map-callout { left: auto; right: 22px; }
        .map-callout-name { font: 600 var(--text-sm)/1.25 var(--font-sans); color: var(--color-ink); overflow: hidden; text-overflow: ellipsis; }
        .map-callout-compare {
          font: 500 var(--text-xs)/1 var(--font-sans); color: var(--color-pin); background: none; border: 0;
          padding: 5px 6px; border-radius: var(--radius-square); cursor: pointer;
        }
        .map-callout-compare:hover { background: var(--color-selected); color: var(--color-pin-hover); }
      `}</style>
    </div>
  );
});

function MapButton({ label, pressed, onClick, children }: { label: string; pressed?: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      aria-pressed={pressed}
      onClick={onClick}
      className="grid size-control place-items-center border-b border-rule-list text-ink last:border-b-0 hover:bg-paper aria-pressed:text-pin"
    >
      {children}
    </button>
  );
}

export default LocatingMap;
