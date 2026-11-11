"use client";

/* eslint-disable @next/next/no-img-element -- the photo is a local blob URL */

import { type PointerEvent, type KeyboardEvent, useRef, useState } from "react";

import type { Region } from "@/lib/api";

const MIN = 0.08; // Smallest region side, as a fraction of the photo (API minimum is 0.05).
const STEP = 0.02;

interface Props {
  src: string | null;
  region: Region | null;
  editing: boolean;
  onChange: (region: Region) => void; // Called when a drag or key press finishes.
  onUnavailable: () => void;
}

interface Box {
  scale: number;
  offX: number;
  offY: number;
  natW: number;
  natH: number;
  boxW: number;
  boxH: number;
}

type Drag = { mode: "move" | "new" | "n" | "s" | "e" | "w"; x0: number; y0: number; start: Region };

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** The part of the photo visible in the cover-fitted box, in normalised photo coords. */
function visible(b: Box): Region {
  const x = -b.offX / (b.natW * b.scale);
  const y = -b.offY / (b.natH * b.scale);
  return { x, y, width: 1 - 2 * x, height: 1 - 2 * y };
}

function fit(r: Region, v: Region): Region {
  const width = clamp(r.width, MIN, v.width);
  const height = clamp(r.height, MIN, v.height);
  return {
    x: clamp(r.x, v.x, v.x + v.width - width),
    y: clamp(r.y, v.y, v.y + v.height - height),
    width,
    height,
  };
}

export default function PhotoRegion({ src, region, editing, onChange, onUnavailable }: Props) {
  const frame = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState<Box | null>(null);
  // While dragging, the region being drawn; otherwise the parent's region is shown.
  const [live, setLive] = useState<Region | null>(null);
  const draft = live ?? region;
  const drag = useRef<Drag | null>(null);
  const [failed, setFailed] = useState(false);

  function measure(img: HTMLImageElement) {
    const el = frame.current;
    if (!el) return;
    const boxW = el.clientWidth;
    const boxH = el.clientHeight;
    const scale = Math.max(boxW / img.naturalWidth, boxH / img.naturalHeight);
    setBox({
      scale,
      natW: img.naturalWidth,
      natH: img.naturalHeight,
      boxW,
      boxH,
      offX: (boxW - img.naturalWidth * scale) / 2,
      offY: (boxH - img.naturalHeight * scale) / 2,
    });
  }

  const toPhoto = (clientX: number, clientY: number) => {
    const rect = frame.current!.getBoundingClientRect();
    const b = box!;
    return {
      x: (clientX - rect.left - b.offX) / (b.natW * b.scale),
      y: (clientY - rect.top - b.offY) / (b.natH * b.scale),
    };
  };

  function onPointerDown(e: PointerEvent<HTMLElement>, fallback: Drag["mode"]) {
    if (!editing || !box) return;
    e.stopPropagation();
    const mode = ((e.target as HTMLElement).dataset.edge as Drag["mode"] | undefined) ?? fallback;
    const p = toPhoto(e.clientX, e.clientY);
    const start = mode === "new" ? { x: p.x, y: p.y, width: 0, height: 0 } : (draft ?? visible(box));
    drag.current = { mode, x0: p.x, y0: p.y, start };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }

  function onPointerMove(e: PointerEvent<HTMLDivElement>) {
    const d = drag.current;
    if (!d || !box) return;
    const p = toPhoto(e.clientX, e.clientY);
    const dx = p.x - d.x0;
    const dy = p.y - d.y0;
    const s = d.start;
    const v = visible(box);
    let r: Region;
    if (d.mode === "move") r = { ...s, x: s.x + dx, y: s.y + dy };
    else if (d.mode === "new")
      r = {
        x: Math.min(d.x0, p.x),
        y: Math.min(d.y0, p.y),
        width: Math.abs(dx),
        height: Math.abs(dy),
      };
    else if (d.mode === "e") r = { ...s, width: s.width + dx };
    else if (d.mode === "w") r = { ...s, x: s.x + dx, width: s.width - dx };
    else if (d.mode === "s") r = { ...s, height: s.height + dy };
    else r = { ...s, y: s.y + dy, height: s.height - dy };
    setLive(fit(r, v));
  }

  function onPointerUp() {
    if (drag.current && live) onChange(live);
    drag.current = null;
    setLive(null);
  }

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (!box || !draft) return;
    const k = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    if (!k) {
      if (e.key === "Enter") onChange(draft);
      return;
    }
    e.preventDefault();
    const [dx, dy] = k.map((v) => v * STEP);
    const r = e.shiftKey
      ? { ...draft, width: draft.width + dx, height: draft.height + dy }
      : { ...draft, x: draft.x + dx, y: draft.y + dy };
    onChange(fit(r, visible(box)));
  }

  const shown = draft && box && {
    left: draft.x * box.natW * box.scale + box.offX,
    top: draft.y * box.natH * box.scale + box.offY,
    width: draft.width * box.natW * box.scale,
    height: draft.height * box.natH * box.scale,
  };

  return (
    <div
      ref={frame}
      className={`relative h-photo w-full overflow-hidden bg-rule-list ${editing && !failed ? "cursor-crosshair touch-none" : ""}`}
      onPointerDown={(e) => onPointerDown(e, "new")}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      {src && !failed && (
        <img
          src={src}
          alt="Your photo"
          className="h-full w-full object-cover select-none"
          draggable={false}
          onLoad={(e) => measure(e.currentTarget)}
          onError={() => {
            setFailed(true);
            onUnavailable();
          }}
        />
      )}
      {failed && (
        <p className="grid h-full place-items-center px-6 text-center text-sm text-ink-2">
          This browser can&apos;t show HEIC photos, so the whole photo is searched. Safari can
          show it, or choose a JPEG copy to select a region.
        </p>
      )}
      {shown && (
        <svg className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
          <path
            fillRule="evenodd"
            fill="var(--color-scrim)"
            d={`M0 0H${box!.boxW}V${box!.boxH}H0Z M${shown.left} ${shown.top}h${shown.width}v${shown.height}h${-shown.width}Z`}
          />
        </svg>
      )}
      {shown && (
        <div
          role="group"
          tabIndex={editing ? 0 : -1}
          aria-label="Selected region. Arrow keys move it, Shift with arrow keys resizes it."
          onKeyDown={onKeyDown}
          onPointerDown={(e) => onPointerDown(e, "move")}
          className={`absolute border-[1.5px] border-surface ${editing ? "cursor-move" : "pointer-events-none"}`}
          style={shown}
        >
          {editing &&
            (["n", "s", "e", "w"] as const).map((edge) => (
              <span
                key={edge}
                aria-hidden="true"
                data-edge={edge}
                className={
                  {
                    n: "absolute -top-2 left-2 right-2 h-4 cursor-ns-resize",
                    s: "absolute -bottom-2 left-2 right-2 h-4 cursor-ns-resize",
                    e: "absolute -right-2 top-2 bottom-2 w-4 cursor-ew-resize",
                    w: "absolute -left-2 top-2 bottom-2 w-4 cursor-ew-resize",
                  }[edge]
                }
              />
            ))}
        </div>
      )}
    </div>
  );
}
