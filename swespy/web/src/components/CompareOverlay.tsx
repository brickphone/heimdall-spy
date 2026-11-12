"use client";

/* eslint-disable @next/next/no-img-element -- local blob/preview and proxied API images */

import { type CSSProperties, type KeyboardEvent, type PointerEvent, useEffect, useRef, useState } from "react";

import { IconClose, IconSideBySide, IconSwap } from "@/components/Icons";
import VerdictControl, { type Verdict } from "@/components/Verdict";
import { type Match, type Region, licenceName, referenceImageUrl, sourceName } from "@/lib/api";
import { distanceM, formatDistance, type LatLng } from "@/lib/geo";

export const EEA_TEXT =
  "Google Street View match. Under Google's EEA terms its imagery can't be shown next to a map; only its position is used.";

type Mode = "side" | "swipe";

interface Props {
  photoUrl: string;
  region: Region | null; // The searched area, normalised to the upright photo.
  match: Match;
  place: string;
  predicted: LatLng;
  verdict: Verdict | undefined;
  onVerdict: (v: Verdict | undefined) => void;
  onClose: () => void;
  style: CSSProperties; // Position: the map area between the floating panels.
}

/** Your searched area drawn into a canvas, cover-fitted like the reference image. */
function RegionCanvas({ src, region }: { src: string; region: Region | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      const c = canvas.current;
      if (!c) return;
      const r = region ?? { x: 0, y: 0, width: 1, height: 1 };
      const sw = r.width * img.naturalWidth;
      const sh = r.height * img.naturalHeight;
      c.width = Math.round(sw);
      c.height = Math.round(sh);
      c.getContext("2d")?.drawImage(img, r.x * img.naturalWidth, r.y * img.naturalHeight, sw, sh, 0, 0, c.width, c.height);
    };
    img.src = src;
  }, [src, region]);
  return <canvas ref={canvas} className="absolute inset-0 h-full w-full object-contain" aria-label="Your photo, searched area" role="img" />;
}

function Label({ children, right }: { children: string; right?: boolean }) {
  return <span className={`absolute top-2 rounded-square bg-surface px-1.5 py-0.5 text-xs font-medium text-ink ${right ? "right-2" : "left-2"}`}>{children}</span>;
}

function Swipe({ photoUrl, region, refUrl }: { photoUrl: string; region: Region | null; refUrl: string }) {
  const [split, setSplit] = useState(50);
  const frame = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const move = (clientX: number) => {
    const rect = frame.current!.getBoundingClientRect();
    setSplit(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)));
  };
  const onKey = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 10 : 2;
    if (e.key === "ArrowLeft") setSplit((s) => Math.max(0, s - step));
    else if (e.key === "ArrowRight") setSplit((s) => Math.min(100, s + step));
    else if (e.key === "Home") setSplit(0);
    else if (e.key === "End") setSplit(100);
    else return;
    e.preventDefault();
  };
  return (
    <div
      ref={frame}
      className="relative h-full w-full touch-none overflow-hidden bg-rule-list"
      onPointerDown={(e: PointerEvent) => {
        dragging.current = true;
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
        move(e.clientX);
      }}
      onPointerMove={(e) => dragging.current && move(e.clientX)}
      onPointerUp={() => {
        dragging.current = false;
      }}
    >
      <RegionCanvas src={photoUrl} region={region} />
      <img src={refUrl} alt="Reference photo" draggable={false} className="absolute inset-0 h-full w-full object-contain" style={{ clipPath: `inset(0 0 0 ${split}%)` }} />
      <div className="absolute inset-y-0 bg-surface" style={{ left: `calc(${split}% - var(--divider-width) / 2)`, width: "var(--divider-width)" }} aria-hidden="true" />
      <button
        type="button"
        role="slider"
        aria-label="Swipe between your photo and the reference"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(split)}
        aria-valuetext={`${Math.round(split)}% of your photo shown`}
        onKeyDown={onKey}
        className="absolute top-1/2 grid size-handle -translate-x-1/2 -translate-y-1/2 place-items-center rounded-control border border-ink bg-surface text-ink"
        style={{ left: `${split}%` }}
      >
        <IconSwap />
      </button>
      <Label>Your photo</Label>
      <Label right>Reference</Label>
    </div>
  );
}

/** Large comparison over the map: your searched area and the reference, with a verdict. */
export default function CompareOverlay(props: Props) {
  const { photoUrl, region, match, place, predicted, verdict, onVerdict, onClose, style } = props;
  const [mode, setMode] = useState<Mode>("swipe");
  const close = useRef<HTMLButtonElement>(null);
  const isStreetView = match.source === "gsv";
  const refUrl = referenceImageUrl(match.image_id);

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    close.current?.focus();
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      opener?.focus?.();
    };
  }, [onClose]);

  const year = match.capture_year ? `, ${match.capture_year}` : "";
  return (
    <section
      role="dialog"
      aria-label={`Compare with ${place}`}
      className="absolute z-30 flex flex-col overflow-hidden rounded-panel border border-rule bg-surface shadow-[var(--shadow-panel)]"
      style={style}
    >
      <header className="flex items-center gap-3 border-b border-rule px-3 py-2">
        <span className="grid size-5 shrink-0 place-items-center rounded-square border border-ink text-xs font-semibold text-ink">{match.rank}</span>
        <h2 className="min-w-0 truncate text-sm font-semibold text-ink">{place}</h2>
        <div role="group" aria-label="View" className="ml-auto flex overflow-hidden rounded-control border border-rule-control">
          {(
            [
              ["swipe", "Swipe", IconSwap],
              ["side", "Side by side", IconSideBySide],
            ] as const
          ).map(([m, label, Icon]) => (
            <button
              key={m}
              type="button"
              aria-pressed={mode === m}
              onClick={() => setMode(m)}
              className="flex h-7 items-center gap-1.5 border-r border-rule-control px-2 text-xs text-ink last:border-r-0 hover:bg-paper aria-pressed:bg-ink aria-pressed:text-surface"
            >
              <Icon />
              {label}
            </button>
          ))}
        </div>
        <button ref={close} type="button" aria-label="Close compare" title="Close (Esc)" onClick={onClose} className="grid size-7 place-items-center rounded-control text-ink hover:bg-paper">
          <IconClose />
        </button>
      </header>
      <div className="min-h-0 flex-1 bg-rule-list">
        {isStreetView ? (
          <p className="p-4 text-sm text-ink-2">{EEA_TEXT}</p>
        ) : mode === "side" ? (
          <div className="grid h-full grid-cols-2 gap-px">
            <div className="relative bg-rule-list">
              <RegionCanvas src={photoUrl} region={region} />
              <Label>Your photo</Label>
            </div>
            <div className="relative bg-rule-list">
              <img src={refUrl} alt="Reference photo" className="absolute inset-0 h-full w-full object-contain" />
              <Label>Reference</Label>
            </div>
          </div>
        ) : (
          <Swipe photoUrl={photoUrl} region={region} refUrl={refUrl} />
        )}
      </div>
      <footer className="flex items-center gap-3 border-t border-rule px-3 py-2">
        <p className="min-w-0 flex-1 truncate text-xs text-ink-2">
          {isStreetView ? (
            "Google Street View, © Google"
          ) : (
            <>
              {match.creator || "Unknown author"},{" "}
              {match.source_url ? <a href={match.source_url} target="_blank" rel="noopener noreferrer">{sourceName(match.source)}</a> : sourceName(match.source)},{" "}
              {match.license_url ? <a href={match.license_url} target="_blank" rel="noopener noreferrer">{licenceName(match.license_url)}</a> : "licence unknown"}
              {year}, {formatDistance(distanceM(match, predicted))} from the prediction
            </>
          )}
        </p>
        <VerdictControl value={verdict} onChange={onVerdict} subject={`candidate ${match.rank}`} withText />
      </footer>
    </section>
  );
}
