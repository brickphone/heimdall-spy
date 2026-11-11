"use client";

/* eslint-disable @next/next/no-img-element -- local blob and proxied API images */

import { type KeyboardEvent, type PointerEvent, useEffect, useRef, useState } from "react";

import { IconSwap } from "@/components/Icons";
import { type Match, type Region, licenceName, referenceImageUrl, sourceName } from "@/lib/api";
import { distanceM, formatDistance, type LatLng } from "@/lib/geo";

export const EEA_TEXT =
  "This match comes from Google Street View. Under Google's terms for the EEA, Street View " +
  "imagery can't be shown next to a map, so only its position is used here.";

interface Props {
  photoUrl: string | null;
  region: Region | null;
  match: Match;
  predicted: LatLng;
  tall?: boolean;
}

/** Your region drawn into a canvas, so both sides fill the same frame. */
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
  return <canvas ref={canvas} className="absolute inset-0 h-full w-full object-cover" aria-label="Your region" role="img" />;
}

export default function Compare({ photoUrl, region, match, predicted, tall }: Props) {
  const [split, setSplit] = useState(50);
  const frame = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const isStreetView = match.source === "gsv";

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

  const metres = distanceM(match, predicted);
  const year = match.capture_year ? `, captured ${match.capture_year}` : "";

  if (isStreetView) {
    return <p className="text-sm text-ink-2">{EEA_TEXT}</p>;
  }

  return (
    <figure className="flex flex-col gap-3">
      <div
        ref={frame}
        className={`relative w-full overflow-hidden bg-rule-list touch-none ${tall ? "h-full min-h-photo" : "h-compare"}`}
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
        {photoUrl && <RegionCanvas src={photoUrl} region={region} />}
        <img
          src={referenceImageUrl(match.image_id)}
          alt="Matched reference photo"
          className="absolute inset-0 h-full w-full object-cover"
          style={{ clipPath: `inset(0 0 0 ${split}%)` }}
          draggable={false}
        />
        <div
          className="absolute inset-y-0 bg-surface"
          style={{ left: `calc(${split}% - var(--divider-width) / 2)`, width: "var(--divider-width)" }}
          aria-hidden="true"
        />
        <button
          type="button"
          role="slider"
          aria-label="Compare your region with the match"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(split)}
          aria-valuetext={`${Math.round(split)}% of your region shown`}
          onKeyDown={onKey}
          className="absolute top-1/2 grid size-handle -translate-x-1/2 -translate-y-1/2 place-items-center border border-ink bg-surface text-ink"
          style={{ left: `${split}%` }}
        >
          <IconSwap />
        </button>
        <span className="absolute left-2 top-2 bg-surface px-2 py-0.5 text-xs text-ink">Your region</span>
        <span className="absolute right-2 top-2 bg-surface px-2 py-0.5 text-xs text-ink">Match</span>
      </div>
      <figcaption className="text-sm text-ink-2">
        Reference photo by {match.creator || "an unknown author"} on{" "}
        {match.source_url ? <a href={match.source_url} target="_blank" rel="noopener noreferrer">{sourceName(match.source)}</a> : sourceName(match.source)}
        ,{" "}
        {match.license_url ? <a href={match.license_url} target="_blank" rel="noopener noreferrer">{licenceName(match.license_url)}</a> : "licence unknown"}
        {year}, taken {formatDistance(metres)} from the predicted point.
      </figcaption>
    </figure>
  );
}
