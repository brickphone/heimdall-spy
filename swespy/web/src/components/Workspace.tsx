"use client";

/* eslint-disable @next/next/no-img-element -- local blob thumbnails */

import * as Dialog from "@radix-ui/react-dialog";
import * as Toggle from "@radix-ui/react-toggle";
import * as Tooltip from "@radix-ui/react-tooltip";
import Link from "next/link";
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";

import Compare from "@/components/Compare";
import {
  IconBack,
  IconClose,
  IconCoverage,
  IconHelp,
  IconNew,
  IconRecent,
  LogoMark,
} from "@/components/Icons";
import MapView, { type MapItem } from "@/components/MapView";
import PhotoRegion from "@/components/PhotoRegion";
import { type Prediction, type Region, UserFacingError, checkFile, predict } from "@/lib/api";
import { distanceM, formatCoords, formatDistance, plainCoords } from "@/lib/geo";
import { type Place, loadPlaces, lookup, placeLabel } from "@/lib/places";
import { reasonText } from "@/lib/reason";

interface Search {
  id: number;
  file: File;
  url: string;
  region: Region | null;
  prediction: Prediction;
  places: Place[];
}

type Status =
  | { kind: "empty" }
  | { kind: "searching"; started: number }
  | { kind: "error"; message: string }
  | { kind: "done" };

const DEFAULT_REGION: Region = { x: 0.2, y: 0.2, width: 0.6, height: 0.6 };
const REGION_DEBOUNCE_MS = 400;

function useElapsed(started: number | null): number {
  // The clock is only read in the timer, never during render (prerendering forbids it).
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (started === null) return;
    const t = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(t);
  }, [started]);
  return started === null || now < started ? 0 : (now - started) / 1000;
}

export default function Workspace({ references }: { references: number }) {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [region, setRegion] = useState<Region | null>(null);
  const [regionMode, setRegionMode] = useState(false);
  const [noPreview, setNoPreview] = useState(false);
  const [status, setStatus] = useState<Status>({ kind: "empty" });
  const [result, setResult] = useState<Search | null>(null);
  const [selected, setSelected] = useState(0);
  const [coverage, setCoverage] = useState(true);
  const [recent, setRecent] = useState<Search[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const request = useRef(0);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);
  const elapsed = useElapsed(status.kind === "searching" ? status.started : null);

  const run = useCallback(async (f: File, u: string, r: Region | null) => {
    const id = ++request.current;
    setStatus({ kind: "searching", started: Date.now() });
    try {
      const [prediction, places] = await Promise.all([predict(f, r), loadPlaces()]);
      if (id !== request.current) return;
      const at = [
        { lat: prediction.predicted_lat, lng: prediction.predicted_lng },
        ...prediction.top_matches.slice(1),
      ];
      const search: Search = { id, file: f, url: u, region: r, prediction, places: at.map((p) => lookup(places, p)) };
      setResult(search);
      setSelected(0);
      setCoverage(false);
      setStatus({ kind: "done" });
      setRecent((list) => [search, ...list.filter((s) => s.url !== u || s.region !== r)].slice(0, 12));
    } catch (err) {
      if (id !== request.current) return;
      const message =
        err instanceof UserFacingError ? err.message : "Something went wrong while searching. Try again.";
      setStatus({ kind: "error", message });
    }
  }, []);

  function choose(f: File | null) {
    if (!f) return;
    try {
      checkFile(f);
    } catch (err) {
      setStatus({ kind: "error", message: (err as Error).message });
      return;
    }
    const u = URL.createObjectURL(f);
    setFile(f);
    setUrl(u);
    setRegion(null);
    setRegionMode(false);
    setNoPreview(false);
    setResult(null);
    void run(f, u, null);
  }

  function reset() {
    request.current++;
    setFile(null);
    setUrl(null);
    setRegion(null);
    setRegionMode(false);
    setResult(null);
    setStatus({ kind: "empty" });
    setCoverage(true);
  }

  function changeRegion(r: Region | null) {
    setRegion(r);
    if (!file || !url) return;
    if (debounce.current) clearTimeout(debounce.current);
    debounce.current = setTimeout(() => void run(file, url, r), REGION_DEBOUNCE_MS);
  }

  function restore(s: Search) {
    request.current++;
    setFile(s.file);
    setUrl(s.url);
    setRegion(s.region);
    setRegionMode(Boolean(s.region));
    setResult(s);
    setSelected(0);
    setStatus({ kind: "done" });
  }

  const items: MapItem[] = useMemo(() => {
    if (!result) return [];
    const p = result.prediction;
    const top = p.top_matches[0];
    return [
      { index: 0, rank: 1, at: { lat: p.predicted_lat, lng: p.predicted_lng }, inliers: top?.inliers ?? null, label: placeLabel(result.places[0]) },
      ...p.top_matches.slice(1).map((m, i) => ({
        index: i + 1, rank: m.rank, at: { lat: m.lat, lng: m.lng }, inliers: m.inliers, label: placeLabel(result.places[i + 1]),
      })),
    ];
  }, [result]);

  const searching = status.kind === "searching";
  const photo = (
    <PhotoSection
      url={url}
      region={region}
      regionMode={regionMode}
      noPreview={noPreview}
      onRegionMode={(on) => {
        setRegionMode(on);
        if (on && !region) changeRegion(DEFAULT_REGION);
      }}
      onWhole={() => {
        setRegionMode(false);
        changeRegion(null);
      }}
      onRegion={changeRegion}
      onUnavailable={() => {
        setNoPreview(true);
        setRegionMode(false);
      }}
    />
  );

  const progress = searching && (
    <p className="text-sm text-ink-2" role="status" aria-live="polite">
      Searching {references.toLocaleString("en-US")} reference photos and checking the closest
      matches. {elapsed.toFixed(0)} s
    </p>
  );
  const error = status.kind === "error" && (
    <div role="alert" className="flex flex-col gap-3">
      <p className="text-body text-danger">{status.message}</p>
      {file && (
        <button type="button" onClick={reset} className="self-start rounded-control border border-rule-control px-3 py-2 text-control text-ink hover:bg-paper">
          Choose another photo
        </button>
      )}
    </div>
  );

  const resultSections = result && (
    <>
      <ResultSection search={result} />
      <Candidates search={result} selected={selected} onSelect={setSelected} />
    </>
  );

  return (
    <Tooltip.Provider delayDuration={300}>
      <div className="flex h-full flex-col desk:grid desk:grid-cols-[var(--spacing-rail)_1fr_var(--spacing-inspector)]">
        {/* Rail (desktop) */}
        <nav aria-label="Workspace" className="hidden flex-col items-center gap-2 border-r border-rule bg-surface py-4 desk:flex">
          <div className="mb-4" title="Heimdall">
            <LogoMark />
          </div>
          <RailButton label="New search" onClick={reset}>
            <IconNew />
          </RailButton>
          <RecentSearches recent={recent} onRestore={restore} />
          <RailButton label={coverage ? "Hide reference coverage" : "Show reference coverage"} pressed={coverage} onClick={() => setCoverage((c) => !c)}>
            <IconCoverage />
          </RailButton>
          <div className="mt-auto">
            <Tooltip.Root>
              <Tooltip.Trigger asChild>
                <Link href="/about" aria-label="How it works" className="grid size-touch place-items-center rounded-control text-ink hover:bg-paper">
                  <IconHelp />
                </Link>
              </Tooltip.Trigger>
              <TipContent>How it works</TipContent>
            </Tooltip.Root>
          </div>
        </nav>

        {/* Map */}
        <div className="relative h-map-mobile shrink-0 desk:h-full">
          <MapView
            prediction={result?.prediction ?? null}
            items={items}
            selected={selected}
            onSelect={setSelected}
            coverage={coverage}
            onCoverage={setCoverage}
          />
          <div className="absolute bottom-10 left-3 flex gap-2 desk:hidden">
            {status.kind !== "empty" && (
              <button type="button" onClick={reset} aria-label="New photo" className="grid size-touch place-items-center rounded-control border border-rule-control bg-surface text-ink">
                <IconBack />
              </button>
            )}
            <Toggle.Root
              pressed={coverage}
              onPressedChange={setCoverage}
              aria-label="Reference coverage"
              className="grid size-touch place-items-center rounded-control border border-rule-control bg-surface text-ink data-[state=on]:border-ink data-[state=on]:bg-ink data-[state=on]:text-surface"
            >
              <IconCoverage />
            </Toggle.Root>
          </div>
        </div>

        {/* Inspector (desktop) / bottom sheet (mobile) */}
        <aside
          aria-label="Inspector"
          className="relative z-10 flex min-h-0 flex-1 flex-col overflow-y-auto rounded-t-sheet border-t border-rule bg-surface desk:rounded-none desk:border-l desk:border-t-0"
        >
          <span className="mx-auto mt-2 h-1 w-10 shrink-0 rounded-control bg-rule desk:hidden" aria-hidden="true" />
          {(status.kind === "empty" || (status.kind === "error" && !file)) && (
            <EmptyState references={references} onFile={choose} error={status.kind === "error" ? status.message : null} />
          )}
          {status.kind !== "empty" && file && (
            <>
              <Section>
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h1 className="truncate text-body font-semibold text-ink">{file?.name ?? "Photo"}</h1>
                    <p className="text-sm text-ink-2">
                      {searching ? "Locating…" : result ? `Located in ${(result.prediction.processing_time_ms / 1000).toFixed(1)} s` : ""}
                    </p>
                  </div>
                  <button type="button" onClick={reset} className="shrink-0 rounded-control border border-rule-control px-3 py-2 text-control text-ink hover:bg-paper">
                    New photo
                  </button>
                </div>
              </Section>
              <div className="hidden desk:block">{photo}</div>
              {(progress || error) && <Section>{progress}{error}</Section>}
              {result && (
                <>
                  <div className="hidden desk:contents">{resultSections}</div>
                  <MobileResult search={result} onCompare={() => setCompareOpen(true)} />
                  <div className="desk:hidden">
                    <Candidates search={result} selected={selected} onSelect={setSelected} />
                    {photo}
                  </div>
                  <Section title="Your region and the match" className="hidden desk:flex">
                    <CompareFor search={result} selected={selected} />
                  </Section>
                </>
              )}
            </>
          )}
        </aside>
      </div>

      {/* Full-screen compare on mobile */}
      <Dialog.Root open={compareOpen} onOpenChange={setCompareOpen}>
        <Dialog.Portal>
          <Dialog.Content className="fixed inset-0 z-50 flex flex-col gap-4 bg-surface p-4" aria-describedby={undefined}>
            <div className="flex items-center justify-between">
              <Dialog.Title className="text-md font-semibold text-ink">Your region and the match</Dialog.Title>
              <Dialog.Close aria-label="Close" className="grid size-touch place-items-center rounded-control border border-rule-control text-ink">
                <IconClose />
              </Dialog.Close>
            </div>
            {result && <CompareFor search={result} selected={selected} tall />}
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </Tooltip.Provider>
  );
}

// ---------------------------------------------------------------- pieces

function Section({ title, children, className = "" }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`flex flex-col gap-3 border-b border-rule px-5 py-4 ${className}`}>
      {title && <h2 className="text-sm font-medium text-ink-2">{title}</h2>}
      {children}
    </section>
  );
}

function TipContent({ children }: { children: ReactNode }) {
  return (
    <Tooltip.Portal>
      <Tooltip.Content side="right" sideOffset={8} className="z-50 rounded-control bg-ink px-2 py-1 text-xs text-surface">
        {children}
      </Tooltip.Content>
    </Tooltip.Portal>
  );
}

function RailButton({ label, onClick, pressed, children }: { label: string; onClick: () => void; pressed?: boolean; children: ReactNode }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        <button
          type="button"
          aria-label={label}
          aria-pressed={pressed}
          onClick={onClick}
          className="grid size-touch place-items-center rounded-control text-ink hover:bg-paper aria-pressed:bg-ink aria-pressed:text-surface"
        >
          {children}
        </button>
      </Tooltip.Trigger>
      <TipContent>{label}</TipContent>
    </Tooltip.Root>
  );
}

function RecentSearches({ recent, onRestore }: { recent: Search[]; onRestore: (s: Search) => void }) {
  return (
    <Dialog.Root>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <Dialog.Trigger aria-label="Recent searches" className="grid size-touch place-items-center rounded-control text-ink hover:bg-paper">
            <IconRecent />
          </Dialog.Trigger>
        </Tooltip.Trigger>
        <TipContent>Recent searches</TipContent>
      </Tooltip.Root>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-scrim" />
        <Dialog.Content className="fixed left-rail top-0 z-50 flex h-full w-inspector flex-col border-r border-rule bg-surface" aria-describedby={undefined}>
          <div className="flex items-center justify-between border-b border-rule px-5 py-4">
            <Dialog.Title className="text-body font-semibold text-ink">Recent searches</Dialog.Title>
            <Dialog.Close aria-label="Close" className="grid size-touch place-items-center rounded-control text-ink hover:bg-paper">
              <IconClose />
            </Dialog.Close>
          </div>
          <p className="px-5 py-3 text-sm text-ink-2">Kept for this session only. Closing the tab clears them.</p>
          {recent.length === 0 && <p className="px-5 text-body text-ink-2">No searches yet.</p>}
          <ul className="flex flex-col">
            {recent.map((s) => (
              <li key={s.id} className="border-b border-rule-list">
                <Dialog.Close asChild>
                  <button type="button" onClick={() => onRestore(s)} className="flex w-full items-center gap-3 px-5 py-3 text-left hover:bg-paper">
                    <img src={s.url} alt="" className="size-touch object-cover" />
                    <span className="min-w-0">
                      <span className="block truncate text-body font-medium text-ink">{placeLabel(s.places[0])}</span>
                      <span className="block truncate text-sm text-ink-2">
                        {s.file.name}
                        {s.region ? ", region" : ""}
                      </span>
                    </span>
                  </button>
                </Dialog.Close>
              </li>
            ))}
          </ul>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function EmptyState({ references, onFile, error }: { references: number; onFile: (f: File | null) => void; error: string | null }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
    <div className="flex flex-1 flex-col">
      <div className="flex flex-col gap-5 px-5 py-6">
        <h1 className="text-lg font-semibold text-ink desk:text-xl desk:tracking-[var(--text-xl--letter-spacing)]">Where was this photo taken?</h1>
        <p className="text-md text-ink">
          Heimdall compares your photo with {references.toLocaleString("en-US")} street-level
          reference photos of Linköping and shows where it was most likely taken, and how sure it is.
        </p>
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            onFile(e.dataTransfer.files[0] ?? null);
          }}
          className={`flex flex-col items-start gap-3 border border-dashed p-5 ${over ? "border-pin bg-selected" : "border-rule-control"}`}
        >
          <input
            ref={input}
            type="file"
            accept="image/jpeg,image/png,image/heic,image/heif,.heic,.heif"
            className="sr-only"
            aria-label="Choose a photo"
            tabIndex={-1}
            onChange={(e) => onFile(e.target.files?.[0] ?? null)}
          />
          <button
            type="button"
            onClick={() => input.current?.click()}
            className="h-touch rounded-control bg-pin px-5 text-control font-medium text-surface hover:bg-pin-hover"
          >
            Choose a photo
          </button>
          <p className="text-sm text-ink-2">Or drop it here. JPEG, PNG or HEIC, taken outdoors in Linköping.</p>
        </div>
        {error && <p role="alert" className="text-body text-danger">{error}</p>}
      </div>
      <Section title="Before you start">
        <p className="text-body text-ink">Location data in the file, such as GPS, is ignored.</p>
        <p className="text-body text-ink">Your photo is processed in memory and not stored.</p>
        <p className="text-body text-ink">
          It works best when buildings or streets fill the frame. Otherwise, select the part of
          the photo that shows the place.
        </p>
      </Section>
      <footer className="mt-auto flex items-center justify-between gap-3 px-5 py-4 text-sm text-ink-2">
        <span className="flex items-center gap-2">
          <svg width="28" height="6" aria-hidden="true">
            <line x1="2" y1="3" x2="26" y2="3" stroke="var(--color-pin)" strokeOpacity="0.45" strokeWidth="2.2" strokeLinecap="round" strokeDasharray="0 7" />
          </svg>
          Streets with reference photos
        </span>
        <Link href="/about">How it works</Link>
      </footer>
    </div>
  );
}

function PhotoSection(props: {
  url: string | null;
  region: Region | null;
  regionMode: boolean;
  noPreview: boolean;
  onRegionMode: (on: boolean) => void;
  onWhole: () => void;
  onRegion: (r: Region) => void;
  onUnavailable: () => void;
}) {
  const { url, region, regionMode, noPreview } = props;
  return (
    <section className="flex flex-col border-b border-rule">
      <PhotoRegion
        src={url}
        region={region}
        editing={regionMode}
        onChange={props.onRegion}
        onUnavailable={props.onUnavailable}
      />
      <div className="flex flex-col gap-2 px-5 py-3">
        <div className="flex gap-2">
          <Toggle.Root
            pressed={regionMode}
            onPressedChange={props.onRegionMode}
            disabled={noPreview || !url}
            className="h-touch rounded-control border border-rule-control px-3 text-control text-ink hover:bg-paper disabled:text-ink-2 data-[state=on]:border-ink data-[state=on]:bg-ink data-[state=on]:text-surface desk:h-auto desk:py-2"
          >
            Select region
          </Toggle.Root>
          <button
            type="button"
            onClick={props.onWhole}
            disabled={!region}
            className="h-touch rounded-control border border-rule-control px-3 text-control text-ink hover:bg-paper disabled:text-ink-2 desk:h-auto desk:py-2"
          >
            Use whole photo
          </button>
        </div>
        {regionMode && (
          <p className="text-sm text-ink-2">Only the highlighted region is searched. Drag its edges to change it.</p>
        )}
      </div>
    </section>
  );
}

function StateLine({ state }: { state: string }) {
  const ok = state === "ok";
  return (
    <p className={`flex items-center gap-2 text-body font-medium ${ok ? "text-sure" : "text-unsure"}`}>
      <span className={`size-state ${ok ? "bg-sure" : "bg-unsure"}`} aria-hidden="true" />
      {ok ? "Confident" : "Low confidence"}
    </p>
  );
}

function CopyButton({ text, tall }: { text: string; tall?: boolean }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        await navigator.clipboard?.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }}
      className={`rounded-control border border-rule-control px-2 text-sm text-ink hover:bg-paper ${tall ? "h-touch flex-1 text-control" : "py-0.5"}`}
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function ResultSection({ search }: { search: Search }) {
  const p = search.prediction;
  const at = { lat: p.predicted_lat, lng: p.predicted_lng };
  return (
    <Section>
      <StateLine state={p.confidence_state} />
      <h2 className="text-xl font-semibold tracking-[var(--text-xl--letter-spacing)] text-ink">{placeLabel(search.places[0])}</h2>
      <p className="flex items-center gap-3 text-sm text-ink-2">
        {formatCoords(at)}
        <CopyButton text={plainCoords(at)} />
      </p>
      <p className="text-body text-ink">{reasonText(p, Boolean(search.region))}</p>
    </Section>
  );
}

function MobileResult({ search, onCompare }: { search: Search; onCompare: () => void }) {
  const p = search.prediction;
  const at = { lat: p.predicted_lat, lng: p.predicted_lng };
  return (
    <section className="flex flex-col gap-3 border-b border-rule px-5 py-4 desk:hidden">
      <div className="flex items-start gap-3">
        <img src={search.url} alt="Your photo" className="size-touch shrink-0 object-cover" />
        <div className="min-w-0">
          <StateLine state={p.confidence_state} />
          <h2 className="text-lg font-semibold text-ink">{placeLabel(search.places[0])}</h2>
          <p className="text-sm text-ink-2">{formatCoords(at)}</p>
        </div>
      </div>
      <p className="text-body text-ink">{reasonText(p, Boolean(search.region))}</p>
      <div className="flex gap-2">
        <button type="button" onClick={onCompare} className="h-touch flex-[2] rounded-control bg-pin text-control font-medium text-surface hover:bg-pin-hover">
          Compare with match
        </button>
        <CopyButton text={plainCoords(at)} tall />
      </div>
    </section>
  );
}

function Candidates({ search, selected, onSelect }: { search: Search; selected: number; onSelect: (i: number) => void }) {
  const p = search.prediction;
  const best = { lat: p.predicted_lat, lng: p.predicted_lng };
  const rows = p.top_matches
    .slice(1)
    .map((m, i) => ({ index: i + 1, match: m, place: search.places[i + 1] }));
  if (!rows.length) return null;
  return (
    <Section title="Other candidates">
      <ul className="-mx-5 flex flex-col">
        {rows.map(({ index, match, place }) => (
          <li key={match.image_id} className="border-b border-rule-list last:border-b-0">
            <button
              type="button"
              aria-pressed={selected === index}
              onClick={() => onSelect(index)}
              className="grid w-full grid-cols-[1.5rem_1fr_auto] items-baseline gap-x-3 px-5 py-2.5 text-left hover:bg-paper aria-pressed:bg-selected"
            >
              <span className="text-sm font-semibold text-ink">{match.rank}</span>
              <span className="min-w-0">
                <span className="block truncate text-body font-medium text-ink">{placeLabel(place)}</span>
                <span className="block text-sm text-ink-2">
                  {formatDistance(distanceM(best, match))} from the best match
                </span>
              </span>
              <span className="text-sm text-ink-2">{match.inliers ?? 0} details</span>
            </button>
          </li>
        ))}
      </ul>
    </Section>
  );
}

function CompareFor({ search, selected, tall }: { search: Search; selected: number; tall?: boolean }) {
  const p = search.prediction;
  const match = p.top_matches[selected] ?? p.top_matches[0];
  if (!match) return null;
  return (
    <Compare
      photoUrl={search.url}
      region={search.region}
      match={match}
      predicted={{ lat: p.predicted_lat, lng: p.predicted_lng }}
      tall={tall}
    />
  );
}
