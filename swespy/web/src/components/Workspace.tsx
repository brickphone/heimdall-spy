"use client";

/* eslint-disable @next/next/no-img-element -- local blob thumbnails */

import * as Dialog from "@radix-ui/react-dialog";
import * as Tooltip from "@radix-ui/react-tooltip";
import Link from "next/link";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import Compare from "@/components/Compare";
import { IconBack, IconClose, IconHelp, IconNew, IconRecent, LogoMark } from "@/components/Icons";
import LocatingMap, { type LocatingMapHandle, type MapItem } from "@/components/LocatingMap";
import PhotoRegion from "@/components/PhotoRegion";
import { type Prediction, type Region, UserFacingError, analyze, checkFile } from "@/lib/api";
import { distanceM, formatCoords, formatDistance, plainCoords } from "@/lib/geo";
import { type Place, loadPlaces, lookup, placeLabel } from "@/lib/places";
import { reasonText } from "@/lib/reason";
import { type CandidatePoint, type StreamEvent, locateStream } from "@/lib/stream";

interface Search {
  id: number;
  file: File;
  url: string;
  region: Region | null;
  prediction: Prediction;
  places: Place[];
  timings: Timings;
}

/** Seconds per stage, measured from the arrival of the stream events. */
interface Timings {
  search: number | null;
  compare: number | null;
  check: number | null;
}

type StageKey = keyof Timings;
type StageState = "pending" | "active" | "done";

type Phase =
  | { kind: "empty" }
  | { kind: "ready" } // Photo chosen, nothing sent yet.
  | { kind: "locating"; stage: StageKey; label: string }
  | { kind: "error"; message: string }
  | { kind: "done" };

const STAGES: { key: StageKey; label: string }[] = [
  { key: "search", label: "Search reference photos" },
  { key: "compare", label: "Compare closest matches" },
  { key: "check", label: "Check confidence" },
];

// Minimum time each phase stays on screen, so fast answers still read clearly (ms).
const HOLD = { begin: 900, candidates: 1000, verification: 1200, verified: 180 } as const;
const HOLD_REDUCED = { begin: 250, candidates: 250, verification: 250, verified: 60 } as const;

// Notes under the photo when Heimdall pre-draws a region.
const NOTES = {
  person: "Most of this photo is a person. The highlighted area shows the place.",
  alternative: "Another part of the photo is highlighted. Adjust it or use the whole photo, then locate again.",
} as const;
type Note = keyof typeof NOTES;

/** True when the result came from part of the photo rather than all of it. */
const usedRegion = (p: Prediction) => p.searched_variant === "manual" || p.searched_variant.startsWith("crop:");

const reducedMotion = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export default function Workspace({ references, area }: { references: number; area: [number, number][] }) {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [region, setRegion] = useState<Region | null>(null);
  const [editing, setEditing] = useState(true);
  const [noPreview, setNoPreview] = useState(false);
  const [phase, setPhase] = useState<Phase>({ kind: "empty" });
  const [result, setResult] = useState<Search | null>(null);
  const [selected, setSelected] = useState(0);
  const [recent, setRecent] = useState<Search[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const [timings, setTimings] = useState<Timings>({ search: null, compare: null, check: null });
  const [emptyError, setEmptyError] = useState<string | null>(null);
  const [note, setNote] = useState<Note | null>(null);
  const mapRef = useRef<LocatingMapHandle>(null);
  const photoRef = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null);
  const analyzing = useRef<AbortController | null>(null);
  const runId = useRef(0);

  useEffect(() => {
    void loadPlaces(); // Warm the place-name index while the user picks a photo.
  }, []);

  function choose(f: File | null) {
    if (!f) return;
    try {
      checkFile(f);
    } catch (err) {
      setEmptyError((err as Error).message);
      if (!file) setPhase({ kind: "empty" });
      return;
    }
    stopRun();
    mapRef.current?.reset();
    setEmptyError(null);
    setFile(f);
    setUrl(URL.createObjectURL(f));
    setRegion(null);
    setNote(null);
    setEditing(true);
    setNoPreview(false);
    setResult(null);
    setPhase({ kind: "ready" });
    suggest(f);
  }

  /** Ask the API whether people dominate the photo; if so, pre-draw the place. */
  function suggest(f: File) {
    analyzing.current?.abort();
    const controller = new AbortController();
    analyzing.current = controller;
    void analyze(f, controller.signal).then((a) => {
      if (controller.signal.aborted || !a?.suggest || !a.suggested_region) return;
      const suggested = a.suggested_region;
      // Never replace a region the user has drawn in the meantime.
      setRegion((current) => {
        if (current) return current;
        setNote("person");
        return suggested;
      });
    });
  }

  function stopAnalyze() {
    analyzing.current?.abort();
    analyzing.current = null;
  }

  /** Low confidence: open region selection with the best automatic alternative drawn. */
  function tryDifferentRegion() {
    if (!result) return;
    stopAnalyze();
    setRegion(result.prediction.alternative_region);
    setNote(result.prediction.alternative_region ? "alternative" : null);
    setEditing(true);
    photoRef.current?.scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" });
  }

  function stopRun() {
    runId.current++;
    abort.current?.abort();
    abort.current = null;
  }

  function reset() {
    stopRun();
    stopAnalyze();
    setNote(null);
    mapRef.current?.reset();
    setFile(null);
    setUrl(null);
    setRegion(null);
    setResult(null);
    setEmptyError(null);
    setPhase({ kind: "empty" });
  }

  function cancel() {
    stopRun();
    mapRef.current?.cancel();
    if (result) {
      // A re-run was cancelled: keep the previous answer on the map.
      mapRef.current?.result(mapItems(result), result.prediction.confidence_state);
      setTimings(result.timings);
      setPhase({ kind: "done" });
    } else {
      setPhase({ kind: "ready" });
    }
    setEditing(true);
  }

  /** Locate: stream events into a queue and play them, each phase held long enough. */
  async function locate() {
    if (!file || !url) return;
    stopRun();
    stopAnalyze();
    setNote(null);
    const id = runId.current;
    const controller = new AbortController();
    abort.current = controller;
    const live = () => id === runId.current;
    const hold = reducedMotion() ? HOLD_REDUCED : HOLD;
    const t0 = performance.now();
    const arrived: Record<string, number> = {};
    const queue: StreamEvent[] = [];
    let wake: (() => void) | null = null;
    let finished = false;
    let failure: string | null = null;

    setEditing(false);
    setTimings({ search: null, compare: null, check: null });
    setPhase({ kind: "locating", stage: "search", label: "Searching Linköping" });
    mapRef.current?.begin();

    const fetching = locateStream(file, region, controller.signal, (ev) => {
      const now = performance.now();
      if (ev.type === "candidates") arrived.candidates = now;
      if (ev.type === "verified") arrived.verified = now;
      if (ev.type === "result") arrived.result = now;
      queue.push(ev);
      wake?.();
    })
      .catch((err) => {
        if (!controller.signal.aborted) {
          failure = err instanceof UserFacingError ? err.message : "Something went wrong while locating. Try again.";
        }
      })
      .finally(() => {
        finished = true;
        wake?.();
      });

    let nextAt = performance.now() + hold.begin;
    let top: CandidatePoint[] = [];
    let candidates: CandidatePoint[] = [];
    const places = loadPlaces();
    while (live()) {
      if (!queue.length) {
        if (finished) break;
        await new Promise<void>((r) => (wake = r));
        wake = null;
        continue;
      }
      const wait = nextAt - performance.now();
      if (wait > 0) await sleep(wait);
      if (!live()) return;
      const ev = queue.shift()!;
      if (ev.type === "candidates") {
        candidates = ev.items;
        setTimings((t) => ({ ...t, search: (arrived.candidates - t0) / 1000 }));
        setPhase({ kind: "locating", stage: "compare", label: "Comparing the 5 closest matches" });
        mapRef.current?.candidates(ev.items);
        nextAt = performance.now() + hold.candidates;
      } else if (ev.type === "stage" && ev.stage === "verification") {
        top = candidates.slice(0, ev.count);
        setPhase({ kind: "locating", stage: "compare", label: `Comparing the ${ev.count} closest matches` });
        mapRef.current?.verifying(top);
        nextAt = performance.now() + hold.verification;
      } else if (ev.type === "verified") {
        mapRef.current?.verified(ev.id, ev.inliers);
        nextAt = performance.now() + hold.verified;
      } else if (ev.type === "result") {
        const prediction: Prediction = ev;
        const lastVerified = arrived.verified ?? arrived.candidates;
        const done: Timings = {
          search: (arrived.candidates - t0) / 1000,
          compare: (lastVerified - arrived.candidates) / 1000,
          check: (arrived.result - lastVerified) / 1000,
        };
        setTimings(done);
        setPhase({ kind: "locating", stage: "check", label: "Checking confidence" });
        const index = await places;
        if (!live()) return;
        const at = [{ lat: prediction.predicted_lat, lng: prediction.predicted_lng }, ...prediction.top_matches.slice(1)];
        const search: Search = { id, file, url, region, prediction, places: at.map((p) => lookup(index, p)), timings: done };
        mapRef.current?.result(mapItems(search), prediction.confidence_state);
        setResult(search);
        setSelected(0);
        setPhase({ kind: "done" });
        setRecent((list) => [search, ...list].slice(0, 12));
        abort.current = null;
        return;
      } else if (ev.type === "error") {
        failure = ev.message;
        break;
      }
    }
    await fetching;
    if (!live()) return;
    mapRef.current?.fail();
    setPhase({ kind: "error", message: failure ?? "The search ended without a result. Try again." });
  }

  function restore(s: Search) {
    stopRun();
    stopAnalyze();
    setNote(null);
    setFile(s.file);
    setUrl(s.url);
    setRegion(s.region);
    setEditing(false);
    setResult(s);
    setTimings(s.timings);
    setSelected(0);
    setPhase({ kind: "done" });
    mapRef.current?.result(mapItems(s), s.prediction.confidence_state);
  }

  function selectCandidate(index: number) {
    setSelected(index);
    mapRef.current?.select(index);
  }

  const locating = phase.kind === "locating";
  const status =
    phase.kind === "locating" ? (
      <StatusLabel text={phase.label} onCancel={cancel} />
    ) : phase.kind === "error" ? (
      <StatusLabel text={phase.message} error onRetry={() => void locate()} />
    ) : null;

  const stageState = (key: StageKey): StageState => {
    if (timings[key] !== null) return "done";
    if (phase.kind === "locating" && phase.stage === key) return "active";
    return "pending";
  };

  const photo = file && (
    <div ref={photoRef}>
      <PhotoSection
        url={url}
        region={region}
        searched={result && !locating ? result.prediction.searched_region : null}
        note={note && editing && !locating ? NOTES[note] : null}
        editing={editing && !locating}
        noPreview={noPreview}
        hasResult={Boolean(result)}
        locating={locating}
        onRegion={setRegion}
        onWhole={() => {
          stopAnalyze();
          setRegion(null);
          setNote(null);
        }}
        onChangeRegion={() => setEditing(true)}
        onLocate={() => void locate()}
        onUnavailable={() => {
          setNoPreview(true);
          setEditing(false);
        }}
      />
    </div>
  );

  // The stage list is shown while locating (and after an error); with a result it
  // collapses into the "Located in" line in the header.
  const stages = file && (locating || phase.kind === "error") && (
    <Section>
      <StageList state={stageState} timings={timings} />
    </Section>
  );

  const results = result && (
    <div className={`desk:contents ${locating ? "pointer-events-none opacity-40" : ""}`} inert={locating}>
      <div className="hidden desk:contents">
        <ResultSection search={result} onTryRegion={tryDifferentRegion} />
        <Section title="Your region and the match">
          <CompareFor search={result} selected={selected} />
        </Section>
        <Candidates search={result} selected={selected} onSelect={selectCandidate} />
      </div>
      <div className="desk:hidden">
        <MobileResult search={result} onCompare={() => setCompareOpen(true)} onTryRegion={tryDifferentRegion} />
        <Candidates search={result} selected={selected} onSelect={selectCandidate} />
      </div>
      <Section title="Stage timings">
        <StageList state={() => "done"} timings={result.timings} />
      </Section>
    </div>
  );

  return (
    <Tooltip.Provider delayDuration={300}>
      <div className="flex h-full flex-col desk:grid desk:grid-cols-[var(--spacing-rail)_1fr_var(--spacing-inspector)]">
        <nav aria-label="Workspace" className="hidden flex-col items-center gap-2 border-r border-rule bg-surface py-4 desk:flex">
          <div className="mb-4" title="Heimdall">
            <LogoMark />
          </div>
          <RailButton label="New search" onClick={reset}>
            <IconNew />
          </RailButton>
          <RecentSearches recent={recent} onRestore={restore} />
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

        <div className="relative h-map-mobile shrink-0 desk:h-full">
          <LocatingMap
            ref={mapRef}
            area={area}
            dimmed={locating}
            status={status}
            onSelect={selectCandidate}
            onDropFile={choose}
          />
          {phase.kind !== "empty" && (
            <button
              type="button"
              onClick={reset}
              aria-label="New photo"
              className="absolute bottom-10 left-3 grid size-touch place-items-center rounded-control border border-rule-control bg-surface text-ink desk:hidden"
            >
              <IconBack />
            </button>
          )}
        </div>

        <aside
          aria-label="Inspector"
          className="relative z-10 flex min-h-0 flex-1 flex-col overflow-y-auto rounded-t-sheet border-t border-rule bg-surface desk:rounded-none desk:border-l desk:border-t-0"
        >
          <span className="mx-auto mt-2 h-1 w-10 shrink-0 rounded-control bg-rule desk:hidden" aria-hidden="true" />
          {!file && <EmptyState references={references} onFile={choose} error={emptyError} />}
          {file && (
            <>
              <Section>
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h1 className="truncate text-body font-semibold text-ink">{file.name}</h1>
                    <LocatedLine phase={phase} timings={timings} hasResult={Boolean(result)} />
                  </div>
                  <button type="button" onClick={reset} className="shrink-0 rounded-control border border-rule-control px-3 py-2 text-control text-ink hover:bg-paper">
                    New photo
                  </button>
                </div>
              </Section>
              {photo}
              {stages}
              {results}
            </>
          )}
        </aside>
      </div>

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

function mapItems(s: Search): MapItem[] {
  const p = s.prediction;
  const top = p.top_matches[0];
  return [
    { index: 0, rank: 1, at: { lat: p.predicted_lat, lng: p.predicted_lng }, inliers: top?.inliers ?? null, label: placeLabel(s.places[0]) },
    ...p.top_matches.slice(1).map((m, i) => ({
      index: i + 1, rank: m.rank, at: { lat: m.lat, lng: m.lng }, inliers: m.inliers, label: placeLabel(s.places[i + 1]),
    })),
  ];
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

function ProgressArc() {
  return (
    <svg viewBox="0 0 12 12" className="size-3 shrink-0 animate-spin motion-reduce:animate-none" aria-hidden="true">
      <circle cx="6" cy="6" r="4.5" fill="none" stroke="var(--color-rule)" strokeWidth="1.4" />
      <path d="M6 1.5a4.5 4.5 0 0 1 4.5 4.5" fill="none" stroke="var(--color-pin)" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function StatusLabel({ text, error, onCancel, onRetry }: { text: string; error?: boolean; onCancel?: () => void; onRetry?: () => void }) {
  return (
    <div role="status" aria-live="polite" className="flex max-w-sm items-center gap-3 border-[0.9px] border-ink bg-surface px-3 py-2">
      {!error && <ProgressArc />}
      <span className={`text-control ${error ? "text-danger" : "text-ink"}`}>{text}</span>
      {onCancel && (
        <button type="button" onClick={onCancel} className="ml-1 text-control font-medium text-pin underline underline-offset-2 hover:text-pin-hover">
          Cancel
        </button>
      )}
      {onRetry && (
        <button type="button" onClick={onRetry} className="ml-1 shrink-0 text-control font-medium text-pin underline underline-offset-2 hover:text-pin-hover">
          Try again
        </button>
      )}
    </div>
  );
}

function StageMarker({ state }: { state: StageState }) {
  if (state === "active") return <ProgressArc />;
  return <span className={`size-3 shrink-0 border border-ink ${state === "done" ? "bg-ink" : "bg-surface"}`} aria-hidden="true" />;
}

function StageList({ state, timings }: { state: (k: StageKey) => StageState; timings: Timings }) {
  return (
    <ol className="flex flex-col gap-2" aria-label="Progress">
      {STAGES.map(({ key, label }) => (
        <li key={key} className="flex items-center gap-3 text-body">
          <StageMarker state={state(key)} />
          <span className={state(key) === "pending" ? "text-ink-2" : "text-ink"}>{label}</span>
          <span className="ml-auto text-sm text-ink-2">
            {timings[key] !== null ? `${timings[key]!.toFixed(1)} s` : state(key) === "active" ? "running" : ""}
          </span>
          <span className="sr-only">{state(key)}</span>
        </li>
      ))}
    </ol>
  );
}

/** "Located in 4.1 s" once a result exists; the stage timings close the inspector. */
function LocatedLine({ phase, timings, hasResult }: { phase: Phase; timings: Timings; hasResult: boolean }) {
  if (phase.kind === "locating") return <p className="text-sm text-ink-2">Locating…</p>;
  if (phase.kind === "ready") return <p className="text-sm text-ink-2">{hasResult ? "Change the region, then locate again." : "Ready to locate."}</p>;
  if (phase.kind === "error") return <p className="text-sm text-danger">Not located</p>;
  if (!hasResult || timings.search === null) return null;
  const total = (timings.search ?? 0) + (timings.compare ?? 0) + (timings.check ?? 0);
  return <p className="text-sm text-ink-2">Located in {total.toFixed(1)} s</p>;
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

function RailButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        <button type="button" aria-label={label} onClick={onClick} className="grid size-touch place-items-center rounded-control text-ink hover:bg-paper">
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
            onChange={(e) => {
              onFile(e.target.files?.[0] ?? null);
              e.target.value = "";
            }}
          />
          <button type="button" onClick={() => input.current?.click()} className="h-touch rounded-control bg-pin px-5 text-control font-medium text-surface hover:bg-pin-hover">
            Choose a photo
          </button>
          <p className="text-sm text-ink-2">Or drop it here or on the map. JPEG, PNG or HEIC, taken outdoors in Linköping.</p>
        </div>
        {error && <p role="alert" className="text-body text-danger">{error}</p>}
      </div>
      <Section title="Before you start">
        <p className="text-body text-ink">Location data in the file, such as GPS, is ignored.</p>
        <p className="text-body text-ink">Your photo is processed in memory and not stored.</p>
        <p className="text-body text-ink">
          It works best when buildings or streets fill the frame. Otherwise, select the part of
          the photo that shows the place before you locate it.
        </p>
      </Section>
      <footer className="mt-auto flex justify-end px-5 py-4 text-sm">
        <Link href="/about">How it works</Link>
      </footer>
    </div>
  );
}

function PhotoSection(props: {
  url: string | null;
  region: Region | null;
  searched: Region | null;
  note: string | null;
  editing: boolean;
  noPreview: boolean;
  hasResult: boolean;
  locating: boolean;
  onRegion: (r: Region) => void;
  onWhole: () => void;
  onChangeRegion: () => void;
  onLocate: () => void;
  onUnavailable: () => void;
}) {
  const { url, region, editing, noPreview, hasResult, locating } = props;
  const canLocate = !locating && (!hasResult || editing);
  return (
    <section className="flex flex-col border-b border-rule">
      <PhotoRegion src={url} region={region} editing={editing} searched={props.searched} onChange={props.onRegion} onUnavailable={props.onUnavailable} />
      <div className="flex flex-col gap-3 px-5 py-3">
        {props.note && !noPreview && <p className="text-body text-ink">{props.note}</p>}
        {editing && !noPreview && (
          <p className="text-sm text-ink-2">
            {region
              ? "Only the highlighted region is searched. Drag its edges to change it."
              : "Drag a box around the part that shows the place, or locate the whole photo."}
          </p>
        )}
        <div className="flex flex-wrap gap-2">
          {canLocate && (
            <button type="button" onClick={props.onLocate} className="h-touch rounded-control bg-pin px-5 text-control font-medium text-surface hover:bg-pin-hover">
              Locate
            </button>
          )}
          {hasResult && !editing && !locating && (
            <button type="button" onClick={props.onChangeRegion} disabled={noPreview} className="h-touch rounded-control border border-rule-control px-3 text-control text-ink hover:bg-paper disabled:text-ink-2">
              Change region
            </button>
          )}
          {editing && (
            <button type="button" onClick={props.onWhole} disabled={!region} className="h-touch rounded-control border border-rule-control px-3 text-control text-ink hover:bg-paper disabled:text-ink-2">
              Use whole photo
            </button>
          )}
        </div>
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

function TryRegionButton({ onClick, tall }: { onClick: () => void; tall?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`h-touch rounded-control border border-rule-control px-3 text-control text-ink hover:bg-paper ${tall ? "flex-1" : "self-start"}`}
    >
      Try a different region
    </button>
  );
}

function ResultSection({ search, onTryRegion }: { search: Search; onTryRegion: () => void }) {
  const p = search.prediction;
  const at = { lat: p.predicted_lat, lng: p.predicted_lng };
  const low = p.confidence_state !== "ok";
  return (
    <Section>
      <StateLine state={p.confidence_state} />
      <h2 className="text-xl font-semibold tracking-[var(--text-xl--letter-spacing)] text-ink">{placeLabel(search.places[0])}</h2>
      <p className="flex items-center gap-3 text-sm text-ink-2">
        {formatCoords(at)}
        <CopyButton text={plainCoords(at)} />
      </p>
      <p className="text-body text-ink">{reasonText(p, usedRegion(p))}</p>
      {low && <TryRegionButton onClick={onTryRegion} />}
    </Section>
  );
}

function MobileResult({ search, onCompare, onTryRegion }: { search: Search; onCompare: () => void; onTryRegion: () => void }) {
  const p = search.prediction;
  const at = { lat: p.predicted_lat, lng: p.predicted_lng };
  return (
    <section className="flex flex-col gap-3 border-b border-rule px-5 py-4">
      <div className="flex items-start gap-3">
        <img src={search.url} alt="Your photo" className="size-touch shrink-0 object-cover" />
        <div className="min-w-0">
          <StateLine state={p.confidence_state} />
          <h2 className="text-lg font-semibold text-ink">{placeLabel(search.places[0])}</h2>
          <p className="text-sm text-ink-2">{formatCoords(at)}</p>
        </div>
      </div>
      <p className="text-body text-ink">{reasonText(p, usedRegion(p))}</p>
      <div className="flex gap-2">
        <button type="button" onClick={onCompare} className="h-touch flex-[2] rounded-control bg-pin text-control font-medium text-surface hover:bg-pin-hover">
          Compare with match
        </button>
        <CopyButton text={plainCoords(at)} tall />
      </div>
      {p.confidence_state !== "ok" && <TryRegionButton onClick={onTryRegion} />}
    </section>
  );
}

function Candidates({ search, selected, onSelect }: { search: Search; selected: number; onSelect: (i: number) => void }) {
  const p = search.prediction;
  const best = { lat: p.predicted_lat, lng: p.predicted_lng };
  const rows = useMemo(
    () => p.top_matches.slice(1).map((m, i) => ({ index: i + 1, match: m, place: search.places[i + 1] })),
    [p, search.places],
  );
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
                <span className="block text-sm text-ink-2">{formatDistance(distanceM(best, match))} from the best match</span>
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
  // "Your region" is the part of the photo whose match won (manual, crop or whole).
  return <Compare photoUrl={search.url} region={p.searched_region ?? search.region} match={match} predicted={{ lat: p.predicted_lat, lng: p.predicted_lng }} tall={tall} />;
}
