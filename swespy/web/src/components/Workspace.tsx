"use client";

/* eslint-disable @next/next/no-img-element -- local blob and preview thumbnails */

import * as Dialog from "@radix-ui/react-dialog";
import * as Tooltip from "@radix-ui/react-tooltip";
import Link from "next/link";
import { type CSSProperties, type ReactNode, type RefObject, useCallback, useEffect, useRef, useState } from "react";

import CompareOverlay from "@/components/CompareOverlay";
import { IconClose, IconCopy, IconHelp, IconNew, IconRecent, IconSideBySide, IconUpload, LogoMark } from "@/components/Icons";
import LocatingMap, { type LocatingMapHandle, type MapInsets, type MapItem } from "@/components/LocatingMap";
import PhotoRegion from "@/components/PhotoRegion";
import VerdictControl, { type Verdict } from "@/components/Verdict";
import { type Prediction, type Region, UserFacingError, analyze, checkFile, sourceName } from "@/lib/api";
import { distanceM, formatCoords, formatDistance, plainCoords } from "@/lib/geo";
import { type Place, loadPlaces, lookup, placeLabel } from "@/lib/places";
import { type CandidatePoint, type StreamEvent, locateStream } from "@/lib/stream";

interface Search {
  id: number;
  file: File;
  url: string; // What the browser can display: the file itself, or the API's preview.
  region: Region | null;
  prediction: Prediction;
  places: Place[];
  timings: Timings;
  verdicts: Record<string, Verdict>; // By reference image id; this session only.
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
  { key: "search", label: "Search references" },
  { key: "compare", label: "Compare closest matches" },
  { key: "check", label: "Check confidence" },
];

// Minimum time each phase stays on screen, so fast answers still read clearly (ms).
const HOLD = { begin: 900, candidates: 1000, verification: 1200, verified: 180 } as const;
const HOLD_REDUCED = { begin: 250, candidates: 250, verification: 250, verified: 60 } as const;

// Notes under the photo when SweSpy pre-draws a region.
const NOTES = {
  person: "Most of this photo is a person. The highlighted area shows the place.",
  alternative: "Suggested: another part of the photo.",
} as const;
type Note = keyof typeof NOTES;

const RIVAL_M = 500; // swespy.model.rerank.DISAGREE_M: a different place.
const NO_INSETS: MapInsets = { top: 0, right: 0, bottom: 0, left: 0 };

const SEARCHED: Record<string, string> = {
  whole: "Whole photo",
  manual: "Your region",
  masked: "Automatic: people masked",
  "crop:left": "Automatic: left part",
  "crop:right": "Automatic: right part",
  "crop:center": "Automatic: centre part",
  "crop:band": "Automatic: middle band",
  "crop:clean": "Automatic: area without people",
};

const isHeic = (f: File) => /^image\/hei[cf]$/.test(f.type) || /\.hei[cf]$/i.test(f.name);
const reducedMotion = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const total = (t: Timings) => (t.search ?? 0) + (t.compare ?? 0) + (t.check ?? 0);

/** Screen space covered by the floating rail and inspector, measured live. */
function useInsets(rail: RefObject<HTMLElement | null>, panel: RefObject<HTMLElement | null>): MapInsets {
  const [insets, setInsets] = useState<MapInsets>(NO_INSETS);
  useEffect(() => {
    const measure = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      const r = rail.current?.getBoundingClientRect();
      const p = panel.current?.getBoundingClientRect();
      const gap = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--spacing-inset")) || 12;
      const side = Boolean(p && p.height > h * 0.8); // Desktop: a full-height side panel.
      setInsets({
        top: gap,
        left: r && r.width ? Math.round(r.right + gap) : gap,
        right: side && p ? Math.round(w - p.left + gap) : gap,
        bottom: !side && p ? Math.round(h - p.top + gap) : gap,
      });
    };
    measure();
    const ro = new ResizeObserver(measure);
    if (rail.current) ro.observe(rail.current);
    if (panel.current) ro.observe(panel.current);
    window.addEventListener("resize", measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [rail, panel]);
  return insets;
}

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
  const [comparing, setComparing] = useState<number | null>(null);
  const [timings, setTimings] = useState<Timings>({ search: null, compare: null, check: null });
  const [emptyError, setEmptyError] = useState<string | null>(null);
  const [note, setNote] = useState<Note | null>(null);
  const mapRef = useRef<LocatingMapHandle>(null);
  const railRef = useRef<HTMLElement>(null);
  const panelRef = useRef<HTMLElement>(null);
  const photoRef = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null);
  const analyzing = useRef<AbortController | null>(null);
  const runId = useRef(0);
  const insets = useInsets(railRef, panelRef);

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
    // HEIC: Chrome cannot show it; wait for the API's upright JPEG preview.
    setUrl(isHeic(f) ? null : URL.createObjectURL(f));
    setRegion(null);
    setNote(null);
    setEditing(true);
    setNoPreview(false);
    setResult(null);
    setComparing(null);
    setPhase({ kind: "ready" });
    suggest(f);
  }

  /** People share, the suggested region and a display preview, before searching. */
  function suggest(f: File) {
    analyzing.current?.abort();
    const controller = new AbortController();
    analyzing.current = controller;
    void analyze(f, controller.signal).then((a) => {
      if (controller.signal.aborted) return;
      if (isHeic(f)) {
        if (a?.preview) setUrl(a.preview);
        else setNoPreview(true);
      }
      if (!a?.suggest || !a.suggested_region) return;
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

  function stopRun() {
    runId.current++;
    abort.current?.abort();
    abort.current = null;
  }

  function reset() {
    stopRun();
    stopAnalyze();
    mapRef.current?.reset();
    setFile(null);
    setUrl(null);
    setRegion(null);
    setNote(null);
    setResult(null);
    setComparing(null);
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

  /** Low confidence: open region selection with the best automatic alternative drawn. */
  function tryDifferentRegion() {
    if (!result) return;
    stopAnalyze();
    setComparing(null);
    setRegion(result.prediction.alternative_region);
    setNote(result.prediction.alternative_region ? "alternative" : null);
    setEditing(true);
    photoRef.current?.scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" });
  }

  /** Locate: stream events into a queue and play them, each phase held long enough. */
  async function locate() {
    if (!file || (!url && !noPreview)) return;
    stopRun();
    stopAnalyze();
    setNote(null);
    setComparing(null);
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
          failure = err instanceof UserFacingError ? err.message : "Locating failed. Try again.";
        }
      })
      .finally(() => {
        finished = true;
        wake?.();
      });

    let nextAt = performance.now() + hold.begin;
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
        setPhase({ kind: "locating", stage: "compare", label: `Comparing the ${ev.count} closest matches` });
        mapRef.current?.verifying(candidates.slice(0, ev.count));
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
        const search: Search = {
          id, file, url: url ?? "", region, prediction, places: at.map((p) => lookup(index, p)), timings: done, verdicts: {},
        };
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
    setPhase({ kind: "error", message: failure ?? "No result returned. Try again." });
  }

  function restore(s: Search) {
    stopRun();
    stopAnalyze();
    setNote(null);
    setComparing(null);
    setFile(s.file);
    setUrl(s.url || null);
    setNoPreview(!s.url);
    setRegion(s.region);
    setEditing(false);
    setResult(s);
    setTimings(s.timings);
    setSelected(0);
    setPhase({ kind: "done" });
    mapRef.current?.result(mapItems(s), s.prediction.confidence_state);
  }

  const selectCandidate = useCallback((index: number) => {
    setSelected(index);
    mapRef.current?.select(index);
  }, []);

  const openCompare = useCallback((index: number) => {
    setSelected(index);
    mapRef.current?.select(index);
    setComparing(index);
  }, []);

  /** Store a verdict on the current result and on its entry in recent searches. */
  function setVerdict(imageId: string, v: Verdict | undefined) {
    if (!result) return;
    const verdicts = { ...result.verdicts };
    if (v) verdicts[imageId] = v;
    else delete verdicts[imageId];
    const next = { ...result, verdicts };
    setResult(next);
    setRecent((list) => list.map((s) => (s.id === next.id ? next : s)));
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

  const overlayStyle: CSSProperties = { top: insets.top, left: insets.left, right: insets.right, bottom: insets.bottom };
  const compareMatch = result && comparing !== null ? result.prediction.top_matches[comparing] : null;

  return (
    <Tooltip.Provider delayDuration={300}>
      <div className="relative h-full overflow-hidden text-sm text-ink">
        {/* The map is the workspace: full-bleed under the floating panels. */}
        <div className="absolute inset-0">
          <LocatingMap
            ref={mapRef}
            area={area}
            status={status}
            insets={insets}
            onSelect={selectCandidate}
            onCompare={openCompare}
            onDropFile={choose}
          />
        </div>

        <nav
          ref={railRef}
          aria-label="Workspace"
          className="absolute bottom-inset left-inset top-inset z-20 hidden w-rail flex-col items-center gap-1 rounded-panel border border-rule bg-surface py-2 shadow-[var(--shadow-panel)] desk:flex"
        >
          <div className="mb-2 grid size-9 place-items-center" title="SweSpy">
            <LogoMark />
          </div>
          <RailButton label="New search" onClick={reset}>
            <IconNew />
          </RailButton>
          <RecentSearches recent={recent} onRestore={restore} />
          <div className="mt-auto">
            <Tooltip.Root>
              <Tooltip.Trigger asChild>
                <Link href="/about" aria-label="How it works" className="grid size-9 place-items-center rounded-control text-ink hover:bg-paper">
                  <IconHelp />
                </Link>
              </Tooltip.Trigger>
              <TipContent>How it works</TipContent>
            </Tooltip.Root>
          </div>
        </nav>

        <aside
          ref={panelRef}
          aria-label="Inspector"
          className="absolute inset-x-inset bottom-inset z-20 flex max-h-sheet flex-col overflow-y-auto rounded-panel border border-rule bg-surface shadow-[var(--shadow-panel)] desk:left-auto desk:top-inset desk:max-h-none desk:w-inspector"
        >
          {!file && <EmptyState references={references} onFile={choose} error={emptyError} />}
          {file && (
            <>
              <header className="flex items-center gap-2 border-b border-rule px-3 py-2">
                <div className="min-w-0 flex-1">
                  <h1 className="truncate text-sm font-medium text-ink">{file.name}</h1>
                  <PhaseLine phase={phase} hasResult={Boolean(result)} />
                </div>
                <button type="button" onClick={reset} aria-label="New photo" title="New photo" className="grid size-8 shrink-0 place-items-center rounded-control border border-rule-control text-ink hover:bg-paper">
                  <IconNew />
                </button>
              </header>
              <div ref={photoRef}>
                <PhotoSection
                  url={url}
                  heicPending={isHeic(file) && !url && !noPreview}
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
              {(locating || phase.kind === "error") && (
                <Section>
                  <StageList state={stageState} timings={timings} />
                </Section>
              )}
              {result && (
                <div className={locating ? "pointer-events-none opacity-40" : ""} inert={locating}>
                  <ResultBlock search={result} onCompare={() => openCompare(0)} onTryRegion={tryDifferentRegion} />
                  <Candidates search={result} selected={selected} onSelect={selectCandidate} onCompare={openCompare} onVerdict={setVerdict} />
                </div>
              )}
            </>
          )}
        </aside>

        {result && compareMatch && result.url && (
          <CompareOverlay
            photoUrl={result.url}
            region={result.prediction.searched_region ?? result.region}
            match={compareMatch}
            place={placeLabel(result.places[comparing ?? 0])}
            predicted={{ lat: result.prediction.predicted_lat, lng: result.prediction.predicted_lng }}
            verdict={result.verdicts[compareMatch.image_id]}
            onVerdict={(v) => setVerdict(compareMatch.image_id, v)}
            onClose={() => setComparing(null)}
            style={overlayStyle}
          />
        )}
      </div>
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

function Section({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-2 border-b border-rule px-3 py-3 last:border-b-0">
      {title && <h2 className="text-xs font-semibold text-ink-2">{title}</h2>}
      {children}
    </section>
  );
}

/** Label–value rows: label 12 px secondary on the left, value 13 px tabular on the right. */
function KeyValues({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-xs leading-5 text-ink-2">{k}</dt>
          <dd className="truncate text-right text-sm leading-5 text-ink">{v}</dd>
        </div>
      ))}
    </dl>
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
    <div role="status" aria-live="polite" className="flex max-w-sm items-center gap-3 rounded-control border border-ink bg-surface px-3 py-1.5">
      {!error && <ProgressArc />}
      <span className={`text-sm ${error ? "text-danger" : "text-ink"}`}>{text}</span>
      {onCancel && (
        <button type="button" onClick={onCancel} className="ml-1 text-sm font-medium text-pin underline underline-offset-2 hover:text-pin-hover">
          Cancel
        </button>
      )}
      {onRetry && (
        <button type="button" onClick={onRetry} className="ml-1 shrink-0 text-sm font-medium text-pin underline underline-offset-2 hover:text-pin-hover">
          Try again
        </button>
      )}
    </div>
  );
}

function StageMarker({ state }: { state: StageState }) {
  if (state === "active") return <ProgressArc />;
  return <span className={`size-2.5 shrink-0 border border-ink ${state === "done" ? "bg-ink" : "bg-surface"}`} aria-hidden="true" />;
}

function StageList({ state, timings }: { state: (k: StageKey) => StageState; timings: Timings }) {
  return (
    <ol className="flex flex-col gap-1.5" aria-label="Progress">
      {STAGES.map(({ key, label }) => (
        <li key={key} className="flex items-center gap-2.5 text-sm">
          <StageMarker state={state(key)} />
          <span className={state(key) === "pending" ? "text-ink-2" : "text-ink"}>{label}</span>
          <span className="ml-auto text-xs text-ink-2">
            {timings[key] !== null ? `${timings[key]!.toFixed(1)} s` : state(key) === "active" ? "Running" : ""}
          </span>
          <span className="sr-only">{state(key)}</span>
        </li>
      ))}
    </ol>
  );
}

function PhaseLine({ phase, hasResult }: { phase: Phase; hasResult: boolean }) {
  const text =
    phase.kind === "locating" ? "Locating" : phase.kind === "error" ? "Not located" : phase.kind === "ready" ? (hasResult ? "Region changed" : "Ready") : null;
  if (!text) return null;
  return <p className={`text-xs ${phase.kind === "error" ? "text-danger" : "text-ink-2"}`}>{text}</p>;
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
        <button type="button" aria-label={label} onClick={onClick} className="grid size-9 place-items-center rounded-control text-ink hover:bg-paper">
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
          <Dialog.Trigger aria-label="Recent searches" className="grid size-9 place-items-center rounded-control text-ink hover:bg-paper">
            <IconRecent />
          </Dialog.Trigger>
        </Tooltip.Trigger>
        <TipContent>Recent searches</TipContent>
      </Tooltip.Root>
      <Dialog.Portal>
        <Dialog.Content
          className="fixed bottom-inset top-inset z-50 flex w-inspector flex-col overflow-hidden rounded-panel border border-rule bg-surface text-sm shadow-[var(--shadow-panel)]"
          style={{ left: "calc(var(--spacing-inset) * 2 + var(--spacing-rail))" }}
          aria-describedby={undefined}
        >
          <div className="flex items-center justify-between border-b border-rule px-3 py-2">
            <Dialog.Title className="text-sm font-semibold text-ink">Recent searches</Dialog.Title>
            <Dialog.Close aria-label="Close" className="grid size-8 place-items-center rounded-control text-ink hover:bg-paper">
              <IconClose />
            </Dialog.Close>
          </div>
          <p className="px-3 py-2 text-xs text-ink-2">This session only</p>
          {recent.length === 0 && <p className="px-3 text-sm text-ink-2">No searches yet</p>}
          <ul className="flex flex-col overflow-y-auto">
            {recent.map((s) => {
              const judged = Object.keys(s.verdicts).length;
              return (
                <li key={s.id} className="border-b border-rule-list">
                  <Dialog.Close asChild>
                    <button type="button" onClick={() => onRestore(s)} className="flex w-full items-center gap-3 px-3 py-2 text-left hover:bg-paper">
                      {s.url ? <img src={s.url} alt="" className="size-9 rounded-square object-cover" /> : <span className="size-9 rounded-square bg-rule-list" />}
                      <span className="min-w-0">
                        <span className="block truncate text-sm font-medium text-ink">{placeLabel(s.places[0])}</span>
                        <span className="block truncate text-xs text-ink-2">
                          {s.file.name}
                          {judged ? `, ${judged} verdict${judged > 1 ? "s" : ""}` : ""}
                        </span>
                      </span>
                    </button>
                  </Dialog.Close>
                </li>
              );
            })}
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
    <div className="flex flex-col">
      <header className="flex items-center gap-2 border-b border-rule px-3 py-2">
        <span className="desk:hidden">
          <LogoMark />
        </span>
        <h1 className="text-sm font-semibold text-ink">SweSpy</h1>
        <span className="text-xs text-ink-2">Locate a photo</span>
      </header>
      <Section>
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
          className={`flex flex-col items-start gap-2 rounded-control border border-dashed p-3 ${over ? "border-pin bg-selected" : "border-rule-control"}`}
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
          <button type="button" onClick={() => input.current?.click()} className="flex h-8 items-center gap-2 rounded-control bg-pin px-3 text-sm font-medium text-surface hover:bg-pin-hover">
            <IconUpload />
            Choose a photo
          </button>
          <p className="text-xs text-ink-2">Or drop it here or on the map</p>
        </div>
        {error && <p role="alert" className="text-sm text-danger">{error}</p>}
      </Section>
      <Section title="Details">
        <KeyValues
          rows={[
            ["Formats", "JPEG, PNG, HEIC, up to 10 MB"],
            ["Coverage", "Central Linköping"],
            ["References", references.toLocaleString("en-US")],
            ["GPS in the file", "Ignored"],
            ["Storage", "In memory only"],
          ]}
        />
      </Section>
    </div>
  );
}

function PhotoSection(props: {
  url: string | null;
  heicPending: boolean;
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
  const { url, region, editing, noPreview, hasResult, locating, heicPending } = props;
  const canLocate = !locating && !heicPending && (!hasResult || editing || noPreview);
  return (
    <section className="flex flex-col border-b border-rule">
      {heicPending ? (
        <p className="grid h-photo place-items-center bg-rule-list text-xs text-ink-2">Preparing preview</p>
      ) : noPreview ? (
        <p className="grid h-photo place-items-center bg-rule-list px-4 text-center text-xs text-ink-2">No preview. The whole photo is searched.</p>
      ) : (
        <PhotoRegion src={url} region={region} editing={editing} searched={props.searched} onChange={props.onRegion} onUnavailable={props.onUnavailable} />
      )}
      <div className="flex flex-col gap-2 px-3 py-2">
        {props.note && !noPreview && <p className="text-xs text-ink">{props.note}</p>}
        {editing && !noPreview && !heicPending && !props.note && (
          <p className="text-xs text-ink-2">{region ? "Drag the edges to adjust" : "Drag to select the part that shows the place"}</p>
        )}
        <div className="flex flex-wrap gap-1.5">
          {canLocate && (
            <button type="button" onClick={props.onLocate} className="h-8 rounded-control bg-pin px-3 text-sm font-medium text-surface hover:bg-pin-hover">
              Locate
            </button>
          )}
          {hasResult && !editing && !locating && !noPreview && (
            <button type="button" onClick={props.onChangeRegion} className="h-8 rounded-control border border-rule-control px-3 text-sm text-ink hover:bg-paper">
              Change region
            </button>
          )}
          {editing && !noPreview && !heicPending && (
            <button type="button" onClick={props.onWhole} disabled={!region} className="h-8 rounded-control border border-rule-control px-3 text-sm text-ink hover:bg-paper disabled:text-ink-2">
              Use whole photo
            </button>
          )}
        </div>
      </div>
    </section>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      aria-label={copied ? "Copied" : "Copy coordinates"}
      title={copied ? "Copied" : "Copy coordinates"}
      onClick={async () => {
        await navigator.clipboard?.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }}
      className={`grid size-7 place-items-center rounded-control hover:bg-paper ${copied ? "text-sure" : "text-ink-2 hover:text-ink"}`}
    >
      <IconCopy />
    </button>
  );
}

/** State, place, coordinates and the facts behind them, as label–value rows. */
function ResultBlock({ search, onCompare, onTryRegion }: { search: Search; onCompare: () => void; onTryRegion: () => void }) {
  const p = search.prediction;
  const at = { lat: p.predicted_lat, lng: p.predicted_lng };
  const ok = p.confidence_state === "ok";
  const [top, ...others] = p.top_matches;
  const rival = others
    .filter((m) => distanceM(top, m) > RIVAL_M)
    .sort((a, b) => (b.inliers ?? 0) - (a.inliers ?? 0))[0];
  const ref = [
    top.source === "gsv" ? "Google Street View, © Google" : sourceName(top.source),
    top.capture_year ? String(top.capture_year) : null,
    `${formatDistance(distanceM(top, at))} away`,
  ]
    .filter(Boolean)
    .join(", ");
  return (
    <Section>
      <p className={`flex items-center gap-2 text-xs font-semibold ${ok ? "text-sure" : "text-unsure"}`}>
        <span className={`size-state ${ok ? "bg-sure" : "bg-unsure"}`} aria-hidden="true" />
        {ok ? "Confident" : "Low confidence"}
      </p>
      <div>
        <h2 className="text-place font-semibold leading-tight text-ink">{placeLabel(search.places[0])}</h2>
        <p className="flex items-center gap-1 text-sm text-ink-2">
          {formatCoords(at)}
          <CopyButton text={plainCoords(at)} />
        </p>
      </div>
      <KeyValues
        rows={[
          ["Matched details", String(top.inliers ?? 0)],
          ["Closest rival", rival ? `${rival.inliers ?? 0}, ${formatDistance(distanceM(top, rival))} away` : "None"],
          ["Searched area", SEARCHED[p.searched_variant] ?? "Automatic"],
          ["Reference", ref],
          ["Time", `${total(search.timings).toFixed(1)} s`],
        ]}
      />
      {!ok && <p className="text-sm text-unsure">Too few matching details to be sure. Compare before relying on it.</p>}
      <div className="flex flex-wrap gap-1.5 pt-1">
        <button type="button" onClick={onCompare} className="h-8 rounded-control bg-pin px-3 text-sm font-medium text-surface hover:bg-pin-hover">
          Compare
        </button>
        {!ok && (
          <button type="button" onClick={onTryRegion} className="h-8 rounded-control border border-rule-control px-3 text-sm text-ink hover:bg-paper">
            Try a different region
          </button>
        )}
      </div>
    </Section>
  );
}

function Candidates({
  search,
  selected,
  onSelect,
  onCompare,
  onVerdict,
}: {
  search: Search;
  selected: number;
  onSelect: (i: number) => void;
  onCompare: (i: number) => void;
  onVerdict: (imageId: string, v: Verdict | undefined) => void;
}) {
  const p = search.prediction;
  const first = p.top_matches[0];
  const most = Math.max(1, first?.inliers ?? 0);
  return (
    <Section title="Candidates">
      <ul className="-mx-3 flex flex-col">
        {p.top_matches.map((m, i) => {
          const details = m.inliers ?? 0;
          return (
            <li key={m.image_id} className={`flex items-center gap-2 px-3 py-1.5 ${selected === i ? "outline-2 -outline-offset-2 outline-pin" : ""}`}>
              <button
                type="button"
                aria-pressed={selected === i}
                onClick={() => onSelect(i)}
                className="grid min-w-0 flex-1 grid-cols-[1rem_1fr_auto] items-center gap-x-2 text-left"
              >
                <span className="text-xs font-semibold text-ink">{m.rank}</span>
                <span className="min-w-0">
                  <span className="block truncate text-sm font-medium text-ink">{placeLabel(search.places[i])}</span>
                  <span className="block text-xs text-ink-2">{i === 0 ? "Top match" : `${formatDistance(distanceM(first, m))} from #1`}</span>
                </span>
                <span className="flex items-center gap-1.5" title={`${details} matched details`}>
                  <span className="text-sm text-ink">{details}</span>
                  <span className="h-1 w-bar overflow-hidden rounded-square bg-rule-list" aria-hidden="true">
                    <span className="block h-full bg-ink-2" style={{ width: `${Math.min(100, (100 * details) / most)}%` }} />
                  </span>
                </span>
              </button>
              <button
                type="button"
                onClick={() => onCompare(i)}
                aria-label={`Compare with candidate ${m.rank}`}
                title="Compare"
                className="grid size-7 shrink-0 place-items-center rounded-control text-pin hover:bg-selected hover:text-pin-hover"
              >
                <IconSideBySide />
              </button>
              <VerdictControl value={search.verdicts[m.image_id]} onChange={(v) => onVerdict(m.image_id, v)} subject={`candidate ${m.rank}`} />
            </li>
          );
        })}
      </ul>
    </Section>
  );
}
