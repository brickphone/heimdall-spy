"use client";

/* eslint-disable @next/next/no-img-element -- previews are blobs and proxied API images */

import { useEffect, useRef, useState } from "react";

import ResultMap from "@/components/ResultMap";
import {
  type LatLng,
  type Prediction,
  isReady,
  licenceName,
  predict,
  referenceImageUrl,
  saveFieldPhoto,
  sourceName,
} from "@/lib/api";

type Run =
  | { state: "idle" }
  | { state: "running"; started: number }
  | { state: "error"; message: string }
  | { state: "done"; prediction: Prediction; totalMs: number };

type Save =
  | { state: "idle" }
  | { state: "saving" }
  | { state: "saved"; id: string }
  | { state: "error"; message: string };

export default function Demo() {
  const [file, setFile] = useState<File | null>(null);
  const [keep, setKeep] = useState(false); // Off by default: nothing is stored.
  const [run, setRun] = useState<Run>({ state: "idle" });
  const [truth, setTruth] = useState<LatLng | null>(null);
  const [save, setSave] = useState<Save>({ state: "idle" });
  const [preview, setPreview] = useState<string | null>(null);
  const previewRef = useRef<string | null>(null);
  const ready = useApiReady();

  // Local object URL for the chosen file (shown next to the match; not uploaded for this).
  function choose(next: File | null) {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = next ? URL.createObjectURL(next) : null;
    setPreview(previewRef.current);
    setFile(next);
    setRun({ state: "idle" });
  }
  useEffect(() => () => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
  }, []);

  async function locate(event: React.FormEvent) {
    event.preventDefault();
    if (!file) return;
    const started = performance.now();
    setRun({ state: "running", started });
    setTruth(null);
    setSave({ state: "idle" });
    try {
      const prediction = await predict(file);
      setRun({ state: "done", prediction, totalMs: performance.now() - started });
    } catch (error) {
      setRun({ state: "error", message: (error as Error).message });
    }
  }

  async function store() {
    if (!file || run.state !== "done" || !truth) return;
    setSave({ state: "saving" });
    try {
      setSave({ state: "saved", id: await saveFieldPhoto(file, run.prediction, truth) });
    } catch (error) {
      setSave({ state: "error", message: (error as Error).message });
    }
  }

  const picking = keep && run.state === "done" && save.state !== "saved";

  return (
    <div className="space-y-4">
      <form
        onSubmit={locate}
        className="flex flex-wrap items-center gap-4 rounded-xl border border-line bg-card p-4"
      >
        <input
          type="file"
          required
          accept="image/jpeg,image/png,image/heic,image/heif,.heic,.heif"
          onChange={(event) => choose(event.target.files?.[0] ?? null)}
          className="text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-accent/10 file:px-3 file:py-2 file:font-medium file:text-accent hover:file:bg-accent/20"
        />
        <label className="flex items-center gap-2 text-sm text-muted">
          <input
            type="checkbox"
            checked={keep}
            onChange={(event) => setKeep(event.target.checked)}
            className="size-4 accent-accent"
          />
          Save this photo for later evaluation
        </label>
        <button
          type="submit"
          disabled={!ready || !file || run.state === "running"}
          className="ml-auto rounded-lg bg-accent px-5 py-2 font-medium text-on-solid transition hover:opacity-90 disabled:opacity-40"
        >
          {run.state === "running" ? "Locating…" : "Locate"}
        </button>
      </form>

      <div role="status" aria-live="polite" className="min-h-6 text-sm text-muted">
        {!ready && "Waiting for the API to load the model and gallery (about a minute)…"}
        {run.state === "running" && "Locating… usually 2–5 s on a Mac, ~20 s on CPU."}
        {run.state === "error" && <span className="text-low">Error: {run.message}</span>}
      </div>

      {run.state === "done" && (
        <Result
          prediction={run.prediction}
          totalMs={run.totalMs}
          preview={preview}
          picking={picking}
          truth={keep ? truth : null}
          onPickTruth={setTruth}
        />
      )}

      {run.state === "done" && keep && (
        <section className="rounded-xl border border-line bg-card p-4">
          <h2 className="font-semibold">Save for later evaluation</h2>
          <p className="mt-1 text-sm text-muted">
            Click the map where the photo was actually taken (drag the green pin to adjust),
            then save. Saved photos are never added to the reference gallery or used for
            tuning. Location metadata is removed from the stored copy.
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <button
              onClick={store}
              disabled={!truth || save.state === "saving" || save.state === "saved"}
              className="rounded-lg bg-ok px-4 py-2 font-medium text-on-solid transition hover:opacity-90 disabled:opacity-40"
            >
              Save photo and true location
            </button>
            <span className="text-sm text-muted">
              {!truth && "No true location yet."}
              {truth && save.state === "idle" && `${truth.lat.toFixed(5)}, ${truth.lng.toFixed(5)}`}
              {save.state === "saving" && "Saving…"}
              {save.state === "saved" && `Saved (${save.id})`}
              {save.state === "error" && <span className="text-low">Error: {save.message}</span>}
            </span>
          </div>
        </section>
      )}
    </div>
  );
}

function Result({
  prediction,
  totalMs,
  preview,
  picking,
  truth,
  onPickTruth,
}: {
  prediction: Prediction;
  totalMs: number;
  preview: string | null;
  picking: boolean;
  truth: LatLng | null;
  onPickTruth: (point: LatLng) => void;
}) {
  const confident = prediction.confidence_state === "ok";
  const best = prediction.top_matches[0];
  return (
    <>
      <section className="flex flex-wrap items-center gap-3 rounded-xl border border-line bg-card p-4">
        <span
          className={`rounded-full px-3 py-1 text-sm font-semibold ${
            confident ? "bg-ok/15 text-ok" : "bg-low/15 text-low"
          }`}
        >
          {confident ? "Confident" : "Low confidence"}
        </span>
        <span className="text-sm text-muted">{prediction.confidence_reasons.join("; ")}</span>
        <span className="ml-auto text-sm tabular-nums text-muted">
          Processed in {(prediction.processing_time_ms / 1000).toFixed(1)} s (server) ·{" "}
          {(totalMs / 1000).toFixed(1)} s total
        </span>
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="rounded-xl border border-line bg-card p-2">
          <ResultMap
            prediction={prediction}
            truth={truth}
            onPickTruth={picking ? onPickTruth : undefined}
          />
          <p className="px-2 pt-2 text-xs text-muted">
            Blue pin: predicted location. Orange dots: candidates 2–5.
            {truth && " Green pin: true location."}
            {picking && !truth && " Click the map to mark the true location."}
          </p>
        </section>

        <section className="space-y-4 rounded-xl border border-line bg-card p-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <figure>
              <Thumb key={preview} src={preview} alt="Your photo" fallback="Preview not supported (HEIC)" />
              <figcaption className="mt-1.5 text-xs text-muted">Your photo</figcaption>
            </figure>
            {best && (
              <figure>
                <Thumb
                  key={best.image_id}
                  src={referenceImageUrl(best.image_id)} alt="Best matching reference" />
                <figcaption className="mt-1.5 text-xs text-muted">
                  Best match:{" "}
                  {best.source_url ? (
                    <a href={best.source_url} target="_blank" rel="noopener noreferrer">
                      {sourceName(best.source)}
                    </a>
                  ) : (
                    sourceName(best.source)
                  )}{" "}
                  · © {best.creator || "unknown"} ·{" "}
                  {best.license_url ? (
                    <a href={best.license_url} target="_blank" rel="noopener noreferrer">
                      {licenceName(best.license_url)}
                    </a>
                  ) : (
                    "licence unknown"
                  )}{" "}
                  · resized{best.capture_year ? ` · ${best.capture_year}` : ""}
                </figcaption>
              </figure>
            )}
          </div>
          <Candidates prediction={prediction} />
        </section>
      </div>

      <details className="rounded-xl border border-line bg-card p-4 text-sm text-muted">
        <summary className="cursor-pointer">Versions (what produced this result)</summary>
        <pre className="mt-2 overflow-x-auto font-mono text-xs whitespace-pre-wrap break-words">
          {JSON.stringify(prediction.versions, null, 2)}
        </pre>
      </details>
    </>
  );
}

function Candidates({ prediction }: { prediction: Prediction }) {
  return (
    <table className="w-full text-sm">
      <thead className="text-left text-muted">
        <tr className="border-b border-line">
          <th className="py-1.5 pr-2 font-medium">#</th>
          <th className="py-1.5 pr-2 font-medium">Source</th>
          <th className="py-1.5 pr-2 text-right font-medium">Similarity</th>
          <th className="py-1.5 pr-2 text-right font-medium">Inliers</th>
          <th className="py-1.5 text-right font-medium">Score</th>
        </tr>
      </thead>
      <tbody className="tabular-nums">
        {prediction.top_matches.map((match) => (
          <tr key={match.image_id} className="border-b border-line last:border-0">
            <td className="py-1.5 pr-2">{match.rank}</td>
            <td className="py-1.5 pr-2">
              {match.source_url ? (
                <a href={match.source_url} target="_blank" rel="noopener noreferrer">
                  {sourceName(match.source)}
                </a>
              ) : (
                sourceName(match.source)
              )}
            </td>
            <td className="py-1.5 pr-2 text-right">{match.similarity_score.toFixed(3)}</td>
            <td className="py-1.5 pr-2 text-right">{match.inliers ?? "–"}</td>
            <td className="py-1.5 text-right">{match.score?.toFixed(3) ?? "–"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Thumb({ src, alt, fallback }: { src: string | null; alt: string; fallback?: string }) {
  const [failed, setFailed] = useState(false); // Reset by remounting (key = src).
  if (!src || failed) {
    return (
      <div className="flex aspect-[4/3] items-center justify-center rounded-lg bg-muted/10 p-3 text-center text-xs text-muted">
        {fallback ?? "Image unavailable"}
      </div>
    );
  }
  return (
    <img
      src={src}
      alt={alt}
      onError={() => setFailed(true)}
      className="aspect-[4/3] w-full rounded-lg bg-muted/10 object-contain"
    />
  );
}

/** Polls GET /ready every 3 s until the API reports the model and gallery are loaded. */
function useApiReady(): boolean {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      if (await isReady()) {
        if (!stopped) setReady(true);
        return;
      }
      if (!stopped) timer = setTimeout(poll, 3000);
    };
    void poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, []);
  return ready;
}
