// Client for POST /predict/stream (NDJSON, one event per line). The proxy passes the
// stream through unbuffered; aborting the signal closes the connection, which makes the
// API stop working on the request.
import { type Prediction, type Region, UserFacingError } from "./api";

export interface CandidatePoint {
  id: number; // Retrieval rank, 0-based.
  lat: number;
  lng: number;
  similarity: number;
}

export type StreamEvent =
  | { type: "stage"; stage: "retrieval" }
  | { type: "stage"; stage: "verification"; count: number }
  | { type: "candidates"; items: CandidatePoint[] }
  | { type: "verified"; id: number; inliers: number; score: number }
  | ({ type: "result" } & Prediction)
  | { type: "error"; message: string };

function withMimeType(file: File): File {
  if (file.type) return file;
  const type = /\.hei[cf]$/i.test(file.name) ? "image/heic" : "image/jpeg";
  return new File([file], file.name, { type });
}

async function explain(response: Response): Promise<UserFacingError> {
  let detail = "";
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // Not JSON.
  }
  if (response.status >= 500) {
    return new UserFacingError(
      "The location service isn't running. Start it with scripts/run_local.sh and try again.",
    );
  }
  if (/supported|unreadable/i.test(detail)) {
    return new UserFacingError("This file isn't an image. Choose a JPEG, PNG or HEIC.");
  }
  if (/10MB|too large/i.test(detail)) {
    return new UserFacingError(
      "This photo is larger than 10 MB. Choose a smaller copy, or export it at a lower resolution.",
    );
  }
  if (/region/i.test(detail)) return new UserFacingError(`${detail}. Draw a larger region, or use the whole photo.`);
  return new UserFacingError(detail || `The search failed (${response.status}). Try again.`);
}

/** Streams events to ``onEvent`` until the result, an error or ``signal`` aborts. */
export async function locateStream(
  file: File,
  region: Region | null,
  signal: AbortSignal,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const body = new FormData();
  body.append("file", withMimeType(file));
  if (region) body.append("region", JSON.stringify(region));
  let response: Response;
  try {
    response = await fetch("/api/predict/stream", { method: "POST", body, signal });
  } catch (err) {
    if (signal.aborted) throw err;
    throw new UserFacingError(
      "The location service isn't reachable. Start it with scripts/run_local.sh and try again.",
    );
  }
  if (!response.ok || !response.body) throw await explain(response);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) onEvent(JSON.parse(line) as StreamEvent);
    }
  }
}
