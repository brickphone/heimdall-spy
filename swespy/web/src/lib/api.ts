// Types mirror swespy/api/schemas.py. The app talks to the public API only; all
// inference happens in FastAPI.

export interface Match {
  rank: number;
  image_id: string;
  lat: number;
  lng: number;
  similarity_score: number;
  creator: string | null;
  source_url: string | null;
  license_url: string | null;
  source: string | null;
  capture_year: number | null;
  inliers: number | null;
  score: number | null;
  variant?: string | null;
}

export interface Region {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Prediction {
  predicted_lat: number;
  predicted_lng: number;
  confidence_state: "ok" | "low" | "unavailable" | string;
  confidence_reasons: string[];
  top_matches: Match[];
  processing_time_ms: number;
  region: Region | null;
  /** Automatic region search: the part of the photo whose match won. */
  searched_variant: string;
  searched_region: Region | null;
  person_share: number | null;
  alternative_region: Region | null;
  versions: Record<string, unknown>;
}

export interface Analysis {
  person_share: number | null;
  suggested_region: Region | null;
  suggest: boolean; // People cover more than 30% of the photo.
  preview: string | null; // Upright JPEG data URL, for browsers that cannot show the file.
}

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // Same limit as the API and its proxy.
const ACCEPTED = /\.(jpe?g|png|hei[cf])$/i;
const API = "/api";

/** An error whose message tells the user what happened and how to fix it. */
export class UserFacingError extends Error {}

export function checkFile(file: File): void {
  const typeOk = /^image\/(jpeg|png|heic|heif)$/.test(file.type) || ACCEPTED.test(file.name);
  if (!typeOk) {
    throw new UserFacingError("This file isn't an image. Choose a JPEG, PNG or HEIC.");
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new UserFacingError(
      "This photo is larger than 10 MB. Choose a smaller copy, or export it at a lower resolution.",
    );
  }
}

/** iOS and some browsers send HEIC with an empty MIME type; the API checks it. */
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
    // Not JSON (e.g. proxy error page).
  }
  if (response.status >= 500 || response.status === 0) {
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
  if (/region/i.test(detail)) {
    return new UserFacingError(`${detail}. Draw a larger region, or use the whole photo.`);
  }
  return new UserFacingError(detail || `The search failed (${response.status}). Try again.`);
}

export async function predict(file: File, region: Region | null): Promise<Prediction> {
  const body = new FormData();
  body.append("file", withMimeType(file));
  if (region) body.append("region", JSON.stringify(region));
  let response: Response;
  try {
    response = await fetch(`${API}/predict`, { method: "POST", body });
  } catch {
    throw new UserFacingError(
      "The location service isn't reachable. Start it with scripts/run_local.sh and try again.",
    );
  }
  if (!response.ok) throw await explain(response);
  return (await response.json()) as Prediction;
}

/** People share and a suggested region, before searching (no retrieval). Never throws. */
export async function analyze(file: File, signal: AbortSignal): Promise<Analysis | null> {
  const body = new FormData();
  body.append("file", withMimeType(file));
  try {
    const response = await fetch(`${API}/analyze`, { method: "POST", body, signal });
    return response.ok ? ((await response.json()) as Analysis) : null;
  } catch {
    return null; // The suggestion is optional; locating still works without it.
  }
}

export function referenceImageUrl(imageId: string): string {
  return `${API}/references/${encodeURIComponent(imageId)}/image`;
}

const SOURCES: Record<string, string> = {
  mapillary: "Mapillary",
  wikimedia_commons: "Wikimedia Commons",
  kartaview: "KartaView",
  gsv: "Google Street View",
};

export function sourceName(source: string | null): string {
  return (source && SOURCES[source]) || source || "an unknown source";
}

export function licenceName(url: string | null): string {
  if (!url) return "licence unknown";
  const cc = url.match(/licenses\/([a-z-]+)\/([0-9.]+)/);
  if (cc) return `CC ${cc[1].toUpperCase()} ${cc[2]}`;
  if (url.includes("publicdomain/zero")) return "CC0";
  if (url.includes("publicdomain/mark")) return "public domain";
  return "licence";
}
