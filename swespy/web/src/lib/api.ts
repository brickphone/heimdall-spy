// Types mirror swespy/api/schemas.py. The app talks to the public API only.

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
}

export interface Prediction {
  predicted_lat: number;
  predicted_lng: number;
  prediction_method: string;
  confidence: number;
  confidence_status: string;
  confidence_state: "ok" | "low" | "unavailable" | string;
  confidence_reasons: string[];
  top_matches: Match[];
  processing_time_ms: number;
  versions: Record<string, unknown>;
}

export interface LatLng {
  lat: number;
  lng: number;
}

const API = "/api";

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // Not JSON (e.g. proxy error page).
  }
  return response.status === 502 || response.status === 504
    ? "The API is not reachable. Is scripts/run_local.sh running?"
    : `Request failed (${response.status})`;
}

/** iOS and some browsers send HEIC with an empty MIME type; the API checks it. */
function withMimeType(file: File): File {
  if (file.type) return file;
  const type = /\.hei[cf]$/i.test(file.name) ? "image/heic" : "image/jpeg";
  return new File([file], file.name, { type });
}

export async function predict(file: File): Promise<Prediction> {
  const body = new FormData();
  body.append("file", withMimeType(file));
  const response = await fetch(`${API}/predict`, { method: "POST", body });
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()) as Prediction;
}

export async function saveFieldPhoto(
  file: File,
  prediction: Prediction,
  truth: LatLng,
): Promise<string> {
  const body = new FormData();
  body.append("file", withMimeType(file));
  body.append("prediction", JSON.stringify(prediction));
  body.append("true_lat", String(truth.lat));
  body.append("true_lng", String(truth.lng));
  const response = await fetch(`${API}/field-photos`, { method: "POST", body });
  if (!response.ok) throw new Error(await errorMessage(response));
  return ((await response.json()) as { saved: string }).saved;
}

/** True once the API has loaded the model and gallery (public GET /ready). */
export async function isReady(): Promise<boolean> {
  try {
    const response = await fetch(`${API}/ready`, { cache: "no-store" });
    return response.ok && ((await response.json()) as { ready?: boolean }).ready === true;
  } catch {
    return false;
  }
}

export function referenceImageUrl(imageId: string): string {
  return `${API}/references/${encodeURIComponent(imageId)}/image`;
}

const SOURCES: Record<string, string> = {
  mapillary: "Mapillary",
  wikimedia_commons: "Wikimedia Commons",
  kartaview: "KartaView",
  self_collected: "Self-collected",
};

export function sourceName(source: string | null): string {
  return (source && SOURCES[source]) || source || "Reference";
}

export function licenceName(url: string | null): string {
  if (!url) return "licence unknown";
  const cc = url.match(/licenses\/([a-z-]+)\/([0-9.]+)/);
  if (cc) return `CC ${cc[1].toUpperCase()} ${cc[2]}`;
  if (url.includes("publicdomain/zero")) return "CC0";
  if (url.includes("publicdomain/mark")) return "Public Domain";
  return "licence";
}
