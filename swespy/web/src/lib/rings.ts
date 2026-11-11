/**
 * Contour rings: the one place uncertainty is drawn.
 *
 * Each candidate gets three rings. Their spacing comes from one function of the number
 * of matching details (RANSAC inliers) between the photo and that candidate's reference:
 *
 *   certainty = clamp(ln(1 + inliers) / ln(1 + 120), 0, 1)
 *   spread    = 1 + 1.4 * (1 - certainty)          // 1.0 (>= 120 details) ... 2.4 (none)
 *   radii     = base * spread                       // screen px at RING_ZOOM
 *
 * base = 16 / 29 / 45 px for the prediction and 22 / 40 / 62 px for other candidates
 * (always wider, so the prediction reads as the tightest cluster). A low-confidence
 * prediction is spread a further 1.25x, because the state itself says the answer is
 * uncertain even when the top match has many details (e.g. a far-away rival that is
 * as strong). Radii are converted to metres at RING_ZOOM, so rings keep their
 * geographic size when the map zooms.
 */
export const RING_ZOOM = 14;
export const PREDICTION_BASE_PX = [16, 29, 45] as const;
export const CANDIDATE_BASE_PX = [22, 40, 62] as const;
const FULL_CERTAINTY_INLIERS = 120;

export function certainty(inliers: number | null): number {
  const v = Math.log1p(Math.max(0, inliers ?? 0)) / Math.log1p(FULL_CERTAINTY_INLIERS);
  return Math.min(1, Math.max(0, v));
}

export function ringRadiiPx(
  inliers: number | null,
  kind: "prediction" | "candidate",
  state: string = "ok",
): number[] {
  const spread = 1 + 1.4 * (1 - certainty(inliers));
  const extra = kind === "prediction" && state !== "ok" ? 1.25 : 1;
  const base = kind === "prediction" ? PREDICTION_BASE_PX : CANDIDATE_BASE_PX;
  return base.map((r) => r * spread * extra);
}

/**
 * Low-confidence result: no ring is drawn tighter than a candidate's. Every location,
 * the best guess included, gets candidate-style dashed rings spread a further 1.25x.
 */
export const LOW_SPREAD = 1.25;

export function lowRadiiPx(inliers: number | null): number[] {
  return ringRadiiPx(inliers, "candidate").map((r) => r * LOW_SPREAD);
}
