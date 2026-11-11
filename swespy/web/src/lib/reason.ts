// The one plain-language reason under the state line, built only from API data.
// "Details" are matching details: keypoints that agree geometrically between the photo
// and a reference (RANSAC inliers). The API's own threshold is 25.
import type { Prediction } from "./api";
import { distanceM, formatDistance } from "./geo";

export const MIN_DETAILS = 25; // swespy.model.rerank.DEFAULT_MIN_INLIERS
const SAME_PLACE_M = 500; // swespy.model.rerank.DISAGREE_M

export function reasonText(p: Prediction, usedRegion: boolean): string {
  const what = usedRegion ? "your region" : "your photo";
  const [top, ...others] = p.top_matches;
  if (!top) return "No reference photo matched.";
  const n = top.inliers ?? 0;
  const rivals = others.filter((m) => distanceM(top, m) > SAME_PLACE_M);
  const rival = rivals[0];
  const strongRival = rivals.find((m) => (m.inliers ?? 0) >= MIN_DETAILS);

  if (p.confidence_state === "ok") {
    const first = `The best match shares ${n} details with ${what}.`;
    if (!rival) return `${first} The other candidates point to the same place.`;
    return (
      `${first} The closest rival place is ${formatDistance(distanceM(top, rival))} away ` +
      `and shares ${rival.inliers ?? 0}, so the answer is unambiguous.`
    );
  }
  if (n < MIN_DETAILS) {
    const first = `The best match shares only ${n} details with ${what}, too few to be sure.`;
    if (strongRival) {
      return `${first} A place ${formatDistance(distanceM(top, strongRival))} away matches better on details, so treat this as a lead to check.`;
    }
    return `${first} Treat this as a lead to check, not an answer.`;
  }
  if (strongRival) {
    return (
      `The best match shares ${n} details, but a place ${formatDistance(distanceM(top, strongRival))} ` +
      `away shares ${strongRival.inliers}, so the photo fits more than one location.`
    );
  }
  return "The matches disagree on where this is. Treat this as a lead to check, not an answer.";
}
