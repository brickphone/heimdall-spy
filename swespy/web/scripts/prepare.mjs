// Runs before `next build` and `next dev`:
// 1. About-page data: every number comes from the frozen evaluation results, nothing is
//    typed by hand.
// 2. MapLibre's web worker, copied to public/ (it must be served as its own file).
import { copyFileSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const results = join(web, "..", "eval", "results", "dev_historic_rerank.json");
const library = join(web, "public", "data", "library.json");
const out = join(web, "src", "generated", "about.json");

const dev = JSON.parse(readFileSync(results, "utf8"));
const lib = JSON.parse(readFileSync(library, "utf8"));
const pick = (m) => ({ count: m.count, n: m.n, rate: m.rate, ci95: m.ci95 });

const about = {
  source: "eval/results/dev_historic_rerank.json",
  label: dev.label,
  createdAt: dev.created_at,
  photos: dev.metrics.query_count,
  medianErrorM: dev.metrics.median_error_m,
  within100m: pick(dev.metrics.within["100m"]),
  within500m: pick(dev.metrics.within["500m"]),
  baselineWithin500m: pick(dev.baselines.majority_location.within["500m"]),
  confident: {
    photos: dev.rerank.confidence_states.ok.photos,
    correct: pick(dev.rerank.confidence_states.ok.within_500m),
  },
  low: {
    photos: dev.rerank.confidence_states.low.photos,
    correct: pick(dev.rerank.confidence_states.low.within_500m),
  },
  failureTypes: Object.fromEntries(
    Object.entries(dev.metrics.failure_type).map(([k, v]) => [k, pick(v)]),
  ),
  minInliers: dev.rerank.min_inliers,
  rerankK: dev.rerank.k,
  references: lib.references,
  bySource: lib.by_source,
};

mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify(about, null, 1));
console.log(`about.json: ${about.photos} dev photos, ${about.references} references`);

const worker = join(web, "public", "maplibre");
mkdirSync(worker, { recursive: true });
copyFileSync(join(web, "node_modules", "maplibre-gl", "dist", "maplibre-gl-worker.mjs"), join(worker, "maplibre-gl-worker.mjs"));
console.log("maplibre worker copied to public/maplibre/");
