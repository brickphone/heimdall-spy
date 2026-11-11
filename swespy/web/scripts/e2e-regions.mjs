// End-to-end check of automatic region search in the UI, with the API mocked so every
// state is deterministic: the suggested region for a person-dominated photo, the
// "Searched area" outline, confident vs low-confidence map modes, "Try a different
// region", and the inspector order. Needs the web app running (any port):
//   BASE=http://127.0.0.1:3100 node scripts/e2e-regions.mjs <photo.jpg> [<person-photo>]
// Screenshots go to docs/screenshots/regions-*.png. With <person-photo>, a last section
// runs it through the real API (no mocks) and checks the suggestion, the map mode for the
// streamed confidence state and the "Searched area". That section takes no screenshots:
// the photo shows a private person and must not end up in the docs.
import { mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const here = dirname(fileURLToPath(import.meta.url));
const shots = join(here, "..", "..", "..", "docs", "screenshots");
mkdirSync(shots, { recursive: true });
const PHOTO = process.argv[2];
const PERSON_PHOTO = process.argv[3];
const BASE = process.env.BASE ?? "http://127.0.0.1:3000";
const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  console.log(`${ok ? "pass" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
};

const SUGGESTED = { x: 0, y: 0.125, width: 0.4375, height: 0.75 };
const ALTERNATIVE = { x: 0.6, y: 0, width: 0.4, height: 1 };
// Five candidates around central Linköping, 300-900 m apart.
const PLACES = [
  [58.4109, 15.6216],
  [58.4135, 15.6280],
  [58.4072, 15.6120],
  [58.4160, 15.6150],
  [58.4050, 15.6300],
];

function prediction(state) {
  const ok = state === "ok";
  const inliers = ok ? [162, 30, 12, 8, 4] : [8, 6, 5, 3, 2];
  return {
    predicted_lat: PLACES[0][0],
    predicted_lng: PLACES[0][1],
    prediction_method: "mock",
    confidence: 0.6,
    confidence_status: "mock",
    confidence_state: state,
    confidence_reasons: ok ? [] : ["top match has 8 < 25 inliers"],
    top_matches: PLACES.map(([lat, lng], i) => ({
      rank: i + 1, image_id: `mock-${i}`, lat, lng, similarity_score: 0.6 - i * 0.01,
      creator: "Mock", source_url: null, license_url: null, source: "mapillary",
      capture_year: 2024, inliers: inliers[i], score: 0.6, variant: ok ? "crop:left" : "whole",
    })),
    processing_time_ms: 4000,
    region: null,
    searched_variant: ok ? "crop:left" : "whole",
    searched_region: ok ? { x: 0, y: 0, width: 0.4, height: 1 } : { x: 0, y: 0, width: 1, height: 1 },
    person_share: 0.55,
    alternative_region: ALTERNATIVE,
    variants: [],
    timings_ms: {},
    versions: {},
  };
}

function stream(state) {
  const events = [
    { type: "stage", stage: "retrieval" },
    { type: "candidates", items: PLACES.map(([lat, lng], id) => ({ id, lat, lng, similarity: 0.6 })) },
    { type: "stage", stage: "verification", count: 5 },
    ...PLACES.map((_, id) => ({ type: "verified", id, inliers: 10, score: 0.6 })),
    { type: "result", ...prediction(state) },
  ];
  return events.map((e) => JSON.stringify(e)).join("\n") + "\n";
}

async function mockApi(page, state) {
  await page.unroute("**/api/predict/stream").catch(() => {});
  await page.route("**/api/analyze", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ person_share: 0.55, suggested_region: SUGGESTED, suggest: true, variants: [], processing_time_ms: 120 }),
    }),
  );
  await page.route("**/api/predict/stream", (route) => route.fulfill({ contentType: "application/x-ndjson", body: stream(state) }));
  await page.route("**/api/references/**", (route) => route.fulfill({ contentType: "image/jpeg", body: readFileSync(PHOTO) }));
}

async function locate(page) {
  await page.getByRole("button", { name: "Locate" }).click();
  await page.locator('[role="status"]:has(button:text("Cancel"))').waitFor({ state: "detached", timeout: 60_000 });
  await page.locator("[data-result]").waitFor({ timeout: 20_000 });
}

const browser = await chromium.launch({ channel: "chrome", headless: true });

// ---------------------------------------------------------------- desktop 1440x900
{
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const sent = [];
  page.on("request", (r) => sent.push(r.url()));
  await mockApi(page, "ok");
  await page.goto(BASE);
  await page.getByRole("heading", { name: "Where was this photo taken?" }).waitFor();

  // Suggested region for a person-dominated photo.
  await page.setInputFiles('input[type="file"]', PHOTO);
  const note = page.getByText("Most of this photo is a person. The highlighted area shows the place.");
  await note.waitFor({ timeout: 10_000 });
  check("analyze is called on photo choice", sent.some((u) => u.endsWith("/api/analyze")));
  check("no search before Locate", !sent.some((u) => u.includes("/api/predict")));
  const region = page.getByRole("group", { name: /Selected region/ });
  check("suggested region is pre-drawn", await region.isVisible());
  await page.screenshot({ path: join(shots, "regions-1-suggested.png") });
  await page.getByRole("button", { name: "Use whole photo" }).click();
  check("'Use whole photo' clears the suggestion", !(await region.isVisible()) && !(await note.isVisible()));

  // Confident: close 3D view; the winning crop is outlined as "Searched area".
  await locate(page);
  await page.waitForTimeout(800);
  const confident = await page.locator("[data-result]").getAttribute("data-result");
  check("confident: close-up at pitch 45, zoom 16", confident === "confident pitch=45 zoom=16.0", confident);
  check("confident: pin uses the pin colour", (await page.locator(".map-pin.unsure").count()) === 0);
  const searched = page.getByTestId("searched-area").first();
  check("'Searched area' outline after the result", (await searched.isVisible()) && (await searched.innerText()) === "Searched area");
  const order = await page.evaluate(() => {
    const y = (sel) => {
      const el = [...document.querySelectorAll("aside h2, aside h1, aside p")].find((e) => e.textContent?.trim() === sel);
      return el ? el.getBoundingClientRect().top : NaN;
    };
    return {
      header: y("mock.jpg") || 0,
      result: y("Confident"),
      compare: y("Your region and the match"),
      candidates: y("Other candidates"),
      timings: y("Stage timings"),
    };
  });
  check(
    "inspector order: result, comparison, other candidates, stage timings",
    order.result < order.compare && order.compare < order.candidates && order.candidates < order.timings,
    JSON.stringify(order),
  );
  const frame = await page.getByRole("slider", { name: "Compare your region with the match" }).locator("..").boundingBox();
  check("comparison is 240 px high", Math.round(frame.height) === 240, `${frame.height}`);
  await page.screenshot({ path: join(shots, "regions-2-confident.png") });

  // Low confidence: overview of all candidates, flat, unsure pin, "Best guess".
  await page.getByRole("button", { name: "New photo" }).first().click();
  await mockApi(page, "low");
  await page.setInputFiles('input[type="file"]', PHOTO);
  await page.getByRole("button", { name: "Locate" }).waitFor();
  await locate(page);
  await page.waitForTimeout(800);
  const low = await page.locator("[data-result]").getAttribute("data-result");
  const zoom = Number(low?.match(/zoom=([\d.]+)/)?.[1]);
  check("low confidence: pitch 0 and below the 3D zoom", /^low pitch=0 /.test(low ?? "") && zoom <= 14.5, low);
  check("low confidence: pin in the unsure colour", (await page.locator(".map-pin.unsure").count()) === 1);
  check("low confidence: callout reads 'Best guess'", (await page.locator(".map-callout-name").innerText()) === "Best guess");
  const mapBox = await page.getByRole("region", { name: "Map of Linköping" }).boundingBox();
  const markers = await page.locator(".map-pin, .map-square").evaluateAll((els) => els.map((e) => e.getBoundingClientRect().toJSON()));
  const inside = markers.every((m) => m.left >= mapBox.x && m.right <= mapBox.x + mapBox.width && m.top >= mapBox.y && m.bottom <= mapBox.y + mapBox.height);
  check("low confidence: all 5 candidates in view", markers.length === 5 && inside, `${markers.length}`);
  await page.screenshot({ path: join(shots, "regions-3-low.png") });
  await page.getByRole("button", { name: "Try a different region" }).first().click();
  check("'Try a different region' opens region selection with the alternative drawn",
    (await page.getByRole("group", { name: /Selected region/ }).isVisible()) &&
      (await page.getByText(/Another part of the photo is highlighted/).isVisible()) &&
      (await page.getByRole("button", { name: "Locate" }).isVisible()));
  await page.screenshot({ path: join(shots, "regions-4-try-region.png") });
  await context.close();
}

// ---------------------------------------------------------------- mobile 390x844
{
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const page = await context.newPage();
  await mockApi(page, "low");
  await page.goto(BASE);
  await page.setInputFiles('input[type="file"]', PHOTO);
  await page.getByRole("button", { name: "Locate" }).waitFor();
  await locate(page);
  await page.waitForTimeout(800);
  const buttons = await page.locator("aside section:has(h2) button:visible").allInnerTexts();
  const firstResultButton = buttons.find((b) => ["Compare with match", "Copy", "Try a different region"].includes(b));
  check("mobile: 'Compare with match' is the first result button", firstResultButton === "Compare with match", buttons.join(" | "));
  check("mobile: no horizontal scroll at 390 px", await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await page.screenshot({ path: join(shots, "regions-5-mobile-low.png") });
  await context.close();
}

// ---------------------------------------------------------------- real API, person photo
if (PERSON_PHOTO) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await context.addInitScript(() => {
    const orig = window.fetch;
    window.fetch = async (...args) => {
      const res = await orig(...args);
      if (String(args[0]).endsWith("/api/predict/stream") && res.body) {
        const [a, b] = res.body.tee();
        void new Response(b).text().then((t) => (window.__lastResult = JSON.parse(t.trim().split("\n").pop())));
        return new Response(a, { headers: res.headers, status: res.status });
      }
      return res;
    };
  });
  const page = await context.newPage();
  await page.goto(BASE);
  await page.setInputFiles('input[type="file"]', PERSON_PHOTO);
  const note = page.getByText("Most of this photo is a person. The highlighted area shows the place.");
  const suggested = await note.waitFor({ timeout: 30_000 }).then(() => true, () => false);
  check("real: person photo gets the suggested region", suggested && (await page.getByRole("group", { name: /Selected region/ }).isVisible()));
  // As in the field test: search the whole photo and let the automatic variants work.
  await page.getByRole("button", { name: "Use whole photo" }).click();
  await page.getByRole("button", { name: "Locate" }).click();
  await page.locator('[role="status"]:has(button:text("Cancel"))').waitFor({ state: "detached", timeout: 180_000 });
  await page.locator("[data-result]").waitFor({ timeout: 30_000 });
  await page.waitForTimeout(800);
  const json = await page.evaluate(() => window.__lastResult);
  const mode = await page.locator("[data-result]").getAttribute("data-result");
  const expected = json.confidence_state === "ok" ? /^confident pitch=45 zoom=16\.0$/ : /^low pitch=0 /;
  check("real: map mode follows the streamed confidence state", expected.test(mode ?? ""), `${json.confidence_state}: ${mode}`);
  check("real: response names the winning variant and region",
    typeof json.searched_variant === "string" && json.searched_region && json.person_share > 0.3,
    `${json.searched_variant}, person share ${json.person_share?.toFixed(2)}`);
  check("real: 'Searched area' is outlined", await page.getByTestId("searched-area").first().isVisible());
  const label = json.confidence_state === "ok" ? "Confident" : "Low confidence";
  check(`real: UI shows "${label}"`, await page.getByText(label, { exact: true }).first().isVisible());
  await context.close();
}

await browser.close();
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
