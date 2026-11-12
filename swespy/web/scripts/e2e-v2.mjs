// End-to-end check of the v2 workspace (floating panels over a full-bleed map) with
// Playwright (playwright-core + the installed Chrome). The API is mocked so each state is
// deterministic; the HEIC section uses the real API (it needs the /analyze preview).
//   BASE=http://127.0.0.1:3100 node scripts/e2e-v2.mjs <photo.jpg> [<photo.heic>]
// Screenshots go to docs/screenshots/v2-*.png.
import { mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const here = dirname(fileURLToPath(import.meta.url));
const shots = join(here, "..", "..", "..", "docs", "screenshots");
mkdirSync(shots, { recursive: true });
const [PHOTO, HEIC] = process.argv.slice(2);
const BASE = process.env.BASE ?? "http://127.0.0.1:3000";
const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  console.log(`${ok ? "pass" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
};

// Spread-out candidates (300-900 m) and a tight cluster (all within 50 m).
const SPREAD = [[58.4109, 15.6216], [58.4135, 15.6280], [58.4072, 15.6120], [58.4160, 15.6150], [58.4050, 15.6300]];
const CLUSTER = [[58.41090, 15.62160], [58.41100, 15.62175], [58.41082, 15.62150], [58.41095, 15.62140], [58.41085, 15.62185]];

function prediction(state, places) {
  const ok = state === "ok";
  const inliers = ok ? [208, 13, 12, 8, 4] : [8, 6, 5, 3, 2];
  return {
    predicted_lat: places[0][0], predicted_lng: places[0][1], prediction_method: "mock", confidence: 0.6,
    confidence_status: "mock", confidence_state: state, confidence_reasons: ok ? [] : ["top match has 8 < 25 inliers"],
    top_matches: places.map(([lat, lng], i) => ({
      rank: i + 1, image_id: `mock-${i}`, lat, lng, similarity_score: 0.6 - i * 0.01, creator: "A. Photographer",
      source_url: "https://www.mapillary.com", license_url: "https://creativecommons.org/licenses/by-sa/4.0/",
      source: "mapillary", capture_year: 2015, inliers: inliers[i], score: 0.6,
    })),
    processing_time_ms: 4000, region: null,
    searched_variant: ok ? "crop:left" : "whole",
    searched_region: ok ? { x: 0, y: 0, width: 0.4, height: 1 } : { x: 0, y: 0, width: 1, height: 1 },
    person_share: 0.1, alternative_region: { x: 0.6, y: 0, width: 0.4, height: 1 }, variants: [], timings_ms: {}, versions: {},
  };
}

function stream(state, places) {
  return [
    { type: "stage", stage: "retrieval" },
    { type: "candidates", items: places.map(([lat, lng], id) => ({ id, lat, lng, similarity: 0.6 })) },
    { type: "stage", stage: "verification", count: 5 },
    ...places.map((_, id) => ({ type: "verified", id, inliers: 10, score: 0.6 })),
    { type: "result", ...prediction(state, places) },
  ].map((e) => JSON.stringify(e)).join("\n") + "\n";
}

async function mock(page, state, places = SPREAD, delayMs = 0) {
  await page.unroute("**/api/**").catch(() => {});
  await page.route("**/api/analyze", (r) =>
    r.fulfill({ contentType: "application/json", body: JSON.stringify({ person_share: 0.1, suggested_region: null, suggest: false, variants: [], preview: null, processing_time_ms: 50 }) }));
  await page.route("**/api/predict/stream", async (r) => {
    if (delayMs) await new Promise((res) => setTimeout(res, delayMs));
    await r.fulfill({ contentType: "application/x-ndjson", body: stream(state, places) });
  });
  await page.route("**/api/references/**", (r) => r.fulfill({ contentType: "image/jpeg", body: readFileSync(PHOTO) }));
}

async function choose(page, path) {
  await page.setInputFiles('input[type="file"]', path);
  await page.getByRole("button", { name: "Locate", exact: true }).waitFor();
}

async function locate(page) {
  await page.getByRole("button", { name: "Locate", exact: true }).click();
  await page.locator('[role="status"]:has(button:text("Cancel"))').waitFor({ state: "detached", timeout: 60_000 });
  await page.locator("[data-result]").waitFor({ timeout: 20_000 });
  await page.waitForTimeout(900);
}

const overlaps = (a, b) => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;

async function markerBoxes(page) {
  return page.locator(".map-pin, .map-square").evaluateAll((els) => els.map((e) => e.getBoundingClientRect().toJSON()));
}

const browser = await chromium.launch({ channel: "chrome", headless: true });
// Under `next dev`, the dev-tools layer sits over the page and takes pointer events.
const hideDevTools = (page) => page.addInitScript(() => {
  const style = document.createElement("style");
  style.textContent = "nextjs-portal { display: none !important; }";
  document.documentElement.appendChild(style);
});

// ---------------------------------------------------------------- desktop 1440x900
{
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  await hideDevTools(page);
  await mock(page, "ok", SPREAD, 3500);
  await page.goto(BASE);
  await page.locator('[data-ready="true"]').waitFor();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(shots, "v2-desktop-1-empty.png") });

  // Layout: full-bleed map, floating rail and inspector at the specified sizes.
  const geo = await page.evaluate(() => {
    const r = (sel) => document.querySelector(sel).getBoundingClientRect().toJSON();
    const map = r('[aria-label="Map of Linköping"]');
    return { map, rail: r('nav[aria-label="Workspace"]'), panel: r('aside[aria-label="Inspector"]'), w: innerWidth, h: innerHeight };
  });
  check("map is full-bleed", geo.map.left === 0 && geo.map.top === 0 && geo.map.width === geo.w && geo.map.height === geo.h, JSON.stringify(geo.map));
  check("rail 48 px wide, 12 px inset", geo.rail.width === 48 && geo.rail.left === 12 && geo.rail.top === 12 && geo.h - geo.rail.bottom === 12, JSON.stringify(geo.rail));
  check("inspector 380 px wide, 12 px from top, right and bottom", geo.panel.width === 380 && geo.w - geo.panel.right === 12 && geo.panel.top === 12 && geo.h - geo.panel.bottom === 12, JSON.stringify(geo.panel));
  check("no 'Linköping' map label", (await page.locator("span", { hasText: /^Linköping$/ }).count()) === 0);
  const controls = await page.locator(".map-controls").boundingBox();
  check("map controls bottom-right, left of the inspector", controls.x + controls.width <= geo.panel.left && controls.y > geo.h / 2, JSON.stringify(controls));
  check("map controls: zoom, north, 3D", await page.getByRole("button", { name: "Reset to north" }).isVisible() && await page.getByRole("button", { name: "3D view" }).isVisible());

  await choose(page, PHOTO);
  await page.getByRole("button", { name: "Locate", exact: true }).click();
  await page.waitForTimeout(1800);
  await page.screenshot({ path: join(shots, "v2-desktop-2-locating.png") });
  await page.locator('[role="status"]:has(button:text("Cancel"))').waitFor({ state: "detached", timeout: 60_000 });
  await page.locator("[data-result]").waitFor();
  await page.waitForTimeout(1200);
  await page.screenshot({ path: join(shots, "v2-desktop-3-confident.png") });

  // Result block: key-value data, no prose.
  const kv = await page.locator("aside dl").first().innerText();
  for (const label of ["Matched details", "Closest rival", "Searched area", "Reference", "Time"]) {
    check(`result row "${label}"`, kv.includes(label));
  }
  check("closest rival uses a comma", /Closest rival\s+\d+, [\d.]+ (m|km) away/.test(kv), kv.replace(/\n/g, " | "));
  check("searched area named", kv.includes("Automatic: left part"));
  check("reference row", /Mapillary, 2015, \d+ m away/.test(kv));
  const aside = await page.locator("aside").innerText();
  check("no explanatory prose", !/unambiguous|Treat this as a lead|so the answer/.test(aside));
  const sizes = await page.evaluate(() => {
    const fs = (el) => getComputedStyle(el).fontSize;
    const h2 = [...document.querySelectorAll("aside h2")].find((e) => e.className.includes("text-place"));
    return { place: fs(h2), base: fs(document.querySelector("aside dd")), label: fs(document.querySelector("aside dt")) };
  });
  check("type: place 18, values 13, labels 12", sizes.place === "18px" && sizes.base === "13px" && sizes.label === "12px", JSON.stringify(sizes));

  // Rings only on the selected result; no overlapping markers.
  const boxes = await markerBoxes(page);
  const bad = boxes.flatMap((a, i) => boxes.slice(i + 1).filter((b) => overlaps(a, b)));
  check("spread candidates: no overlapping markers", boxes.length === 5 && bad.length === 0, `${boxes.length} markers`);
  check("callout: place name and Compare only", (await page.locator(".map-callout").innerText()).trim().endsWith("Compare") && !(await page.locator(".map-callout").innerText()).includes("°"));

  // Candidates: verdicts, selection outline.
  const rows = page.locator('section:has(h2:text("Candidates")) li');
  check("5 candidate rows", (await rows.count()) === 5);
  await rows.nth(2).getByRole("button", { name: "Matches" }).click();
  check("verdict set", (await rows.nth(2).getByRole("button", { name: "Matches" }).getAttribute("aria-pressed")) === "true");
  await rows.nth(2).getByRole("button", { pressed: false }).first().click().catch(() => {});
  await rows.nth(2).locator("button").first().click();
  const outline = await rows.nth(2).evaluate((e) => getComputedStyle(e).outlineWidth + " " + getComputedStyle(e).outlineStyle);
  check("selected row: 2px outline, not a fill", outline === "2px solid", outline);

  // Compare overlay over the map area.
  await rows.nth(0).getByRole("button", { name: /Compare with candidate/ }).click();
  const dialog = page.getByRole("dialog", { name: /Compare with/ });
  await dialog.waitFor();
  await page.waitForTimeout(600);
  const d = await dialog.boundingBox();
  check("compare overlay covers the map area, not the inspector", d.x >= geo.rail.right && d.x + d.width <= geo.panel.left && d.width > 800, JSON.stringify(d));
  check("compare labels", (await dialog.getByText("Your photo", { exact: true }).first().isVisible()) && (await dialog.getByText("Reference", { exact: true }).first().isVisible()));
  check("compare has a verdict control", await dialog.getByRole("group", { name: /Your verdict/ }).isVisible());
  check("swipe is the default view", await dialog.getByRole("slider").isVisible());
  await page.screenshot({ path: join(shots, "v2-desktop-5-compare-swipe.png") });
  await dialog.getByRole("button", { name: "Side by side" }).click();
  await page.waitForTimeout(400);
  await page.screenshot({ path: join(shots, "v2-desktop-4-compare.png") });
  await page.keyboard.press("Escape");
  check("Escape closes compare", (await dialog.count()) === 0);

  // Low confidence.
  await page.getByRole("button", { name: "New photo" }).click();
  await mock(page, "low");
  await choose(page, PHOTO);
  await locate(page);
  const lowText = await page.locator("aside").innerText();
  check("low confidence: one unsure line", lowText.includes("Too few matching details to be sure. Compare before relying on it."));
  const mode = await page.locator("[data-result]").getAttribute("data-result");
  check("low confidence map: flat overview", /^low pitch=0 /.test(mode ?? ""), mode);
  check("low confidence pin in the unsure colour", (await page.locator(".map-pin.unsure").count()) === 1);
  await page.screenshot({ path: join(shots, "v2-desktop-6-low.png") });

  // Collision: 5 candidates within 50 m.
  await page.getByRole("button", { name: "New photo" }).click();
  await mock(page, "ok", CLUSTER);
  await choose(page, PHOTO);
  await locate(page);
  for (const zoomOut of [0, 2]) {
    for (let k = 0; k < zoomOut; k++) await page.getByRole("button", { name: "Zoom out" }).click();
    await page.waitForTimeout(700);
    const b = await markerBoxes(page);
    const o = b.flatMap((a, i) => b.slice(i + 1).filter((c) => overlaps(a, c)));
    check(`5 candidates within 50 m: no overlap (zoomed out ${zoomOut})`, b.length === 5 && o.length === 0, `${o.length} overlapping pairs`);
  }
  await page.screenshot({ path: join(shots, "v2-desktop-7-cluster.png") });
  await context.close();
}

// ---------------------------------------------------------------- mobile 390x844
{
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const page = await context.newPage();
  await hideDevTools(page);
  await mock(page, "ok");
  await page.goto(BASE);
  await page.locator('[data-ready="true"]').waitFor();
  await page.waitForTimeout(1200);
  await page.screenshot({ path: join(shots, "v2-mobile-1-empty.png") });
  await choose(page, PHOTO);
  await locate(page);
  await page.screenshot({ path: join(shots, "v2-mobile-2-confident.png") });
  check("mobile: no horizontal scroll", await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  const first = await page.locator("aside section button:visible").evaluateAll((els) =>
    els.map((e) => e.textContent?.trim()).find((t) => ["Compare", "Change region", "Locate"].includes(t ?? "") && t !== "Change region"));
  check("mobile: Compare is the first result button", first === "Compare", String(first));
  await page.getByRole("button", { name: "Compare", exact: true }).first().click();
  await page.getByRole("dialog", { name: /Compare with/ }).waitFor();
  await page.waitForTimeout(500);
  await page.screenshot({ path: join(shots, "v2-mobile-3-compare.png") });
  await context.close();
}

// ---------------------------------------------------------------- HEIC through the real API
if (HEIC) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  await hideDevTools(page);
  await page.goto(BASE);
  await page.locator('[data-ready="true"]').waitFor();
  await page.waitForTimeout(1500);
  await page.setInputFiles('input[type="file"]', HEIC);
  const img = page.locator('img[alt="Your photo"]');
  const shown = await img.waitFor({ timeout: 30_000 }).then(() => true, () => false);
  const src = shown ? await img.getAttribute("src") : "";
  check("HEIC: preview shown in Chromium", shown && src.startsWith("data:image/jpeg"), src.slice(0, 30));
  const box = await img.boundingBox();
  await page.mouse.move(box.x + box.width * 0.1, box.y + box.height * 0.2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.8, { steps: 8 });
  await page.mouse.up();
  check("HEIC: region drawn on the preview", await page.getByRole("group", { name: /Selected region/ }).isVisible());
  await page.screenshot({ path: join(shots, "v2-desktop-8-heic-region.png") });
  await context.close();
}

await browser.close();
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
