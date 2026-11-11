// End-to-end check of the Heimdall workspace with Playwright (playwright-core + the
// installed Chrome; no browser download). Needs scripts/run_local.sh running.
//   node scripts/e2e.mjs <expected.json>
// expected.json: [{image_id, path, state, top5: [...], inliers: [...]}] from the eval.
// Screenshots go to docs/screenshots/.
import { readFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const here = dirname(fileURLToPath(import.meta.url));
const shots = join(here, "..", "..", "..", "docs", "screenshots");
mkdirSync(shots, { recursive: true });
const expected = JSON.parse(readFileSync(process.argv[2], "utf8"));
const BASE = "http://127.0.0.1:3000";
const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  console.log(`${ok ? "pass" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
};
const browser = await chromium.launch({ channel: "chrome", headless: true });
const STATE = /^(Confident|Low confidence)$/;

async function open(context) {
  const page = await context.newPage();
  const requests = [];
  page.on("request", (r) => requests.push(r.url()));
  // Long tasks (> 50 ms) from the start of the page; the sequence is checked for > 100 ms.
  await page.addInitScript(() => {
    window.__long = [];
    new PerformanceObserver((list) => list.getEntries().forEach((e) => window.__long.push({ start: e.startTime, dur: e.duration })))
      .observe({ type: "longtask", buffered: true });
  });
  await page.goto(BASE);
  await page.getByRole("heading", { name: "Where was this photo taken?" }).waitFor();
  await page.waitForLoadState("networkidle").catch(() => {});
  return { page, requests };
}

async function choose(page, path) {
  await page.setInputFiles('input[type="file"]', path);
  await page.getByRole("button", { name: "Locate" }).waitFor();
}

async function locateAndWait(page) {
  const res = page.waitForResponse((r) => r.url().endsWith("/api/predict/stream"), { timeout: 180_000 });
  await page.getByRole("button", { name: "Locate" }).click();
  await res;
  // A previous result stays visible (greyed) during a re-run, so wait for the progress
  // label to go away rather than for a state label to appear.
  await page.locator('[role="status"]:has(button:text("Cancel"))').waitFor({ state: "detached", timeout: 120_000 });
  await page.getByText(STATE).locator("visible=true").first().waitFor({ timeout: 60_000 });
}

/** The final result event of the stream, read in the page (same body the UI parsed). */
async function captureStream(page) {
  return page.evaluate(() => window.__lastResult);
}

// ---------------------------------------------------------------- desktop 1440x900
{
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await context.addInitScript(() => {
    // Record the last "result" event the page receives, without changing the stream.
    const orig = window.fetch;
    window.fetch = async (...args) => {
      const res = await orig(...args);
      if (String(args[0]).endsWith("/api/predict/stream") && res.body) {
        const [a, b] = res.body.tee();
        (async () => {
          const text = await new Response(b).text();
          const last = text.trim().split("\n").pop();
          window.__lastResult = JSON.parse(last);
        })();
        return new Response(a, { headers: res.headers, status: res.status });
      }
      return res;
    };
  });
  const { page, requests } = await open(context);
  await page.screenshot({ path: join(shots, "desktop-1-empty.png") });

  // Keyboard and focus ring.
  let reachedChoose = false;
  let ring = "";
  for (let i = 0; i < 40 && !reachedChoose; i++) {
    await page.keyboard.press("Tab");
    const info = await page.evaluate(() => {
      const el = document.activeElement;
      const s = el ? getComputedStyle(el) : null;
      return { text: el?.textContent?.trim(), outline: s ? `${s.outlineWidth} ${s.outlineStyle} ${s.outlineColor} ${s.outlineOffset}` : "" };
    });
    if (info.text === "Choose a photo") {
      reachedChoose = true;
      ring = info.outline;
    }
  }
  check("keyboard reaches 'Choose a photo'", reachedChoose);
  check("focus ring is 2px solid cobalt with 2px offset", ring === "2px solid rgb(32, 71, 201) 2px", ring);

  // Photo first, nothing sent until Locate.
  await choose(page, expected[0].path);
  await page.waitForTimeout(600);
  check("nothing is sent before Locate", !requests.some((u) => u.includes("/api/predict")));
  check("region selection is active by default", await page.getByText("Drag a box around the part that shows the place").isVisible());
  await page.screenshot({ path: join(shots, "desktop-2-ready.png") });

  // The locating sequence: 10 screenshots at fixed times after Locate.
  const times = [0.3, 0.9, 1.5, 2.1, 2.7, 3.3, 4.0, 4.8, 5.8, 7.5];
  const t0 = Date.now();
  await page.getByRole("button", { name: "Locate" }).click();
  const seqStart = await page.evaluate(() => performance.now());
  const labels = [];
  for (const [i, t] of times.entries()) {
    const wait = t * 1000 - (Date.now() - t0);
    if (wait > 0) await page.waitForTimeout(wait);
    labels.push(await page.locator('[role="status"]').allInnerTexts().catch(() => []));
    await page.screenshot({ path: join(shots, `sequence-${String(i + 1).padStart(2, "0")}-${t.toFixed(1)}s.png`) });
  }
  await page.getByText(STATE).locator("visible=true").first().waitFor({ timeout: 60_000 });
  void seqStart; // Screenshots block the main thread, so frame timing is measured below.
  const sawSearching = labels.some((l) => l.join(" ").includes("Searching Linköping"));
  const sawComparing = labels.some((l) => l.join(" ").includes("Comparing the 5 closest matches"));
  check("status label shows the stages from the stream", sawSearching && sawComparing, JSON.stringify(labels.map((l) => l.join(" ").replace(/\s+/g, " ").slice(0, 40))));
  const located = await page.getByText(/^Located in /).first().innerText();
  check("result header shows 'Located in … s'", /^Located in \d+\.\d s$/.test(located), located);

  // API-equals for 3 dev photos (UI = stream result = eval).
  for (const [n, exp] of expected.entries()) {
    if (n > 0) {
      // These runs take no screenshots, so they measure frame timing honestly.
      await page.getByRole("button", { name: "New photo" }).first().click();
      await choose(page, exp.path);
      const a = await page.evaluate(() => performance.now());
      await locateAndWait(page);
      await page.waitForTimeout(2500); // Result flight and rings.
      const b = await page.evaluate(() => performance.now());
      const long = await page.evaluate(([x, y]) => window.__long.filter((e) => e.start >= x && e.start <= y), [a, b]);
      const worst = Math.max(0, ...long.map((e) => e.dur));
      console.log(`  note: ${exp.image_id} in the long session: ${long.length} long tasks > 50 ms, worst ${worst.toFixed(0)} ms`);
    }
    const json = await captureStream(page);
    check(`${exp.image_id}: streamed top 5 = eval top 5`, JSON.stringify(json.top_matches.map((m) => m.image_id)) === JSON.stringify(exp.top5));
    check(`${exp.image_id}: streamed state = eval state`, json.confidence_state === exp.state, json.confidence_state);
    const label = exp.state === "ok" ? "Confident" : "Low confidence";
    check(`${exp.image_id}: UI shows "${label}"`, await page.getByText(label, { exact: true }).locator("visible=true").first().isVisible());
    const rows = await page.locator('section:has(h2:text("Other candidates")) li:visible').allInnerTexts();
    const details = rows.map((t) => Number((t.match(/(\d+) details/) || [])[1]));
    check(`${exp.image_id}: UI candidates 2-5 show the eval's details`, JSON.stringify(details) === JSON.stringify(exp.inliers.slice(1)), `${details}`);
  }
  await page.screenshot({ path: join(shots, "desktop-3-result.png") });

  // Change region, then locate again with a keyboard-moved region.
  await page.getByRole("button", { name: "Change region" }).click();
  const photo = page.locator('img[alt="Your photo"]').first();
  const box = await photo.boundingBox();
  await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.25);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.8, box.y + box.height * 0.85, { steps: 8 });
  await page.mouse.up();
  const regionBox = page.getByRole("group", { name: /Selected region/ });
  await regionBox.focus();
  await page.keyboard.press("ArrowLeft");
  check("region drawn and adjustable before Locate", await regionBox.isVisible());
  await locateAndWait(page);
  const withRegion = await captureStream(page);
  check("Locate sends the drawn region", withRegion.region && withRegion.region.width > 0.3, JSON.stringify(withRegion.region));
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(shots, "desktop-4-region-result.png") });

  // Candidate 3: list and map stay in sync.
  const row3 = page.locator('section:has(h2:text("Other candidates")) li:visible button').nth(1);
  await row3.focus();
  await page.keyboard.press("Enter");
  check("candidate 3 selected from the list (Enter)", (await row3.getAttribute("aria-pressed")) === "true");
  const square3 = page.locator('.map-square:text-is("3")');
  check("candidate 3 highlighted on the map", (await square3.getAttribute("aria-pressed")) === "true");
  await page.waitForTimeout(1200);
  await page.screenshot({ path: join(shots, "desktop-5-candidate3.png") });
  await page.locator('.map-square:text-is("2")').focus();
  await page.keyboard.press("Enter");
  const row2 = page.locator('section:has(h2:text("Other candidates")) li:visible button').nth(0);
  check("selecting on the map selects the list row", (await row2.getAttribute("aria-pressed")) === "true");

  // Compare.
  const slider = page.getByRole("slider", { name: /Compare/ });
  await slider.scrollIntoViewIfNeeded();
  await slider.focus();
  await page.keyboard.press("ArrowLeft");
  check("compare divider moves with the keyboard", (await slider.getAttribute("aria-valuenow")) === "48");
  const credit = await page.locator("figcaption").first().innerText();
  check("compare shows the credit line", /Reference photo by .+ on .+ from the predicted point\./.test(credit));
  await page.screenshot({ path: join(shots, "desktop-6-compare.png") });

  // Cancel mid-verification: photo and region kept, previous answer restored.
  await page.getByRole("button", { name: "New photo" }).first().click();
  await choose(page, expected[1].path);
  await page.getByRole("button", { name: "Locate" }).click();
  await page.getByText("Comparing the 5 closest matches").waitFor({ timeout: 60_000 });
  await page.waitForTimeout(1300); // Into the verification flight.
  await page.screenshot({ path: join(shots, "cancel-1-before.png") });
  await page.getByRole("button", { name: "Cancel" }).click();
  await page.waitForTimeout(900);
  await page.screenshot({ path: join(shots, "cancel-2-after.png") });
  check("cancel removes the status label", (await page.locator('[role="status"]:has-text("Comparing")').count()) === 0);
  check("cancel keeps the photo and offers Locate again", (await page.locator('img[alt="Your photo"]').count()) > 0 && (await page.getByRole("button", { name: "Locate" }).isVisible()));

  // Error: the API is unreachable; the label turns into the message with Try again.
  await page.route("**/api/predict/stream", (route) => route.fulfill({ status: 502, body: "Bad gateway" }));
  await page.getByRole("button", { name: "Locate" }).click();
  await page.getByRole("button", { name: "Try again" }).waitFor({ timeout: 10_000 });
  const err = await page.locator('[role="status"]').first().innerText();
  check("error label explains and offers Try again", /location service isn't running/.test(err), err.replace(/\s+/g, " "));
  await page.screenshot({ path: join(shots, "error.png") });
  await page.unroute("**/api/predict/stream");

  // Network: no coverage layer or gallery points in the workspace.
  const coverage = requests.filter((u) => /coverage|library\.json/.test(u));
  const refs = requests.filter((u) => u.includes("/api/references/"));
  check("no coverage or gallery-point requests in the workspace", coverage.length === 0, `${coverage.length}`);
  console.log(`  note: ${refs.length} reference image requests, all from the compare view after a result`);

  await page.goto(`${BASE}/about`);
  await page.getByRole("heading", { name: "How it works", level: 1 }).waitFor();
  await page.waitForLoadState("networkidle").catch(() => {});
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(shots, "desktop-7-about.png"), fullPage: true });
  check("coverage is shown on /about", requests.some((u) => u.includes("coverage.geojson")));
  await context.close();
}

// ---------------------------------------------------------------- frame timing
// Each run: a fresh page, one dev photo, no screenshots (they block the main thread).
{
  const worst = [];
  for (const exp of expected) {
    for (let run = 0; run < 2; run++) {
      const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
      const { page } = await open(context);
      await choose(page, exp.path);
      const a = await page.evaluate(() => performance.now());
      await locateAndWait(page);
      await page.waitForTimeout(2500); // Result flight and rings.
      const b = await page.evaluate(() => performance.now());
      const long = await page.evaluate(([x, y]) => window.__long.filter((e) => e.start >= x && e.start <= y), [a, b]);
      worst.push(Math.max(0, ...long.map((e) => e.dur)));
      await context.close();
    }
  }
  const over = worst.filter((w) => w > 100).length;
  check("no long task > 100 ms in isolated sequence runs", over === 0,
    `worst per run (ms): ${worst.map((w) => w.toFixed(0)).join(", ")}`);
}

// ---------------------------------------------------------------- reduced motion
{
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
  const { page } = await open(context);
  await choose(page, expected[0].path);
  const t0 = Date.now();
  await page.getByRole("button", { name: "Locate" }).click();
  for (const [i, t] of [0.5, 2.0].entries()) {
    await page.waitForTimeout(Math.max(0, t * 1000 - (Date.now() - t0)));
    await page.screenshot({ path: join(shots, `reduced-${i + 1}.png`) });
  }
  await page.getByText(STATE).locator("visible=true").first().waitFor({ timeout: 60_000 });
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(shots, "reduced-3-result.png") });
  const pitch = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--duration-ease").trim());
  check("reduced motion: result shown, no eased transitions", pitch === "0ms" || pitch === "0s", pitch);
  await context.close();
}

// ---------------------------------------------------------------- mobile 390x844
{
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const { page } = await open(context);
  await page.screenshot({ path: join(shots, "mobile-1-empty.png") });
  await choose(page, expected[0].path);
  await locateAndWait(page);
  await page.waitForTimeout(2200);
  await page.screenshot({ path: join(shots, "mobile-2-result.png") });
  check("mobile: no horizontal scroll at 390 px", await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await page.getByRole("button", { name: "Compare with match" }).click();
  await page.getByRole("dialog").waitFor();
  await page.waitForTimeout(500);
  await page.screenshot({ path: join(shots, "mobile-3-compare.png") });
  await context.close();
}

await browser.close();
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
