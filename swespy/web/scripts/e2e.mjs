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

async function settle(page) {
  await page.waitForLoadState("networkidle").catch(() => {});
  await page.waitForTimeout(900); // Map tiles and eased camera.
}

async function upload(page, path) {
  const response = page.waitForResponse((r) => r.url().endsWith("/api/predict"), { timeout: 180_000 });
  await page.setInputFiles('input[type="file"]', path);
  const res = await response;
  const json = await res.json();
  await page.getByText(/^(Confident|Low confidence)$/).locator("visible=true").first().waitFor({ timeout: 60_000 });
  return json;
}

// ---------------------------------------------------------------- desktop 1440x900
{
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(BASE);
  await page.getByRole("heading", { name: "Where was this photo taken?" }).waitFor();
  await settle(page);
  await page.screenshot({ path: join(shots, "desktop-1-empty.png") });

  // Keyboard: the first Tab stops reach the rail and the primary action, with the focus ring.
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

  // API-equals: the UI's top 5 and state equal the evaluation for 3 dev photos.
  for (const [n, exp] of expected.entries()) {
    if (n > 0) {
      await page.getByRole("button", { name: "New photo" }).first().click();
      await page.getByRole("heading", { name: "Where was this photo taken?" }).waitFor();
    }
    const json = await upload(page, exp.path);
    const ids = json.top_matches.map((m) => m.image_id);
    const inl = json.top_matches.map((m) => m.inliers);
    check(`${exp.image_id}: API top 5 = eval top 5`, JSON.stringify(ids) === JSON.stringify(exp.top5));
    check(`${exp.image_id}: API state = eval state`, json.confidence_state === exp.state, json.confidence_state);
    const label = exp.state === "ok" ? "Confident" : "Low confidence";
    const shown = await page.getByText(label, { exact: true }).locator("visible=true").first().isVisible();
    check(`${exp.image_id}: UI shows "${label}"`, shown);
    const rows = await page.locator('section:has(h2:text("Other candidates")) li:visible').allInnerTexts();
    const details = rows.map((t) => Number((t.match(/(\d+) details/) || [])[1]));
    check(`${exp.image_id}: UI candidates 2-5 show the eval's details`,
      JSON.stringify(details) === JSON.stringify(inl.slice(1).map((v) => v)), `${details} vs ${inl.slice(1)}`);
    if (n === 0) {
      await settle(page);
      await page.screenshot({ path: join(shots, "desktop-2-result.png") });
    }
  }

  // Region selection: default region, then move it with the keyboard; the search reruns.
  // The API echoes the region it searched, so the test reads it from the response.
  const regionRes = page.waitForResponse((r) => r.url().endsWith("/api/predict"), { timeout: 180_000 });
  await page.getByRole("button", { name: "Select region" }).click();
  const first = await (await regionRes).json();
  check("Select region searches the default region", first.region?.width === 0.6, JSON.stringify(first.region));
  const box = page.getByRole("group", { name: /Selected region/ });
  await box.focus();
  const moved = page.waitForResponse((r) => r.url().endsWith("/api/predict"), { timeout: 180_000 });
  await page.keyboard.press("ArrowLeft");
  const second = await (await moved).json();
  check("keyboard moves the region and searches again", Math.abs((second.region?.x ?? 0) - 0.18) < 1e-6, JSON.stringify(second.region));
  await page.getByText(/^(Confident|Low confidence)$/).locator("visible=true").first().waitFor();
  await settle(page);
  await page.screenshot({ path: join(shots, "desktop-3-region.png") });

  // Candidate 3: in the list, then on the map; both stay in sync.
  const row3 = page.locator('section:has(h2:text("Other candidates")) li:visible button').nth(1);
  await row3.focus();
  await page.keyboard.press("Enter");
  check("candidate 3 selected from the list (Enter)", (await row3.getAttribute("aria-pressed")) === "true");
  const square3 = page.locator('.map-square:text-is("3")');
  check("candidate 3 highlighted on the map", (await square3.getAttribute("aria-pressed")) === "true");
  await settle(page);
  await page.screenshot({ path: join(shots, "desktop-4-candidate3.png") });
  await page.locator('.map-square:text-is("2")').focus();
  await page.keyboard.press("Enter"); // Map markers are buttons: keyboard works there too.
  const row2 = page.locator('section:has(h2:text("Other candidates")) li:visible button').nth(0);
  check("selecting on the map selects the list row", (await row2.getAttribute("aria-pressed")) === "true");
  await square3.focus();
  await page.keyboard.press("Enter");

  // Compare view with the keyboard slider.
  const slider = page.getByRole("slider", { name: /Compare/ });
  await slider.scrollIntoViewIfNeeded();
  await slider.focus();
  await page.keyboard.press("ArrowLeft");
  check("compare divider moves with the keyboard", (await slider.getAttribute("aria-valuenow")) === "48");
  const credit = await page.locator("figcaption").first().innerText();
  check("compare shows the credit line", /Reference photo by .+ on .+, .+ from the predicted point\./.test(credit), credit);
  await page.locator('section:has(h2:text("Your region and the match"))').scrollIntoViewIfNeeded();
  await page.waitForTimeout(400);
  await page.screenshot({ path: join(shots, "desktop-5-compare.png") });

  // Error state for a non-image file.
  await page.getByRole("button", { name: "New photo" }).first().click();
  await page.setInputFiles('input[type="file"]', { name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("hello") });
  // Next.js adds its own (empty) route announcer with role="alert"; take ours.
  const err = await page.getByRole("alert").filter({ hasText: /\S/ }).first().innerText();
  check("non-image file gives a plain error", err.includes("This file isn't an image. Choose a JPEG, PNG or HEIC."), err);

  await page.goto(`${BASE}/about`);
  await page.getByRole("heading", { name: "How it works", level: 1 }).waitFor();
  await page.screenshot({ path: join(shots, "desktop-6-about.png"), fullPage: true });
  const aboutText = await page.locator("main").innerText();
  check("about page shows evaluation numbers", /27 of 66/.test(aboutText) && /31 of 66/.test(aboutText));
  await page.close();
}

// ---------------------------------------------------------------- mobile 390x844
{
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.goto(BASE);
  await page.getByRole("heading", { name: "Where was this photo taken?" }).waitFor();
  await settle(page);
  await page.screenshot({ path: join(shots, "mobile-1-empty.png") });
  await upload(page, expected[0].path);
  await settle(page);
  await page.screenshot({ path: join(shots, "mobile-2-result.png") });
  const wide = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
  check("mobile: no horizontal scroll at 390 px", wide);
  await page.getByRole("button", { name: "Compare with match" }).click();
  await page.getByRole("dialog").waitFor();
  await page.waitForTimeout(500);
  await page.screenshot({ path: join(shots, "mobile-3-compare.png") });
  await ctx.close();
}

await browser.close();
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
