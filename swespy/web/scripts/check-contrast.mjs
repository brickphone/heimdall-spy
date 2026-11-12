// WCAG 2.2 contrast check for every text/background pair the UI uses, with colours read
// from the token file. AA: 4.5:1 for normal text, 3:1 for large text (>= 24 px, or
// >= 18.66 px bold) and for the state squares and focus ring (non-text, 1.4.11).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const css = readFileSync(join(fileURLToPath(import.meta.url), "..", "..", "src", "app", "globals.css"), "utf8");
const tokens = Object.fromEntries([...css.matchAll(/--color-([a-z0-9-]+):\s*(#[0-9a-f]{6})/gi)].map((m) => [m[1], m[2]]));

function lum(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
function ratio(a, b) {
  const [l1, l2] = [lum(tokens[a]), lum(tokens[b])].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
}

// [foreground, background, minimum, where]. Map labels are checked against their halo colour.
const PAIRS = [
  ["ink", "surface", 4.5, "body text on panels"],
  ["ink", "paper", 4.5, "text on the app background"],
  ["ink-2", "surface", 4.5, "secondary text on panels"],
  ["ink-2", "paper", 4.5, "secondary text on hover rows / about"],
  ["ink-2", "selected", 4.5, "secondary text in a selected row"],
  ["ink", "selected", 4.5, "text in a selected row"],
  ["sure", "surface", 4.5, "Confident label"],
  ["unsure", "surface", 4.5, "Low confidence label"],
  ["danger", "surface", 4.5, "error messages"],
  ["pin", "surface", 4.5, "links"],
  ["pin-hover", "surface", 4.5, "hovered links"],
  ["surface", "pin", 4.5, "primary button text"],
  ["surface", "pin-hover", 4.5, "primary button text, hover"],
  ["surface", "ink", 4.5, "pressed toggles, Unsure verdict and tooltips"],
  ["surface", "sure", 4.5, "Matches verdict, pressed"],
  ["surface", "danger", 4.5, "Doesnt match verdict, pressed"],
  ["ink-2", "rule-list", 4.5, "secondary text on photo placeholders"],
  ["pin", "selected", 4.5, "selected candidate square on the map"],
  ["map-label", "map-land", 4.5, "district names (14 px, on a land halo)"],
  ["map-label", "map-landuse", 4.5, "district names over residential areas"],
  ["ink-map", "map-land", 4.5, "scale bar on the map"],
  ["ink-map", "surface", 4.5, "attribution on the map"],
  ["map-street-label", "map-road", 4.5, "street names (white halo on white roads)"],
  ["map-water-label", "map-land", 4.5, "water names (12 px italic, on a land halo)"],
  ["unsure", "map-land", 3.0, "low-confidence pin and rings on the map (non-text)"],
  ["pin", "map-land", 3.0, "prediction pin and rings on the map (non-text)"],
  ["ink", "map-land", 3.0, "candidate square borders on the map (non-text)"],
  ["sure", "surface", 3.0, "Confident state square (non-text)"],
  ["unsure", "surface", 3.0, "Low confidence state square (non-text)"],
  ["pin", "surface", 3.0, "focus ring (non-text)"],
  ["pin", "paper", 3.0, "focus ring on the map/background (non-text)"],
];

let failed = 0;
for (const [fg, bg, min, where] of PAIRS) {
  const r = ratio(fg, bg);
  const ok = r >= min;
  if (!ok) failed++;
  console.log(`${ok ? "pass" : "FAIL"}  ${r.toFixed(2).padStart(5)}:1  (needs ${min})  ${fg} on ${bg}  ${where}`);
}
if (failed) {
  console.error(`${failed} pair(s) below WCAG AA.`);
  process.exit(1);
}
console.log("Contrast check: all pairs pass WCAG AA.");
