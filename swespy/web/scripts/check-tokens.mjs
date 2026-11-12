// Fails if a hex colour, an rgb()/hsl() colour, a px border radius or a literal font size
// appears outside the token file (src/app/globals.css). Generated data and the token file
// are exempt.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(fileURLToPath(import.meta.url), "..", "..", "src");
const TOKEN_FILE = join(root, "app", "globals.css");
const RULES = [
  [/#[0-9a-fA-F]{3,8}\b/g, "hex colour"],
  [/\b(rgb|rgba|hsl|hsla)\(/g, "colour function"],
  [/border-radius\s*:\s*[0-9.]+px/g, "px border radius"],
  [/rounded-\[[0-9.]+px\]/g, "px border radius"],
  [/text-\[[0-9.]+(px|rem|em)\]/g, "off-token font size"],
  [/font-size\s*:\s*[0-9.]+(px|rem|em)/g, "off-token font size"],
  [/font:\s*[0-9]{3}\s+[0-9.]+(px|rem)/g, "off-token font size"],
];

function* files(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) {
      if (name !== "generated") yield* files(p);
    } else if (/\.(tsx?|css|mjs)$/.test(name) && p !== TOKEN_FILE) yield p;
  }
}

const problems = [];
for (const file of files(root)) {
  readFileSync(file, "utf8").split("\n").forEach((line, i) => {
    for (const [re, what] of RULES) {
      for (const m of line.matchAll(re)) problems.push(`${relative(root, file)}:${i + 1}  ${what}: ${m[0]}`);
    }
  });
}
if (problems.length) {
  console.error(`Off-token values found:\n${problems.join("\n")}`);
  process.exit(1);
}
console.log("Token check: no hex colours, colour functions, px radii or font sizes outside globals.css.");
