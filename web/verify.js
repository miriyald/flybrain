/* Check the browser port against the Python model.
 *
 * Python writes cases to data/web_check.json; this replays them in JavaScript and requires
 * an exact match - the same 8x8 input, the same Kenyon cells firing, the same verdict.
 *
 * Exactness is reachable only because both sides quantise the drive before ranking. Gain
 * normalisation makes each cell's weights sum to one, so a cell fed only saturated pixels
 * scores exactly 16, and a bold drawing can leave three hundred cells sitting there with
 * the winner boundary inside that group. Left raw, which of them fire is decided by the
 * last bits of a float64 sum, and numpy's BLAS and a sparse row loop do not agree on those.
 *
 * Run:  node web/verify.js
 */

const fs = require("fs");
const path = require("path");

require("./flybrain.js");

const MIN_CODE_OVERLAP = 1.0;

const root = path.join(__dirname, "..", "data");
const model = JSON.parse(fs.readFileSync(path.join(root, "flybrain_web.json"), "utf8"));
const cases = JSON.parse(fs.readFileSync(path.join(root, "web_check.json"), "utf8"));

let digitMismatch = 0;
let labelMismatch = 0;
let worstOverlap = 1;
let differingCells = 0;

for (const item of cases) {
  const result = FlyBrain.read(Float64Array.from(item.ink), item.width, item.height, model);

  if (Array.from(result.digit).join(",") !== item.digit.join(",")) digitMismatch++;
  if (result.ranked[0] !== item.prediction) labelMismatch++;

  const expected = new Set(item.code);
  let shared = 0;
  for (let i = 0; i < result.code.length; i++) if (result.code[i] && expected.has(i)) shared++;
  worstOverlap = Math.min(worstOverlap, shared / expected.size);
  differingCells += expected.size - shared;
}

console.log(`cases                 : ${cases.length}`);
console.log(`8x8 mismatches        : ${digitMismatch}   (must be 0)`);
console.log(`prediction mismatches : ${labelMismatch}   (must be 0)`);
console.log(`worst code overlap    : ${(worstOverlap * 100).toFixed(2)}%   (must be ${MIN_CODE_OVERLAP * 100}%)`);
console.log(`differing cells       : ${differingCells} of ${cases.length * model.k}`);

if (digitMismatch || labelMismatch || worstOverlap < MIN_CODE_OVERLAP) {
  console.log("\nFAIL - the browser port does not agree with Python");
  process.exit(1);
}
console.log("\nOK - preprocessing and predictions match Python exactly");
