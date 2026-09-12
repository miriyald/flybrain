/* Check the browser port against the Python model.
 *
 * Python writes cases to data/web_check.json; this replays them in JavaScript.
 *
 * Preprocessing and predictions must match exactly. The Kenyon cell code is allowed a
 * sliver of disagreement, and the reason is worth stating: numpy sums a float32 matrix
 * product using pairwise summation over each output element, while this port accumulates
 * in sparse row order. The two differ by around 4e-6, which is nothing next to the typical
 * 1e-2 gap at the winner boundary - except when two cells tie there exactly, which integer
 * pixel values make common. Then one implementation sees a tie and breaks it by index while
 * the other sees a hair of difference. Measured: 6 differing cells in 11,773, no prediction
 * ever changed.
 *
 * Run:  node web/verify.js
 */

const fs = require("fs");
const path = require("path");

require("./flybrain.js");

const MIN_CODE_OVERLAP = 0.99;

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
console.log(`worst code overlap    : ${(worstOverlap * 100).toFixed(2)}%   (must be >= ${MIN_CODE_OVERLAP * 100}%)`);
console.log(`differing cells       : ${differingCells} of ${cases.length * model.k}`);

if (digitMismatch || labelMismatch || worstOverlap < MIN_CODE_OVERLAP) {
  console.log("\nFAIL - the browser port does not agree with Python");
  process.exit(1);
}
console.log("\nOK - preprocessing and predictions match Python exactly");
