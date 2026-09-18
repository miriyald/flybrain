/* Check the browser runner against the Python it was trained in.
 *
 * Python writes runs to data/dino_check.json; this replays them in JavaScript and requires an
 * exact match - the same obstacles, the same Kenyon cells firing, the same action every frame.
 *
 * A game is less forgiving than a classifier. A digit misread is one wrong answer; one wrong
 * decision here sends the run down a different future and never rejoins. So "close" is not a
 * weaker pass, it is a failure that has not shown up yet, and the bar is zero.
 *
 * Exactness is reachable because both sides quantise the drive before ranking, both narrow the
 * glomerular activation to float32 before the matrix multiply, and both draw obstacles from the
 * same generator - mulberry32, chosen precisely because it can be written twice and checked.
 *
 * Run:  node web/dino/verify.js
 */

const fs = require("fs");
const path = require("path");

const DinoGame = require("./dino-game.js");
const FlyDino = require("./flydino.js");

const MIN_CODE_OVERLAP = 1.0;

const root = path.join(__dirname, "..", "..", "data");
const model = JSON.parse(fs.readFileSync(path.join(root, "flydino_web.json"), "utf8"));
const check = JSON.parse(fs.readFileSync(path.join(root, "dino_check.json"), "utf8"));

let frames = 0;
let actionMismatch = 0;
let stateMismatch = 0;
let eventMismatch = 0;
let fieldsLost = 0;
let sampleCount = 0;
let worstOverlap = 1;
let worstScoreGap = 0;

for (const run of check.runs) {
  const game = new DinoGame.Game(run.seed);
  let decisions = 0;
  let sampleAt = 0;

  for (const expected of run.frames) {
    const decision = FlyDino.decide(game, model);
    frames++;

    if (decision.committed) {
      if (expected.action !== -1) actionMismatch++;
      game.step(DinoGame.RUN);
    } else {
      if (decision.action !== expected.action) actionMismatch++;

      const sample = run.samples[sampleAt];
      if (sample && sample.frame === game.frame && decisions % 8 === 0) {
        sampleCount++;
        const mine = new Set();
        for (let i = 0; i < decision.code.length; i++) if (decision.code[i]) mine.add(i);
        let shared = 0;
        for (const cell of sample.code) if (mine.has(cell)) shared++;
        worstOverlap = Math.min(worstOverlap, shared / sample.code.length);
        for (let a = 0; a < sample.scores.length; a++) {
          worstScoreGap = Math.max(worstScoreGap, Math.abs(decision.scores[a] - sample.scores[a]));
        }
        sampleAt++;
      }
      decisions++;
      const step = game.step(decision.action);
      if (step.cleared !== expected.cleared || step.crashed !== expected.crashed) eventMismatch++;
    }

    if (game.y !== expected.y || game.alive !== expected.alive) stateMismatch++;

    /* Python carries every field when an obstacle moves; JavaScript has to be told to. The
     * renderer is the only reader of `kind` and `bird`, so losing them left the rules and this
     * check both perfectly happy while the page threw on its second frame. */
    for (const obstacle of game.obstacles) {
      if (typeof obstacle.bird !== "boolean" || typeof obstacle.kind !== "number") fieldsLost++;
    }
  }
}

console.log(`frames replayed       : ${frames}`);
console.log(`action mismatches     : ${actionMismatch}   (must be 0)`);
console.log(`game state mismatches : ${stateMismatch}   (must be 0)`);
console.log(`event mismatches      : ${eventMismatch}   (must be 0)`);
console.log(`obstacles missing kind: ${fieldsLost}   (must be 0)`);
console.log(`sampled decisions     : ${sampleCount}`);
console.log(`worst code overlap    : ${(worstOverlap * 100).toFixed(2)}%   (must be ${MIN_CODE_OVERLAP * 100}%)`);
console.log(`worst score gap       : ${worstScoreGap.toExponential(2)}`);

if (actionMismatch || stateMismatch || eventMismatch || fieldsLost || worstOverlap < MIN_CODE_OVERLAP) {
  console.log("\nFAIL - the browser runner does not agree with Python");
  process.exit(1);
}
console.log("\nOK - the browser plays the identical game, decision for decision");
