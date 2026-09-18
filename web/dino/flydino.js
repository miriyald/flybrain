/* The mushroom body choosing what the runner does next.
 *
 * A port of flylab/pilot.py. Only the readout differs from the digit demo: three actions
 * instead of ten digits, and a game state instead of pixels.
 *
 * The kenyonCode kernel below is deliberately a copy of the one in ../digits/flybrain.js
 * rather than a shared import. Each demo is meant to stand on its own - one folder, openable
 * and deployable without the other. If you change the quantisation here, change it there too;
 * both are checked against Python by their own verify.js.
 */

(function (global) {
  "use strict";

  const DRIVE_GRID = 1e-4;

  /* Drive every Kenyon cell, then let the strongest k percent win - APL's job. */
  function kenyonCode(activation, model) {
    const drive = new Float64Array(model.kenyonCells);
    const { indptr, indices, data } = model.pnToKc;
    for (let g = 0; g < activation.length; g++) {
      const value = activation[g];
      if (value === 0) continue;
      for (let p = indptr[g]; p < indptr[g + 1]; p++) drive[indices[p]] += value * data[p];
    }

    /* Snap to a coarse grid before ranking, matching flylab.model.quantise. Ties are broken
     * by index so the winner set is the same set Python picked. floor(x/grid + 0.5) avoids
     * the differing half-way rules of numpy's round and JavaScript's Math.round. */
    const ranked = new Float64Array(model.kenyonCells);
    for (let i = 0; i < drive.length; i++) ranked[i] = Math.floor(drive[i] / DRIVE_GRID + 0.5);
    const order = Array.from(ranked.keys()).sort((a, b) => ranked[b] - ranked[a] || a - b);
    const code = new Uint8Array(model.kenyonCells);
    for (let i = 0; i < model.k; i++) code[order[i]] = 1;
    return code;
  }

  /* The readout ships as approach minus avoid, because only the difference is ever read. */
  function score(code, model) {
    const scores = new Float32Array(model.actions);
    for (let cell = 0; cell < model.kenyonCells; cell++) {
      if (!code[cell]) continue;
      const row = cell * model.actions;
      for (let a = 0; a < model.actions; a++) scores[a] += model.readout[row + a];
    }
    return scores;
  }

  /* Ties break by index, matching numpy's argmax and the k-winners-take-all convention.
   * An untaught circuit scores every action at exactly zero, so this makes it run. */
  function choose(scores) {
    let best = 0;
    for (let a = 1; a < scores.length; a++) {
      if (scores[a] > scores[best]) best = a;
    }
    return best;
  }

  /* One decision from a live game. Mid-air there is nothing to decide: the arc is committed,
   * which is exactly why training did not credit those frames either. */
  function decide(game, model) {
    if (game.jumping()) return { action: 0, code: null, scores: null, committed: true };
    const code = kenyonCode(game.glomeruli(), model);
    const scores = score(code, model);
    return { action: choose(scores), code: code, scores: scores, committed: false };
  }

  const FlyDino = { kenyonCode: kenyonCode, score: score, choose: choose, decide: decide };

  if (typeof module !== "undefined" && module.exports) module.exports = FlyDino;
  else global.FlyDino = FlyDino;
})(typeof globalThis !== "undefined" ? globalThis : this);
