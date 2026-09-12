/* The fly's mushroom body, running in a browser.
 *
 * A direct port of flylab/digits.py and flylab/classifier.py. It must agree with Python
 * exactly - tests/verify_web.js checks that against the real model on real digits.
 */

(function (global) {
  "use strict";

  const BITMAP = 32;
  const BLOCK = BITMAP / 8;
  const DRIVE_GRID = 1e-4;

  /* Python's round() breaks ties to even; Math.round breaks them upward. */
  function roundHalfEven(value) {
    const floor = Math.floor(value);
    const diff = value - floor;
    if (diff > 0.5) return floor + 1;
    if (diff < 0.5) return floor;
    return floor % 2 === 0 ? floor : floor + 1;
  }

  /* Area-average resize. Averaging rather than sampling keeps thin strokes alive. */
  function boxResize(image, srcW, srcH, height, width) {
    const rowEdge = [], colEdge = [];
    for (let i = 0; i <= height; i++) rowEdge.push(Math.trunc((i * srcH) / height));
    for (let j = 0; j <= width; j++) colEdge.push(Math.trunc((j * srcW) / width));

    const out = new Float64Array(height * width);
    for (let i = 0; i < height; i++) {
      const top = rowEdge[i];
      const bottom = Math.max(rowEdge[i + 1], top + 1);
      for (let j = 0; j < width; j++) {
        const left = colEdge[j];
        const right = Math.max(colEdge[j + 1], left + 1);
        let total = 0;
        for (let y = top; y < bottom; y++) for (let x = left; x < right; x++) total += image[y * srcW + x];
        out[i * width + j] = total / ((bottom - top) * (right - left));
      }
    }
    return out;
  }

  /* Canvas ink -> an 8x8 digit scaled 0-16, framed the way optdigits frames digits.
   *
   * Every digit in the source set spans all eight rows and none spans all eight columns:
   * they are height-normalised, with width following the digit's own proportions. Squaring
   * the crop instead - the obvious thing to do - widens every digit by about a third, which
   * is enough to close the loop of a 6 into an 8.
   *
   * Coverage stays fractional. Counting only fully-inked pixels leaves a drawn digit with
   * just two values, 8 and 16, where the source digits spread smoothly across 1-16. */
  function bitmapToDigit(ink, width, height) {
    let top = height, left = width, bottom = -1, right = -1;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (ink[y * width + x] > 0) {
          if (y < top) top = y;
          if (y > bottom) bottom = y;
          if (x < left) left = x;
          if (x > right) right = x;
        }
      }
    }
    if (bottom < 0) return new Float32Array(64);

    const cropH = bottom - top + 1;
    const cropW = right - left + 1;
    const crop = new Float64Array(cropH * cropW);
    for (let y = 0; y < cropH; y++) {
      for (let x = 0; x < cropW; x++) crop[y * cropW + x] = ink[(top + y) * width + left + x];
    }

    const target = Math.min(BITMAP, Math.max(1, roundHalfEven((BITMAP * cropW) / cropH)));
    const scaled = boxResize(crop, cropW, cropH, BITMAP, target);
    const offset = Math.floor((BITMAP - target) / 2);

    /* Accumulate at double precision and narrow once, the way numpy sums then casts.
     * Adding straight into a Float32Array rounds after every term and drifts apart. */
    const totals = new Float64Array(64);
    for (let i = 0; i < BITMAP; i++) {
      for (let j = 0; j < target; j++) {
        const coverage = Math.min(1, Math.max(0, scaled[i * target + j]));
        totals[Math.floor(i / BLOCK) * 8 + Math.floor((offset + j) / BLOCK)] += coverage;
      }
    }
    return Float32Array.from(totals);
  }

  /* Drive every Kenyon cell, then let the strongest k percent win - APL's job. */
  function kenyonCode(activation, model) {
    const drive = new Float64Array(model.kenyonCells);
    const { indptr, indices, data } = model.pnToKc;
    for (let g = 0; g < activation.length; g++) {
      const value = activation[g];
      if (value === 0) continue;
      for (let p = indptr[g]; p < indptr[g + 1]; p++) drive[indices[p]] += value * data[p];
    }

    /* Snap to a coarse grid before ranking, matching flylab.model.quantise. A cell fed only
     * saturated pixels scores exactly 16, and hundreds can sit there together with the
     * winner boundary inside that group; float noise of a few parts in 10^8 would otherwise
     * decide which of them fire. floor(x/grid + 0.5) avoids the differing half-way rules of
     * numpy's round and JavaScript's Math.round. */
    const ranked = new Float64Array(model.kenyonCells);
    for (let i = 0; i < drive.length; i++) ranked[i] = Math.floor(drive[i] / DRIVE_GRID + 0.5);
    const order = Array.from(ranked.keys()).sort((a, b) => ranked[b] - ranked[a] || a - b);
    const code = new Uint8Array(model.kenyonCells);
    for (let i = 0; i < model.k; i++) code[order[i]] = 1;
    return code;
  }

  function score(code, model) {
    const scores = new Float32Array(model.classes);
    for (let cell = 0; cell < model.kenyonCells; cell++) {
      if (!code[cell]) continue;
      const row = cell * model.classes;
      for (let c = 0; c < model.classes; c++) scores[c] += model.readout[row + c];
    }
    return scores;
  }

  /* Ink in, verdict out. */
  function read(ink, width, height, model) {
    return readDigit(bitmapToDigit(ink, width, height), model);
  }

  /* An 8x8 digit in, verdict out - the entry point for dataset samples, which are already
   * in the circuit's input format and must not be pushed back through the canvas pipeline.
   * Re-framing and re-quantising an already-8x8 digit only degrades it. */
  function readDigit(digit, model) {
    if (!digit.some((value) => value > 0)) return null;

    const activation = new Float32Array(model.pixels.length);
    for (let i = 0; i < model.pixels.length; i++) activation[i] = digit[model.pixels[i]];

    const code = kenyonCode(activation, model);
    const scores = score(code, model);
    const ranked = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]);
    let active = 0;
    for (let i = 0; i < code.length; i++) active += code[i];
    return { digit, code, scores, ranked, active };
  }

  global.FlyBrain = { bitmapToDigit, kenyonCode, score, read, readDigit };
})(typeof window !== "undefined" ? window : globalThis);
