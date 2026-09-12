/* The fly's mushroom body, running in a browser.
 *
 * A direct port of flylab/digits.py and flylab/classifier.py. It must agree with Python
 * exactly - tests/verify_web.js checks that against the real model on real digits.
 */

(function (global) {
  "use strict";

  const BITMAP = 32;
  const BLOCK = BITMAP / 8;
  const MARGIN = 0.15;

  /* Python's round() breaks ties to even; Math.round breaks them upward. */
  function roundHalfEven(value) {
    const floor = Math.floor(value);
    const diff = value - floor;
    if (diff > 0.5) return floor + 1;
    if (diff < 0.5) return floor;
    return floor % 2 === 0 ? floor : floor + 1;
  }

  /* Canvas ink -> an 8x8 digit scaled 0-16, matching how optdigits was built. */
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

    const cropHeight = bottom - top + 1;
    const cropWidth = right - left + 1;
    const side = roundHalfEven(Math.max(cropHeight, cropWidth) * (1 + 2 * MARGIN));
    const padTop = Math.floor((side - cropHeight) / 2);
    const padLeft = Math.floor((side - cropWidth) / 2);

    const square = new Float64Array(side * side);
    for (let y = 0; y < cropHeight; y++) {
      for (let x = 0; x < cropWidth; x++) {
        square[(padTop + y) * side + padLeft + x] = ink[(top + y) * width + left + x];
      }
    }

    /* Area-average down to 32x32, then threshold. Averaging keeps thin strokes alive. */
    const edges = [];
    for (let i = 0; i <= BITMAP; i++) edges.push(Math.trunc((i * side) / BITMAP));

    const digit = new Float32Array(64);
    for (let i = 0; i < BITMAP; i++) {
      const rowStart = edges[i];
      const rowEnd = Math.max(edges[i + 1], rowStart + 1);
      for (let j = 0; j < BITMAP; j++) {
        const colStart = edges[j];
        const colEnd = Math.max(edges[j + 1], colStart + 1);
        let total = 0;
        for (let y = rowStart; y < rowEnd; y++) {
          for (let x = colStart; x < colEnd; x++) total += square[y * side + x];
        }
        if (total > 0) {
          digit[Math.floor(i / BLOCK) * 8 + Math.floor(j / BLOCK)] += 1;
        }
      }
    }
    return digit;
  }

  /* Drive every Kenyon cell, then let the strongest k percent win - APL's job. */
  function kenyonCode(activation, model) {
    const drive = new Float32Array(model.kenyonCells);
    const { indptr, indices, data } = model.pnToKc;
    for (let g = 0; g < activation.length; g++) {
      const value = activation[g];
      if (value === 0) continue;
      for (let p = indptr[g]; p < indptr[g + 1]; p++) drive[indices[p]] += value * data[p];
    }

    const order = Array.from(drive.keys()).sort((a, b) => drive[b] - drive[a]);
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
   * Doing that binarises them, saturating every lit cell to 16 and losing the grey levels
   * the circuit was taught on. */
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
