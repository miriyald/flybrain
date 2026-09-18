"""Step 11 - rebuild the fixture the digit demo's parity check reads.

`web/digits/verify.js` has always checked that the browser port agrees with Python, but nothing
in the repository produced the file it reads, and that file is gitignored - so on a fresh clone
the check could not be run at all. This writes it.

The cases are drawn canvases rather than dataset rasters, because the canvas path is where the
interesting failure lives: cropping, aspect-preserving rescaling and fractional block coverage
all have to match, and a browser that rounds any of them differently will read a different
digit before the circuit ever sees it.

Run:  python 11_check_the_web.py
"""

from __future__ import annotations

import json

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_digits

from flylab import data, digits, model
from flylab.circuit import Circuit
from flylab.classifier import FlyClassifier

CHECK_PATH = data.DATA_DIR / "web_check.json"
CANVAS = 280
STROKE = 28
SEED = 0
VARIANTS = 4


def as_ink(digit: npt.NDArray[np.float64], rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Paint an 8x8 digit onto a canvas the way a hand would leave it: thick, soft, off-centre."""
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.float64)
    grid = digit.reshape(8, 8) / 16.0
    shift = rng.integers(-18, 19, size=2)
    scale = float(rng.uniform(0.82, 1.0))
    span = int(STROKE * scale)

    for row in range(8):
        for col in range(8):
            if grid[row, col] <= 0.0:
                continue
            top = int((CANVAS - 8 * span) / 2 + row * span + shift[0])
            left = int((CANVAS - 8 * span) / 2 + col * span + shift[1])
            top, left = max(0, min(CANVAS - span, top)), max(0, min(CANVAS - span, left))
            canvas[top : top + span, left : left + span] += grid[row, col]

    blurred = canvas.copy()
    for axis in (0, 1):
        blurred = (blurred + np.roll(blurred, 1, axis=axis) + np.roll(blurred, -1, axis=axis)) / 3.0
    return np.clip(blurred, 0.0, 1.0)


def main() -> None:
    print(__doc__)
    circuit = Circuit.load()
    classifier = FlyClassifier.load(circuit)
    weights = model.normalise_gain(circuit.pn_to_kc)
    rng = np.random.default_rng(SEED)

    images, labels = load_digits(return_X_y=True)
    cases = []
    for label in range(10):
        sources = np.flatnonzero(labels == label)[:VARIANTS]
        for index in sources:
            ink = as_ink(images[index], rng)
            digit = digits.bitmap_to_digit(ink)
            activation = digit[classifier.pixels]
            code = model.kenyon_code(activation, weights, classifier.sparsity)
            cases.append(
                {
                    "ink": [float(f"{v:.9g}") for v in ink.ravel()],
                    "width": CANVAS,
                    "height": CANVAS,
                    # Not rounded. The 8x8 is compared exactly, and both languages print the
                    # shortest string that round-trips a float64, so full precision agrees
                    # while nine significant digits does not.
                    "digit": [float(v) for v in digit],
                    "code": [int(i) for i in np.flatnonzero(code)],
                    "prediction": int(classifier.predict(activation)[0]),
                }
            )

    CHECK_PATH.write_text(json.dumps(cases, separators=(",", ":")), encoding="utf-8")
    blank = sum(1 for case in cases if not any(case["digit"]))
    print(f"  {len(cases)} drawn cases, {blank} of them blank (blank cases teach nothing - expect 0)")
    print(f"\nwrote {CHECK_PATH}  ({CHECK_PATH.stat().st_size / 1e6:.2f} MB)")
    print("\nNext: node web/digits/verify.js, which must report zero mismatches.")


if __name__ == "__main__":
    main()
