"""Step 7 - draw a number in the square and let the fly read it.

The window shows three things at once, on purpose:

  the square you draw in
  what the circuit actually receives, after the drawing is reduced to 8x8
  which Kenyon cells fired, and what they voted for

The middle panel matters most. A digit that looks fine to you can still reach the circuit as
something unrecognisable, and when that happens the preview shows it immediately rather than
leaving you guessing at a wrong answer.

Run:  python 07_draw_a_digit.py
"""

from __future__ import annotations

import tkinter as tk

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_digits

from flylab import digits
from flylab.circuit import Circuit
from flylab.classifier import MEMORY_PATH, FlyClassifier

CANVAS = 280
BRUSH = 11
INK = "#e8eef5"
PAPER = "#1d2530"


def _stamp(sheet: npt.NDArray[np.float64], x: int, y: int, radius: int = BRUSH) -> None:
    """Mark a filled disc of ink, clipped to the sheet."""
    top, bottom = max(0, y - radius), min(sheet.shape[0], y + radius + 1)
    left, right = max(0, x - radius), min(sheet.shape[1], x + radius + 1)
    if top >= bottom or left >= right:
        return
    rows = np.arange(top, bottom)[:, None] - y
    cols = np.arange(left, right)[None, :] - x
    sheet[top:bottom, left:right][rows**2 + cols**2 <= radius**2] = 1.0


class DigitPad:
    """A square to draw in, wired to the fly's circuit."""

    def __init__(self, classifier: FlyClassifier) -> None:
        self.classifier = classifier
        self.sheet = np.zeros((CANVAS, CANVAS), dtype=np.float64)
        self.last: tuple[int, int] | None = None

        self.root = tk.Tk()
        self.root.title("Draw a number - read by a fly")
        self.root.configure(bg=PAPER)

        frame = tk.Frame(self.root, bg=PAPER)
        frame.pack(padx=12, pady=12)

        self.canvas = tk.Canvas(frame, width=CANVAS, height=CANVAS, bg=PAPER, highlightthickness=1, highlightbackground="#3a4654")
        self.canvas.grid(row=0, column=0, rowspan=2, padx=(0, 12))
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)

        self.preview = tk.Label(frame, font=("Consolas", 9), bg=PAPER, fg="#8fa3b8", justify="left", anchor="nw")
        self.preview.grid(row=0, column=1, sticky="nw")

        self.verdict = tk.Label(frame, font=("Consolas", 11), bg=PAPER, fg=INK, justify="left", anchor="nw")
        self.verdict.grid(row=1, column=1, sticky="nw")

        tk.Button(self.root, text="clear", command=self._clear, bg="#2d3f52", fg=INK, relief="flat", padx=18).pack(pady=(0, 12))
        self._clear()

    def _press(self, event: tk.Event[tk.Canvas]) -> None:
        self.last = (event.x, event.y)
        self._drag(event)

    def _drag(self, event: tk.Event[tk.Canvas]) -> None:
        x, y = event.x, event.y
        if self.last is not None:
            previous_x, previous_y = self.last
            steps = max(abs(x - previous_x), abs(y - previous_y), 1)
            for step in range(steps + 1):
                _stamp(self.sheet, previous_x + (x - previous_x) * step // steps, previous_y + (y - previous_y) * step // steps)
            self.canvas.create_line(previous_x, previous_y, x, y, fill=INK, width=BRUSH * 2, capstyle="round", smooth=True)
        self.last = (x, y)

    def _release(self, _: tk.Event[tk.Canvas]) -> None:
        self.last = None
        self._read()

    def _clear(self) -> None:
        self.sheet[:] = 0.0
        self.canvas.delete("all")
        self.last = None
        self.preview.configure(text="what the circuit receives\n\n" + "\n".join(["                "] * 8))
        self.verdict.configure(text="draw a number in the square")

    def _read(self) -> None:
        digit = digits.bitmap_to_digit(self.sheet)
        if digit.sum() == 0:
            return
        self.preview.configure(text="what the circuit receives\n\n" + "\n".join(digits.render(digit)))

        activation = digits.to_glomeruli(digit[None, :], self.classifier.pixels)
        code = self.classifier.encode(activation)[0]
        scores = self.classifier.scores(activation)[0]
        ranked = np.argsort(scores)[::-1]
        spread = float(scores.max() - scores.min()) or 1.0

        lines = [f"  {int(ranked[0])}", "", f"{int(code.sum())} of {code.size} Kenyon cells fired", ""]
        for candidate in ranked[:3]:
            share = (scores[candidate] - scores.min()) / spread
            lines.append(f"  {int(candidate)}  {'#' * round(share * 18):<18} {share:>5.0%}")
        self.verdict.configure(text="\n".join(lines))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    print(__doc__)
    circuit = Circuit.load()
    try:
        classifier = FlyClassifier.load(circuit)
        print(f"recalling what it learned in step 6: {MEMORY_PATH.name}")
    except FileNotFoundError:
        images, labels = load_digits(return_X_y=True)
        pixels = digits.live_pixels(images, keep=circuit.n_glomeruli)
        print(f"no saved memory, teaching it {len(images)} digits now ...")
        classifier = FlyClassifier.from_circuit(circuit, pixels).fit(digits.to_glomeruli(images, pixels), labels)
        classifier.save()
    print("ready - draw a number in the square, release the mouse to read it\n")
    DigitPad(classifier).run()


if __name__ == "__main__":
    main()
