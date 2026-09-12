"""Turn digit images into something the fly's antennal lobe could accept.

The fly has 55 glomeruli. An 8x8 digit has 64 pixels, but the border pixels of a digit
raster are blank in essentially every sample - the left column is exactly zero across all
1,797 - so dropping the nine most useless ones costs 0.003% of the total variance and
leaves one pixel driving one glomerulus.

The canvas half reframes a drawing the way optdigits frames its digits - height-normalised,
centred, reduced to 8x8 with values 0-16. Even then a freehand stroke is not quite what the
source set contains, so `augment` teaches the circuit the drawn form alongside the original.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

GLOMERULI = 55
BITMAP = 32
BLOCK = BITMAP // 8
_UPSCALE = 24
_SMEAR = 6
DRAWN_BLURS = (1, 3)


def live_pixels(images: npt.NDArray[np.floating[Any]], keep: int = GLOMERULI) -> npt.NDArray[np.int64]:
    """Indices of the `keep` most informative pixels, measured rather than assumed."""
    if keep > images.shape[1]:
        raise ValueError(f"cannot keep {keep} of {images.shape[1]} pixels")
    return np.sort(np.argsort(images.var(axis=0))[-keep:])


def to_glomeruli(images: npt.NDArray[np.floating[Any]], pixels: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    """Select the live pixels and present them as glomerular activation."""
    return np.atleast_2d(images)[:, pixels].astype(np.float32)


def render(digit: npt.NDArray[np.float32], shades: str = " .:-=+*#%@") -> list[str]:
    """An 8x8 digit as ASCII, so a mismatch between drawn and trained input is visible."""
    grid = np.asarray(digit, dtype=np.float32).reshape(8, 8)
    scale = max(float(grid.max()), 1.0)
    return ["".join(shades[min(int(value / scale * len(shades)), len(shades) - 1)] * 2 for value in row) for row in grid]


def bitmap_to_digit(ink: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float32]:
    """Canvas ink -> an 8x8 digit scaled 0-16, framed the way optdigits frames digits.

    Every digit in the source set spans all eight rows and none spans all eight columns:
    they are height-normalised, with width left to follow the digit's own proportions
    (mean box 8 x 5.9). So the ink is scaled until it fills the height, keeps its aspect
    ratio, and is centred horizontally. Squaring it instead - the obvious thing to do -
    widens every digit by about a third, which is enough to close the loop of a 6 into an 8.

    Coverage is kept fractional rather than thresholded. Counting only fully-inked pixels
    gives a drawn digit just two values, 8 and 16, where the training digits spread smoothly
    across 1-16 - the grey levels live at the edges of a stroke, and rounding them away is
    what makes a hand-drawn digit unrecognisable to a circuit taught on soft ones.
    A blank canvas returns all zeros.
    """
    marked = np.argwhere(ink > 0.0)
    if marked.size == 0:
        return np.zeros(64, dtype=np.float32)

    (top, left), (bottom, right) = marked.min(axis=0), marked.max(axis=0) + 1
    cropped = ink[top:bottom, left:right]

    width = min(BITMAP, max(1, round(BITMAP * cropped.shape[1] / cropped.shape[0])))
    scaled = np.clip(_box_resize(cropped, BITMAP, width), 0.0, 1.0)

    frame = np.zeros((BITMAP, BITMAP), dtype=np.float64)
    offset = (BITMAP - width) // 2
    frame[:, offset:offset + width] = scaled

    blocks = frame.reshape((8, BLOCK, 8, BLOCK)).sum(axis=(1, 3))
    digit: npt.NDArray[np.float32] = blocks.astype(np.float32).ravel()
    return digit


def _box_resize(image: npt.NDArray[np.floating[Any]], height: int, width: int) -> npt.NDArray[np.float64]:
    """Area-average resize. Averaging rather than sampling keeps thin strokes alive."""
    rows = (np.arange(height + 1) * image.shape[0] / height).astype(int)
    cols = (np.arange(width + 1) * image.shape[1] / width).astype(int)
    out = np.zeros((height, width), dtype=np.float64)
    for i in range(height):
        top, bottom = rows[i], max(rows[i + 1], rows[i] + 1)
        for j in range(width):
            left, right = cols[j], max(cols[j + 1], cols[j] + 1)
            out[i, j] = image[top:bottom, left:right].mean()
    return out


def as_drawn(image: npt.NDArray[np.floating[Any]], blur: int = 1) -> npt.NDArray[np.float32]:
    """An 8x8 digit as it would come back if someone had drawn that shape.

    Freehand strokes land outside the distribution the source digits occupy - they are
    thicker, harder-edged, and framed by where the hand happened to put them. Training on
    these alongside the originals teaches the circuit both, and costs about a point of
    accuracy on the clean set to gain twelve on drawn input.
    """
    canvas = np.kron(np.asarray(image, dtype=np.float64).reshape(8, 8), np.ones((_UPSCALE, _UPSCALE)))
    for _ in range(blur):
        canvas = (
            canvas
            + np.roll(canvas, _SMEAR, axis=0)
            + np.roll(canvas, -_SMEAR, axis=0)
            + np.roll(canvas, _SMEAR, axis=1)
            + np.roll(canvas, -_SMEAR, axis=1)
        ) / 5.0
    return bitmap_to_digit(canvas)


def augment(images: npt.NDArray[np.floating[Any]], blurs: tuple[int, ...] = DRAWN_BLURS) -> npt.NDArray[np.float32]:
    """Stack the originals with one drawn-style copy per blur level."""
    drawn = [np.array([as_drawn(image, blur) for image in images]) for blur in blurs]
    return np.vstack([images.astype(np.float32), *drawn])
