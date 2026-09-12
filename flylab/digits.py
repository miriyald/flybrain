"""Turn digit images into something the fly's antennal lobe could accept.

The fly has 55 glomeruli. An 8x8 digit has 64 pixels, but the border pixels of a digit
raster are blank in essentially every sample - the left column is exactly zero across all
1,797 - so dropping the nine most useless ones costs 0.003% of the total variance and
leaves one pixel driving one glomerulus.

The canvas half reproduces how the optdigits dataset was originally built: 32x32 binary
bitmaps reduced by summing 4x4 blocks, giving values 0-16. A hand-drawn digit that skips
that pipeline will not resemble the training data, however good the classifier is.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

GLOMERULI = 55
BITMAP = 32
BLOCK = BITMAP // 8


def live_pixels(images: npt.NDArray[np.float64], keep: int = GLOMERULI) -> npt.NDArray[np.int64]:
    """Indices of the `keep` most informative pixels, measured rather than assumed."""
    if keep > images.shape[1]:
        raise ValueError(f"cannot keep {keep} of {images.shape[1]} pixels")
    return np.sort(np.argsort(images.var(axis=0))[-keep:])


def to_glomeruli(images: npt.NDArray[np.float64], pixels: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    """Select the live pixels and present them as glomerular activation."""
    return np.atleast_2d(images)[:, pixels].astype(np.float32)


def render(digit: npt.NDArray[np.float32], shades: str = " .:-=+*#%@") -> list[str]:
    """An 8x8 digit as ASCII, so a mismatch between drawn and trained input is visible."""
    grid = np.asarray(digit, dtype=np.float32).reshape(8, 8)
    scale = max(float(grid.max()), 1.0)
    return ["".join(shades[min(int(value / scale * len(shades)), len(shades) - 1)] * 2 for value in row) for row in grid]


def bitmap_to_digit(ink: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
    """Canvas ink -> an 8x8 digit scaled 0-16, matching the optdigits pipeline.

    Crop to the ink, pad back to square so the aspect ratio survives, reduce to a 32x32
    binary bitmap, then sum each 4x4 block. A blank canvas returns all zeros.
    """
    marked = np.argwhere(ink > 0.0)
    if marked.size == 0:
        return np.zeros(64, dtype=np.float32)

    (top, left), (bottom, right) = marked.min(axis=0), marked.max(axis=0) + 1
    cropped = ink[top:bottom, left:right]
    squared = _pad_to_square(cropped)
    binary = _box_resize(squared, BITMAP) > 0.0
    blocks = binary.reshape(8, BLOCK, 8, BLOCK).sum(axis=(1, 3))
    digit: npt.NDArray[np.float32] = blocks.astype(np.float32).ravel()
    return digit


def _pad_to_square(image: npt.NDArray[np.float64], margin: float = 0.15) -> npt.NDArray[np.float64]:
    """Centre the image in a square with a margin, the way the source digits are framed."""
    side = round(max(image.shape) * (1.0 + 2.0 * margin))
    canvas = np.zeros((side, side), dtype=image.dtype)
    top = (side - image.shape[0]) // 2
    left = (side - image.shape[1]) // 2
    canvas[top:top + image.shape[0], left:left + image.shape[1]] = image
    return canvas


def _box_resize(image: npt.NDArray[np.float64], size: int) -> npt.NDArray[np.float64]:
    """Area-average resize. Averaging rather than sampling keeps thin strokes alive."""
    rows = (np.arange(size + 1) * image.shape[0] / size).astype(int)
    cols = (np.arange(size + 1) * image.shape[1] / size).astype(int)
    out = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        top, bottom = rows[i], max(rows[i + 1], rows[i] + 1)
        for j in range(size):
            left, right = cols[j], max(cols[j + 1], cols[j] + 1)
            out[i, j] = image[top:bottom, left:right].mean()
    return out
