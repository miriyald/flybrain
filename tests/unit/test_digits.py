"""Pixel-to-glomerulus encoding and the canvas preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from flylab import digits


@pytest.fixture
def images() -> np.ndarray:
    rng = np.random.default_rng(0)
    samples = rng.random((40, 64)) * 16.0
    samples[:, :6] = 0.0
    return samples


def test_live_pixels_drops_the_dead_ones(images: np.ndarray) -> None:
    chosen = digits.live_pixels(images, keep=55)
    assert len(chosen) == 55
    assert not set(chosen) & set(range(6))


def test_live_pixels_returns_sorted_indices(images: np.ndarray) -> None:
    chosen = digits.live_pixels(images, keep=55)
    assert list(chosen) == sorted(chosen)


def test_live_pixels_rejects_impossible_request(images: np.ndarray) -> None:
    with pytest.raises(ValueError, match="cannot keep"):
        digits.live_pixels(images, keep=99)


def test_to_glomeruli_selects_the_chosen_columns(images: np.ndarray) -> None:
    chosen = digits.live_pixels(images, keep=55)
    encoded = digits.to_glomeruli(images, chosen)
    assert encoded.shape == (40, 55)
    assert np.allclose(encoded, images[:, chosen])


def test_blank_canvas_produces_no_digit() -> None:
    assert digits.bitmap_to_digit(np.zeros((280, 280))).sum() == 0.0


def test_output_matches_the_optdigits_format() -> None:
    sheet = np.zeros((280, 280))
    sheet[60:220, 120:160] = 1.0
    digit = digits.bitmap_to_digit(sheet)
    assert digit.shape == (64,)
    assert digit.min() >= 0.0
    assert digit.max() <= 16.0


def _stroke(top: int, left: int, height: int, width: int, size: int = 280) -> np.ndarray:
    sheet = np.zeros((size, size))
    sheet[top:top + height, left:left + width] = 1.0
    return sheet


def test_drawing_position_does_not_matter() -> None:
    """Cropping to the ink means a digit drawn in any corner reaches the circuit the same."""
    corner = digits.bitmap_to_digit(_stroke(10, 10, 120, 40))
    middle = digits.bitmap_to_digit(_stroke(150, 200, 120, 40))
    assert np.array_equal(corner, middle)


def test_drawing_size_does_not_matter() -> None:
    """Aspect ratio is preserved, so a large and a small version agree closely."""
    small = digits.bitmap_to_digit(_stroke(40, 40, 60, 20))
    large = digits.bitmap_to_digit(_stroke(40, 40, 180, 60))
    assert np.corrcoef(small, large)[0, 1] > 0.95


def test_render_produces_eight_rows() -> None:
    lines = digits.render(np.arange(64, dtype=np.float32))
    assert len(lines) == 8
    assert all(len(line) == 16 for line in lines)


def test_render_handles_an_empty_digit() -> None:
    assert digits.render(np.zeros(64, dtype=np.float32)) == [" " * 16] * 8
