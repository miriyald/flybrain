"""The four model functions, and the two properties the result depends on."""

from __future__ import annotations

import numpy as np
import pytest

from flylab import model


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_kwta_keeps_exactly_k() -> None:
    drive = np.arange(100, dtype=np.float32)
    assert model.kwta(drive, k=7).sum() == 7


def test_kwta_keeps_the_strongest() -> None:
    drive = np.array([5.0, 1.0, 9.0, 3.0], dtype=np.float32)
    assert list(np.flatnonzero(model.kwta(drive, k=2))) == [0, 2]


def test_kenyon_code_hits_the_sparsity_target(rng: np.random.Generator) -> None:
    weights = model.normalise_gain(rng.random((20, 400), dtype=np.float32))
    code = model.kenyon_code(model.odour(20, rng), weights, sparsity=0.05)
    assert code.sum() == 20


def test_normalise_gain_equalises_columns(rng: np.random.Generator) -> None:
    weights = model.normalise_gain(rng.random((10, 50), dtype=np.float32) * 100.0)
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-5)


def test_normalise_gain_survives_unconnected_cells() -> None:
    weights = np.zeros((4, 3), dtype=np.float32)
    weights[:, 1] = 2.0
    normalised = model.normalise_gain(weights)
    assert normalised[:, 0].sum() == 0.0
    assert np.isclose(normalised[:, 1].sum(), 1.0)


def test_blend_at_full_similarity_returns_the_original(rng: np.random.Generator) -> None:
    base = model.odour(30, rng)
    assert np.allclose(model.blend(base, rng, similarity=1.0), base)


def test_depress_only_touches_active_rows_and_target_columns(rng: np.random.Generator) -> None:
    weights = rng.random((6, 4), dtype=np.float32) + 1.0
    code = np.array([True, False, True, False, False, False])
    targets = np.array([True, False, True, False])
    updated = model.depress(weights, code, targets, rate=0.5)

    assert np.array_equal(updated[~code], weights[~code])
    assert np.array_equal(updated[:, ~targets], weights[:, ~targets])
    assert np.allclose(updated[np.ix_(code, targets)], weights[np.ix_(code, targets)] * 0.5)


def test_depress_does_not_mutate_its_input(rng: np.random.Generator) -> None:
    weights = rng.random((5, 3), dtype=np.float32)
    original = weights.copy()
    model.depress(weights, np.ones(5, dtype=bool), np.ones(3, dtype=bool), rate=0.4)
    assert np.array_equal(weights, original)


def test_repeated_pairing_monotonically_weakens(rng: np.random.Generator) -> None:
    weights = rng.random((8, 3), dtype=np.float32) + 1.0
    code = np.array([True] * 4 + [False] * 4)
    targets = np.ones(3, dtype=bool)
    totals = []
    for _ in range(5):
        weights = model.depress(weights, code, targets, rate=0.3)
        totals.append(float(weights[code].sum()))
    assert totals == sorted(totals, reverse=True)


def test_valence_is_bounded_and_signed() -> None:
    signs = np.array([1, 1, -1, -1], dtype=np.int8)
    assert model.valence(np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32), signs) == pytest.approx(1.0)
    assert model.valence(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32), signs) == pytest.approx(-1.0)
    assert model.valence(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32), signs) == pytest.approx(0.0)


def test_valence_of_silence_is_neutral() -> None:
    assert model.valence(np.zeros(4, dtype=np.float32), np.array([1, -1, 1, -1], dtype=np.int8)) == 0.0
