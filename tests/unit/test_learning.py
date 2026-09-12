"""The end-to-end claim: learning is specific to the odour that was punished.

Needs the built circuit, so it skips cleanly on a fresh checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from flylab import model
from flylab.circuit import CIRCUIT_PATH, Circuit

pytestmark = pytest.mark.skipif(not CIRCUIT_PATH.exists(), reason="run 01_get_the_data.py and 03_build_circuit.py first")

TRIALS = 6
RATE = 0.30


@pytest.fixture(scope="module")
def circuit() -> Circuit:
    return Circuit.load()


def _train(circuit: Circuit, seed: int) -> tuple[float, float, float]:
    """Punish odour A, then report A's shift, B's shift, and their Kenyon cell overlap."""
    rng = np.random.default_rng(seed)
    weights = model.normalise_gain(circuit.pn_to_kc)
    code_a = model.kenyon_code(model.odour(circuit.n_glomeruli, rng), weights)
    code_b = model.kenyon_code(model.odour(circuit.n_glomeruli, rng), weights)

    def valence(kc_to_mbon: np.ndarray, code: np.ndarray) -> float:
        return model.valence(model.mbon_response(code, kc_to_mbon), circuit.mbon_valence)

    trained = circuit.kc_to_mbon
    for _ in range(TRIALS):
        trained = model.depress(trained, code_a, circuit.mbon_valence == 1, RATE)

    overlap = float((code_a & code_b).sum()) / max(1, int(code_a.sum()))
    shift_a = valence(trained, code_a) - valence(circuit.kc_to_mbon, code_a)
    shift_b = valence(trained, code_b) - valence(circuit.kc_to_mbon, code_b)
    return shift_a, shift_b, overlap


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13])
def test_punished_odour_shifts_towards_avoidance(circuit: Circuit, seed: int) -> None:
    shift_a, _, _ = _train(circuit, seed)
    assert shift_a < -0.2


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13])
def test_unpaired_odour_is_left_alone(circuit: Circuit, seed: int) -> None:
    """The result the whole project exists to show: no catastrophic interference."""
    shift_a, shift_b, _ = _train(circuit, seed)
    assert abs(shift_b) < 0.1 * abs(shift_a)


def test_sparseness_matches_published_range(circuit: Circuit) -> None:
    rng = np.random.default_rng(3)
    weights = model.normalise_gain(circuit.pn_to_kc)
    code = model.kenyon_code(model.odour(circuit.n_glomeruli, rng), weights)
    assert 0.03 <= code.mean() <= 0.10


def test_gain_normalisation_decorrelates_unrelated_odours(circuit: Circuit) -> None:
    """Without it the same high-gain cells win every time; see model.normalise_gain."""
    rng = np.random.default_rng(4)
    overlaps = {}
    for label, weights in (("raw", circuit.pn_to_kc), ("normalised", model.normalise_gain(circuit.pn_to_kc))):
        pairs = []
        for _ in range(20):
            first = model.kenyon_code(model.odour(circuit.n_glomeruli, rng), weights)
            second = model.kenyon_code(model.odour(circuit.n_glomeruli, rng), weights)
            pairs.append(float((first & second).sum()) / max(1, int(first.sum())))
        overlaps[label] = float(np.mean(pairs))
    assert overlaps["normalised"] < 0.15
    assert overlaps["normalised"] < overlaps["raw"] / 2.0


def test_derived_valence_covers_both_directions(circuit: Circuit) -> None:
    assert (circuit.mbon_valence == 1).sum() > 0
    assert (circuit.mbon_valence == -1).sum() > 0
