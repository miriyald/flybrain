"""The fly circuit used as a digit classifier.

Needs the built circuit, so it skips cleanly on a fresh checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from flylab.circuit import CIRCUIT_PATH, Circuit
from flylab.classifier import FlyClassifier

pytestmark = pytest.mark.skipif(not CIRCUIT_PATH.exists(), reason="run 01_get_the_data.py and 03_build_circuit.py first")


@pytest.fixture(scope="module")
def circuit() -> Circuit:
    return Circuit.load()


@pytest.fixture
def samples(circuit: Circuit) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    activations = rng.random((60, circuit.n_glomeruli)).astype(np.float32) * 16.0
    labels = np.arange(60) % 10
    return activations, labels


def test_readout_starts_undifferentiated(circuit: Circuit) -> None:
    """Nothing distinguishes any class until depression carves it out."""
    classifier = FlyClassifier.from_circuit(circuit)
    assert classifier.readout.shape == (circuit.n_kc, 10)
    assert np.all(classifier.readout == 1.0)


def test_encoding_is_sparse(circuit: Circuit, samples: tuple[np.ndarray, np.ndarray]) -> None:
    codes = FlyClassifier.from_circuit(circuit).encode(samples[0])
    assert codes.shape == (60, circuit.n_kc)
    assert np.allclose(codes.sum(axis=1), round(0.05 * circuit.n_kc))


def test_fit_only_ever_weakens(circuit: Circuit, samples: tuple[np.ndarray, np.ndarray]) -> None:
    classifier = FlyClassifier.from_circuit(circuit)
    classifier.fit(*samples)
    assert classifier.readout.max() <= 1.0
    assert classifier.readout.min() < 1.0


def test_predictions_are_valid_classes(circuit: Circuit, samples: tuple[np.ndarray, np.ndarray]) -> None:
    classifier = FlyClassifier.from_circuit(circuit).fit(*samples)
    predictions = classifier.predict(samples[0])
    assert predictions.shape == (60,)
    assert set(predictions.tolist()) <= set(range(10))


def test_prediction_ignores_how_hard_you_press(circuit: Circuit, samples: tuple[np.ndarray, np.ndarray]) -> None:
    """k-winners-take-all is scale invariant, so stroke intensity cannot change the answer."""
    activations, labels = samples
    classifier = FlyClassifier.from_circuit(circuit).fit(activations, labels)
    louder = (activations * 7.5).astype(np.float32)
    assert np.array_equal(classifier.predict(activations), classifier.predict(louder))


def test_the_circuit_learns_something(circuit: Circuit) -> None:
    """Ten well-separated patterns should be recalled better than chance after one pass."""
    rng = np.random.default_rng(1)
    patterns = rng.random((10, circuit.n_glomeruli)).astype(np.float32) * 16.0
    labels = np.arange(10)
    classifier = FlyClassifier.from_circuit(circuit).fit(patterns, labels)
    assert (classifier.predict(patterns) == labels).mean() > 0.5
