"""The fly circuit used as a digit classifier.

Needs the built circuit, so it skips cleanly on a fresh checkout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flylab.circuit import CIRCUIT_PATH, Circuit
from flylab.classifier import FlyClassifier

Samples = tuple[np.ndarray, np.ndarray]

pytestmark = pytest.mark.skipif(not CIRCUIT_PATH.exists(), reason="run 01_get_the_data.py and 03_build_circuit.py first")


@pytest.fixture(scope="module")
def circuit() -> Circuit:
    return Circuit.load()


@pytest.fixture
def pixels(circuit: Circuit) -> np.ndarray:
    return np.arange(circuit.n_glomeruli, dtype=np.int64)


@pytest.fixture
def samples(circuit: Circuit) -> Samples:
    rng = np.random.default_rng(0)
    activations = (rng.random((60, circuit.n_glomeruli)) * 16.0).astype(np.float32)
    return activations, np.arange(60) % 10


def test_readouts_start_undifferentiated(circuit: Circuit, pixels: np.ndarray) -> None:
    """Nothing distinguishes any class until dopamine carves it out."""
    classifier = FlyClassifier.from_circuit(circuit, pixels)
    assert classifier.approach.shape == (circuit.n_kc, 10)
    assert np.all(classifier.approach == 1.0)
    assert np.all(classifier.avoid == 1.0)


def test_encoding_is_sparse(circuit: Circuit, pixels: np.ndarray, samples: Samples) -> None:
    classifier = FlyClassifier.from_circuit(circuit, pixels)
    codes = classifier.encode(samples[0])
    assert codes.shape == (60, circuit.n_kc)
    assert np.allclose(codes.sum(axis=1), round(classifier.sparsity * circuit.n_kc))


def test_both_readouts_only_ever_weaken(circuit: Circuit, pixels: np.ndarray, samples: Samples) -> None:
    """Dopamine depresses. Neither system ever strengthens a synapse."""
    classifier = FlyClassifier.from_circuit(circuit, pixels).fit(*samples)
    assert classifier.approach.max() <= 1.0
    assert classifier.avoid.max() <= 1.0
    assert classifier.approach.min() < 1.0
    assert classifier.avoid.min() < 1.0


def test_predictions_are_valid_classes(circuit: Circuit, pixels: np.ndarray, samples: Samples) -> None:
    classifier = FlyClassifier.from_circuit(circuit, pixels).fit(*samples)
    predictions = classifier.predict(samples[0])
    assert predictions.shape == (60,)
    assert set(predictions.tolist()) <= set(range(10))


def test_prediction_ignores_how_hard_you_press(circuit: Circuit, pixels: np.ndarray, samples: Samples) -> None:
    """k-winners-take-all is scale invariant, so stroke intensity cannot change the answer."""
    activations, labels = samples
    classifier = FlyClassifier.from_circuit(circuit, pixels).fit(activations, labels)
    louder = (activations * 7.5).astype(np.float32)
    assert np.array_equal(classifier.predict(activations), classifier.predict(louder))


def test_reward_adds_signal_that_punishment_alone_lacks(circuit: Circuit, pixels: np.ndarray, samples: Samples) -> None:
    """Silencing the reward readout must change the verdict, or it is doing nothing."""
    classifier = FlyClassifier.from_circuit(circuit, pixels).fit(*samples)
    both = classifier.scores(samples[0])
    classifier.avoid = np.ones_like(classifier.avoid)
    assert not np.allclose(both, classifier.scores(samples[0]))


def test_the_circuit_learns_something(circuit: Circuit, pixels: np.ndarray) -> None:
    """Ten well-separated patterns should be recalled after one pass."""
    rng = np.random.default_rng(1)
    patterns = (rng.random((10, circuit.n_glomeruli)) * 16.0).astype(np.float32)
    labels = np.arange(10)
    classifier = FlyClassifier.from_circuit(circuit, pixels).fit(patterns, labels)
    assert (classifier.predict(patterns) == labels).mean() > 0.5


def test_memory_survives_a_round_trip(circuit: Circuit, pixels: np.ndarray, samples: Samples, tmp_path: Path) -> None:
    """What the circuit learned is separable from its wiring, and reloads unchanged."""
    trained = FlyClassifier.from_circuit(circuit, pixels).fit(*samples)
    path = trained.save(tmp_path / "memory.npz")
    recalled = FlyClassifier.load(circuit, path)
    assert np.array_equal(trained.approach, recalled.approach)
    assert np.array_equal(trained.avoid, recalled.avoid)
    assert np.array_equal(trained.pixels, recalled.pixels)
    assert recalled.sparsity == trained.sparsity
    assert np.array_equal(trained.predict(samples[0]), recalled.predict(samples[0]))


def test_loading_absent_memory_is_explicit(circuit: Circuit, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="06_recognise_digits"):
        FlyClassifier.load(circuit, tmp_path / "nothing.npz")
