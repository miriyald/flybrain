"""A classifier made of the fly's own circuit.

The expansion layer is the measured connectome, the sparsening is the same k-winners-take-all
that stands in for APL, and learning is the same dopamine-gated depression used for odours.
Nothing here adds new mathematics - it only points the existing circuit at pixels.

The fly has two dopamine systems, not one. PPL1 neurons signal punishment and teach the
compartments whose outputs promote approach; PAM neurons signal reward and teach the
compartments whose outputs promote avoidance. Both act by depression - the difference is
which population they act on.

So there are two readouts. A labelled sample punishes every wrong answer and rewards the
right one, and the verdict is the difference between them. Using only punishment, as an
earlier version did, throws away half the circuit and about seven points of accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from flylab import data, model
from flylab.circuit import Circuit

CLASSES = 10
SPARSITY = 0.10
RATE = 0.25
EPOCHS = 1
MEMORY_PATH = data.DATA_DIR / "digit_memory.npz"


@dataclass
class FlyClassifier:
    """Glomeruli -> Kenyon cells -> one approach and one avoid output per digit."""

    pn_to_kc: npt.NDArray[np.float32]
    approach: npt.NDArray[np.float32]
    avoid: npt.NDArray[np.float32]
    pixels: npt.NDArray[np.int64]
    sparsity: float = SPARSITY
    rate: float = RATE

    @classmethod
    def from_circuit(
        cls,
        circuit: Circuit,
        pixels: npt.NDArray[np.int64],
        *,
        classes: int = CLASSES,
        sparsity: float = SPARSITY,
        rate: float = RATE,
    ) -> FlyClassifier:
        """An untaught circuit: both readouts flat, so no class is distinguishable yet."""
        shape = (circuit.n_kc, classes)
        return cls(
            pn_to_kc=model.normalise_gain(circuit.pn_to_kc),
            approach=np.ones(shape, dtype=np.float32),
            avoid=np.ones(shape, dtype=np.float32),
            pixels=pixels,
            sparsity=sparsity,
            rate=rate,
        )

    @property
    def n_classes(self) -> int:
        return int(self.approach.shape[1])

    def encode(self, activations: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        """Sparse Kenyon cell codes, one row per sample."""
        return np.array([model.kenyon_code(row, self.pn_to_kc, self.sparsity) for row in np.atleast_2d(activations)])

    def fit(self, activations: npt.NDArray[np.float32], labels: npt.NDArray[np.int64], epochs: int = EPOCHS) -> FlyClassifier:
        """Punish every wrong answer, reward the right one.

        One pass is the default because extra passes do not help: depression only ever
        removes weight, so repeated exposure erodes the differences it built.
        """
        codes = self.encode(activations)
        for _ in range(epochs):
            for code, label in zip(codes, labels, strict=True):
                wrong = np.ones(self.n_classes, dtype=np.bool_)
                wrong[label] = False
                self.approach = model.depress(self.approach, code, wrong, self.rate)

                right = np.zeros(self.n_classes, dtype=np.bool_)
                right[label] = True
                self.avoid = model.depress(self.avoid, code, right, self.rate)
        return self

    def scores(self, activations: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        code = self.encode(activations).astype(np.float32)
        return code @ self.approach - code @ self.avoid

    def predict(self, activations: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]:
        chosen: npt.NDArray[np.int64] = np.argmax(self.scores(activations), axis=1)
        return chosen

    def save(self, path: Path = MEMORY_PATH) -> Path:
        """Persist what the circuit learned. The wiring lives in circuit.npz; this is memory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            approach=self.approach,
            avoid=self.avoid,
            pixels=self.pixels,
            sparsity=np.float64(self.sparsity),
            rate=np.float64(self.rate),
        )
        return path

    @classmethod
    def load(cls, circuit: Circuit, path: Path = MEMORY_PATH) -> FlyClassifier:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run 06_recognise_digits.py first")
        with np.load(path, allow_pickle=False) as memory:
            return cls(
                pn_to_kc=model.normalise_gain(circuit.pn_to_kc),
                approach=memory["approach"],
                avoid=memory["avoid"],
                pixels=memory["pixels"],
                sparsity=float(memory["sparsity"]),
                rate=float(memory["rate"]),
            )
