"""A classifier made of the fly's own circuit.

The expansion layer is the measured connectome, the sparsening is the same k-winners-take-all
that stands in for APL, and the learning rule is the same dopamine-gated depression used for
odours. Nothing here adds new mathematics - it only points the existing circuit at pixels.

The readout starts as all ones, so at the outset every class looks identical. All
discrimination is carved out by weakening synapses onto the wrong answers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from flylab import model
from flylab.circuit import Circuit

CLASSES = 10
RATE = 0.001
EPOCHS = 1


@dataclass
class FlyClassifier:
    """Glomeruli -> Kenyon cells -> one output per digit."""

    pn_to_kc: npt.NDArray[np.float32]
    readout: npt.NDArray[np.float32]
    sparsity: float = model.SPARSITY
    rate: float = RATE

    @classmethod
    def from_circuit(cls, circuit: Circuit, classes: int = CLASSES, sparsity: float = model.SPARSITY, rate: float = RATE) -> FlyClassifier:
        return cls(
            pn_to_kc=model.normalise_gain(circuit.pn_to_kc),
            readout=np.ones((circuit.n_kc, classes), dtype=np.float32),
            sparsity=sparsity,
            rate=rate,
        )

    @property
    def n_classes(self) -> int:
        return int(self.readout.shape[1])

    def encode(self, activations: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        """Sparse Kenyon cell codes, one row per sample."""
        return np.array([model.kenyon_code(row, self.pn_to_kc, self.sparsity) for row in np.atleast_2d(activations)])

    def fit(self, activations: npt.NDArray[np.float32], labels: npt.NDArray[np.int64], epochs: int = EPOCHS) -> FlyClassifier:
        """Show each labelled sample and depress its active cells onto every wrong class.

        One pass is the default because extra passes measurably hurt: depression only ever
        removes weight, so repeated exposure drives the readout towards zero and erodes the
        differences it built. The circuit learns in one shot or not at all.
        """
        codes = self.encode(activations)
        for _ in range(epochs):
            for code, label in zip(codes, labels, strict=True):
                wrong = np.ones(self.n_classes, dtype=np.bool_)
                wrong[label] = False
                self.readout = model.depress(self.readout, code, wrong, self.rate)
        return self

    def scores(self, activations: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return self.encode(activations).astype(np.float32) @ self.readout

    def predict(self, activations: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]:
        chosen: npt.NDArray[np.int64] = np.argmax(self.scores(activations), axis=1)
        return chosen
