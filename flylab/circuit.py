"""Extract the mushroom body circuit from the hemibrain tables.

Two weight matrices are pulled straight out of the connectome and nothing about them is
invented: glomerulus -> Kenyon cell, and Kenyon cell -> output neuron.

Compartments come from the `instance` string rather than the ROI table, because hemibrain
ROIs stop at lobe level (gL, aL, PED) while the mushroom body's functional units are finer
(y1, y2, a'3). Both MBONs and dopaminergic neurons spell their compartment the same way -
`MBON11(y1pedc>a/B)_R` and `PPL101(y1pedc)_R` - so the two can be matched directly.

Greek letters are transliterated in this dataset: gamma -> y, alpha -> a, beta -> B.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from flylab import data

UNIGLOMERULAR_PN = re.compile(r"^(?P<glomerulus>[A-Z][A-Za-z0-9]*)_(?:ad|lv|il2|il|l2|l|i|v)PN\d*$")
_COMPARTMENT_TOKEN = re.compile(r"a'\d|b'\d|B'\d|a\d|b\d|B\d|y\d|pedc|calyx")
_INSTANCE_COMPARTMENT = re.compile(r"\((.*?)\)")
_ARROW = re.compile(r"[<>]")

CIRCUIT_PATH = data.DATA_DIR / "circuit.npz"


@dataclass(frozen=True)
class Circuit:
    """The mushroom body as two matrices plus the labels needed to interpret them."""

    glomeruli: npt.NDArray[np.str_]
    kc_ids: npt.NDArray[np.int64]
    mbon_labels: npt.NDArray[np.str_]
    pn_to_kc: npt.NDArray[np.float32]
    kc_to_mbon: npt.NDArray[np.float32]
    mbon_valence: npt.NDArray[np.int8]

    @property
    def n_glomeruli(self) -> int:
        return len(self.glomeruli)

    @property
    def n_kc(self) -> int:
        return len(self.kc_ids)

    @property
    def n_mbon(self) -> int:
        return len(self.mbon_labels)

    def save(self, path: Path = CIRCUIT_PATH) -> Path:
        np.savez_compressed(
            path,
            glomeruli=self.glomeruli,
            kc_ids=self.kc_ids,
            mbon_labels=self.mbon_labels,
            pn_to_kc=self.pn_to_kc,
            kc_to_mbon=self.kc_to_mbon,
            mbon_valence=self.mbon_valence,
        )
        return path

    @classmethod
    def load(cls, path: Path = CIRCUIT_PATH) -> Circuit:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run 03_build_circuit.py first")
        with np.load(path, allow_pickle=False) as bundle:
            return cls(**{key: bundle[key] for key in bundle.files})


def compartments(instance: str) -> frozenset[str]:
    """Atomic mushroom body compartments a neuron belongs to, read from its instance name.

    An arrow marks the neuron's other end and is excluded: `MBON11(y1pedc>a/B)` has its
    dendrites in y1/pedc and writes out to a/B, while `PAM07(y4<y1y2)` releases dopamine
    into y4 and reads from y1/y2. In both directions the neuron's own compartment is the
    part before the arrow.
    """
    match = _INSTANCE_COMPARTMENT.search(instance)
    if not match:
        return frozenset()
    own = _ARROW.split(match.group(1))[0]
    return frozenset(_COMPARTMENT_TOKEN.findall(own))


def _right_hemisphere(neurons: pd.DataFrame) -> pd.DataFrame:
    return neurons[neurons["instance"].fillna("").str.endswith("_R")]


def _label(cell_type: str, instance: str) -> str:
    return f"{cell_type}({'/'.join(sorted(compartments(instance))) or '?'})"


def _weight_matrix(
    edges: pd.DataFrame, row_index: dict[int, int], col_index: dict[int, int], shape: tuple[int, int]
) -> npt.NDArray[np.float32]:
    matrix = np.zeros(shape, dtype=np.float32)
    rows = edges["bodyId_pre"].map(row_index).to_numpy()
    cols = edges["bodyId_post"].map(col_index).to_numpy()
    np.add.at(matrix, (rows, cols), edges["weight"].to_numpy(dtype=np.float32))
    return matrix


def build() -> Circuit:
    """Slice the mushroom body out of the whole-brain tables."""
    neurons = data.load_neurons()
    neurons["type"] = neurons["type"].fillna("")
    neurons["instance"] = neurons["instance"].fillna("")
    connections = data.load_connections()

    kcs = neurons[neurons["type"].str.startswith("KC")]
    kc_ids = kcs["bodyId"].to_numpy(dtype=np.int64)
    kc_index = {int(body_id): i for i, body_id in enumerate(kc_ids)}

    pns = _right_hemisphere(neurons[neurons["type"].str.match(UNIGLOMERULAR_PN)]).copy()
    pns["glomerulus"] = pns["type"].str.extract(UNIGLOMERULAR_PN)["glomerulus"]
    glomeruli = np.array(sorted(pns["glomerulus"].unique()), dtype=np.str_)
    glomerulus_index = {name: i for i, name in enumerate(glomeruli)}
    pn_row = {int(body_id): glomerulus_index[str(name)] for body_id, name in zip(pns["bodyId"], pns["glomerulus"], strict=True)}

    mbons = _right_hemisphere(neurons[neurons["type"].str.startswith("MBON")]).copy()
    mbon_ids = mbons["bodyId"].to_numpy(dtype=np.int64)
    mbon_index = {int(body_id): i for i, body_id in enumerate(mbon_ids)}
    pairs = zip(mbons["type"], mbons["instance"], strict=True)
    mbon_labels = np.array([_label(str(cell_type), str(instance)) for cell_type, instance in pairs], dtype=np.str_)

    pn_edges = connections[connections["bodyId_pre"].isin(pn_row) & connections["bodyId_post"].isin(kc_index)]
    kc_edges = connections[connections["bodyId_pre"].isin(kc_index) & connections["bodyId_post"].isin(mbon_index)]

    pn_to_kc = _weight_matrix(pn_edges, pn_row, kc_index, (len(glomeruli), len(kc_ids)))
    kc_to_mbon = _weight_matrix(kc_edges, kc_index, mbon_index, (len(kc_ids), len(mbon_ids)))

    valence = _derive_valence(neurons, mbons)
    return Circuit(glomeruli, kc_ids, mbon_labels, pn_to_kc, kc_to_mbon, valence)


def dan_compartments(neurons: pd.DataFrame, prefix: str) -> frozenset[str]:
    """Compartments innervated by a dopaminergic cluster (PPL1 = punishment, PAM = reward)."""
    dans = _right_hemisphere(neurons[neurons["type"].str.startswith(prefix)])
    return frozenset().union(*(compartments(instance) for instance in dans["instance"])) if len(dans) else frozenset()


def _derive_valence(neurons: pd.DataFrame, mbons: pd.DataFrame) -> npt.NDArray[np.int8]:
    """+1 approach-promoting, -1 avoidance-promoting, 0 undetermined.

    Aso et al. 2014: compartments taught by punishment dopamine carry approach-promoting
    outputs, and compartments taught by reward dopamine carry avoidance-promoting outputs.
    Depressing an approach output is therefore how punishment produces avoidance.
    """
    punishment = dan_compartments(neurons, "PPL1")
    reward = dan_compartments(neurons, "PAM")
    signs = []
    for instance in mbons["instance"]:
        own = compartments(instance)
        punished, rewarded = bool(own & punishment), bool(own & reward)
        signs.append(1 if punished and not rewarded else -1 if rewarded and not punished else 0)
    return np.array(signs, dtype=np.int8)
