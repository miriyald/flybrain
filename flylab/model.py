"""The whole model: four functions.

An odour drives glomeruli, the connectome's wiring expands that into Kenyon cell drive,
inhibition keeps only the strongest few percent, and the output neurons read the survivors
out into an approach/avoid balance. Learning depresses the synapses that were active when
dopamine arrived.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

SPARSITY = 0.05
DRIVE_GRID = 1e-4


def odour(n_glomeruli: int, rng: np.random.Generator) -> npt.NDArray[np.float32]:
    """A synthetic odour: a broad, graded activation across all glomeruli.

    Projection neuron responses in the fly are broad rather than selective - an odour moves
    most glomeruli by some amount, with a long tail of strong responders - so the input is
    dense and lognormal, not a sparse set of switches. That density is the point: the
    sparsening downstream is what turns it into a compact tag.

    The wiring is real; the smells are invented. Measured odour-to-glomerulus maps exist as
    a separate dataset, and the computational result here does not depend on them.
    """
    return rng.lognormal(mean=0.0, sigma=0.6, size=n_glomeruli).astype(np.float32)


def blend(base: npt.NDArray[np.float32], rng: np.random.Generator, similarity: float) -> npt.NDArray[np.float32]:
    """An odour a given fraction of the way between `base` and an unrelated smell."""
    fresh = odour(base.size, rng)
    return (similarity * base + (1.0 - similarity) * fresh).astype(np.float32)


def normalise_gain(pn_to_kc: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Scale each Kenyon cell's inputs so that no cell wins on raw gain alone.

    THIS IS A MODELLING ASSUMPTION, NOT CONNECTOME DATA - and it is the clearest example
    of what the connectome does not contain. Total input weight per Kenyon cell spans 0 to
    254 synapses, so without normalisation the same high-gain cells win every competition
    and every odour produces almost the same code (measured: 32% overlap between unrelated
    odours, against a 5% chance level).

    Flies do not have that problem, because a neuron's firing threshold scales with the
    drive it normally receives - homeostatic excitability regulation, plus the divisive
    effect of APL feedback. The wiring diagram records synapse counts but no thresholds or
    conductances, so that scaling has to be reintroduced by hand. Normalising each cell's
    input to sum to one brings overlap to 8%, close to chance.
    """
    total = pn_to_kc.sum(axis=0, keepdims=True)
    return (pn_to_kc / np.where(total > 0.0, total, 1.0)).astype(np.float32)


def kenyon_code(
    activation: npt.NDArray[np.float32], pn_to_kc: npt.NDArray[np.float32], sparsity: float = SPARSITY
) -> npt.NDArray[np.bool_]:
    """Expand an odour into a sparse Kenyon cell code.

    Expects gain-normalised weights from `normalise_gain`.

    The drive is accumulated at double precision. In float32 the rounding differs between
    numpy's BLAS and any other summation order by a few parts per million, which is enough
    to reorder cells sitting close together at the winner boundary and leave the browser
    port picking a slightly different set.

    APL is a single inhibitory neuron contacting every Kenyon cell - all 1,927 of them, as
    the connectome confirms - so the more the population fires, the harder it is held down.
    The steady state of that negative feedback is a k-winners-take-all.
    """
    drive = activation.astype(np.float64) @ pn_to_kc.astype(np.float64)
    return kwta(drive, k=max(1, round(sparsity * drive.size)))


def kwta(drive: npt.NDArray[np.floating[Any]], k: int) -> npt.NDArray[np.bool_]:
    """Keep the k strongest units, silence the rest.

    Ranking happens on a quantised drive with ties broken by index, so the winner set is
    reproducible across implementations. See `quantise` for why that is not fussiness.
    """
    code = np.zeros(drive.size, dtype=np.bool_)
    code[np.argsort(-quantise(drive), kind="stable")[:k]] = True
    return code


def quantise(drive: npt.NDArray[np.floating[Any]], grid: float = DRIVE_GRID) -> npt.NDArray[np.float64]:
    """Snap the drive to a coarse grid so that genuine ties compare equal everywhere.

    Gain normalisation makes each cell's input weights sum to one, so a cell fed only
    saturated pixels scores exactly 16 in real arithmetic - and a bold drawing can leave
    three hundred cells sitting there at once, with the winner boundary entirely inside that
    group. In floating point those sums land a few parts in 10^8 either side of 16, which is
    precisely the rounding midpoint of the nearest float32; rounding to float32 therefore
    balances them on a knife edge instead of merging them. A grid far coarser than the error
    but far finer than any real difference in drive settles it identically in any language.

    Rounding is written as floor(x / grid + 0.5) rather than round(), whose half-way rule
    differs between numpy and JavaScript.
    """
    snapped: npt.NDArray[np.float64] = np.floor(drive / grid + 0.5)
    return snapped


def mbon_response(code: npt.NDArray[np.bool_], kc_to_mbon: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Output neuron activity: each MBON sums the synaptic weight of the active Kenyon cells."""
    return code.astype(np.float32) @ kc_to_mbon


def valence(response: npt.NDArray[np.float32], signs: npt.NDArray[np.int8]) -> float:
    """Approach/avoid balance in [-1, +1]; 0 means the two drives cancel."""
    total = np.abs(response[signs != 0]).sum()
    return float((response * signs).sum() / total) if total else 0.0


def depress(
    kc_to_mbon: npt.NDArray[np.float32], code: npt.NDArray[np.bool_], targets: npt.NDArray[np.bool_], rate: float
) -> npt.NDArray[np.float32]:
    """The learning rule, in one line of arithmetic.

    Three factors must coincide: the Kenyon cell was active, the output neuron sits in the
    compartment receiving dopamine, and dopamine is present. Where they do, the synapse
    weakens. Nothing is strengthened and no error signal travels backwards.
    """
    updated = kc_to_mbon.copy()
    updated[np.ix_(code, targets)] *= 1.0 - rate
    return updated
