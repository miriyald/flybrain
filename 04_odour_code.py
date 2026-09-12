"""Step 4 - watch the connectome turn odours into near-orthogonal tags.

This is the computation, and it happens before any learning.

Odours arrive as broad, overlapping activity across ~55 glomeruli - in the fly, most
glomeruli respond to most odours by some amount. The wiring expands that into 1,927 Kenyon
cells, inhibition keeps only the strongest 5%, and what survives barely overlaps even for
odours that looked similar going in.

Dasgupta, Stevens & Navlakha (Science, 2017) identified this as locality-sensitive hashing:
random projection into high dimension followed by a winner-take-all yields a compact tag
where similar inputs stay distinguishable.

This step also shows the one place the connectome is not enough on its own.

Run:  python 04_odour_code.py
"""

from __future__ import annotations

import numpy as np

from flylab import model
from flylab.circuit import Circuit

SEED = 7
TRIALS = 60


def overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Fraction of active Kenyon cells shared, relative to the smaller set."""
    return int((left & right).sum()) / max(1, min(int(left.sum()), int(right.sum())))


def meter(fraction: float, width: int = 40) -> str:
    return "#" * round(fraction * width) + "." * (width - round(fraction * width))


def mean_overlap(
    weights: np.ndarray, n_glomeruli: int, rng: np.random.Generator, similarity: float, trials: int = TRIALS
) -> tuple[float, float]:
    correlations, overlaps = [], []
    for _ in range(trials):
        base = model.odour(n_glomeruli, rng)
        other = model.blend(base, rng, similarity)
        correlations.append(np.corrcoef(base, other)[0, 1])
        overlaps.append(overlap(model.kenyon_code(base, weights), model.kenyon_code(other, weights)))
    return float(np.mean(correlations)), float(np.mean(overlaps))


def main() -> None:
    print(__doc__)
    built = Circuit.load()
    rng = np.random.default_rng(SEED)
    raw = built.pn_to_kc
    weights = model.normalise_gain(raw)
    chance = model.SPARSITY

    totals = raw.sum(axis=0)
    print("--- the wiring alone is not enough ---")
    print(f"  total input synapses per Kenyon cell: min {totals.min():.0f}, median {np.median(totals):.0f}, max {totals.max():.0f}")
    print("  Left uncorrected, the strongest cells win every competition regardless of odour.")
    print("  The connectome records synapse counts but no firing thresholds, so the excitability")
    print("  scaling a real fly has must be put back by hand. The cost of skipping it:\n")
    print(f"  {'weights':<24} {'overlap between unrelated odours':>34}")
    for label, matrix in (("raw connectome", raw), ("gain-normalised", weights)):
        _, value = mean_overlap(matrix, built.n_glomeruli, rng, similarity=0.0, trials=30)
        print(f"  {label:<24} {value:>33.1%}   {meter(value)}")
    print(f"  {'chance level':<24} {chance:>33.1%}")

    print("\n--- two unrelated odours, gain-normalised ---")
    odour_a = model.odour(built.n_glomeruli, rng)
    odour_b = model.odour(built.n_glomeruli, rng)
    code_a = model.kenyon_code(odour_a, weights)
    code_b = model.kenyon_code(odour_b, weights)
    print(f"  input correlation across {built.n_glomeruli} glomeruli: {np.corrcoef(odour_a, odour_b)[0, 1]:>6.1%}")
    for name, code in (("A", code_a), ("B", code_b)):
        print(f"  odour {name}: {int(code.sum()):>4} of {built.n_kc} Kenyon cells active   sparseness {code.mean():>5.1%}")
    print(f"  shared Kenyon cells: {int((code_a & code_b).sum())}   overlap {overlap(code_a, code_b):.1%}")
    print("\n  Published expectation is that 5-10% of Kenyon cells respond to an odour")
    print("  (Turner et al. 2008). The sparsity target is set once; the wiring does the rest.")

    print(f"\n--- how similar must two odours be before their tags collide? ({TRIALS} pairs each) ---")
    print(f"  {'input corr':>12}   {'code overlap':>12}   (chance = {chance:.1%})")
    for similarity in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99):
        correlation, value = mean_overlap(weights, built.n_glomeruli, rng, similarity)
        print(f"  {correlation:>11.1%}   {value:>11.1%}   {meter(value)}")

    print("\nOdours have to be genuinely similar before they share many Kenyon cells.")
    print("Unrelated ones land on near-disjoint sets - which is what lets the next step teach")
    print("the fly about one odour without touching what it knows about any other.")
    print("\nNext: python 05_learn.py")


if __name__ == "__main__":
    main()
