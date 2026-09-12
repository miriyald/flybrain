"""Step 5 - teach the fly to fear a smell.

The rule is one line of arithmetic and it only ever weakens synapses:

    kc_to_mbon[active_kenyon_cells, punished_compartment] *= (1 - rate)

Three things must coincide - the Kenyon cell fired, the output neuron sits in a compartment
receiving dopamine, and dopamine is present. No error signal travels backwards and nothing
is strengthened.

Punishment depresses the APPROACH-promoting outputs, because those are the ones sitting in
the compartments that punishment dopamine teaches. The fly comes to avoid an odour by
unlearning its attraction to it.

Then the part that matters: odour B is left alone, because it was never encoded on the same
Kenyon cells. Sparse coding buys one-shot learning without catastrophic interference.

Run:  python 05_learn.py
"""

from __future__ import annotations

import numpy as np

from flylab import model
from flylab.circuit import Circuit

SEED = 7
TRIALS = 6
RATE = 0.30


def gauge(value: float, width: int = 33) -> str:
    """A -1..+1 valence meter with a marked centre."""
    slot = round((value + 1.0) / 2.0 * (width - 1))
    cells = ["-"] * width
    cells[width // 2] = "|"
    cells[slot] = "#"
    return f"avoid [{''.join(cells)}] approach"


def main() -> None:
    print(__doc__)
    built = Circuit.load()
    rng = np.random.default_rng(SEED)
    weights = model.normalise_gain(built.pn_to_kc)

    punished = built.mbon_valence == 1
    print(f"punishment dopamine (PPL1) teaches {int(punished.sum())} approach-promoting outputs:")
    print(f"  {', '.join(sorted(set(built.mbon_labels[punished]))[:8])} ...")

    odour_a = model.odour(built.n_glomeruli, rng)
    odour_b = model.odour(built.n_glomeruli, rng)
    code_a = model.kenyon_code(odour_a, weights)
    code_b = model.kenyon_code(odour_b, weights)
    shared = int((code_a & code_b).sum())
    print(f"\nodour A and odour B share {shared} of {int(code_a.sum())} Kenyon cells "
          f"({shared / int(code_a.sum()):.1%}) - remember this number")

    def read(kc_to_mbon: np.ndarray, code: np.ndarray) -> float:
        return model.valence(model.mbon_response(code, kc_to_mbon), built.mbon_valence)

    kc_to_mbon = built.kc_to_mbon
    before_a, before_b = read(kc_to_mbon, code_a), read(kc_to_mbon, code_b)

    print("\n--- before training ---")
    print(f"  odour A  {before_a:+.3f}  {gauge(before_a)}")
    print(f"  odour B  {before_b:+.3f}  {gauge(before_b)}")

    print(f"\n--- pairing odour A with punishment, {TRIALS} trials ---")
    print(f"  {'trial':>5}  {'odour A':>8}  {'odour B':>8}")
    print(f"  {0:>5}  {before_a:>+8.3f}  {before_b:>+8.3f}   (naive)")
    for trial in range(1, TRIALS + 1):
        kc_to_mbon = model.depress(kc_to_mbon, code_a, punished, RATE)
        print(f"  {trial:>5}  {read(kc_to_mbon, code_a):>+8.3f}  {read(kc_to_mbon, code_b):>+8.3f}")

    after_a, after_b = read(kc_to_mbon, code_a), read(kc_to_mbon, code_b)
    print("\n--- after training ---")
    print(f"  odour A  {after_a:+.3f}  {gauge(after_a)}   shifted {after_a - before_a:+.3f}  <- taught")
    print(f"  odour B  {after_b:+.3f}  {gauge(after_b)}   shifted {after_b - before_b:+.3f}  <- untouched")

    drift = abs(after_b - before_b) / max(abs(after_a - before_a), 1e-9)
    print(f"\n  collateral damage: odour B moved {drift:.1%} as much as odour A")

    print("\n--- generalisation: what else did the fly learn to fear? ---")
    print(f"  {'similarity to A':>16}  {'KC overlap':>11}  {'valence shift':>14}")
    for similarity in (1.0, 0.9, 0.75, 0.5, 0.25, 0.0):
        shifts, overlaps = [], []
        for _ in range(20):
            probe = model.blend(odour_a, rng, similarity) if similarity < 1.0 else odour_a
            code = model.kenyon_code(probe, weights)
            overlaps.append(float((code & code_a).sum()) / max(1, int(code.sum())))
            shifts.append(read(kc_to_mbon, code) - read(built.kc_to_mbon, code))
        print(f"  {similarity:>15.0%}  {np.mean(overlaps):>10.1%}  {np.mean(shifts):>+14.3f}")

    print("\nThe learned aversion falls away exactly as fast as the Kenyon cell codes stop")
    print("overlapping. That gradient is the whole point: the fly generalises to smells that")
    print("resemble the punished one, and leaves everything else intact. A dense code would")
    print("have overwritten all of it at once.")


if __name__ == "__main__":
    main()
