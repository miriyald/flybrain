"""Step 3 - slice the mushroom body out of the whole brain.

Two matrices come out of this, and neither is invented:

    glomerulus -> Kenyon cell    the odour input, ~54 x ~1900
    Kenyon cell -> MBON          the readout,     ~1900 x ~34   <- this one learns

We also need to know which outputs mean "approach" and which mean "avoid". Rather than
hardcoding that from memory, it is derived: dopaminergic neurons and output neurons name
their compartment the same way, so matching them recovers the published organisation
(Aso et al. 2014) - punishment-taught compartments carry approach-promoting outputs.

The derivation is then checked against three cases from the literature.

Run:  python 03_build_circuit.py
"""

from __future__ import annotations

import numpy as np

from flylab import circuit, data

LITERATURE_CHECKS = (
    ("MBON11", 1, "y1pedc>a/B, taught by PPL1-y1 punishment; approach-promoting (Hige et al. 2015)"),
    ("MBON12", 1, "y2a'1, taught by PPL1-y2a'1 punishment; approach-promoting (Aso et al. 2014)"),
    ("MBON01", -1, "y5B'2a, taught by PAM reward; avoidance-promoting (Owald et al. 2015)"),
)


def main() -> None:
    print(__doc__)

    neurons = data.load_neurons()
    neurons["type"] = neurons["type"].fillna("")
    neurons["instance"] = neurons["instance"].fillna("")
    punishment = circuit.dan_compartments(neurons, "PPL1")
    reward = circuit.dan_compartments(neurons, "PAM")
    print(f"punishment compartments (PPL1): {sorted(punishment)}")
    print(f"reward compartments     (PAM) : {sorted(reward)}")
    print(f"taught by both                : {sorted(punishment & reward)}")

    built = circuit.build()

    print("\n--- matrices ---")
    print(f"glomerulus -> KC : {built.pn_to_kc.shape}   {int((built.pn_to_kc > 0).sum()):,} nonzero")
    print(f"KC -> MBON       : {built.kc_to_mbon.shape}   {int((built.kc_to_mbon > 0).sum()):,} nonzero")

    inputs_per_kc = (built.pn_to_kc > 0).sum(axis=0)
    connected = inputs_per_kc[inputs_per_kc > 0]
    print(f"\nglomeruli per Kenyon cell: mean {connected.mean():.1f}, median {np.median(connected):.0f}")
    print("published expectation is ~6-7 (Caron et al. 2013), and the sampling is near-random")

    print("\n--- derived valence ---")
    for sign, label in ((1, "APPROACH"), (-1, "AVOID"), (0, "undetermined")):
        members = built.mbon_labels[built.mbon_valence == sign]
        print(f"  {label:<13} {len(members):>3}  {', '.join(sorted(members)[:6])}{' ...' if len(members) > 6 else ''}")

    print("\n--- cross-check against the literature ---")
    failures = 0
    for mbon_type, expected, note in LITERATURE_CHECKS:
        matches = [i for i, label in enumerate(built.mbon_labels) if label.startswith(f"{mbon_type}(")]
        if not matches:
            print(f"  {mbon_type}: not present in this hemisphere - skipped")
            continue
        got = int(built.mbon_valence[matches[0]])
        ok = got == expected
        failures += not ok
        print(f"  [{'OK ' if ok else 'MISS'}] {built.mbon_labels[matches[0]]:<26} derived {got:+d}, expected {expected:+d}")
        print(f"         {note}")
    print(f"\n{'derivation reproduces the literature' if not failures else f'{failures} disagreement(s) - reported, not overridden'}")

    path = built.save()
    print(f"\nsaved: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print("\nNext: python 04_odour_code.py")


if __name__ == "__main__":
    main()
