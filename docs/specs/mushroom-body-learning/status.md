# Status: Associative learning on real mushroom body wiring

_Last updated: 2026-09-12_

## Current state

**Done.** All five steps run end to end. 47 tests pass, `lint.cmd` reports 10.00/10 with
mypy clean.

## Completed

- Anonymous 45.9 MB hemibrain v1.2 download, size-verified, safely extracted.
- Cell-type discovery step that de-risked the two naming assumptions that would otherwise
  have been wrong.
- Circuit extraction: glomerulus → Kenyon cell (55 × 1,927) and Kenyon cell → MBON
  (1,927 × 44), both from measured synapse counts.
- Valence derived from dopamine compartment membership, cross-checked against three
  published cases — all three agree.
- Rate model: gain normalisation, k-winners-take-all, MBON readout, dopamine-gated depression.
- Punishment learning with a generalisation gradient that tracks Kenyon cell overlap.
- Tests including a regression test for the `<` arrow bug and specificity across five seeds.
- README, design, plan, pinned dependencies, lint script.

## Results held against published work

| Claim | Measured | Published |
|---|---|---|
| Kenyon cells | 1,927 | ~2,000 |
| glomeruli per Kenyon cell | 6.0 | ~6–7 (Caron 2013) |
| odour sparseness | 5.0% | 5–10% (Turner 2008) |
| MBONs per hemisphere | 44 neurons / 34 types | 34 types (Aso 2014) |
| APL → Kenyon cells | 1,927 of 1,927 | all |
| valence cross-checks | 3 of 3 agree | Hige 2015, Aso 2014, Owald 2015 |

## Blocked / open issues

None blocking. Known limitations, all documented in the README:

- Odours are synthetic; real wiring, invented smells.
- Kenyon cell gain normalisation is a modelling assumption, not connectome data. Step 4
  demonstrates the failure mode it corrects rather than hiding it.
- Firing-rate model, right hemisphere only, no spiking or dopamine timing.

## Next steps

Optional extensions, none required for the current goal:

1. Real odour-to-glomerulus maps (DoOR) to make the input layer measured too.
2. Reward learning via PAM compartments — valence is already derived, only a training loop
   is missing.
3. Explicit APL feedback inhibition in place of k-winners-take-all.
4. Memory decay and extinction, to show forgetting as well as learning.
