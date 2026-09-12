# Status: Digit recognition with the fly connectome

_Last updated: 2026-09-12_

## Current state

**Done, with one caveat.** Training and evaluation are complete and verified. The drawing
canvas is complete and its prediction path is verified headlessly, but it has not been driven
with a real mouse — that needs a human.

## Completed

- Pixel-to-glomerulus encoding justified by measured variance, not assumption.
- `FlyClassifier` built entirely from the existing circuit and learning rule;
  `flylab/model.py` unmodified, so scripts 01–05 still run.
- Both dopamine systems: punishment teaches the approach readout, reward teaches the avoid
  readout, and the verdict is their difference.
- Three-way split with tuning confined to validation; test scored once.
- Learned state persisted separately from the wiring, and recalled by step 7.
- Confusion matrix, ASCII rendering of failures, class-overlap diagnostic, sparsity sweep.
- Canvas preprocessing reproducing the optdigits construction.
- 19 new tests (66 total), lint clean at 10.00/10.

## Results

| Measure | Value |
|---|---|
| test accuracy | **92.8%** (chance 10%) |
| punishment only, best rate | 84.1% validation |
| punishment + reward, best rate | 91.9% validation |
| optimal sparsity | 9–10%, gently above the 5% used for odours |
| best classes | 0 at 100%, 5 at 95% |
| most-shared code | 1 and 8 at 88%; 3 and 9 at 82% |
| variance retained by the 64→55 encoding | 99.9985% |
| training | one pass, under a second |

## Corrections made

- **Test-set leakage.** The first version tuned sparsity and rate against the test split and
  reported 84.2%. Retuned on a proper validation split; the current 92.8% is scored once.
- **Unfair ablation.** The punishment-only comparison initially used the rate tuned for the
  bidirectional model, inflating the apparent gain to 22 points. Giving each variant its own
  best rate shows the honest gain: about 8 points.
- **Float round-trip.** Saving `sparsity` as float32 meant 0.10 reloaded as 0.10000000149.
  Stored as float64; covered by a round-trip test.

## Blocked / open issues

- `07_draw_a_digit.py` is unverified interactively. If hand-drawn digits predict poorly, the
  live 8×8 preview shows whether the drawing reaches the circuit in a recognisable form,
  which distinguishes a preprocessing bug from a model limit.
- 1 and 8 remain the hardest pair, explained by code overlap but not fixed.

## Next steps

1. Drive the canvas by hand and confirm predictions are sensible.
2. Optional: let the canvas teach the circuit — correct a wrong prediction and depress on the
   spot, which is exactly what the odour training loop already does.
3. Optional: a decay term, which might make multi-epoch training useful rather than harmful.
