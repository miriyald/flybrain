# Status: Digit recognition with the fly connectome

_Last updated: 2026-09-12_

## Current state

**Done, with one caveat.** Training and evaluation are complete and verified. The drawing
canvas is complete and its prediction path is verified headlessly, but it has not been driven
with a real mouse — that needs a human.

## Completed

- Pixel-to-glomerulus encoding justified by measured variance, not assumption.
- `FlyClassifier` built entirely from the existing circuit and learning rule; `flylab/model.py`
  unmodified.
- 84.2% test accuracy on held-out digits from a single training pass.
- Confusion matrix, ASCII rendering of failures, class-overlap diagnostic, sparsity sweep.
- Canvas preprocessing reproducing the optdigits construction.
- tkinter square canvas with a live 8×8 preview of what the circuit actually receives.
- 16 new tests (63 total), lint clean at 10.00/10.

## Results

| Measure | Value |
|---|---|
| test accuracy | 84.2% (chance 10%) |
| training | 1,437 digits, one pass, ~0.2s |
| optimal sparsity | 5% — the same value used for odours |
| best classes | 0 and 1, both 100% |
| worst class | 8, at 46%; 12 of its errors go to class 1 |
| variance retained by the 64→55 encoding | 99.997% |

The headline is not the accuracy. It is that **the sparsity the fly uses for smells turned
out to be optimal for pixels**, and that **one training pass beat three** — the circuit is a
general-purpose one-shot classifier, not an olfactory specialisation.

## Blocked / open issues

- `07_draw_a_digit.py` is unverified interactively. If hand-drawn digits predict poorly, the
  live 8×8 preview will show whether the drawing is reaching the circuit in a recognisable
  form; that distinguishes a preprocessing bug from a model limitation.
- Class 8 at 46% is a real weakness, explained but not fixed.

## Next steps

1. Drive the canvas by hand and confirm predictions are sensible.
2. If 8s and 9s read poorly, compare the drawn preview against training samples of the same
   digit rendered by `digits.render`.
3. Optional: a reward/punishment variant using the derived MBON valence, so the digit task
   and the odour task share a readout.
