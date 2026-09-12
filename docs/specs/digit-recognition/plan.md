# Plan: Digit recognition with the fly connectome

## Scope

In scope: pixel-to-glomerulus encoding, a classifier built from the existing circuit, a
training and evaluation script, and a square drawing canvas for live prediction.

Out of scope: MNIST, competitive accuracy, any sklearn-trained model, modifying
`flylab/model.py`, gradient-based learning.

## Steps

- [x] 1. Install and pin scikit-learn; bump the package to 0.2.0.
- [x] 2. `flylab/digits.py` — measure per-pixel variance, select the 55 live pixels, encode
      to glomeruli. Verify the dead-pixel assumption before relying on it.
- [x] 3. `flylab/classifier.py` — `FlyClassifier` over the existing `normalise_gain`,
      `kwta` and `depress`. No new mathematics.
- [x] 4. `06_recognise_digits.py` — accuracy, confusion matrix, misclassified digits as
      ASCII, class-overlap diagnostic, sparsity sweep.
- [x] 5. Canvas preprocessing — crop, centre, 32×32 binary, 4×4 block-sum to 0–16.
- [x] 6. `07_draw_a_digit.py` — tkinter square canvas with a live 8×8 preview.
- [x] 7. Tests, docs, lint, and confirmation that the odour scripts still run.

## Risks & mitigations

| Risk | Outcome |
|---|---|
| Hand-drawn digits mispredict despite good test accuracy | Mitigated by reproducing the optdigits pipeline, and by showing the 8×8 preview live in the GUI so a mismatch is visible. Verified headlessly on synthetic strokes; **not verified with a real mouse** |
| Fewer than 55 pixels genuinely dead | **Not hit.** Dropping 9 retains 99.997% of variance; 4 pixels have exactly zero variance |
| Depression-only weights underflow | **Partly hit.** At rate 0.02 over 3 epochs the readout minimum reached 1.7e-26. Resolved by the tuning below rather than by clamping |
| Accuracy poor (<70%) | Not hit: 84.2% |
| Odour sparsity wrong for pixels | **Not hit, and the interesting result.** 5% — the value used for odours — was optimal across the sweep |
| More training helps | **Inverted.** Three epochs scored worse than one (0.786 vs 0.831); ten worse again. One-shot is not just sufficient, it is better |

## Verification / acceptance criteria

All met except where noted:

- 84.2% test accuracy on 360 held-out digits, against a 10% chance baseline.
- Sparsity sweep: 5% optimal, matching the odour model's value.
- Class-overlap diagnostic explains the errors — 1 and 8 share 83% of their code and account
  for 12 confusions; 8 is the weakest class at 46%.
- Canvas pipeline verified headlessly: synthetic vertical line → 1, seven-shape → 7,
  ellipse → 0; blank canvas → zeros; output within 0–16; position and scale invariant.
- **`07_draw_a_digit.py` has not been driven with a real mouse.** The GUI constructs and its
  prediction path is tested headlessly, but interactive behaviour needs a human.
- `python -m pytest` — 63 passed. `lint.cmd` — 10.00/10, mypy clean over 11 files.
- Scripts 01–05 still run unchanged; `flylab/model.py` was not modified.
