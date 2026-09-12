# Plan: Associative learning on real mushroom body wiring

## Scope

In scope: downloading hemibrain v1.2, extracting the mushroom body circuit, a rate model
with sparse coding, punishment learning, and a demonstration that learning is specific to
the punished odour.

Out of scope: spiking dynamics, reward learning, real odour maps, both hemispheres, any
closed-loop environment or game.

## Steps

- [x] 1. `flylab/data.py` + `01_get_the_data.py` — stream the 45.9 MB tarball, verify size,
      extract safely (`filter="data"`), load the three CSVs.
- [x] 2. `02_explore.py` — discover the real naming schemes before anything hardcodes them.
- [x] 3. `flylab/circuit.py` + `03_build_circuit.py` — build both weight matrices, derive
      valence from dopamine compartments, cross-check against literature, save `circuit.npz`.
- [x] 4. `flylab/model.py` + `04_odour_code.py` — odour generation, gain normalisation,
      k-winners-take-all, MBON readout; measure sparseness and decorrelation.
- [x] 5. `05_learn.py` — baseline, punish odour A, re-measure both odours, report the
      generalisation gradient.
- [x] 6. `tests/unit/` — model arithmetic, compartment parsing, end-to-end specificity.
- [x] 7. `README.md`, `docs/specs/`, `lint.cmd`, pinned dependencies.

## Risks & mitigations

| Risk | Outcome |
|---|---|
| Type/ROI naming differs from assumption | **Hit.** MBONs are numeric (`MBON01`) with compartment in the `instance`; ROI table stops at lobe level. Step 2 caught it before step 3 was written. |
| Fine compartments absent from ROI table | **Hit.** Fell back to parsing the instance string, as planned. |
| Valence principle misapplied | **Hit.** `PAM07(y4<y1y2)` uses `<`, which the parser treated as the neuron's own compartment, making γ1/γ2 look like reward territory and flipping MBON11 and MBON12. The literature cross-check caught it; fixed and covered by a regression test. |
| Sparseness far off published range | Not hit. Measured 5.0%, within the published 5–10%. |
| Learning wipes both odours | Not hit. Unpaired odour moves 0.5% as much as the punished one. |
| Codes not decorrelated | **Hit, and informative.** Raw connectome weights gave 34% overlap between unrelated odours against 5% chance, because Kenyon cell input gain spans 0–254 synapses. Fixed with explicit gain normalisation, documented as a modelling assumption and demonstrated in step 4. |

## Verification / acceptance criteria

All met:

- Tarball is 45,872,577 bytes; 21,739 neurons and 3,550,403 connections load.
- 1,927 Kenyon cells; mean 6.0 glomeruli each (published ~6–7, Caron et al. 2013).
- APL contacts all 1,927 Kenyon cells.
- Derived valence reproduces all three literature cross-checks (MBON11, MBON12, MBON01).
- Sparseness 5.0%, within the published 5–10% (Turner et al. 2008).
- Punished odour shifts −0.883; unpaired odour shifts −0.004 (0.5% collateral).
- Generalisation gradient decays monotonically with Kenyon cell overlap.
- `python -m pytest` — 47 passed. `lint.cmd` — 10.00/10, mypy clean.
