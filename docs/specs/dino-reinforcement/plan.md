# Plan: Learning to play Dino with the same circuit

## Scope

In:

- A deterministic, seedable Python port of the Chrome Dino game from the Processing original.
- A tuning-curve encoding of game state into the connectome's 55 glomeruli.
- `FlyPilot`: eligibility trace, both dopamine systems, recovery.
- A training script reporting honest numbers against do-nothing and random baselines.
- A browser demo, in its own folder, provably identical to the Python.
- The site becoming a landing page plus two self-contained demos.

Out:

- Any change to `flylab/model.py`.
- Any change to the digit demo's behaviour or generated numbers. Its files move; its logic
  does not.
- Beating the source repository's genetic algorithm.

## Steps

1. [x] `flylab/dino.py` — game constants, `step`, collision, spawning, `mulberry32`.
       Tests first. A seed must replay identically before anything else is built.
2. [x] Tuning-curve encoding + a Kenyon-code separability diagnostic.
       **Go/no-go gate**: passed, but only after narrowing the gap range from 900 to 500 —
       see `status.md`.
3. [x] `flylab/pilot.py` — trace, `punish`, `reward`, `recover`, persistence. The episode loop
       moved here too, as `run_episode`, after a bug in an untested copy of it in `09` cost a
       day's worth of wrong conclusions.
4. [ ] `09_teach_the_dino.py` — train, ablate one dopamine system, report against baselines.
       Written; final run pending the mid-air fix.
5. [x] `flylab/export.py` extracted from `08`; `10_export_the_dino.py`; parity fixture.
6. [x] Web restructure: landing page, `web/digits/`, `web/dino/`. Digits demo verified at its
       new path before any dino web code was written. `pages.yml` needed no change — rsync
       excludes match a basename at any depth.
7. [x] `web/dino/` — `dino-game.js`, `flydino.js`, p5 sketch page, `verify.js`.
8. [ ] README section, sprite attribution, spec trio finalised.

Added along the way, not in the original plan:

9. [x] `11_check_the_web.py` — the digit demo's parity fixture had no generator and was
       gitignored, so that check could not be run from a clean clone at all.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Depression plus recovery does not converge | Sweep rate x recovery as `06` sweeps rate x sparsity. Fallback: normalise the score by active-cell count. A documented negative result against the GA baseline is still a result |
| Credit diluted across too long a trace | Decisions only while grounded, which shortens the trace roughly fivefold; discharge per obstacle rather than on a fixed window |
| Sparsity tuned for classification is wrong for control | Sweep it, as the digit work did |
| Python too slow to train in | Roughly 0.3 s per episode, so 2,000 episodes is about ten minutes. If tight, `argpartition` with a stable sort across the boundary group only |
| p5 shadows `window.random` and silently breaks parity | p5 in instance mode, rendering only. `dino-game.js` takes no p5 dependency and runs headless under Node, which would catch any leak |
| Python and JavaScript diverge | Positions are integer-exact; `quantise` already settles k-WTA ties; `mulberry32` asserted against the same reference vector in both languages |
| The digit demo breaks during the move | Only paths change. Step 6 verifies it before any new code lands |

## Verification / acceptance criteria

- `pytest` green, including new `test_dino.py` and `test_pilot.py`.
- `lint.cmd` clean, with the new modules and scripts added to its targets.
- `09_teach_the_dino.py` prints a learning curve and a baseline table. **Acceptance: median
  score over 100 unseen seeds beats uniform-random by a margin obviously outside noise, and
  both dopamine systems beat punishment alone** — the analogue of the digit task's 91.9%
  against 84.1%.
- `node web/dino/verify.js` — zero action mismatches, 100% winner-set overlap.
- `node web/digits/verify.js` — 61 cases, zero mismatches, proving the move was inert.
- Both demos open and run from `web/index.html`.
