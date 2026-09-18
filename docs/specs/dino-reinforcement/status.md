# Status: Learning to play Dino with the same circuit

_Last updated: 2026-09-18_

## Current state

done, with a weak result — everything builds, tests and verifies; the circuit learns, but not
well. See Results, and read the caveats before quoting any number.

## Completed

- **`flylab/dino.py`** — the game, frame-deterministic and seeded. Constants taken from the
  Processing sources directly, not from a summary. 17 tests.
- **`flylab/pilot.py`** — eligibility trace, both dopamine systems, recovery. 12 tests.
  `flylab/model.py` is untouched, as required.
- **`flylab/export.py`** — the CSR packer and numeric rounding, lifted out of
  `08_export_for_web.py` so both exporters cannot drift. Verified inert: re-running `08` after
  the change produced a byte-identical `model.js` (same md5) and the digits parity check still
  passes 61/61.
- **Web restructured** into `web/index.html` + `web/digits/` + `web/dino/`. The digits demo
  verifies identically at its new path.
- **`web/dino/dino-game.js`** — matches Python exactly across 2,400 frames, including the
  float32-narrowed glomerular activation (worst difference: 0).
- `09_teach_the_dino.py`, `10_export_the_dino.py`, `web/dino/verify.js`, the p5 demo page.

## Results

Median frames survived over 100 held-out seeds, cap 3,000. `09_teach_the_dino.py`, training
seed 0, 3,000 runs.

| Policy | Median | Mean | Best |
|---|---|---|---|
| do nothing | 139 | 142.2 | 236 |
| act at random | 148 | 167.8 | 362 |
| always jump | 159 | 173.4 | 512 |
| **the taught circuit** | **176** | **218.6** | **702** |
| a hand-written policy (the ceiling) | 3,000 | 3,000 | 3,000 |

**This does not clear the bar the plan set.** The acceptance criterion was "beats uniform-random
by a margin obviously outside noise". On the shipped model, 176 against 148 is not that. The
mean and the best run separate more clearly (218.6 against 167.8; 702 against 362), so the
circuit is doing something — but a median 19% above chance is a weak result and should be read
as one.

Across five training seeds, at 2,000 runs each, the picture is better and much more variable:

| | median of medians | range |
|---|---|---|
| taught circuit | 267 | 161 – 541 |
| random | ~148 | — |

**The second dopamine system does not clearly earn its keep here**, which the digit task's
ablation did (84.1% against 91.9%):

| | Median | Mean | Best |
|---|---|---|---|
| punishment alone (PPL1) | 163 | 186.3 | 738 |
| punishment and reward | 176 | 218.6 | 702 |

Thirteen frames of median, and punishment-alone has the *better* best run. This is within the
seed-to-seed noise measured above and should not be reported as a win. The likely reason is
structural: rewards fire when an obstacle is cleared, but most obstacles are cleared mid-jump,
and mid-jump outcomes are deliberately not credited — so the PAM pathway sees far fewer events
than PPL1 does. Testing reward-on-clear separately from punishment-on-crash, rather than
gating both on the same rule, is the obvious next experiment and is not done.

**What it actually learned**, from the policy dump:

```
                 0   50  100  150  200  250  300  350  400  500
  short cactus run  JUMP JUMP JUMP JUMP JUMP run  JUMP JUMP duck
  tall cactus  run  duck JUMP JUMP JUMP JUMP run  run  run  run
  low bird     run  run  run  JUMP run  run  run  run  run  run
  middle bird  run  run  run  run  run  run  run  run  run  run
  high bird    run  run  run  run  run  run  run  run  run  run
```

Cactus jumping is largely right, if smeared wider than the expert's 150-300 window. Birds are
essentially unlearned: the low bird should duck and the middle bird should jump, and it does
neither. The high bird is correct, but only because doing nothing is correct there. So the
result is "it half-learned one of the two obstacle families".

So the method works, unreliably. Two cautions that matter more than the headline:

- **Single seeds mean nothing here.** An earlier draft of this file recorded 707, taken from a
  hyperparameter sweep; the same settings scored 176 when run with a different RNG seed. Any
  number quoted from one seed on this task is noise.
- **Training longer made it worse.** Seed 0 scores 234 at 2,000 runs and 176 at 3,000. That
  echoes the digit task, where one epoch beat three for the same reason — depression only
  removes weight. `EPISODES` has deliberately *not* been tuned down, because the measurement
  showing 2,000 is better used the same held-out seeds, and selecting on it would be leakage.
  A proper validation split is the fix, and it is not done.

## What was learned the hard way

**The gap encoding was spread over the wrong range.** Tuning curves spanning 0-900 pixels put
the decision boundary well inside one curve's width, so a cactus 300 away and one 360 away
shared about four fifths of their code — and those two situations need opposite actions. A
scripted expert showed the entire decision lives between 150 and 300 pixels, so the range was
clamped to 500. Everything beyond that means the same thing: keep running.

**An unbounded eligibility trace learns nothing.** This was the single biggest finding, and it
cost several wrong turns. An obstacle becomes visible about fifty frames before it can be hit,
so a crash was blaming fifty decisions, forty-nine of which were taken while the track was
empty and could not have mattered. The signal was outnumbered roughly thirty to one. Bounding
the trace to the last three decisions took the median from 146 to 707.

**A peaked credit kernel did not help, though the reasoning was sound.** Since the decision
that matters is about twenty frames before impact, and exponential decay puts maximum blame at
zero delay, a kernel peaked at the right delay should have been better — and real KC→MBON
plasticity does depend on the interval that way. Measured: every peaked variant scored 133-169,
no better than random. What matters here is the trace's *length*, not its shape. Recorded
because it is the kind of plausible idea worth not repeating.

**Outcomes that happen mid-jump must be ignored, and that is not a bug.** The training loop
advances airborne frames without learning from what they return. Since a jump lasts 34 frames,
most obstacles are both cleared and crashed into off the ground, so this looked like an obvious
defect — it was found by reading, fixed, and covered with regression tests before being
measured.

Measured across five training seeds, the "fix" was much worse:

| | median of medians | range |
|---|---|---|
| ignoring mid-air outcomes | **267** | 161 – 541 |
| crediting mid-air outcomes | 150 | 150 – 548 |

Three of five seeds collapsed to chance. The mechanism is visible in the learned policies:
crediting mid-air outcomes makes JUMP disappear and the circuit ducks instead. Neighbouring
gaps share 77–99% of their Kenyon cell code, so punishing a jump mistimed by ten pixels also
punishes the jump that would have worked; crediting mid-air crashes roughly doubles the
punishment reaching JUMP and extinguishes it everywhere. Jumping is learned by elimination —
running into things is punished at impact, and jumping is what survives.

The behaviour is now deliberate, documented in `FlyPilot.run_episode`, and pinned by two tests
named so that nobody repeats the same correction.

**Two of three mechanism hypotheses were wrong.** The peaked kernel and the mid-air fix were
both well-reasoned and both refuted by measurement; only the bounded trace survived. Worth
recording as a fact about this problem rather than about any one idea.

**Recovery had to become per-event rather than per-frame.** Relaxing every weight once per
frame is the honest reading of "synapses decay in time", but it costs about 10^10 float
operations over a training run. It now fires on each dopamine event, which keeps the moving
average interpretation exact and the cost proportional to the number of things actually
learned from.

## Browser demo, and three bugs it hid

The page was verified by rendering it in headless Chrome and driving its controls, not by
reasoning about it. Three real defects only showed up that way:

1. **`file://` cannot fetch, so p5 never started.** `loadImage("sprites.png")` was blocked by
   CORS, `preload()` never resolved, and the page sat on p5's "Loading..." with a blank canvas.
   `08_export_for_web.py` had already recorded this lesson for the model bundle; it was not
   applied to the sprite sheet. Fixed twice over: the sheet now ships as a `data:` URI from a
   generated `sprites.js`, and `preload()` is gone entirely, so the game starts immediately and
   falls back to drawing plain hitboxes if the art never arrives.
2. **The JavaScript game silently dropped obstacle fields.** `advanceObstacles` rebuilt each
   moving obstacle from the four fields the *rules* use, discarding `kind` and `bird`, which
   only the renderer reads. Python moves obstacles with `dataclasses.replace` and never had the
   problem. The rules and the parity check both stayed perfectly happy while the page threw on
   its second frame. `verify.js` now asserts those fields survive, and the assertion was checked
   by reintroducing the bug.
3. **The score bars were inverted.** Both readouts only ever shrink, so after training every
   score is negative and the *least* wanted action has the largest absolute value. Sizing bars
   by magnitude drew the longest bar beside the action the circuit least wanted. Bars are now
   scaled between the worst and best score on screen, so the fullest is always the one taken.

The demo plays on demand rather than autoplaying, and stops when the fly crashes instead of
rolling into the next seed, so a crash can be looked at. Verified in Chrome across three
consecutive runs: 227, 208 and 155 frames, each ending with the run halted and the button
offering another.

## Blocked / open issues

- **The headline result is weak and the acceptance criterion is not met.** See Results.
- **The likely root cause is the encoding, not the learning rule.** A cactus 150 pixels away
  and one 400 away share 65% of their Kenyon cell code, and those two states need opposite
  actions. Obstacle *type* separates cleanly (11% overlap between a cactus and a low bird);
  *distance* does not. Every failure observed — jumping too early, jumping too late, extra
  punishment generalising onto correct jumps — traces back to that 65%. Giving the gap its own
  larger share of the 55 glomeruli, or spacing its tuning curves non-uniformly, is the first
  thing to try and was not tried.
- **Birds are unlearned.** Two of five obstacle types get the wrong action at every distance.
- **Reward is starved by design.** Most obstacles are cleared mid-jump, and mid-jump outcomes
  are not credited, so PAM sees far fewer events than PPL1. Decoupling the two would test this.
- p5.min.js is 1.03 MB, which conflicts with the repository rule against committing artifacts
  over 100 KB. Vendoring was the approved choice; a CDN with subresource integrity is the
  alternative. Needs a decision.
- The venv runs Python 3.14.5 while `pyproject.toml` pins mypy to 3.13. Pre-existing, and
  everything passes, but worth knowing.
- `web/digits/verify.js` had its 8x8 comparison relaxed from bit-identical to one float32 ulp.
  That assertion had never been exercised on fractional stroke coverage before, because the old
  fixture had no generator and its cases produced whole numbers. Kenyon cells and predictions
  are still compared at zero tolerance.

## Deviations from the Processing original

| Deviation | Reason |
|---|---|
| Frames, not `millis()` | The original's difficulty depends on the frame rate the machine achieves, so training would not be reproducible. Spawn interval is uniform in `[30, 90)` frames |
| Score is frames survived | The original's score is a `millis() > 50` sampling artefact, not distance |
| Speed is a live input | The original's `(speed - 15) / (30 - 15)` is Java integer division and is 0 throughout — an accidental dead input |
| Three exclusive actions | The original thresholds two ReLU outputs independently; pressing both resolves to duck anyway |
| Decisions only while grounded | A mid-air action cannot change the jump arc, so crediting it only dilutes the eligibility trace |
| One fixed runner position | The original gives each of its 1,000 agents a random x. There is only one fly here |

## Next steps

In the order most likely to move the number:

1. **Give distance more of the input layer.** 65% code overlap between states needing opposite
   actions is the binding constraint. Try 25 glomeruli for the gap and fewer for width and
   speed, which barely matter, or non-uniform spacing that is dense below 300 pixels.
2. **Split the two dopamine events.** Credit mid-jump clears as reward while continuing to
   ignore mid-jump crashes, which is the asymmetry the A/B never tested.
3. **Add a validation split** so training duration can be chosen without leaking.
4. Only then revisit the trace and rate constants.
