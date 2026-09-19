# Status: Learning to play Dino with the same circuit

_Last updated: 2026-09-18_

## Current state

done — everything builds, tests and verifies, and the circuit plays at about 1.8x random after
the recovery rate was tuned on a proper validation split. Still far below a scripted policy,
and one obstacle type is never handled. Read the caveats before quoting any number.

## Completed

- **`flylab/dino.py`** — the game, frame-deterministic and seeded. Constants taken from the
  Processing sources directly, not from a summary. 17 tests.
- **`flylab/pilot.py`** — eligibility trace, both dopamine systems, recovery, and the episode
  loop. 14 tests. `flylab/model.py` is untouched, as required.
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

Median frames survived over 100 test seeds (9000-9099), cap 3,000. `09_teach_the_dino.py`,
training seed 0, 4,000 runs, `recovery=0.0002`.

| Policy | Median | Mean | Best |
|---|---|---|---|
| do nothing | 139 | 142.2 | 236 |
| act at random | 148 | 167.8 | 362 |
| always jump | 159 | 173.4 | 512 |
| **the taught circuit** | **262** | **301.4** | **902** |
| a hand-written policy (the ceiling) | 3,000 | 3,000 | 3,000 |

**This now clears the bar the plan set** — 262 against 148 is well outside noise, and the same
settings over three training seeds give 262, 184 and 309 on test data they never influenced.
It is still a long way from the 3,000-frame ceiling a scripted policy reaches on every seed.

An earlier version of this file reported 176 and concluded the criterion was not met. That was
the same code with `recovery` ten times too fast; see below.

### Hyperparameters were chosen honestly the second time

The first pass chose settings by scoring the same seeds the result was reported on, which is
leakage and was recorded here as an open issue. Redone: choose on 8000-8099, report on
9000-9099.

| recovery | episodes | validation |
|---|---|---|
| **0.0002** | **4,000** | **226** |
| 0.0002 | 8,000 | 197 |
| 0.0005 | 4,000 | 202 |
| 0.0005 | 8,000 | 182 |

Two things fall out. **Recovery is the most consequential number in the task** — 0.002 gives
176, 0.0002 gives 262 — which is a pleasing result, because it is the one piece of arithmetic
this task adds to the fly's rule and it now has a measurement behind it rather than a guess.
Forget an order of magnitude too fast and a run's learning is gone before the next run can
build on it. And **more training is worse at every recovery rate tested**, the same saturation
the digit task hit: depression only removes weight.

**Whether the second dopamine system earns its keep is unresolved**, where the digit task got a
clean eight-point answer (84.1% against 91.9%):

| | Median | Mean | Best |
|---|---|---|---|
| punishment alone (PPL1) | 296 | 382.0 | 1,702 |
| punishment and reward | 262 | 301.4 | 902 |

Punishment alone scores *higher*. That reads as an argument against half the design, and it is
not: one training seed per arm, when the same configuration varies from 184 to 309 across
seeds, cannot separate a 34-frame difference from noise. Several seeds per arm would settle it
and have not been run. There is also a structural reason to expect PAM to be starved — rewards
fire when an obstacle is cleared, most obstacles are cleared mid-jump, and mid-jump outcomes
are deliberately not credited.

**What it actually learned**, from the policy dump:

```
                 0   50  100  150  200  250  300  350  400  500
  short cactus run  duck JUMP JUMP JUMP run  run  run  run  run
  tall cactus  duck duck JUMP JUMP duck duck duck duck duck duck
  low bird     duck duck duck duck duck duck duck duck duck duck
  middle bird  run  run  run  run  run  run  run  run  run  run
  high bird    run  run  run  run  run  run  run  run  run  run
```

Three of five obstacle families are now handled correctly: cacti are jumped inside the window
that works, the low bird is ducked, and the high bird is correctly ignored. The middle bird at
480 is the outstanding failure — low enough to catch a crouching dino, so it must be jumped,
and the circuit runs straight into it. That is roughly one obstacle in six and most of what
still ends runs.

**Why the middle bird is hard, specifically.** Jumping into a bird is a mid-air collision, and
mid-air outcomes are deliberately not credited, so a jump that flies into a bird is never
punished and a jump that clears one is never rewarded. The bird case gets almost no teaching
signal in either direction. Crediting mid-air outcomes globally was measured and is much worse
(see below), so the fix is not simply to turn that on.

So the method works, and unevenly. Two cautions that matter more than the headline:

- **Single seeds mean little here.** An earlier draft recorded 707 from a sweep; the same
  settings scored 176 with a different RNG seed. Even the tuned configuration spans 184 to 309
  across three seeds. Quote the spread, not a number.
- **Training longer makes it worse**, at every recovery rate tested. Depression only removes
  weight, so past a few thousand runs more exposure erodes the differences it built — the same
  effect that made one epoch beat three on digits. `EPISODES` is now chosen on validation seeds
  rather than left untuned, which was the open issue the first pass left behind.

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
the trace to the last three decisions took the median from 146 — chance — to 333 in that sweep.
Both of those are single-seed numbers from before the validation split existed, so read them as
"nothing" against "clearly something", not as precise quantities.

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

4. **The controls offered actions that made no sense.** Dying during play left the button
   reading "Pause" on a run that had already ended, because `advance()` set the paused flag
   without refreshing the controls — only the single-step path did. There are exactly three
   states and the button now derives from them rather than being poked individually:

   | state | button |
   |---|---|
   | stopped | Play, or Resume once started |
   | running | Pause |
   | crashed | Run again |

   A single-step button existed briefly and was removed as clutter. Worth noting that it was
   also the only way to drive the game deterministically from a headless browser — Chrome's
   virtual clock does not advance p5's `deltaTime`, so the render loop stalls at frame zero.
   Testing the crash path in a browser again would mean reinstating something equivalent.

The demo plays on demand rather than autoplaying, and stops when the fly crashes instead of
rolling into the next seed, so a crash can be looked at. Every transition was verified by
driving the real page in headless Chrome, and three consecutive runs gave 227, 208 and 155
frames, each halting with the button offering another.

## Blocked / open issues

- **The middle bird is never jumped.** One obstacle in six, and most of what still ends runs.
  It is the clearest single thing left to fix.
- **The likely root cause is the encoding, not the learning rule.** A cactus 150 pixels away
  and one 400 away share 65% of their Kenyon cell code, and those two states need opposite
  actions. Obstacle *type* separates cleanly (11% overlap between a cactus and a low bird);
  *distance* does not. Every failure observed — jumping too early, jumping too late, extra
  punishment generalising onto correct jumps — traces back to that 65%. Giving the gap its own
  larger share of the 55 glomeruli, or spacing its tuning curves non-uniformly, is the first
  thing to try and was not tried.
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

1. **Teach it the middle bird.** One obstacle in six and the main remaining killer. It needs a
   jump, and a jump into a bird is a mid-air collision, which is never credited — so the case
   gets almost no teaching signal. Splitting the two dopamine events is the cheapest probe:
   credit mid-jump *clears* as reward while continuing to ignore mid-jump crashes. That is the
   asymmetry the A/B never tested, and unlike the symmetric version it adds no punishment to
   generalise onto correct jumps.
2. **Settle the dopamine ablation properly** — several training seeds per arm. One seed each
   cannot separate 296 from 262 when the same configuration spans 184 to 309.
3. **Reconsider the glomerular layout, but carefully.** Distance resolution is poor (65% overlap
   between a cactus 150 away and one 400 away) and sharpening it looks obvious. Measured on the
   code-overlap proxy, every layout that sharpens distance *degrades* low-bird-versus-middle-bird
   separation, from 36% to 52-58% — and that is exactly the discrimination the remaining failure
   needs. The proxy is also non-monotonic: one layout with more gap glomeruli separated distance
   worse. Do not adopt a layout on overlap numbers alone; train it.
4. Only then revisit the trace and rate constants.

Done since this list was written: the validation split (item 3 of the old list), which is what
exposed the recovery rate.
