# Status: Learning to play Dino with the same circuit

_Last updated: 2026-09-18_

## Current state

done — everything builds, tests and verifies, and the circuit plays at about 2.7x random
(median-of-medians across training seeds) after a credit-assignment defect was found and fixed.
Still far below a scripted policy. Read the caveats before quoting any number.

**Three claims in earlier versions of this file were wrong and are corrected below**: that the
encoding was the binding constraint, that the second dopamine system could not be shown to earn
its keep, and that training longer always makes things worse. All three were downstream of one
bug.

## Completed

- **`flylab/dino.py`** — the game, frame-deterministic and seeded. Constants taken from the
  Processing sources directly, not from a summary. 17 tests.
- **`flylab/pilot.py`** — eligibility trace, both dopamine systems, recovery, and the episode
  loop. 15 tests. `flylab/model.py` is untouched, as required.
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
| **the taught circuit** | **749** | **779.4** | **3,000** |
| a hand-written policy (the ceiling) | 3,000 | 3,000 | 3,000 |

That row is the shipped training seed and it is one of the better ones — the ninth of ten when
the same configuration is retrained on seeds 0-9. **The defensible headline is the
median-of-medians, 406.** The shipped model reached the 3,000-frame cap on one test seed, which
no previous version of this circuit ever did.

Across ten training seeds, on test seeds the choice never saw:

| arm | median-of-medians | mean | spread |
|---|---|---|---|
| **punishment and reward** | **406** | 443 | 182 - 854 |
| the same code before this change | 256 | 261 | 184 - 338 |
| punishment alone (PPL1) | 214 | 211 | 173 - 229 |

Earlier versions of this file reported 176, then 262. The first was `recovery` ten times too
fast; the second was the credit-assignment defect below.

### The defect: reward could not reach the action that needed it

A jump leaves the ground on the frame it is chosen and the obstacle passes some fifteen frames
later, so **every successful jump finishes in the air**. `run_episode` learned only on grounded
frames, so a clear that happened mid-jump returned through a `continue` and taught nothing. The
JUMP entry sat in the eligibility trace until three grounded decisions after landing evicted
it, always long before the next obstacle resolved.

The consequence was invisible until the trained weights were dumped:

| readout | approach mean | avoid mean | cells taught |
|---|---|---|---|
| RUN | 0.830 | 0.920 | 633 |
| **JUMP** | 0.882 | **1.000** | **0 of 1,927** |
| DUCK | 0.825 | 0.924 | 616 |

`avoid[:, JUMP]` was exactly 1.0 on every cell: PAM had never fired on a jump in 4,000 runs.
A score is `approach - avoid` and depression only removes weight, so **JUMP's score could not
exceed zero** while RUN and DUCK reached +0.998. Jumping was not competing; it won only where
the other two had been punished harder. `09_teach_the_dino.py` now prints this table every run,
because a readout that never moves is otherwise invisible.

The fix is to call the existing `reward()` on the existing trace when a clear happens in the
air. Crashes in the air are still ignored - that asymmetry is load-bearing, and the symmetric
version was measured and is much worse (see below).

| validation seeds, 25 training seeds per arm | median-of-medians | mean | seeds past 400 |
|---|---|---|---|
| mid-air clears rewarded | **327** | 413 | 44% |
| mid-air clears ignored | 262 | 268 | 4% |

One-sided Mann-Whitney *p* = 0.013; a run of the new arm beats a run of the old one 69% of the
time. Ten seeds per arm gave *p* = 0.145 and were not enough to call it - worth recording,
because the first two attempts at this measurement would both have been reported as wins.

### Hyperparameters were chosen honestly the second time

The first pass chose settings by scoring the same seeds the result was reported on, which is
leakage and was recorded here as an open issue. Redone: choose on 8000-8099, report on
9000-9099.

These are pre-fix numbers, kept because they are what motivated the recovery rate still in use:

| recovery | episodes | validation |
|---|---|---|
| **0.0002** | **4,000** | **226** |
| 0.0002 | 8,000 | 197 |
| 0.0005 | 4,000 | 202 |
| 0.0005 | 8,000 | 182 |

**Recovery is the most consequential of the constants** — 0.002 gives 176, 0.0002 gives 262 —
which is a pleasing result, because it is the one piece of arithmetic this task adds to the
fly's rule and it now has a measurement behind it rather than a guess. It was re-swept after
the defect above was fixed, since it fires once per dopamine event and that change roughly
doubled how many events there are; 0.0002 won again, ahead of 0.0001 (282) and 0.0005 (144).
`trace_len` must stay at 3: at 6 the policy collapses to about 133 frames, which is below
doing nothing.

**"More training is worse" was an artefact of the defect and no longer holds.** It was measured
when reward never fired, and with depression the only force acting, more exposure eroded what
structure existed. With PAM working the curve saturates instead of falling (validation seeds,
ten training seeds each):

| episodes | 1,000 | 2,000 | 4,000 | 8,000 |
|---|---|---|---|---|
| median-of-medians | 166 | 299 | **367** | 374 |

4,000 is kept: 8,000 costs twice as much and 374 against 367 is not a difference.

**The second dopamine system does earn its keep, and now it can be shown.** Ten training seeds
per arm on the test seeds:

| | median-of-medians | Mean | spread |
|---|---|---|---|
| punishment and reward | **406** | 443 | 182 - 854 |
| punishment alone (PPL1) | 214 | 211 | 173 - 229 |

One-sided Mann-Whitney *p* = 0.016. An earlier version of this file recorded the opposite —
punishment alone appearing to *win*, 296 against 262 — and called it unresolved noise. It was
not noise, it was the defect: PAM had no function to lose, so removing it cost nothing. This is
the closest analogue to the digit task's clean eight-point answer (84.1% against 91.9%).

**What it actually learned**, from the policy dump:

```
                 0   50  100  150  200  250  300  350  400  500
  short cactus run  run  JUMP JUMP JUMP JUMP run  run  run  run
  tall cactus  run  run  JUMP JUMP run  run  run  run  run  run
  low bird     run  run  JUMP JUMP JUMP run  run  run  run  run
  middle bird  run  run  JUMP JUMP JUMP JUMP JUMP JUMP run  run
  high bird    run  run  run  run  run  run  run  run  run  run
```

**The middle bird is solved and the low bird broke.** The bird at 480 is low enough to catch a
crouching dino, so it must be jumped; the previous circuit never jumped it at any distance and
this one jumps it across the widest window of any obstacle. The high bird is still correctly
ignored.

But DUCK has been extinguished — it appears nowhere in the table, and its readouts barely moved
(approach 0.992, avoid 1.000, 248 cells touched against 860 for RUN). The low bird, which the
old circuit ducked reliably, is now the single largest killer. What ended each of the 100 test
runs:

| cause | runs |
|---|---|
| low bird | 54 |
| tall cactus | 23 |
| middle bird | 12 |
| short cactus | 10 |
| reached the 3,000-frame cap | 1 |

Jumping does clear a low bird, but on a tighter window than the one it learned. One failure has
been traded for another on much better terms — 262 to 749 on the shipped seed — but it is a
trade, not a clean win, and it is the obvious next thing to attack.

The mechanism is the same 65% code overlap that limits everything else here: praise for jumping
generalises across distance far more readily than the narrow band in which ducking is the
better answer, so once JUMP could be rewarded at all it swamped DUCK.

**Single seeds mean little here.** An earlier draft recorded 707 from a sweep; the same settings
scored 176 with a different RNG seed. The current configuration spans 182 to 854 across ten
training seeds. Quote the spread, not a number.

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

**Mid-jump outcomes split: crashes must be ignored, clears must not.** The original loop ignored
both, which cost this project its largest single result — see the defect section above. The
half that has to stay ignored is punishment.

Measured across five training seeds, crediting *both* was much worse:

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

Three of five seeds collapsed to chance, and the learned policies visibly switched to ducking.
The asymmetry is what makes the current version work: **praise generalising onto a neighbouring
jump costs nothing, punishment generalising onto one destroys the policy.** Both halves are now
pinned by tests named so that neither gets "corrected" again.

**Reading a regression test as a specification is how the defect survived.** The test asserting
that a mid-air clear teaches nothing was written to lock in a measured result, and it did its
job so well that the missing reward looked deliberate for the rest of the project. A test can
only pin the behaviour someone thought to question; it cannot tell you the behaviour is right.
What found it was dumping the weights and asking which of them had ever moved.

**Recovery had to become per-event rather than per-frame.** Relaxing every weight once per
frame is the honest reading of "synapses decay in time", but it costs about 10^10 float
operations over a training run. It now fires on each dopamine event, which keeps the moving
average interpretation exact and the cost proportional to the number of things actually
learned from.

## Browser demo, and five bugs it hid

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
driving the real page in headless Chrome.

5. **The track was the same track every time.** The starting seed was hard-coded to 1, so every
   page load replayed one identical game and the demo looked like a canned animation rather
   than a circuit playing. Each run now draws a fresh seed. Only the *track* is randomised: the
   circuit's own choices stay deterministic argmax, which is exactly what the parity check
   against Python asserts, so this cannot weaken that check.

Driving the browser's own `dino-game.js` and `flydino.js` headlessly over a 6,000-frame budget,
with the currently exported weights:

| | before the reward fix | now |
|---|---|---|
| runs in 6,000 frames | 17 | **9** |
| average run | 353 | **667** |
| obstacles cleared | 60 | **82** |

## Blocked / open issues

- **DUCK has been extinguished, and the low bird now causes 54% of deaths.** This is the
  regression introduced by fixing the reward path, and the clearest single thing left to fix.
  The net was strongly positive, but it is a trade rather than a clean win.
- **The encoding is a real limit, but it was not the binding one** — an earlier version of this
  file said it was, and that was wrong. A cactus 150 pixels away and one 400 away share 65% of
  their Kenyon cell code while needing opposite actions; obstacle *type* separates cleanly (11%
  between a cactus and a low bird), *distance* does not. That 65% is now the best explanation
  for why DUCK collapsed. Giving the gap a larger share of the 55 glomeruli is still untried,
  and the one measurement against it warns that every layout which sharpens distance blurs
  low-bird-versus-middle-bird (36% to 52-58%).
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
| Decisions only while grounded | A mid-air action cannot change the jump arc. Mid-air *outcomes* are a separate question and are credited asymmetrically - clears yes, crashes no |
| One fixed runner position | The original gives each of its 1,000 agents a random x. There is only one fly here |

## Next steps

In the order most likely to move the number:

1. **Bring DUCK back without losing the middle bird.** Ducking clears the low bird across a far
   wider window than jumping does, and the circuit has stopped using it. The suspicion is that
   reward for jumping generalises across distance while the useful ducking band is narrow, so
   the cheapest probe is per-action reward scaling, or crediting the *cleared obstacle's* code
   rather than the launching code. Measure before believing either.
2. **Per-obstacle credit.** Replace "the last three grounded decisions" — a proxy that only
   works because of code overlap — with a tag keyed to the obstacle the decision was about,
   discharged when that obstacle resolves. It is the biologically straightforward reading (the
   tag persists while the stimulus is present), it structurally excludes the empty-track
   decisions that made an unbounded trace fail, and it removes the arbitrary 3.
3. **Reconsider the glomerular layout, but carefully.** See the open issues; do not adopt a
   layout on overlap numbers alone, train it.
4. Raise the 3,000-frame cap. One test seed now reaches it, so it has started to censor the
   distribution; it is not yet doing real damage to the medians.

Done since the previous list was written: the mid-air reward asymmetry (item 1), which turned
out to be the largest result in the task; the dopamine ablation (item 2), now settled at
*p* = 0.016 in favour of both systems; and the validation split before that.
