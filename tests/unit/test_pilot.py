"""The learning rule is the claim being made, so its arithmetic is pinned down here.

Most of these build a pilot over a toy expansion layer rather than the connectome: the rule
does not care how wide the layer is, and a six-cell circuit makes the arithmetic checkable by
hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from flylab import dino
from flylab.pilot import FlyPilot

CELLS = 6


def _pilot(**overrides: float) -> FlyPilot:
    settings: dict[str, float] = {"sparsity": 0.5, "rate": 0.5, "recovery": 0.0, "trace_decay": 0.5, "trace_len": 8}
    settings.update(overrides)
    return FlyPilot(
        pn_to_kc=np.eye(dino.GLOMERULI, CELLS, dtype=np.float32),
        approach=np.ones((CELLS, dino.ACTIONS), dtype=np.float32),
        avoid=np.ones((CELLS, dino.ACTIONS), dtype=np.float32),
        **settings,  # type: ignore[arg-type]
    )


def _code(*cells: int) -> npt.NDArray[np.bool_]:
    code = np.zeros(CELLS, dtype=np.bool_)
    code[list(cells)] = True
    return code


def test_an_untaught_pilot_prefers_nothing() -> None:
    """Both readouts start flat, so an untaught fly acts at random without a special case."""
    scores = _pilot().scores(_code(0, 1))

    assert np.array_equal(scores, np.zeros(dino.ACTIONS, dtype=np.float32))


def test_punishment_lowers_only_the_action_that_was_taken() -> None:
    pilot = _pilot()
    pilot.remember(_code(0, 1), dino.JUMP)

    pilot.punish()
    scores = pilot.scores(_code(0, 1))

    assert scores[dino.JUMP] < 0.0
    assert scores[dino.RUN] == 0.0
    assert scores[dino.DUCK] == 0.0


def test_reward_raises_only_the_action_that_was_taken() -> None:
    pilot = _pilot()
    pilot.remember(_code(0, 1), dino.DUCK)

    pilot.reward()
    scores = pilot.scores(_code(0, 1))

    assert scores[dino.DUCK] > 0.0
    assert scores[dino.RUN] == 0.0
    assert scores[dino.JUMP] == 0.0


def test_a_state_that_was_never_visited_is_untouched() -> None:
    """Depression is local to the cells that fired - this is why odour B stayed safe."""
    pilot = _pilot()
    pilot.remember(_code(0, 1), dino.JUMP)

    pilot.punish()

    assert np.array_equal(pilot.scores(_code(2, 3)), np.zeros(dino.ACTIONS, dtype=np.float32))


def test_recent_decisions_are_blamed_more_than_old_ones() -> None:
    """The eligibility trace is what lets a crash teach the decision that caused it."""
    pilot = _pilot()
    pilot.remember(_code(0), dino.JUMP)
    pilot.remember(_code(1), dino.JUMP)

    pilot.punish()

    older = pilot.scores(_code(0))[dino.JUMP]
    newer = pilot.scores(_code(1))[dino.JUMP]
    assert newer < older < 0.0


def test_the_trace_only_holds_the_most_recent_decisions() -> None:
    """A bounded tag is what keeps a crash from blaming the empty track thirty frames earlier.

    Measured: an unbounded trace scored no better than acting at random, because the decisions
    made before the obstacle was even visible outnumbered the one that mattered.
    """
    pilot = _pilot(trace_len=2)
    pilot.remember(_code(0), dino.JUMP)
    pilot.remember(_code(1), dino.JUMP)
    pilot.remember(_code(2), dino.JUMP)

    pilot.punish()

    assert pilot.scores(_code(0))[dino.JUMP] == 0.0
    assert pilot.scores(_code(1))[dino.JUMP] < 0.0
    assert pilot.scores(_code(2))[dino.JUMP] < 0.0


def test_discharging_the_trace_empties_it() -> None:
    """A second crash must not re-punish decisions the first one already accounted for."""
    pilot = _pilot()
    pilot.remember(_code(0, 1), dino.JUMP)
    pilot.punish()
    once = pilot.scores(_code(0, 1))[dino.JUMP]

    pilot.punish()

    assert pilot.scores(_code(0, 1))[dino.JUMP] == once


def test_recovery_relaxes_weights_toward_one_without_overshooting() -> None:
    pilot = _pilot(recovery=0.25)
    pilot.remember(_code(0, 1), dino.JUMP)
    pilot.punish()
    depressed = float(pilot.approach[0, dino.JUMP])

    pilot.recover()

    assert depressed < float(pilot.approach[0, dino.JUMP]) < 1.0


def test_recovery_never_pushes_a_weight_past_baseline() -> None:
    pilot = _pilot(recovery=1.0)
    pilot.remember(_code(0, 1), dino.JUMP)
    pilot.punish()

    pilot.recover()

    assert float(pilot.approach[0, dino.JUMP]) == pytest.approx(1.0)


def test_a_settled_pilot_picks_the_action_it_was_rewarded_for() -> None:
    pilot = _pilot()
    pilot.remember(_code(0, 1), dino.DUCK)
    pilot.reward()

    assert pilot.act(_code(0, 1)) == dino.DUCK


def test_the_expansion_layer_is_never_rewired() -> None:
    """Only the readout learns, as in the fly. The connectome is fixed."""
    pilot = _pilot()
    wiring = pilot.pn_to_kc.copy()

    pilot.remember(_code(0, 1), dino.JUMP)
    pilot.punish()
    pilot.recover()

    assert np.array_equal(pilot.pn_to_kc, wiring)


def _always_jumps() -> FlyPilot:
    """Force JUMP by flattening the approach to the other two, leaving JUMP's own weights at
    baseline so a later reward or punishment is still visible in them."""
    pilot = _pilot()
    pilot.approach[:, dino.RUN] = 0.0
    pilot.approach[:, dino.DUCK] = 0.0
    return pilot


def test_a_crash_in_mid_air_teaches_nothing() -> None:
    """Deliberate, counterintuitive, and measured. Do not "fix" this without rerunning the A/B.

    A jump lasts 34 frames, so most crashes happen while airborne, and crediting them looks
    obviously right. It is not: across five training seeds it took the median from 267 frames
    down to 150, which is chance. Neighbouring gaps share 77-99% of their Kenyon code, so
    punishing a jump mistimed by ten pixels also punishes the jump that would have worked, and
    JUMP is extinguished everywhere.

    The high bird makes the case constructible - it sits at 370, where a grounded dino passes
    safely underneath but a jumping one rises into it.
    """
    pilot = _always_jumps()
    game = dino.Game.new(0)
    game.obstacles.clear()
    game.spawn_timer = 10_000
    game.obstacles.append(dino.Obstacle(x=350.0, y=370, width=84, height=40))

    episode = pilot.run_episode(game, max_frames=40)

    assert not game.alive
    assert episode.frames < 40
    assert float(pilot.approach[:, dino.JUMP].min()) == 1.0


def test_an_obstacle_cleared_in_mid_air_is_still_counted() -> None:
    """It does not teach, but it does have to be reported, or the run's tally is wrong."""
    pilot = _always_jumps()
    game = dino.Game.new(0)
    game.obstacles.clear()
    game.spawn_timer = 10_000
    game.obstacles.append(dino.Obstacle(x=350.0, y=470, width=30, height=66))

    episode = pilot.run_episode(game, max_frames=40)

    assert game.alive
    assert episode.cleared == 1
    assert float(pilot.avoid[:, dino.JUMP].min()) == 1.0


def test_memory_survives_a_round_trip(tmp_path: Path) -> None:
    """Storing the rates as float32 once made 0.10 reload as 0.10000000149."""
    pilot = _pilot(rate=0.1, recovery=0.001)
    pilot.remember(_code(0, 1), dino.JUMP)
    pilot.punish()
    path = pilot.save(tmp_path / "memory.npz")

    restored = FlyPilot.load(pilot.pn_to_kc, path)

    assert restored.rate == 0.1
    assert restored.recovery == 0.001
    assert np.array_equal(restored.approach, pilot.approach)
    assert np.array_equal(restored.avoid, pilot.avoid)
