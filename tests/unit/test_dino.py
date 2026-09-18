"""The game has to be exactly reproducible, or none of the learning results mean anything."""

from __future__ import annotations

import pytest

from flylab import dino

# Generated from the canonical JavaScript mulberry32, which is the specification the browser
# port must also satisfy. Regenerate with:
#   node -e "function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;
#            var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;
#            return((t^t>>>14)>>>0)/4294967296;};} ..."
JS_SEED_0 = [
    0.26642920868471265,
    0.00032974570058286190,
    0.22327202744781971,
    0.14620214793831110,
    0.46732782293111086,
    0.54504908272065222,
]
JS_SEED_12345 = [0.97972826776094735, 0.30675226449966431, 0.48420542152598500]


def _jump_arc(game: dino.Game) -> list[int]:
    """Every vertical position from take-off until the runner is back on the ground."""
    game.step(dino.JUMP)
    heights = [game.y]
    while game.jumping:
        game.step(dino.RUN)
        heights.append(game.y)
    return heights


def _history(seed: int, frames: int = 400) -> list[tuple[object, ...]]:
    game = dino.Game.new(seed)
    trail: list[tuple[object, ...]] = []
    for _ in range(frames):
        game.step(dino.RUN)
        trail.append((game.frame, game.alive, game.y, tuple(game.obstacles)))
    return trail


@pytest.mark.parametrize(("seed", "expected"), [(0, JS_SEED_0), (12345, JS_SEED_12345)])
def test_the_generator_matches_the_javascript_reference(seed: int, expected: list[float]) -> None:
    """Identical random numbers in both languages are what makes the parity check possible."""
    rng = dino.Mulberry32(seed)

    drawn = [rng.next() for _ in expected]

    assert drawn == expected


def test_the_jump_peaks_172_pixels_above_the_ground() -> None:
    arc = _jump_arc(dino.Game.new(0))

    assert min(arc) == dino.GROUND_Y - 172


def test_the_jump_lasts_34_frames_and_ends_on_the_ground() -> None:
    arc = _jump_arc(dino.Game.new(0))

    assert len(arc) == 34
    assert arc[-1] == dino.GROUND_Y
    assert all(height < dino.GROUND_Y for height in arc[:-1])


def test_an_action_taken_in_mid_air_is_ignored() -> None:
    """Decisions only count while grounded, so the eligibility trace is not diluted by no-ops."""
    game = dino.Game.new(0)
    game.step(dino.JUMP)

    airborne = game.y
    game.step(dino.DUCK)

    assert game.jumping
    assert game.y != dino.DUCK_Y
    assert game.y != airborne


def test_boxes_that_only_touch_do_not_collide() -> None:
    """The original's bounds are exclusive; an off-by-one here silently changes the difficulty."""
    left = (0.0, 0.0, 10, 10)

    assert not dino.overlaps(left, (10.0, 0.0, 10, 10))
    assert dino.overlaps(left, (9.0, 0.0, 10, 10))


def test_the_high_bird_passes_over_a_running_dino() -> None:
    """Bird type 2 sits at y=370 and never reaches the standing box - doing nothing is correct."""
    game = dino.Game.new(0)
    game.obstacles.append(dino.Obstacle(x=float(dino.RUNNER_X), y=370, width=84, height=40))

    step = game.step(dino.RUN)

    assert not step.crashed


def test_ducking_clears_the_low_bird() -> None:
    game = dino.Game.new(0)
    game.obstacles.append(dino.Obstacle(x=float(dino.RUNNER_X), y=435, width=84, height=40))

    step = game.step(dino.DUCK)

    assert not step.crashed


def test_ducking_does_not_clear_the_middle_bird() -> None:
    """Bird type 1 at y=480 overlaps the crouched box too; only a jump clears it."""
    game = dino.Game.new(0)
    game.obstacles.append(dino.Obstacle(x=float(dino.RUNNER_X), y=480, width=84, height=40))

    step = game.step(dino.DUCK)

    assert step.crashed


def test_standing_still_into_a_cactus_is_fatal() -> None:
    game = dino.Game.new(0)
    game.obstacles.append(dino.Obstacle(x=float(dino.RUNNER_X), y=470, width=30, height=66))

    step = game.step(dino.RUN)

    assert step.crashed
    assert not game.alive


def test_clearing_an_obstacle_is_reported_once() -> None:
    """The reward event has to fire exactly once per obstacle, or credit is double counted.

    The high bird is used because it passes the runner without ever touching it, so the
    clearing can be observed without the run ending first.
    """
    game = dino.Game.new(0)
    game.obstacles.append(dino.Obstacle(x=float(dino.RUNNER_X + 200), y=370, width=84, height=40))

    reports = [game.step(dino.RUN).cleared for _ in range(30)]

    assert reports.count(True) == 1


def test_the_same_seed_replays_identically() -> None:
    assert _history(7) == _history(7)


def test_different_seeds_diverge() -> None:
    assert _history(7) != _history(8)


def test_speed_rises_by_a_thousandth_each_frame() -> None:
    game = dino.Game.new(0)

    for _ in range(50):
        game.step(dino.RUN)

    assert game.speed == pytest.approx(dino.SPEED_0 + 50 * dino.SPEED_GAIN)


def test_an_empty_track_reads_as_the_furthest_possible_gap() -> None:
    """The original signals 'nothing ahead' with an out-of-range vector; a clamp is cleaner."""
    game = dino.Game.new(0)
    game.obstacles.clear()

    assert game.features()[0] == dino.GAP_MAX


def test_the_encoding_fills_every_glomerulus_the_connectome_has() -> None:
    activation = dino.to_glomeruli(dino.Game.new(0).features())

    assert activation.shape == (dino.GLOMERULI,)
    assert activation.sum() > 0.0


def test_nearby_states_are_encoded_more_alike_than_distant_ones() -> None:
    """Tuning curves exist so the Kenyon layer can tell 'obstacle near' from 'obstacle far'."""
    near = dino.to_glomeruli((100.0, 470.0, 66.0, 30.0, 15.0))
    close = dino.to_glomeruli((140.0, 470.0, 66.0, 30.0, 15.0))
    far = dino.to_glomeruli((800.0, 470.0, 66.0, 30.0, 15.0))

    assert float(near @ close) > float(near @ far)
