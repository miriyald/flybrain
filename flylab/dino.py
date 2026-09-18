"""The Dino game, ported for determinism, and its encoding into glomeruli.

The Processing original advances physics once per rendered frame but spawns obstacles and
counts score against the wall clock, so how hard the game is depends on the frame rate the
machine happens to achieve. Everything here is measured in frames instead, which is the same
game and a reproducible one.

Game state reaches the circuit the way a smell would: five continuous quantities, each driving
a bank of glomeruli tuned to preferred values along its range. Five banks of eleven is exactly
the 55 glomeruli the connectome provides.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

RUN = 0
JUMP = 1
DUCK = 2
ACTIONS = 3

GROUND_Y = 450
DUCK_Y = 484
STAND_WIDTH = 80
STAND_HEIGHT = 86
DUCK_WIDTH = 110
DUCK_HEIGHT = 52
RUNNER_X = 200

JUMP_HEIGHT = 172
JUMP_START = 0.0001
JUMP_STEP = 0.03

SPAWN_X = 1350.0
SPEED_0 = 15.0
SPEED_GAIN = 0.001
SPAWN_MIN_FRAMES = 30
SPAWN_MAX_FRAMES = 90

CACTUS_WIDTHS = (30, 64, 98, 46, 96, 146)
CACTUS_HEIGHTS = (66, 66, 66, 96, 96, 96)
CACTUS_Y = (470, 470, 470, 444, 444, 444)
BIRD_WIDTH = 84
BIRD_HEIGHT = 40
BIRD_Y = (435, 480, 370)

GAP_MAX = 500.0
GLOMERULI = 55
TUNING_PER_FEATURE = 11
FEATURE_RANGES = ((0.0, GAP_MAX), (370.0, 480.0), (40.0, 96.0), (30.0, 146.0), (15.0, 20.0))

_MASK32 = 0xFFFFFFFF
_TWO32 = 4294967296.0

Box = tuple[float, float, int, int]


def _tuning() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    centres = np.array([np.linspace(low, high, TUNING_PER_FEATURE) for low, high in FEATURE_RANGES])
    widths = np.array([[(high - low) / (TUNING_PER_FEATURE - 1)] for low, high in FEATURE_RANGES])
    return centres, widths


_CENTRES, _WIDTHS = _tuning()


def to_glomeruli(features: tuple[float, ...]) -> npt.NDArray[np.float32]:
    """Five readings into 55 graded glomerular responses.

    An absent obstacle reports zeros for its geometry, which falls far outside every tuning
    curve and so reads as silence on those banks - a distinct, well defined 'nothing ahead'
    rather than the original's out-of-range sentinel values.

    The gap saturates at 500 rather than spanning the full track. A scripted expert clears
    every obstacle by jumping when the gap is between 150 and 300 and dies if it jumps at 350,
    so the whole decision lives inside 500 pixels; spreading the curves across 900 put that
    boundary well inside one curve's width and left neighbouring states sharing four fifths of
    their code. Anything further away means the same thing - keep running - so the resolution
    spent out there bought nothing.
    """
    values = np.asarray(features, dtype=np.float64).reshape(len(FEATURE_RANGES), 1)
    response = np.exp(-0.5 * (((values - _CENTRES) / _WIDTHS) ** 2))
    return response.reshape(GLOMERULI).astype(np.float32)


def overlaps(first: Box, second: Box) -> bool:
    """Axis-aligned overlap with exclusive bounds, as the original has it."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax + aw > bx and ax < bx + bw and ay + ah > by and ay < by + bh


def _imul(left: int, right: int) -> int:
    return (left * right) & _MASK32


class Mulberry32:
    """The browser's generator, reimplemented so both languages draw the same numbers.

    numpy's generators are excellent and unavailable in JavaScript. Parity between the trained
    circuit and the deployed one needs an identical obstacle sequence, so the generator has to
    be one that can be written twice and verified.
    """

    def __init__(self, seed: int) -> None:
        self._state = seed & _MASK32

    def next(self) -> float:
        self._state = (self._state + 0x6D2B79F5) & _MASK32
        state = self._state
        mixed = _imul(state ^ (state >> 15), 1 | state)
        mixed = ((mixed + _imul(mixed ^ (mixed >> 7), 61 | mixed)) & _MASK32) ^ mixed
        return ((mixed ^ (mixed >> 14)) & _MASK32) / _TWO32

    def below(self, bound: int) -> int:
        return int(self.next() * bound)


class Step(NamedTuple):
    """What one frame produced: the two events dopamine responds to."""

    cleared: bool
    crashed: bool


@dataclass(frozen=True)
class Obstacle:
    """A cactus or a bird. Frozen so a recorded run is a snapshot, not a live reference."""

    x: float
    y: int
    width: int
    height: int

    @property
    def box(self) -> Box:
        return (self.x, float(self.y), self.width, self.height)


@dataclass
class Game:
    """One run, advanced a frame at a time."""

    rng: Mulberry32
    speed: float = SPEED_0
    frame: int = 0
    y: int = GROUND_Y
    ducking: bool = False
    jump_stage: float = 0.0
    obstacles: list[Obstacle] = field(default_factory=list)
    alive: bool = True
    spawn_timer: int = 0

    @classmethod
    def new(cls, seed: int) -> Game:
        rng = Mulberry32(seed)
        game = cls(rng=rng)
        game.spawn_timer = game._interval()
        return game

    @property
    def jumping(self) -> bool:
        return self.jump_stage > 0.0

    @property
    def box(self) -> Box:
        if self.ducking:
            return (float(RUNNER_X), float(self.y), DUCK_WIDTH, DUCK_HEIGHT)
        return (float(RUNNER_X), float(self.y), STAND_WIDTH, STAND_HEIGHT)

    def step(self, action: int) -> Step:
        if not self.alive:
            return Step(cleared=False, crashed=False)

        if not self.jumping:
            self._act(action)
        if self.jumping:
            self._advance_jump()

        speed = int(self.speed)
        cleared = self._advance_obstacles(speed)
        self._maybe_spawn()
        crashed = any(overlaps(self.box, obstacle.box) for obstacle in self.obstacles)
        if crashed:
            self.alive = False

        self.speed += SPEED_GAIN
        self.frame += 1
        return Step(cleared=cleared, crashed=crashed)

    def features(self) -> tuple[float, ...]:
        ahead = self._next_obstacle()
        if ahead is None:
            return (GAP_MAX, 0.0, 0.0, 0.0, self.speed)
        gap = min(max(ahead.x - RUNNER_X, 0.0), GAP_MAX)
        return (gap, float(ahead.y), float(ahead.height), float(ahead.width), self.speed)

    def glomeruli(self) -> npt.NDArray[np.float32]:
        return to_glomeruli(self.features())

    def _act(self, action: int) -> None:
        if action == JUMP:
            self.ducking = False
            self.y = GROUND_Y
            self.jump_stage = JUMP_START
        elif action == DUCK:
            self.ducking = True
            self.y = DUCK_Y
        else:
            self.ducking = False
            self.y = GROUND_Y

    def _advance_jump(self) -> None:
        stage = self.jump_stage
        self.y = int(GROUND_Y - ((-4 * stage * (stage - 1)) * JUMP_HEIGHT))
        self.jump_stage += JUMP_STEP
        if self.jump_stage > 1.0:
            self.jump_stage = 0.0
            self.y = GROUND_Y

    def _advance_obstacles(self, speed: int) -> bool:
        moved = [replace(obstacle, x=obstacle.x - speed) for obstacle in self.obstacles]
        cleared = any(
            before.x + before.width >= RUNNER_X > after.x + after.width
            for before, after in zip(self.obstacles, moved, strict=True)
        )
        self.obstacles = [obstacle for obstacle in moved if obstacle.x + obstacle.width >= 0]
        return cleared

    def _maybe_spawn(self) -> None:
        self.spawn_timer -= 1
        if self.spawn_timer > 0:
            return
        self.obstacles.append(self._cactus() if self.rng.next() < 0.5 else self._bird())
        self.spawn_timer = self._interval()

    def _interval(self) -> int:
        return SPAWN_MIN_FRAMES + self.rng.below(SPAWN_MAX_FRAMES - SPAWN_MIN_FRAMES)

    def _cactus(self) -> Obstacle:
        kind = self.rng.below(len(CACTUS_WIDTHS))
        return Obstacle(x=SPAWN_X, y=CACTUS_Y[kind], width=CACTUS_WIDTHS[kind], height=CACTUS_HEIGHTS[kind])

    def _bird(self) -> Obstacle:
        kind = self.rng.below(len(BIRD_Y))
        return Obstacle(x=SPAWN_X, y=BIRD_Y[kind], width=BIRD_WIDTH, height=BIRD_HEIGHT)

    def _next_obstacle(self) -> Obstacle | None:
        for obstacle in self.obstacles:
            if obstacle.x + obstacle.width >= RUNNER_X:
                return obstacle
        return None
