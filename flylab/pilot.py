"""The same circuit, pointed at a control problem.

The digit classifier learns from a label present at the moment the Kenyon cells fire. A crash
is not like that: it arrives some thirty frames after the decision that caused it. The fly's
answer, and the one used here, is that the synapse carries a decaying tag which dopamine acts
on when it eventually arrives. So decisions are remembered as they are made, and a dopamine
event replays them into the same `model.depress` used for odours and digits, weighted by how
recent each one was.

Both dopamine systems appear here exactly as in the classifier. Punishment depresses the
approach readout for the action that was taken; reward depresses the avoid readout for it. The
verdict is the difference, so a score can move either way although neither readout ever grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from flylab import data, dino, model
from flylab.circuit import Circuit
from flylab.dino import ACTIONS

SPARSITY = 0.05
RATE = 0.1
RECOVERY = 0.0002
TRACE_DECAY = 0.97
TRACE_LEN = 3
MAX_FRAMES = 3000
MEMORY_PATH = data.DATA_DIR / "dino_memory.npz"


class Episode(NamedTuple):
    """How one run went."""

    frames: int
    cleared: int


@dataclass
class FlyPilot:
    """Glomeruli -> Kenyon cells -> one approach and one avoid output per action."""

    pn_to_kc: npt.NDArray[np.float32]
    approach: npt.NDArray[np.float32]
    avoid: npt.NDArray[np.float32]
    sparsity: float = SPARSITY
    rate: float = RATE
    recovery: float = RECOVERY
    trace_decay: float = TRACE_DECAY
    trace_len: int = TRACE_LEN
    trace: list[tuple[npt.NDArray[np.bool_], int]] = field(default_factory=list)

    @classmethod
    def from_circuit(
        cls,
        circuit: Circuit,
        *,
        sparsity: float = SPARSITY,
        rate: float = RATE,
        recovery: float = RECOVERY,
        trace_decay: float = TRACE_DECAY,
        trace_len: int = TRACE_LEN,
    ) -> FlyPilot:
        shape = (circuit.n_kc, ACTIONS)
        return cls(
            pn_to_kc=model.normalise_gain(circuit.pn_to_kc),
            approach=np.ones(shape, dtype=np.float32),
            avoid=np.ones(shape, dtype=np.float32),
            sparsity=sparsity,
            rate=rate,
            recovery=recovery,
            trace_decay=trace_decay,
            trace_len=trace_len,
        )

    def encode(self, activation: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        return model.kenyon_code(activation, self.pn_to_kc, self.sparsity)

    def scores(self, code: npt.NDArray[np.bool_]) -> npt.NDArray[np.float32]:
        active = code.astype(np.float32)
        difference: npt.NDArray[np.float32] = active @ self.approach - active @ self.avoid
        return difference

    def act(self, code: npt.NDArray[np.bool_], rng: np.random.Generator | None = None, temperature: float = 0.0) -> int:
        """Pick an action. Ties break by index, which is how the browser port breaks them too."""
        scores = self.scores(code)
        if rng is None or temperature <= 0.0:
            return int(np.argmax(scores))
        weights = np.exp((scores - scores.max()) / temperature)
        return int(rng.choice(ACTIONS, p=weights / weights.sum()))

    def remember(self, code: npt.NDArray[np.bool_], action: int) -> None:
        """Tag this decision, and let the oldest one fade.

        The bound is not a convenience. Measured on this game, an unbounded tag learned nothing
        at all: an obstacle becomes visible some fifty frames before it can be hit, so a crash
        blamed fifty decisions, forty-nine of which were made while the track was empty and the
        action taken could not have mattered. Keeping only the most recent few concentrates the
        blame where it belongs.
        """
        self.trace.append((code, action))
        if len(self.trace) > self.trace_len:
            del self.trace[: -self.trace_len]

    def punish(self) -> None:
        """PPL1 dopamine: the run ended badly, so weaken the approach to what was done."""
        for code, action, weight in self._discharge():
            self.approach = model.depress(self.approach, code, _only(action), self.rate * weight)

    def reward(self) -> None:
        """PAM dopamine: an obstacle was cleared, so weaken the avoidance of what was done."""
        for code, action, weight in self._discharge():
            self.avoid = model.depress(self.avoid, code, _only(action), self.rate * weight)

    def recover(self) -> None:
        """Weights relax back toward baseline.

        THIS IS A MODELLING ASSUMPTION, NOT CONNECTOME DATA - and it is the one piece of
        arithmetic this project adds to the fly's rule. Depression only ever removes weight.
        Over a few thousand samples that is fine, which is why the digit classifier needs
        nothing else, but a run of Dino makes hundreds of thousands of updates and every
        synapse would decay to zero, leaving the readout flat and the fly blind.

        Relaxing toward baseline turns each synapse into an exponential moving average of how
        often that action was punished in that state. Flies do forget - memories decay over
        hours, and extinction is measured - but the connectome records no such rate, so the
        number here is chosen, not derived.

        It is also the single most effective number in this task, and it wants to be small. At
        0.002 the circuit scored a median of 176 frames; at 0.0002 it scores 262, against 148
        for acting at random. Forgetting an order of magnitude faster than that erases what a
        run taught before the next run can build on it. The value was chosen on a separate set
        of validation seeds, never on the seeds the result is reported against.
        """
        self.approach += self.recovery * (1.0 - self.approach)
        self.avoid += self.recovery * (1.0 - self.avoid)

    def forget(self) -> None:
        self.trace.clear()

    def run_episode(
        self,
        game: dino.Game,
        *,
        rng: np.random.Generator | None = None,
        temperature: float = 0.0,
        learn: bool = True,
        reward: bool = True,
        max_frames: int = MAX_FRAMES,
    ) -> Episode:
        """Play one run, learning from it as it goes.

        Dopamine acts only at decision points. While the dino is airborne the arc is committed,
        nothing is tagged, and - this is the part that looks wrong - what those frames return is
        deliberately not learned from either, even though a jump lasts thirty-four frames and so
        most obstacles are both cleared and crashed into while off the ground.

        Crediting them was tried, because ignoring an outcome looks exactly like a bug. Measured
        over five training seeds it was much worse: a median of 150 frames against 267, with
        three of the five seeds collapsing to chance. The reason is that neighbouring gaps share
        between 77% and 99% of their Kenyon cell code, so punishing a jump that was mistimed by
        ten pixels also punishes the jump that would have worked. Crediting mid-air crashes
        roughly doubles the punishment landing on JUMP, it generalises onto the correct jumps,
        and the circuit stops jumping at all - the learned policies visibly switch to ducking.

        So jumping is learned by elimination rather than by praise: running into things is
        punished at the moment of impact, and jumping is what remains.
        """
        self.forget()
        cleared = 0
        while game.alive and game.frame < max_frames:
            if game.jumping:
                cleared += int(game.step(dino.RUN).cleared)
                continue

            code = self.encode(game.glomeruli())
            action = self.act(code, rng, temperature)
            if learn:
                self.remember(code, action)
            step = game.step(action)

            cleared += int(step.cleared)
            if learn and (step.crashed or step.cleared):
                if step.crashed:
                    self.punish()
                elif reward:
                    self.reward()
                else:
                    self.forget()
                self.recover()
        return Episode(frames=game.frame, cleared=cleared)

    def save(self, path: Path = MEMORY_PATH) -> Path:
        """The wiring lives in circuit.npz; this is what the fly learned about the game."""
        return data.save_arrays(
            path,
            approach=self.approach,
            avoid=self.avoid,
            sparsity=np.float64(self.sparsity),
            rate=np.float64(self.rate),
            recovery=np.float64(self.recovery),
            trace_decay=np.float64(self.trace_decay),
            trace_len=np.int64(self.trace_len),
        )

    @classmethod
    def load(cls, pn_to_kc: npt.NDArray[np.float32], path: Path = MEMORY_PATH) -> FlyPilot:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run 09_teach_the_dino.py first")
        with np.load(path, allow_pickle=False) as memory:
            return cls(
                pn_to_kc=pn_to_kc,
                approach=memory["approach"],
                avoid=memory["avoid"],
                sparsity=float(memory["sparsity"]),
                rate=float(memory["rate"]),
                recovery=float(memory["recovery"]),
                trace_decay=float(memory["trace_decay"]),
                trace_len=int(memory["trace_len"]),
            )

    def _discharge(self) -> list[tuple[npt.NDArray[np.bool_], int, float]]:
        """Recent decisions carry more of the blame, then the tag is spent."""
        newest_first = list(reversed(self.trace))
        self.trace = []
        return [(code, action, self.trace_decay**age) for age, (code, action) in enumerate(newest_first)]


def _only(action: int) -> npt.NDArray[np.bool_]:
    targets: npt.NDArray[np.bool_] = np.zeros(ACTIONS, dtype=np.bool_)
    targets[action] = True
    return targets
