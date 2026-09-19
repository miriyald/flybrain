"""Step 9 - teach the same circuit to play Dino, using nothing but crashes.

Steps 5 and 6 taught this circuit with a signal that arrived at the same moment as the Kenyon
cell code: a shock paired with an odour, a label attached to a digit. Control is not like that.
Nobody says which key to press. The only feedback is a crash, and by the time it arrives the
decision that caused it is thirty frames in the past.

The fly's own answer is that the synapse keeps a fading tag of what fired recently, so dopamine
arriving late still finds something to act on. That is all this adds: decisions are remembered
as they are made, and an outcome replays the last few of them into the same `model.depress`
used for odours and digits. Clearing an obstacle is reward and weakens the avoidance of what
was just done. Crashing is punishment and weakens the approach to it. Nothing is strengthened.

One number here is not in the connectome: a slow relaxation of every weight back toward
baseline. Without it a run of this length drives every synapse to zero and the readout goes
flat. See `FlyPilot.recover`.

The settings it trains with are `flylab.pilot`'s own defaults. They were chosen by scoring on
seeds 8000-8099 and are reported below on seeds 9000-9099, which the choice never saw - the
recovery rate alone is worth a hundred frames, and is exactly the kind of number it would be
easy to fool yourself about.

Run:  python 09_teach_the_dino.py
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from flylab import dino
from flylab.circuit import Circuit
from flylab.pilot import Episode, FlyPilot

SEED = 0
EPISODES = 4000
MAX_FRAMES = 3000
TEMPERATURE = 2.0
HELD_OUT = range(9000, 9100)

EXPERT_WINDOW = (150.0, 300.0)
NAMES = {dino.RUN: "run ", dino.JUMP: "JUMP", dino.DUCK: "duck"}
GAPS = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 500.0)
SHAPES = (
    ("short cactus", 470.0, 66.0, 30.0),
    ("tall cactus ", 444.0, 96.0, 96.0),
    ("low bird    ", 435.0, 40.0, 84.0),
    ("middle bird ", 480.0, 40.0, 84.0),
    ("high bird   ", 370.0, 40.0, 84.0),
)


def play(pilot: FlyPilot, seed: int, rng: np.random.Generator | None, temperature: float, *, learn: bool, reward: bool) -> Episode:
    return pilot.run_episode(dino.Game.new(seed), rng=rng, temperature=temperature, learn=learn, reward=reward)


def fixed(seed: int, action: int) -> int:
    game = dino.Game.new(seed)
    while game.alive and game.frame < MAX_FRAMES:
        game.step(dino.RUN if game.jumping else action)
    return game.frame


def shuffled(seed: int, rng: np.random.Generator) -> int:
    game = dino.Game.new(seed)
    while game.alive and game.frame < MAX_FRAMES:
        game.step(dino.RUN if game.jumping else int(rng.integers(dino.ACTIONS)))
    return game.frame


def scripted(seed: int) -> int:
    """The ceiling: what a hand-written policy achieves, so the learned one has a yardstick."""
    game = dino.Game.new(seed)
    while game.alive and game.frame < MAX_FRAMES:
        if game.jumping:
            game.step(dino.RUN)
            continue
        gap, top, _height, _width, _speed = game.features()
        if top == 435.0:
            game.step(dino.DUCK if gap < 200 else dino.RUN)
        elif top in (0.0, 370.0):
            game.step(dino.RUN)
        else:
            game.step(dino.JUMP if EXPERT_WINDOW[0] <= gap <= EXPERT_WINDOW[1] else dino.RUN)
    return game.frame


def train(circuit: Circuit, *, reward: bool, episodes: int = EPISODES) -> tuple[FlyPilot, list[int]]:
    """The tuned settings are `flylab.pilot`'s own defaults, so there is one place to change them."""
    rng = np.random.default_rng(SEED)
    pilot = FlyPilot.from_circuit(circuit)
    curve = [play(pilot, episode, rng, TEMPERATURE, learn=True, reward=reward).frames for episode in range(episodes)]
    return pilot, curve


def evaluate(pilot: FlyPilot) -> list[int]:
    return [play(pilot, seed, None, 0.0, learn=False, reward=False).frames for seed in HELD_OUT]


def report(label: str, scores: list[int]) -> None:
    array = np.asarray(scores)
    print(f"  {label:<26} median {int(np.median(array)):>5}   mean {array.mean():>7.1f}   best {array.max():>5}")


def separability(pilot: FlyPilot) -> None:
    """The gate everything else depends on: can the code tell these situations apart?"""

    def code(gap: float, top: float, height: float, width: float) -> npt.NDArray[np.bool_]:
        return pilot.encode(dino.to_glomeruli((gap, top, height, width, 15.0)))

    def overlap(first: npt.NDArray[np.bool_], second: npt.NDArray[np.bool_]) -> float:
        return float((first & second).sum()) / float(first.sum())

    near_cactus = code(150.0, 470.0, 66.0, 30.0)
    print(f"  a cactus 150 away vs the same cactus 160 away : {overlap(near_cactus, code(160.0, 470.0, 66.0, 30.0)):.0%}")
    print(f"  a cactus 150 away vs a cactus 400 away        : {overlap(near_cactus, code(400.0, 470.0, 66.0, 30.0)):.0%}")
    print(f"  a cactus 150 away vs a low bird 150 away      : {overlap(near_cactus, code(150.0, 435.0, 40.0, 84.0)):.0%}")
    print(f"  a cactus 150 away vs an empty track           : {overlap(near_cactus, code(500.0, 0.0, 0.0, 0.0)):.0%}")
    print(f"  (chance is {pilot.sparsity:.0%} - the fraction of cells firing at all)")


def policy(pilot: FlyPilot) -> None:
    print("                 " + "".join(f"{int(gap):>5}" for gap in GAPS))
    for label, top, height, width in SHAPES:
        chosen = [NAMES[pilot.act(pilot.encode(dino.to_glomeruli((gap, top, height, width, 15.0))))] for gap in GAPS]
        print(f"  {label} " + " ".join(chosen))


def main() -> None:
    print(__doc__)
    circuit = Circuit.load()
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("What the circuit receives\n")
    untaught = FlyPilot.from_circuit(circuit)
    separability(untaught)

    print("\n" + "=" * 78)
    print(f"Learning from {EPISODES:,} runs\n")
    pilot, curve = train(circuit, reward=True)
    block = EPISODES // 6
    for start in range(0, EPISODES, block):
        window = np.asarray(curve[start : start + block])
        print(f"  runs {start:>5}-{start + block:<5} median frames survived {int(np.median(window)):>5}")

    print("\n" + "=" * 78)
    print("Against the alternatives, on 100 runs it never trained on\n")
    report("do nothing", [fixed(seed, dino.RUN) for seed in HELD_OUT])
    report("always jump", [fixed(seed, dino.JUMP) for seed in HELD_OUT])
    report("act at random", [shuffled(seed, rng) for seed in HELD_OUT])
    taught = evaluate(pilot)
    report("the taught circuit", taught)
    report("a hand-written policy", [scripted(seed) for seed in HELD_OUT])

    print("\n" + "=" * 78)
    print("Does the second dopamine system earn its keep?\n")
    punished_only, _ = train(circuit, reward=False)
    report("punishment alone (PPL1)", evaluate(punished_only))
    report("punishment and reward", taught)

    print("\n" + "=" * 78)
    print("What it decided to do, by how far away the obstacle is\n")
    policy(pilot)

    print("\n" + "=" * 78)
    memory = pilot.save()
    print(f"\nwrote {memory}")
    print("Next: python 10_export_the_dino.py, to put this circuit in a browser.")


if __name__ == "__main__":
    main()
