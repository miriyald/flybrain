"""Step 10 - export the trained runner so a browser can play it, and prove the two agree.

Same shape as step 8: the connectome layer ships sparse, and the two readouts collapse into
one because scoring only ever reads `approach - avoid`.

The second output is the part worth caring about. A browser reimplementation that is merely
close is not good enough - if it picks even slightly different Kenyon cells at the winner
boundary it will make different decisions, and a game diverges from one different decision
forever. So this also writes a fixture: several runs, frame by frame, which
`node web/dino/verify.js` replays in JavaScript and must reproduce exactly.

Run:  python 10_export_the_dino.py
"""

from __future__ import annotations

import base64
import json

import numpy as np

from flylab import data, dino, export, model
from flylab.circuit import Circuit
from flylab.pilot import FlyPilot

EXPORT_PATH = data.DATA_DIR / "flydino_web.json"
WEB_PATH = data.PROJECT_ROOT / "web" / "dino" / "model.js"
SPRITE_SOURCE = data.PROJECT_ROOT / "web" / "dino" / "sprites.png"
SPRITE_PATH = data.PROJECT_ROOT / "web" / "dino" / "sprites.js"
CHECK_PATH = data.DATA_DIR / "dino_check.json"

CHECK_SEEDS = (1, 2, 3, 7, 4242)
CHECK_FRAMES = 1500
SAMPLE_EVERY = 8


def record(pilot: FlyPilot, seed: int) -> dict[str, object]:
    """Play one run exactly as the browser will, keeping enough to catch any divergence."""
    game = dino.Game.new(seed)
    frames: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    decisions = 0

    while game.frame < CHECK_FRAMES and game.alive:
        if game.jumping:
            game.step(dino.RUN)
            frames.append({"action": -1, "y": game.y, "alive": game.alive, "cleared": False, "crashed": False})
            continue

        code = pilot.encode(game.glomeruli())
        scores = pilot.scores(code)
        action = pilot.act(code)
        if decisions % SAMPLE_EVERY == 0:
            samples.append(
                {
                    "frame": game.frame,
                    "code": [int(i) for i in np.flatnonzero(code)],
                    "scores": [float(f"{v:.9g}") for v in scores],
                    "action": action,
                }
            )
        decisions += 1
        step = game.step(action)
        frames.append({"action": action, "y": game.y, "alive": game.alive, "cleared": step.cleared, "crashed": step.crashed})

    return {"seed": seed, "frames": frames, "samples": samples}


def write_sprites() -> int:
    """Ship the sprite sheet as a data URI, for the same reason the model ships as a script.

    A page opened from the filesystem cannot fetch anything - `file://` is not in the list of
    schemes cross-origin requests are allowed for - so `loadImage("sprites.png")` fails and p5
    sits on its loading screen forever. A `data:` URI is on that list.
    """
    encoded = base64.b64encode(SPRITE_SOURCE.read_bytes()).decode("ascii")
    SPRITE_PATH.write_text(f'window.FLYDINO_SPRITES="data:image/png;base64,{encoded}";\n', encoding="utf-8")
    return SPRITE_PATH.stat().st_size


def main() -> None:
    print(__doc__)
    circuit = Circuit.load()
    weights = model.normalise_gain(circuit.pn_to_kc)
    pilot = FlyPilot.load(weights)

    net = pilot.approach - pilot.avoid
    bundle = {
        "provenance": "Janelia hemibrain v1.2, CC-BY. Glomerulus->Kenyon cell weights are measured synapse counts.",
        "glomeruli": int(weights.shape[0]),
        "kenyonCells": int(weights.shape[1]),
        "actions": int(net.shape[1]),
        "sparsity": float(pilot.sparsity),
        "k": int(round(pilot.sparsity * weights.shape[1])),
        "featureRanges": [list(pair) for pair in dino.FEATURE_RANGES],
        "tuningPerFeature": int(dino.TUNING_PER_FEATURE),
        "pnToKc": export.compressed_rows(weights),
        "readout": export.rounded(net.ravel()),
    }

    payload = json.dumps(bundle, separators=(",", ":"))
    export.write_bundle(payload, EXPORT_PATH, WEB_PATH, "FLYDINO_MODEL")

    sprite_bytes = write_sprites()
    runs = [record(pilot, seed) for seed in CHECK_SEEDS]
    CHECK_PATH.write_text(json.dumps({"runs": runs}, separators=(",", ":")), encoding="utf-8")

    decisions = sum(len(run["samples"]) for run in runs)  # type: ignore[arg-type]
    played = sum(len(run["frames"]) for run in runs)  # type: ignore[arg-type]
    print(f"  glomerulus -> Kenyon cell : {weights.shape}, {len(bundle['pnToKc']['indices']):,} nonzero")
    print(f"  readout (approach - avoid): {net.shape}")
    print(f"  top {bundle['k']} of {bundle['kenyonCells']} Kenyon cells fire per decision")
    print(f"\nwrote {EXPORT_PATH}  ({EXPORT_PATH.stat().st_size / 1e6:.2f} MB)")
    print(f"wrote {WEB_PATH}  ({WEB_PATH.stat().st_size / 1e6:.2f} MB)")
    print(f"wrote {SPRITE_PATH}  ({sprite_bytes / 1e3:.1f} kB, embedded so a file:// page can load it)")
    print(f"wrote {CHECK_PATH}  ({played:,} frames, {decisions:,} sampled decisions)")
    print("\nNext: node web/dino/verify.js, which must report zero mismatches.")


if __name__ == "__main__":
    main()
