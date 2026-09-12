"""Step 8 - export the trained circuit so a browser can run it.

Everything the demo needs is small. The connectome layer is 10% dense, so it ships as a
sparse matrix, and the two readouts collapse into one: scoring only ever uses
`approach - avoid`, so the difference is all that has to travel.

Nothing is retrained here. This is the same circuit from step 6, written out in a form
JavaScript can read.

Run:  python 08_export_for_web.py
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.datasets import load_digits

from flylab import data, model
from flylab.circuit import Circuit
from flylab.classifier import FlyClassifier

EXPORT_PATH = data.DATA_DIR / "flybrain_web.json"
WEB_PATH = data.PROJECT_ROOT / "web" / "model.js"
PRECISION = 9


def _round(values: np.ndarray) -> list[float]:
    return [float(f"{value:.{PRECISION}g}") for value in values]


def main() -> None:
    print(__doc__)
    circuit = Circuit.load()
    classifier = FlyClassifier.load(circuit)
    weights = model.normalise_gain(circuit.pn_to_kc)

    rows, cols = np.nonzero(weights)
    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]
    indptr = np.searchsorted(rows, np.arange(weights.shape[0] + 1))

    net = classifier.approach - classifier.avoid
    images, labels = load_digits(return_X_y=True)
    examples = {int(digit): [int(v) for v in images[list(labels).index(digit)]] for digit in range(10)}

    bundle = {
        "provenance": "Janelia hemibrain v1.2, CC-BY. Glomerulus->Kenyon cell weights are measured synapse counts.",
        "glomeruli": int(weights.shape[0]),
        "kenyonCells": int(weights.shape[1]),
        "classes": int(net.shape[1]),
        "sparsity": float(classifier.sparsity),
        "k": int(round(classifier.sparsity * weights.shape[1])),
        "pixels": [int(p) for p in classifier.pixels],
        "pnToKc": {"indptr": [int(v) for v in indptr], "indices": [int(v) for v in cols], "data": _round(weights[rows, cols])},
        "readout": _round(net.ravel()),
        "examples": examples,
    }

    payload = json.dumps(bundle, separators=(",", ":"))
    EXPORT_PATH.write_text(payload, encoding="utf-8")

    # A browser page cannot fetch the JSON - the artifact CSP blocks XHR - so the same
    # bundle also ships as a script that simply assigns it.
    WEB_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEB_PATH.write_text(f"window.FLYBRAIN_MODEL={payload};\n", encoding="utf-8")

    print(f"  glomerulus -> Kenyon cell : {weights.shape}, {len(cols):,} nonzero ({(weights > 0).mean():.1%} dense)")
    print(f"  readout (approach - avoid): {net.shape}")
    print(f"  top {bundle['k']} of {bundle['kenyonCells']} Kenyon cells fire per digit")
    print(f"\nwrote {EXPORT_PATH}  ({EXPORT_PATH.stat().st_size / 1e6:.2f} MB)")
    print(f"wrote {WEB_PATH}  ({WEB_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
