"""Turning a trained circuit into something a browser can read.

Both demos ship the same two things: the connectome layer, which is about a tenth dense and so
travels as a sparse matrix, and a readout, which collapses to `approach - avoid` because the
difference is all that scoring ever uses.

The numeric formatting lives here rather than in either exporter so the two cannot drift. A
browser that rounds the weights differently from the Python that trained them will pick
slightly different Kenyon cells at the winner boundary, which is exactly the failure the
parity checks exist to catch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

PRECISION = 9


def rounded(values: npt.NDArray[np.floating[Any]]) -> list[float]:
    """Nine significant digits - far finer than any real difference in synaptic weight."""
    return [float(f"{value:.{PRECISION}g}") for value in values]


def compressed_rows(weights: npt.NDArray[np.float32]) -> dict[str, list[Any]]:
    """The gain-normalised connectome as compressed sparse rows, one row per glomerulus."""
    rows, cols = np.nonzero(weights)
    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]
    indptr = np.searchsorted(rows, np.arange(weights.shape[0] + 1))
    return {"indptr": [int(v) for v in indptr], "indices": [int(v) for v in cols], "data": rounded(weights[rows, cols])}


def write_bundle(payload: str, json_path: Path, web_path: Path, global_name: str) -> None:
    """Write the bundle twice: once as JSON, once as a script that assigns it.

    A page opened from the filesystem cannot fetch the JSON, so the script form is what the
    demos actually load. The JSON is what the Node parity checks read.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(payload, encoding="utf-8")
    web_path.parent.mkdir(parents=True, exist_ok=True)
    web_path.write_text(f"window.{global_name}={payload};\n", encoding="utf-8")
