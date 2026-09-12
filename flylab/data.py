"""Fetch and load the Janelia hemibrain v1.2 traced adjacency tables.

The tarball is served anonymously over HTTPS under CC-BY. It contains plain CSVs:
a typed neuron table and two weighted edge lists (total, and split by brain region).
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

HEMIBRAIN_URL = "https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz"
HEMIBRAIN_BYTES = 45_872_577

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TARBALL_PATH = DATA_DIR / "exported-traced-adjacencies-v1.2.tar.gz"

NEURONS = "traced-neurons.csv"
CONNECTIONS = "traced-total-connections.csv"
ROI_CONNECTIONS = "traced-roi-connections.csv"
CSV_NAMES = (NEURONS, CONNECTIONS, ROI_CONNECTIONS)

_DOWNLOAD_CHUNK = 1 << 20


def download(url: str = HEMIBRAIN_URL, dest: Path = TARBALL_PATH, expected_bytes: int = HEMIBRAIN_BYTES) -> Path:
    """Stream the tarball to `dest`, skipping the download if it is already complete."""
    if dest.exists() and dest.stat().st_size == expected_bytes:
        logger.info("tarball already present", extra={"path": str(dest), "bytes": expected_bytes})
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    logger.info("downloading hemibrain tarball", extra={"url": url, "expected_bytes": expected_bytes})

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK):
                handle.write(chunk)

    size = partial.stat().st_size
    if size != expected_bytes:
        partial.unlink()
        raise RuntimeError(f"downloaded {size} bytes, expected {expected_bytes} - refusing to use a truncated archive")

    partial.replace(dest)
    return dest


def extract(tarball: Path = TARBALL_PATH, dest: Path = DATA_DIR) -> dict[str, Path]:
    """Extract the archive and return the three CSV paths keyed by filename."""
    with tarfile.open(tarball, "r:gz") as archive:
        archive.extractall(dest, filter="data")
    return csv_paths(dest)


def csv_paths(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Locate the three CSVs anywhere under `data_dir`, raising if any is missing."""
    found: dict[str, Path] = {}
    for name in CSV_NAMES:
        matches = sorted(data_dir.rglob(name))
        if not matches:
            raise FileNotFoundError(f"{name} not found under {data_dir} - run 01_get_the_data.py first")
        found[name] = matches[0]
    return found


def load_neurons(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Neuron table: bodyId, instance, type."""
    return pd.read_csv(csv_paths(data_dir)[NEURONS])


def load_connections(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Whole-brain weighted edge list: bodyId_pre, bodyId_post, weight (synapse count)."""
    return pd.read_csv(csv_paths(data_dir)[CONNECTIONS])


def load_roi_connections(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Per-region weighted edge list: bodyId_pre, bodyId_post, roi, weight."""
    return pd.read_csv(csv_paths(data_dir)[ROI_CONNECTIONS])
