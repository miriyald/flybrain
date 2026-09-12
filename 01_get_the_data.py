"""Step 1 - get a real connectome onto your laptop.

What you are downloading is NOT a brain scan. It is a wiring diagram: a fly brain was
sliced into ~8nm sections, imaged by electron microscope (~100 TB of images), traced by
AI segmentation and proofread by humans. The result was reduced to three plain CSVs -
which neurons exist, and how many synapses connect each pair. That is 46 MB.

Run:  python 01_get_the_data.py
"""

from __future__ import annotations

import logging

from flylab import data

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    print(__doc__)
    print(f"source : {data.HEMIBRAIN_URL}")
    print("licence: CC-BY (Janelia hemibrain v1.2)\n")

    tarball = data.download()
    print(f"\narchive: {tarball}  ({tarball.stat().st_size:,} bytes)")

    paths = data.extract(tarball)
    print("\nextracted:")
    for name, path in paths.items():
        print(f"  {name:<32} {path.stat().st_size / 1e6:>8.1f} MB")

    neurons = data.load_neurons()
    connections = data.load_connections()

    print("\n--- what is actually in here ---")
    print(f"neurons     : {len(neurons):,}")
    print(f"connections : {len(connections):,}")
    print(f"synapses    : {int(connections['weight'].sum()):,}")

    print("\nneuron table columns:", list(neurons.columns))
    print(neurons.head(3).to_string(index=False))

    print("\nconnection table columns:", list(connections.columns))
    print(connections.head(3).to_string(index=False))

    print(
        "\nEach connection row is one directed edge, and `weight` is the number of\n"
        "synapses. That is the whole dataset: a weighted directed graph.\n"
        "\nNext: python 02_explore.py"
    )


if __name__ == "__main__":
    main()
