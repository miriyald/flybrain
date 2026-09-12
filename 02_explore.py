"""Step 2 - find the mushroom body inside 21,739 neurons.

This step exists so that step 3 never has to guess. Cell-type naming and brain-region
naming are conventions, not laws, so we print what the dataset actually contains and
build against that.

We are looking for four populations:
  PN    projection neurons  - carry odour identity in from the antennal lobe
  KC    Kenyon cells        - the sparse expansion layer
  MBON  output neurons      - read the KCs out into behaviour
  DAN   dopaminergic        - PPL1 = punishment, PAM = reward

Run:  python 02_explore.py
"""

from __future__ import annotations

import pandas as pd

from flylab import data

pd.set_option("display.width", 140)


def _summarise(neurons: pd.DataFrame, mask: pd.Series, label: str, top: int = 12) -> None:
    subset = neurons[mask]
    counts = subset["type"].value_counts()
    print(f"\n{label}: {len(subset):,} neurons across {len(counts)} types")
    for cell_type, count in counts.head(top).items():
        print(f"    {str(cell_type):<28} {count:>5}")
    if len(counts) > top:
        print(f"    ... and {len(counts) - top} more types")


def main() -> None:
    print(__doc__)
    neurons = data.load_neurons()
    types = neurons["type"].fillna("")

    _summarise(neurons, types.str.startswith("KC"), "KENYON CELLS (KC*)")
    _summarise(neurons, types.str.startswith("MBON"), "OUTPUT NEURONS (MBON*)", top=40)
    _summarise(neurons, types.str.startswith("PPL1"), "PUNISHMENT DOPAMINE (PPL1*)")
    _summarise(neurons, types.str.startswith("PAM"), "REWARD DOPAMINE (PAM*)")
    _summarise(neurons, types.str.contains("APL"), "APL INHIBITION")
    _summarise(neurons, types.str.contains("PN", na=False) & ~types.str.startswith("MBON"), "ANYTHING WITH 'PN'", top=20)

    print("\n--- MBON instances (do the names carry a compartment?) ---")
    mbons = neurons[types.str.startswith("MBON")]
    for _, row in mbons.head(12).iterrows():
        print(f"    {row['type']:<12} {row['instance']}")

    print("\n--- brain regions in the ROI edge table ---")
    rois = pd.read_csv(data.csv_paths()[data.ROI_CONNECTIONS], usecols=["roi"])["roi"].astype(str)
    unique_rois = sorted(rois.unique())
    print(f"    {len(unique_rois)} distinct ROIs")
    mb_rois = [r for r in unique_rois if any(tag in r for tag in ("MB", "CA", "PED", "a'", "b'", "aL", "bL", "gL"))]
    print(f"    mushroom-body-ish ROIs ({len(mb_rois)}): {mb_rois}")

    print("\nNext: python 03_build_circuit.py")


if __name__ == "__main__":
    main()
