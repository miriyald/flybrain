"""Compartment parsing and cell-type selection."""

from __future__ import annotations

import pytest

from flylab.circuit import UNIGLOMERULAR_PN, compartments


@pytest.mark.parametrize(
    ("instance", "expected"),
    [
        ("MBON11(y1pedc>a/B)_R", {"y1", "pedc"}),
        ("MBON12(y2a'1)_R", {"y2", "a'1"}),
        ("MBON24(B2y5)_R", {"B2", "y5"}),
        ("MBON14(a3)_R", {"a3"}),
        ("MBON18(a2sc)_R", {"a2"}),
        ("MBON22(calyx)_R", {"calyx"}),
        ("PPL101(y1pedc)_R", {"y1", "pedc"}),
        ("PAM01_a(y5)_R", {"y5"}),
        ("PPL107_R", set()),
    ],
)
def test_compartments_read_from_instance(instance: str, expected: set[str]) -> None:
    assert compartments(instance) == expected


@pytest.mark.parametrize(
    ("instance", "expected"),
    [
        ("PAM07(y4<y1y2)_R", {"y4"}),
        ("MBON05(y4>y1y2)_R", {"y4"}),
        ("MBON06(B1>a)_R", {"B1"}),
    ],
)
def test_arrow_marks_the_other_end_in_both_directions(instance: str, expected: set[str]) -> None:
    """`<` and `>` both point away from the neuron's own compartment.

    Reading `PAM07(y4<y1y2)` as belonging to y1/y2 makes gamma1 and gamma2 look like reward
    compartments, which silently flips the valence of the two best-documented output
    neurons in the literature.
    """
    assert compartments(instance) == expected


@pytest.mark.parametrize("cell_type", ["DA1_lPN", "DL2d_adPN", "VA5_lPN", "VC5_lvPN", "DM1_lPN"])
def test_uniglomerular_pns_are_selected(cell_type: str) -> None:
    assert UNIGLOMERULAR_PN.match(cell_type)


@pytest.mark.parametrize("cell_type", ["M_vPNml74", "WEDPN8C", "VP1m+VP2_lvPN2", "KCg-m", "MBON11"])
def test_multiglomerular_and_non_olfactory_are_rejected(cell_type: str) -> None:
    assert not UNIGLOMERULAR_PN.match(cell_type)


def test_glomerulus_name_is_captured() -> None:
    match = UNIGLOMERULAR_PN.match("DL2d_adPN")
    assert match is not None
    assert match.group("glomerulus") == "DL2d"
