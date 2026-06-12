from __future__ import annotations

from pathlib import Path

import pytest

from kemp_evb.reference_profile import KJ_PER_HARTREE, KJ_PER_KCAL, canonical_label, load_reference_profile


def test_load_generic_kcal_profile(tmp_path: Path):
    path = tmp_path / "profile.yaml"
    path.write_text("""reference_profile:
  method_label: DFT
  energy_unit: kcal/mol
  zero: RC
  points:
    RC: {relative_energy: 0.0}
    TS: {relative_energy: 18.14}
    PROD: {relative_energy: -37.35}
""", encoding="utf-8")

    profile = load_reference_profile(path)

    assert profile.method_label == "DFT"
    assert profile.points["transition_state"].relative_kj_mol == pytest.approx(18.14 * KJ_PER_KCAL)
    assert profile.reaction_free_energy_kj_mol() == pytest.approx(-37.35 * KJ_PER_KCAL)


def test_load_hartree_absolute_profile(tmp_path: Path):
    path = tmp_path / "profile.yaml"
    path.write_text("""reference_profile:
  method_label: user
  energy_unit: hartree
  zero: reactant
  points:
    reactant: {absolute_energy: -10.0}
    transition_state: {absolute_energy: -9.99}
""", encoding="utf-8")

    profile = load_reference_profile(path)

    assert profile.barrier_kj_mol() == pytest.approx(0.01 * KJ_PER_HARTREE)


def test_aliases_are_generic():
    assert canonical_label("RC") == "reactant"
    assert canonical_label("TS") == "transition_state"
    assert canonical_label("PROD") == "product"
