from __future__ import annotations

from pathlib import Path
import importlib.util

import pytest

from evb.evb_core import diabatic_gap
from kemp_evb.config import load_config, validate_config
from kemp_evb.plumed import PlumedSettings, PlumedUnavailableError, attach_plumed_force, load_plumed_script


def test_evb_package_alias_imports():
    assert diabatic_gap(5.0, 3.0, 1.0) == 1.0


def test_modern_config_rejects_geometry_dependent_h12(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
states:
  state1: {topology: a.prmtop, coordinates: a.inpcrd}
  state2: {topology: b.prmtop, coordinates: b.inpcrd}
evb:
  coupling_model:
    model: geometry_dependent
    parameters:
      delta_alpha_kj_mol: 0.0
      h12_kj_mol: 1.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="constant EVB coupling"):
        load_config(config_path)


def test_validate_config_reports_missing_window_definitions(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
states:
  state1: {topology: a.prmtop, coordinates: a.inpcrd}
  state2: {topology: b.prmtop, coordinates: b.inpcrd}
evb:
  coupling_model:
    model: constant
    parameters:
      delta_alpha_kj_mol: 0.0
      h12_kj_mol: 1.0
sampling:
  mode: mapping
""",
        encoding="utf-8",
    )
    errors = validate_config(load_config(config_path))
    assert any("lambda_values" in error for error in errors)


def test_plumed_script_loader_adds_restart_and_colvar():
    script = load_plumed_script(PlumedSettings(enabled=True, script="d1: DISTANCE ATOMS=1,2", restart=True))
    assert script.startswith("RESTART")
    assert "FILE=COLVAR" in script


def test_plumed_optional_import_behavior():
    pytest.importorskip("openmm")
    if importlib.util.find_spec("openmmplumed") is not None:
        pytest.skip("openmm-plumed is installed; unavailable-path test does not apply.")
    import openmm

    system = openmm.System()
    with pytest.raises(PlumedUnavailableError):
        attach_plumed_force(system, PlumedSettings(enabled=True, script="d1: DISTANCE ATOMS=1,2"))


def test_make_toy_template_writes_openmm_bundles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("openmm")
    from kemp_evb.cli import run_make_template

    monkeypatch.chdir(tmp_path)
    run_make_template("toy")

    assert (tmp_path / "examples" / "toy_evb.yaml").is_file()
    assert (tmp_path / "examples" / "toy_state1" / "system.xml").is_file()
    assert (tmp_path / "examples" / "toy_state1" / "coordinates.pdb").is_file()
    assert (tmp_path / "examples" / "toy_state2" / "system.xml").is_file()
    assert (tmp_path / "examples" / "toy_state2" / "coordinates.pdb").is_file()
