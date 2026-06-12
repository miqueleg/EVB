from __future__ import annotations

import json
from pathlib import Path

from kemp_evb.config import load_config
from kemp_evb.hg317_qregion import make_hg317_qregion_candidates
from kemp_evb.hg317_stability import run_hg317_qregion_stability_check


def _toy_q_region_config(tmp_path: Path) -> Path:
    config_path = Path("examples/toy_evb.yaml")
    summary = make_hg317_qregion_candidates(load_config(config_path), config_path, tmp_path / "candidates", include_reaction_atoms=True)
    return Path(summary["candidate_dir"]) / "local_pme_q_atoms_cutoff_0.8.yaml"


def test_stability_quick_mode_writes_outputs(tmp_path: Path):
    config_path = _toy_q_region_config(tmp_path)

    summary = run_hg317_qregion_stability_check(config_path, tmp_path / "stability", "CPU", plain_steps=5, table_metad_steps=5, quick=True, report_interval=1)

    assert len(summary["runs"]) == 2
    assert summary["runs"][0]["completed_steps"] == 5
    assert summary["runs"][1]["completed_steps"] == 5
    assert (tmp_path / "stability" / "plain_md" / "observables.csv").exists()
    assert (tmp_path / "stability" / "table_metad" / "observables.csv").exists()
    assert (tmp_path / "stability" / "plain_md" / "final.pdb").exists()
    assert (tmp_path / "stability" / "table_metad" / "stability_summary.json").exists()


def test_selected_candidate_path_resolves_from_validation_summary(tmp_path: Path):
    config_path = _toy_q_region_config(tmp_path)
    root = tmp_path / "calibrated"
    selected_dir = root / "validation"
    selected_dir.mkdir(parents=True)
    (selected_dir / "selected_candidate.json").write_text(json.dumps({"fitted_config": str(config_path)}), encoding="utf-8")

    summary = run_hg317_qregion_stability_check(root / "selected_candidate" / "fitted_config.yaml", tmp_path / "stability", "CPU", plain_steps=2, table_metad_steps=0, quick=True, report_interval=1)

    assert summary["runs"][0]["completed_steps"] == 2
    assert summary["runs"][0]["stable"] is True


def test_stability_module_does_not_add_opes():
    source = Path("src/kemp_evb/hg317_stability.py").read_text(encoding="utf-8").lower()

    assert "opes_implemented" in source
    assert "opes_metad" not in source
