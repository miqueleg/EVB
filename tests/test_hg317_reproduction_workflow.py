
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from kemp_evb.config import load_config
from kemp_evb.hg317_reproduction import (
    SELECTED_CANDIDATE,
    SELECTED_DELTA_ALPHA,
    SELECTED_H12,
    analyze_metad_replicas,
    calculate_barrier_from_pmf,
    compute_speedup_summary,
    generate_reproduction_configs,
    run_reproduction_workflow,
    settings_for_mode,
    write_reproduction_scripts,
)


def _candidate_yaml(path: Path) -> Path:
    payload = {
        "project": {"output_dir": str(path.parent / "run")},
        "state1": {"format": "openmm", "system": "state1.xml", "topology": "state1.pdb", "coordinates": "state1.pdb"},
        "state2": {"format": "openmm", "system": "state2.xml", "topology": "state2.pdb", "coordinates": "state2.pdb"},
        "evb": {
            "representation": "q_region",
            "coupling_model": {
                "model": "constant",
                "parameters": {
                    "delta_alpha_kj_mol": SELECTED_DELTA_ALPHA,
                    "h12_kj_mol": SELECTED_H12,
                },
            },
            "q_region": {
                "enabled": True,
                "selected_candidate": SELECTED_CANDIDATE,
                "q_atoms": [0, 1],
                "nonbonded": {"mode": "local_pme_approx", "local_approx_enabled": True},
            },
        },
        "simulation": {"platform": "CPU", "steps": 1},
        "sampling": {"mode": "none"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_workflow_config_generation(tmp_path: Path):
    selected = _candidate_yaml(tmp_path / "selected" / "fitted_config.yaml")
    settings = settings_for_mode("quick")

    summary = generate_reproduction_configs("examples/toy_evb.yaml", selected, tmp_path / "out", settings)

    umbrella = yaml.safe_load(Path(summary["qregion_gap_umbrella"]).read_text(encoding="utf-8"))
    metad = yaml.safe_load(Path(summary["qregion_gap_table_metad"]).read_text(encoding="utf-8"))
    assert umbrella["evb"]["representation"] == "q_region"
    assert umbrella["evb"]["q_region"]["selected_candidate"] == SELECTED_CANDIDATE
    assert umbrella["evb"]["coupling_model"]["parameters"]["delta_alpha_kj_mol"] == SELECTED_DELTA_ALPHA
    assert metad["sampling"]["mode"] == "gap_table_metadynamics"
    assert "native_gap_bias" in metad["sampling"]
    assert summary["h12_kj_mol"] == SELECTED_H12


def test_barrier_calculation_synthetic_pmf():
    summary = calculate_barrier_from_pmf([-10.0, 0.0, 10.0], [0.0, 5.0, 1.0])

    assert summary["barrier_from_left_kcal"] == 5.0
    assert summary["barrier_from_right_kcal"] == 4.0
    assert summary["deltaG_reaction_kcal"] == 1.0
    assert summary["pmf_at_gap0_kcal"] == 5.0


def test_metad_replica_analysis(tmp_path: Path):
    colvars = []
    for idx, offset in enumerate([0.0, 4.184], start=1):
        path = tmp_path / f"rep{idx:02d}" / "colvar.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("step,shifted_gap\n")
            for i in range(20):
                handle.write(f"{i},{offset + (i - 10) * 4.184}\n")
        colvars.append(path)

    result = analyze_metad_replicas(colvars, tmp_path / "analysis")

    assert len(result["replica_curves"]) == 2
    assert Path(result["mean_fel_csv"]).exists()
    assert (tmp_path / "analysis" / "qregion_gap_metad_hist_fel.png").exists()
    assert result["barrier_summary"]["pmf_at_gap0_kcal"] is not None


def test_speedup_summary_from_synthetic_timings(tmp_path: Path):
    umbrella = tmp_path / "umbrella" / "windows" / "w000" / "plain_md"
    metad = tmp_path / "metad" / "rep01" / "stability" / "table_metad"
    umbrella.mkdir(parents=True)
    metad.mkdir(parents=True)
    (umbrella / "stability_summary.json").write_text(json.dumps({"steps_per_s": 100.0, "ns_per_day": 2.16, "duplicated_full_nonbonded": False}), encoding="utf-8")
    (metad / "stability_summary.json").write_text(json.dumps({"steps_per_s": 200.0, "ns_per_day": 4.32, "pme_approximation": True}), encoding="utf-8")

    summary = compute_speedup_summary(tmp_path, {"available": False})

    assert summary["methods"][0]["steps_per_s"] == 100.0
    assert summary["methods"][1]["steps_per_s"] == 200.0
    assert summary["legacy_reference_available"] is False


def test_run_script_generation(tmp_path: Path):
    scripts = write_reproduction_scripts(tmp_path / "out", tmp_path / "templates")

    for name, entries in scripts.items():
        path = Path(entries["template_copy"])
        assert path.exists(), name
        assert os.access(path, os.X_OK), name
        text = path.read_text(encoding="utf-8")
        if name.startswith("run_") or name.startswith("resume"):
            assert "hg317-qregion-reproduce" in text


def test_quick_mode_dry_run_on_toy_config(tmp_path: Path, monkeypatch):
    selected = _candidate_yaml(tmp_path / "selected" / "fitted_config.yaml")
    monkeypatch.setattr("kemp_evb.hg317_reproduction.find_selected_qregion_config", lambda: selected)
    config = load_config("examples/toy_evb.yaml")

    result = run_reproduction_workflow(
        config,
        "examples/toy_evb.yaml",
        "examples/hg317_gxtb_reference_profile.yaml",
        tmp_path / "workflow",
        platform="CPU",
        mode="quick",
        write_run_scripts_only=True,
    )

    assert result["status"] == "scripts_only"
    assert Path(result["configs"]["qregion_gap_umbrella"]).exists()
    assert Path(result["run_scripts"]["run_all_qregion_reproduction.sh"]["template_copy"]).exists()


def test_no_qregion_opes_command_added():
    source = Path("src/kemp_evb/hg317_reproduction.py").read_text(encoding="utf-8").lower()
    script = Path("scripts/hg317_qregion_reproduce_pmf_metad.py").read_text(encoding="utf-8").lower()

    assert "qregion-opes" not in source
    assert "qregion-opes" not in script
    assert "native_opes" not in source
