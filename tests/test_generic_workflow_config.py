from __future__ import annotations

from pathlib import Path

import yaml

from kemp_evb.sampling_workflow import derive_gap_windows, generate_workflow_configs, validate_workflow_config


def test_generic_workflow_config_generation(tmp_path: Path):
    config = tmp_path / "config.yaml"
    payload = yaml.safe_load(Path("examples/generic_q_region_local_pme_template.yaml").read_text(encoding="utf-8"))
    payload["states"]["state1"]["topology"] = "state1.xml"
    payload["states"]["state2"]["topology"] = "state2.xml"
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = generate_workflow_configs(config, tmp_path / "out", mode="quick")

    assert Path(result["umbrella_config"]).exists()
    assert Path(result["metad_config"]).exists()
    assert result["umbrella_windows"]["n_windows"] == 41
    assert validate_workflow_config(config)["valid"] is True


def test_derive_windows_writes_csv(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(Path("examples/generic_full_state_template.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    result = derive_gap_windows(config, tmp_path / "windows")

    assert Path(result["windows_csv"]).exists()
