from __future__ import annotations

from pathlib import Path

import numpy as np
import json
import yaml

from kemp_evb.evb import EVBParameters
from kemp_evb.hg317_gxtb_calibration import (
    GxtbFitResult,
    GxtbReferenceProfile,
    _evb_profile,
    fit_gxtb_profile,
    load_gxtb_reference_profile,
    select_gxtb_candidate,
    smoke_hg317_qregion_gxtb,
    write_gxtb_fitted_config,
)


def test_reference_profile_loader():
    reference = load_gxtb_reference_profile("examples/hg317_gxtb_reference_profile.yaml", "relative_to_RC")

    assert reference.target_kcal_mol["TS"] == 18.13981382
    assert reference.target_kcal_mol["PROD"] == -37.35192092
    assert reference.target_kj_mol["TS"] == 18.13981382 * 4.184
    assert reference.target_kj_mol["PROD"] == -37.35192092 * 4.184


def test_evb_parameter_grid_fit_recovers_synthetic_parameters():
    labels = ["RC", "TS", "PROD"]
    e1 = np.asarray([0.0, 120.0, 260.0])
    e2 = np.asarray([250.0, 80.0, -180.0])
    target, _w1, _w2, _gap = _evb_profile(e1, e2, 75.0, 30.0)

    fit = fit_gxtb_profile(labels, e1.tolist(), e2.tolist(), target.tolist(), delta_alpha_initial_kj_mol=50.0, h12_initial_kj_mol=20.0)

    assert abs(fit.delta_alpha_kj_mol - 75.0) < 1.0e-4
    assert abs(fit.h12_kj_mol - 30.0) < 1.0e-4
    assert fit.rms_residual_kj_mol < 1.0e-6


def test_large_gap_offset_is_absorbed_by_delta_alpha():
    labels = ["RC", "TS", "PROD"]
    e1 = np.asarray([0.0, 120.0, 260.0])
    e2 = np.asarray([250.0, 80.0, -180.0])
    target, _w1, _w2, _gap = _evb_profile(e1, e2, 75.0, 30.0)

    fit = fit_gxtb_profile(labels, (e1 + 15000.0).tolist(), (e2 - 15000.0).tolist(), target.tolist(), delta_alpha_initial_kj_mol=50.0, h12_initial_kj_mol=20.0)

    assert abs(fit.delta_alpha_kj_mol - 30075.0) < 1.0e-3
    assert abs(fit.h12_kj_mol - 30.0) < 1.0e-3
    assert fit.rms_residual_kj_mol < 1.0e-6


def test_candidate_selection_prefers_fitted_profile_over_raw_gap_match():
    rows = [
        {
            "candidate": "raw_gap_match_bad_fit",
            "fit_success": True,
            "rms_residual_kj_mol": 20.0,
            "max_residual_kj_mol": 25.0,
            "force_sanity": {"max_force_abs_kj_mol_nm": 10.0},
            "exploratory_valid": False,
        },
        {
            "candidate": "huge_raw_gap_good_fit",
            "fit_success": True,
            "rms_residual_kj_mol": 0.2,
            "max_residual_kj_mol": 0.4,
            "force_sanity": {"max_force_abs_kj_mol_nm": 20.0},
            "exploratory_valid": True,
        },
    ]

    selected = select_gxtb_candidate(rows)

    assert selected["candidate"] == "huge_raw_gap_good_fit"


def test_smoke_policy_exploratory_skips_non_exploratory_candidate(tmp_path: Path):
    calibrated = tmp_path / "calibrated"
    validation = calibrated / "validation"
    validation.mkdir(parents=True)
    (validation / "gxtb_validation_summary.json").write_text(
        json.dumps({"selected_candidate": {"candidate": "candidate_a", "exploratory_valid": False, "fitted_config": "missing.yaml"}}),
        encoding="utf-8",
    )

    result = smoke_hg317_qregion_gxtb(calibrated, tmp_path / "smoke", platform=None, steps=1, smoke_policy="exploratory")

    assert result["ran"] is False
    assert "not exploratory_valid" in result["reason"]


def test_fitted_config_writing(tmp_path: Path):
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(
        """
project:
  output_dir: old
evb:
  coupling_model:
    model: constant
    parameters:
      delta_alpha_kj_mol: 1.0
      h12_kj_mol: 2.0
  q_region:
    enabled: true
""",
        encoding="utf-8",
    )
    reference = GxtbReferenceProfile("HG3.17", "g-xtb", "relative_to_RC", {"RC": 0.0}, {"RC": 0.0}, {})
    fit = GxtbFitResult(3.0, 4.0, 0.1, 0.2, [0.0], [0.0], [1.0], [0.0], [0.0])
    output = tmp_path / "fitted.yaml"

    write_gxtb_fitted_config(candidate, output, EVBParameters(3.0, 4.0), reference, fit, tmp_path / "run")

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["evb"]["coupling_model"]["parameters"]["delta_alpha_kj_mol"] == 3.0
    assert payload["evb"]["coupling_model"]["parameters"]["h12_kj_mol"] == 4.0
    assert payload["evb"]["q_region"]["calibration"]["status"] == "gxtb_calibrated"


def test_no_opes_added_to_gxtb_module():
    source = Path("src/kemp_evb/hg317_gxtb_calibration.py").read_text(encoding="utf-8").lower()

    assert "opes" not in source
