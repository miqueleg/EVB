from __future__ import annotations

import json
import re
from pathlib import Path

from kemp_evb.config import load_config
from kemp_evb.evb import EVBHamiltonian, EVBParameters, fit_evb_reference_profile
from kemp_evb.io import write_json
from kemp_evb.openmm_backend import AmberSystemLoader, EVBSystemBuilder, OpenMMStateEvaluator, load_positions_file

HARTREE_TO_KJ_MOL = 2625.4996394798254


def _read_xyz_energy_hartree(path: Path) -> float:
    lines = path.read_text(encoding="utf-8").splitlines()
    match = re.search(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?", lines[1])
    if match is None:
        raise ValueError(f"Could not find a floating-point energy in line 2 of {path}")
    return float(match.group(0))


def _relative_qm_energies(rc_xyz: Path, ts_xyz: Path, prod_xyz: Path) -> dict[str, float]:
    e_rc = _read_xyz_energy_hartree(rc_xyz)
    e_ts = _read_xyz_energy_hartree(ts_xyz)
    e_prod = _read_xyz_energy_hartree(prod_xyz)
    return {
        "RC": 0.0,
        "TS": (e_ts - e_rc) * HARTREE_TO_KJ_MOL,
        "PROD": (e_prod - e_rc) * HARTREE_TO_KJ_MOL,
    }


def _profile(ham: EVBHamiltonian, mm_values: dict[str, float]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    rc_evb, _, _ = ham.lower_eigenvalue(mm_values["min1_state1"], mm_values["min1_state2"])
    for label, key in (("RC", "min1"), ("TS", "ts"), ("PROD", "min2")):
        e1 = mm_values[f"{key}_state1"]
        e2 = mm_values[f"{key}_state2"]
        evb_energy, w1, w2 = ham.lower_eigenvalue(e1, e2)
        payload[label] = {
            "E1_kj_mol": float(e1),
            "E2_kj_mol": float(e2),
            "Eevb_kj_mol": float(evb_energy),
            "weight1": float(w1),
            "weight2": float(w2),
            "relative_to_RC_kj_mol": float(evb_energy - rc_evb),
            "shifted_gap_kj_mol": float(e1 - (e2 + ham.parameters.delta_alpha)),
        }
    return payload


def _crossing_diagnostics(mm_values: dict[str, float], qm_rel: dict[str, float]) -> dict[str, float]:
    ts_gap = mm_values["ts_state1"] - mm_values["ts_state2"]
    best_error = float("inf")
    best_h12 = 0.0
    best_barrier = 0.0
    best_reaction = 0.0
    best_w2 = 0.0
    for h12 in [float(i) for i in range(0, 500001, 100)]:
        ham = EVBHamiltonian(EVBParameters(delta_alpha=ts_gap, h12=h12))
        rc_evb, _, _ = ham.lower_eigenvalue(mm_values["min1_state1"], mm_values["min1_state2"])
        ts_evb, _, ts_w2 = ham.lower_eigenvalue(mm_values["ts_state1"], mm_values["ts_state2"])
        prod_evb, _, _ = ham.lower_eigenvalue(mm_values["min2_state1"], mm_values["min2_state2"])
        barrier = ts_evb - rc_evb
        error = abs(barrier - qm_rel["TS"])
        if error < best_error:
            best_error = error
            best_h12 = h12
            best_barrier = barrier
            best_reaction = prod_evb - rc_evb
            best_w2 = ts_w2
    return {
        "delta_alpha_for_exact_ts_crossing_kj_mol": float(ts_gap),
        "best_h12_under_exact_ts_crossing_kj_mol": float(best_h12),
        "best_barrier_under_exact_ts_crossing_kj_mol": float(best_barrier),
        "best_reaction_free_energy_under_exact_ts_crossing_kj_mol": float(best_reaction),
        "barrier_error_under_exact_ts_crossing_kj_mol": float(best_error),
        "ts_weight2_under_exact_ts_crossing": float(best_w2),
    }


def _minima_offset_diagnostics(mm_values: dict[str, float], qm_rel: dict[str, float]) -> dict[str, float]:
    delta_alpha_from_minima = qm_rel["PROD"] + mm_values["min1_state1"] - mm_values["min2_state2"]
    shifted_gap_ts = mm_values["ts_state1"] - (mm_values["ts_state2"] + delta_alpha_from_minima)
    return {
        "delta_alpha_from_state_pure_minima_kj_mol": float(delta_alpha_from_minima),
        "ts_shifted_gap_using_minima_offset_kj_mol": float(shifted_gap_ts),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate EVB delta_alpha/H12 against QM RC/TS/PROD reference energies.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rc-xyz", default="systems/KEMP_implicit/RC_final.xyz")
    parser.add_argument("--ts-xyz", default="systems/KEMP_implicit/TS_final.xyz")
    parser.add_argument("--prod-xyz", default="systems/KEMP_implicit/PROD_final.xyz")
    parser.add_argument("--ts-pdb", default="prep/kemp_qm_openmm/05_templates/TS_solvated_template.pdb")
    parser.add_argument("--output", default="outputs/qm_xyz_calibration.json")
    args = parser.parse_args()

    config = load_config(args.config)
    qm_rel = _relative_qm_energies(Path(args.rc_xyz), Path(args.ts_xyz), Path(args.prod_xyz))
    loader = AmberSystemLoader(
        nonbonded_method=config.simulation.nonbonded_method,
        constraints=config.simulation.constraints,
    )
    builder = EVBSystemBuilder(loader)
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    eval1 = OpenMMStateEvaluator(state1, platform_name=config.simulation.platform)
    eval2 = OpenMMStateEvaluator(state2, platform_name=config.simulation.platform)
    coords = {
        "min1": load_positions_file(config.state1.inpcrd),
        "min2": load_positions_file(config.state2.inpcrd),
        "ts": load_positions_file(args.ts_pdb),
    }
    mm_values = {}
    for label, positions in coords.items():
        e1, _ = eval1.evaluate(positions)
        e2, _ = eval2.evaluate(positions)
        mm_values[f"{label}_state1"] = e1
        mm_values[f"{label}_state2"] = e2
    fit = fit_evb_reference_profile(
        e_mm_min1_state1=mm_values["min1_state1"],
        e_mm_min1_state2=mm_values["min1_state2"],
        e_mm_min2_state1=mm_values["min2_state1"],
        e_mm_min2_state2=mm_values["min2_state2"],
        e_mm_ts_state1=mm_values["ts_state1"],
        e_mm_ts_state2=mm_values["ts_state2"],
        e_qmmm_min1=0.0,
        e_qmmm_min2=qm_rel["PROD"],
        e_qmmm_ts=qm_rel["TS"],
        ts_mixing_weight=1.0,
    )
    ham = EVBHamiltonian(fit.parameters)
    payload = {
        "config": args.config,
        "qm_relative_energies_kj_mol": qm_rel,
        "mm_reference_energies_kj_mol": mm_values,
        "fit": {
            "delta_alpha_kj_mol": fit.parameters.delta_alpha,
            "h12_kj_mol": fit.parameters.h12,
            "objective_value": fit.objective_value,
            "fitted_reaction_free_energy_kj_mol": fit.fitted_reaction_free_energy,
            "fitted_barrier_kj_mol": fit.fitted_barrier,
            "ts_weight2": fit.ts_weight2,
        },
        "minima_offset_diagnostics": _minima_offset_diagnostics(mm_values, qm_rel),
        "ts_crossing_diagnostics": _crossing_diagnostics(mm_values, qm_rel),
        "evb_profile": _profile(ham, mm_values),
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
