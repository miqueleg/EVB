from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import EVBConfig, load_config, resolve_native_gap_bias_settings
from .evb import EVBParameters, calibrate_evb_parameters
from .io import write_json
from .native_bias import NativeGapBiasTable1D, NativeWellTemperedGapMetadynamics1D
from .openmm_backend import AmberSystemLoader, EVBSystemBuilder, load_positions_file, write_pdb
from .q_region import QRegionSystemBuilder, derive_q_region_spec, q_region_spec_from_config, q_region_to_evb_openmm_system, validate_q_region_against_legacy
from .simulation import create_integrator, ensure_output_dir


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required for HG3.17 Q-region candidate workflows.") from exc
    return yaml


def _read_yaml(path: str | Path) -> dict[str, Any]:
    yaml = _yaml_module()
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    yaml = _yaml_module()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _builder(config: EVBConfig) -> EVBSystemBuilder:
    return EVBSystemBuilder(AmberSystemLoader(config.simulation.nonbonded_method, config.simulation.constraints))


def _set_output_dir(payload: dict[str, Any], output_dir: str) -> None:
    payload.setdefault("project", {})["output_dir"] = output_dir


def _set_parameters(payload: dict[str, Any], parameters: EVBParameters) -> None:
    coupling = payload.setdefault("evb", {}).setdefault("coupling_model", {})
    coupling.setdefault("model", "constant")
    params = coupling.setdefault("parameters", {})
    params["delta_alpha_kj_mol"] = float(parameters.delta_alpha)
    params["h12_kj_mol"] = float(parameters.h12)


def _candidate_payload(
    base_payload: dict[str, Any],
    q_atoms: list[int],
    *,
    baseline_state: str,
    mode: str,
    correction_policy: str = "auto",
    cutoff_nm: float = 1.2,
    output_dir: str,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(base_payload))
    payload.setdefault("evb", {})["representation"] = "q_region"
    payload["evb"]["q_region"] = {
        "enabled": True,
        "q_atoms": [int(atom) for atom in q_atoms],
        "baseline_state": baseline_state,
        "changed_atom_policy": "require_subset",
        "common_force_placement": "outer_system",
        "bonded": {
            "derive_from_state_differences": True,
            "include_bonds": True,
            "include_angles": True,
            "include_torsions": True,
            "include_impropers": True,
        },
        "nonbonded": {
            "mode": mode,
            "pme_policy": "shared_baseline" if mode == "shared_nonbonded_model" else "local_direct_space_correction",
            "local_approx_enabled": mode == "local_pme_approx",
            "correction_atoms": correction_policy,
            "correction_cutoff_nm": float(cutoff_nm),
            "include_q_q": True,
            "include_q_environment": True,
            "include_q_water": True,
            "include_exceptions": True,
        },
        "constraints": {"q_atom_constraint_policy": "fail"},
        "validation": {
            "compare_to_legacy": True,
            "max_energy_error_kj_mol_local_approx": 5.0,
            "max_gap_error_kj_mol_local_approx": 5.0,
            "max_force_rmsd_local_approx": 100.0,
        },
    }
    sampling = payload.setdefault("sampling", {})
    sampling["mode"] = "gap_table_metadynamics"
    native = sampling.setdefault("native_gap_bias", {})
    meta = sampling.get("metadynamics", {})
    native.setdefault("method", "well_tempered_metadynamics")
    native.setdefault("cv", "gap")
    for key in ("min_value", "max_value", "bias_width", "height_kj_mol", "bias_factor", "frequency", "save_frequency", "bias_dir", "wall_force_constant_kj_mol2"):
        if key in meta and key not in native:
            native[key] = meta[key]
    native.setdefault("grid_width", 1001)
    native.setdefault("restart", False)
    _set_output_dir(payload, output_dir)
    return payload


def make_hg317_qregion_candidates(config: EVBConfig, config_path: str | Path, output: str | Path, include_reaction_atoms: bool = False) -> dict[str, Any]:
    output_dir = ensure_output_dir(output)
    candidate_dir = ensure_output_dir(output_dir / "candidates")
    builder = _builder(config)
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    spec, report = derive_q_region_spec(config, state1, state2, include_reaction_atoms=include_reaction_atoms)
    report_dict = asdict(report)
    write_json(output_dir / "q_region_derivation_report.json", report_dict)
    base_payload = _read_yaml(config_path)
    candidates: list[dict[str, Any]] = []

    definitions = [
        ("shared_nonbonded_state1", "state1", "shared_nonbonded_model", "auto", 1.2),
        ("shared_nonbonded_state2", "state2", "shared_nonbonded_model", "auto", 1.2),
        ("local_pme_q_atoms_cutoff_0.8", "state1", "local_pme_approx", "q_atoms", 0.8),
        ("local_pme_q_atoms_cutoff_1.0", "state1", "local_pme_approx", "q_atoms", 1.0),
        ("local_pme_q_atoms_cutoff_1.2", "state1", "local_pme_approx", "q_atoms", 1.2),
        ("local_pme_q_atoms_cutoff_1.5", "state1", "local_pme_approx", "q_atoms", 1.5),
        ("local_pme_q_plus_shell_cutoff_1.2", "state1", "local_pme_approx", "q_plus_shell", 1.2),
        ("local_pme_q_plus_shell_cutoff_1.5", "state1", "local_pme_approx", "q_plus_shell", 1.5),
        ("local_pme_q_plus_shell_cutoff_2.0", "state1", "local_pme_approx", "q_plus_shell", 2.0),
        ("local_pme_all_atoms_cutoff_1.2", "state1", "local_pme_approx", "all_atoms", 1.2),
    ]
    for name, baseline, mode, policy, cutoff in definitions:
        path = candidate_dir / f"{name}.yaml"
        payload = _candidate_payload(
            base_payload,
            spec.q_atoms,
            baseline_state=baseline,
            mode=mode,
            correction_policy=policy,
            cutoff_nm=cutoff,
            output_dir=str(output_dir / "runs" / name),
        )
        _write_yaml(path, payload)
        candidates.append({"name": name, "path": str(path), "mode": mode, "baseline_state": baseline, "correction_atoms": policy, "cutoff_nm": cutoff})
    summary = {"candidate_dir": str(candidate_dir), "q_atoms": spec.q_atoms, "candidates": candidates, "derivation_report": str(output_dir / "q_region_derivation_report.json")}
    write_json(output_dir / "candidate_generation_summary.json", summary)
    return summary


def _q_region_diabatic_energies(q_system, positions_nm: np.ndarray, platform_name: str = "CPU") -> tuple[float, float]:
    import openmm
    from openmm import unit

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name) if platform_name else None
    context = openmm.Context(q_system.system, integrator, platform) if platform else openmm.Context(q_system.system, integrator)
    if q_system.box_vectors_nm is not None:
        context.setPeriodicBoxVectors(*[v * unit.nanometer for v in q_system.box_vectors_nm])
    context.setPositions(positions_nm * unit.nanometer)
    vals = [float(v) for v in q_system.evb_force.getCollectiveVariableValues(context)]
    common = float(context.getState(getEnergy=True, groups={30}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    return common + vals[0], common + vals[1]


def q_region_fit_irc(config: EVBConfig, config_path: str | Path, output: str | Path) -> dict[str, Any]:
    output_dir = ensure_output_dir(output)
    builder = _builder(config)
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    original = EVBParameters(config.evb_parameters.delta_alpha or 0.0, config.evb_parameters.h12 or 0.0)
    q_system = QRegionSystemBuilder(q_region_spec_from_config(config)).build(state1, state2, original.delta_alpha, original.h12)
    frame_paths = {}
    if config.calibration is not None:
        frame_paths = {
            "min1": config.calibration.coordinates.min1,
            "min2": config.calibration.coordinates.min2,
            "ts": config.calibration.coordinates.ts,
        }
    default_positions = {"min1": state1.positions_nm, "min2": state2.positions_nm, "ts": None}
    positions = {}
    missing = []
    for label in ("min1", "min2", "ts"):
        path = frame_paths.get(label)
        if path:
            positions[label] = load_positions_file(path)
        elif default_positions[label] is not None:
            positions[label] = default_positions[label]
        else:
            missing.append(label)
    fit_success = config.calibration is not None and not missing
    mm_values: dict[str, float] = {}
    fitted = original
    if fit_success:
        for label, coords in positions.items():
            e1, e2 = _q_region_diabatic_energies(q_system, coords)
            mm_values[f"{label}_state1"] = e1
            mm_values[f"{label}_state2"] = e2
        fitted = calibrate_evb_parameters(
            e_mm_min1_state1=mm_values["min1_state1"],
            e_mm_min1_state2=mm_values["min1_state2"],
            e_mm_min2_state1=mm_values["min2_state1"],
            e_mm_min2_state2=mm_values["min2_state2"],
            e_mm_ts_state1=mm_values["ts_state1"],
            e_mm_ts_state2=mm_values["ts_state2"],
            e_qmmm_min1=config.calibration.e_qmmm_min1,
            e_qmmm_min2=config.calibration.e_qmmm_min2,
            e_qmmm_ts=config.calibration.e_qmmm_ts,
        )
    payload = _read_yaml(config_path)
    _set_parameters(payload, fitted)
    _set_output_dir(payload, str(output_dir / "run"))
    fitted_config = output_dir / "fitted_config.yaml"
    _write_yaml(fitted_config, payload)
    report = {
        "fit_success": bool(fit_success),
        "reason": None if fit_success else "Calibration data with min1/min2/ts coordinates is required for a full Q-region IRC refit; original parameters were retained.",
        "original_parameters": asdict(original),
        "fitted_parameters": asdict(fitted),
        "mm_energies": mm_values,
        "missing_frames": missing,
        "fitted_config": str(fitted_config),
    }
    write_json(output_dir / "fitted_parameters.json", asdict(fitted))
    write_json(output_dir / "q_region_fit_report.json", report)
    with (output_dir / "diabatic_scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "E1_kj_mol", "E2_kj_mol"])
        for label in sorted(mm_values):
            if label.endswith("_state1"):
                frame = label[:-7]
                writer.writerow([frame, mm_values.get(f"{frame}_state1"), mm_values.get(f"{frame}_state2")])
    return report


def _validation_frames(config: EVBConfig, state1, state2) -> tuple[list[tuple[str, np.ndarray]], list[str]]:
    frames = [("state1_initial", state1.positions_nm), ("state2_initial", state2.positions_nm)]
    missing = []
    for label, path in (
        ("irc_rc", getattr(config.calibration.coordinates, "min1", None) if config.calibration else None),
        ("irc_ts", getattr(config.calibration.coordinates, "ts", None) if config.calibration else None),
        ("irc_product", getattr(config.calibration.coordinates, "min2", None) if config.calibration else None),
    ):
        if path and Path(path).exists():
            frames.append((label, load_positions_file(path)))
        else:
            missing.append(label)
    return frames, missing


def _passes_validation(config: EVBConfig, q_report: dict[str, Any], frame_reports: list[dict[str, Any]]) -> bool:
    if not frame_reports:
        return False
    max_energy = max(abs(row["energy_error_kj_mol"]) for row in frame_reports)
    max_gap = max(abs(row["gap_error_kj_mol"]) for row in frame_reports)
    max_force = max(abs(row["force_rmsd_kj_mol_nm"]) for row in frame_reports)
    if q_report.get("exactness_status") == "approximate":
        return (
            max_energy <= config.q_region.get("validation", {}).get("max_energy_error_kj_mol_local_approx", 5.0)
            and max_gap <= config.q_region.get("validation", {}).get("max_gap_error_kj_mol_local_approx", 5.0)
            and max_force <= config.q_region.get("validation", {}).get("max_force_rmsd_local_approx", 100.0)
        )
    # Shared-nonbonded candidates are exact for a changed Hamiltonian, but this
    # validation command alone cannot prove the required IRC refit/barrier sanity.
    return False


def _candidate_name(path: Path) -> str:
    return path.stem


def validate_hg317_qregion_candidates(
    candidate_dir: str | Path,
    output: str | Path,
    platform: str | None = None,
    *,
    run_smoke: bool = False,
    smoke_steps: int = 2000,
    force_smoke: bool = False,
) -> dict[str, Any]:
    candidate_dir = Path(candidate_dir)
    config_dir = candidate_dir / "candidates" if (candidate_dir / "candidates").exists() else candidate_dir
    output_dir = ensure_output_dir(output)
    per_dir = ensure_output_dir(output_dir / "per_candidate")
    rows = []
    for path in sorted(config_dir.glob("*.yaml")):
        name = _candidate_name(path)
        config = load_config(path)
        if platform:
            config.simulation.platform = platform
        builder = _builder(config)
        state1, state2 = builder.build_from_state_files(config.state1, config.state2)
        parameters = EVBParameters(config.evb_parameters.delta_alpha or 0.0, config.evb_parameters.h12 or 0.0)
        legacy = builder.build_openmm_evb_system(state1, state2, parameters.delta_alpha, parameters.h12)
        frames, missing = _validation_frames(config, state1, state2)
        t0 = time.perf_counter()
        error = None
        q_system = None
        try:
            q_system = QRegionSystemBuilder(q_region_spec_from_config(config)).build(state1, state2, parameters.delta_alpha, parameters.h12)
            build_time = time.perf_counter() - t0
            frame_reports = []
            for frame_name, coords in frames:
                report = validate_q_region_against_legacy(q_system, legacy, coords, parameters, platform_name="CPU")
                report["frame"] = frame_name
                frame_reports.append(report)
        except Exception as exc:
            build_time = time.perf_counter() - t0
            error = str(exc)
            frame_reports = []
        q_report = {} if q_system is None else q_system.q_region_report
        max_energy = max([abs(r["energy_error_kj_mol"]) for r in frame_reports], default=None)
        max_gap = max([abs(r["gap_error_kj_mol"]) for r in frame_reports], default=None)
        max_force = max([abs(r["force_rmsd_kj_mol_nm"]) for r in frame_reports], default=None)
        max_force_abs = max([abs(r["force_max_abs_kj_mol_nm"]) for r in frame_reports], default=None)
        passed = False if error else _passes_validation(config, q_report, frame_reports)
        record = {
            "candidate": name,
            "config": str(path),
            "build_error": error,
            "exactness_status": q_report.get("exactness_status"),
            "pme_approximation": q_report.get("pme_approximation"),
            "nonbonded_model_changed": q_report.get("nonbonded_model_changed"),
            "legacy_equivalence": q_report.get("legacy_equivalence"),
            "duplicated_full_nonbonded": q_report.get("duplicated_full_nonbonded"),
            "max_energy_error_kj_mol": max_energy,
            "max_gap_error_kj_mol": max_gap,
            "force_rmsd_kj_mol_nm": max_force,
            "force_max_abs_kj_mol_nm": max_force_abs,
            "context_creation_time_s": build_time,
            "validation_frames": [name for name, _coords in frames],
            "missing_frames": missing,
            "pass": passed,
            "frame_reports": frame_reports,
            "q_region_report": q_report,
        }
        rows.append(record)
        write_json(per_dir / f"{name}_validation.json", record)
    selected_full = _select_best_candidate(rows)
    selected = None if selected_full is None else _compact_validation_record(selected_full)
    compact_rows = [_compact_validation_record(row) for row in rows]
    summary = {"candidates": compact_rows, "selected_best_candidate": selected, "smoke": None}
    if run_smoke:
        if selected_full and (selected_full.get("pass") or force_smoke):
            summary["smoke"] = run_hg317_qregion_smoke(selected_full["config"], output_dir / "smoke", platform, smoke_steps, forced=force_smoke and not selected_full.get("pass"))
        else:
            summary["smoke"] = {"ran": False, "reason": "No candidate passed validation thresholds; use force_smoke to override."}
    write_json(output_dir / "candidate_validation_summary.json", summary)
    _write_validation_csv(output_dir / "candidate_validation_summary.csv", rows)
    _write_validation_md(output_dir / "candidate_validation_summary.md", summary)
    return summary



def _compact_validation_record(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "candidate",
        "config",
        "build_error",
        "exactness_status",
        "pme_approximation",
        "nonbonded_model_changed",
        "legacy_equivalence",
        "duplicated_full_nonbonded",
        "max_energy_error_kj_mol",
        "max_gap_error_kj_mol",
        "force_rmsd_kj_mol_nm",
        "force_max_abs_kj_mol_nm",
        "context_creation_time_s",
        "validation_frames",
        "missing_frames",
        "pass",
    ]
    return {key: row.get(key) for key in keys}

def _select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [row for row in rows if row.get("pass")]
    if passed:
        shared = [row for row in passed if row.get("exactness_status") == "exact_for_shared_nonbonded_model"]
        if shared:
            return sorted(shared, key=lambda r: r.get("context_creation_time_s") or 1e99)[0]
        return sorted(passed, key=lambda r: (r.get("force_rmsd_kj_mol_nm") if r.get("force_rmsd_kj_mol_nm") is not None else 1e99))[0]
    buildable = [row for row in rows if row.get("build_error") is None]
    if buildable:
        return sorted(buildable, key=lambda r: (r.get("force_rmsd_kj_mol_nm") if r.get("force_rmsd_kj_mol_nm") is not None else 1e99))[0]
    return rows[0] if rows else None


def _write_validation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["candidate", "exactness_status", "pme_approximation", "nonbonded_model_changed", "max_energy_error_kj_mol", "max_gap_error_kj_mol", "force_rmsd_kj_mol_nm", "force_max_abs_kj_mol_nm", "pass", "build_error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_validation_md(path: Path, summary: dict[str, Any]) -> None:
    rows = summary.get("candidates", [])
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# HG3.17 Q-Region Candidate Validation\n\n")
        handle.write("| candidate | exactness | PME approx | model changed | max energy | max gap | force RMSD | pass |\n")
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.get('candidate')} | {row.get('exactness_status')} | {row.get('pme_approximation')} | "
                f"{row.get('nonbonded_model_changed')} | {row.get('max_energy_error_kj_mol')} | "
                f"{row.get('max_gap_error_kj_mol')} | {row.get('force_rmsd_kj_mol_nm')} | {row.get('pass')} |\n"
            )
        selected = summary.get("selected_best_candidate") or {}
        handle.write(f"\nSelected best candidate: `{selected.get('candidate')}`\n")
        if summary.get("smoke"):
            handle.write(f"\nSmoke: `{summary['smoke'].get('ran')}` {summary['smoke'].get('reason', '')}\n")


def run_hg317_qregion_smoke(config_path: str | Path, output: str | Path, platform: str | None, smoke_steps: int, *, forced: bool = False) -> dict[str, Any]:
    import openmm
    from openmm import unit

    config = load_config(config_path)
    config.simulation.steps = int(smoke_steps)
    config.simulation.report_interval = max(1, min(int(smoke_steps), int(config.simulation.report_interval or smoke_steps)))
    config.simulation.minimize_steps = 0
    if platform:
        config.simulation.platform = platform
    output_dir = ensure_output_dir(output)
    config.output_dir = str(output_dir)
    config.project.output_dir = str(output_dir)
    config.sampling.native_gap_bias.restart = False
    config.sampling.native_gap_bias.bias_dir = "native_bias"
    native = resolve_native_gap_bias_settings(config)
    table = NativeGapBiasTable1D(float(native.min_value), float(native.max_value), int(native.grid_width))
    metad = NativeWellTemperedGapMetadynamics1D(
        table,
        bias_width=float(native.bias_width),
        height_kj_mol=float(native.height_kj_mol),
        bias_factor=float(native.bias_factor),
        temperature_k=float(native.temperature_k),
        frequency=max(1, int(native.frequency)),
        save_frequency=None,
        bias_dir=ensure_output_dir(output_dir / "native_bias"),
        restart=False,
    )
    builder = _builder(config)
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    params = EVBParameters(config.evb_parameters.delta_alpha or 0.0, config.evb_parameters.h12 or 0.0)
    t0 = time.perf_counter()
    q_system = QRegionSystemBuilder(q_region_spec_from_config(config)).build(state1, state2, params.delta_alpha, params.h12, native_gap_bias_table=table)
    evb_system = q_region_to_evb_openmm_system(q_system)
    integrator = create_integrator(config.simulation.timestep_fs, config.simulation.temperature_k, config.simulation.friction_per_ps, config.simulation.integrator)
    platform_obj = openmm.Platform.getPlatformByName(config.simulation.platform) if config.simulation.platform else None
    context = openmm.Context(evb_system.system, integrator, platform_obj) if platform_obj else openmm.Context(evb_system.system, integrator)
    if evb_system.box_vectors_nm is not None:
        context.setPeriodicBoxVectors(*[v * unit.nanometer for v in evb_system.box_vectors_nm])
    context.setPositions(state1.positions_nm * unit.nanometer)
    context.setVelocitiesToTemperature(config.simulation.temperature_k * unit.kelvin, config.simulation.seed)
    context_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    current = 0
    while current < smoke_steps:
        advance = min(max(1, int(native.frequency)), smoke_steps - current)
        integrator.step(advance)
        current += advance
        values = [float(v) for v in evb_system.evb_force.getCollectiveVariableValues(context)]
        gap = values[0] - values[1] - params.delta_alpha
        metad.maybe_deposit(current, gap, evb_system.evb_force, context)
    wall = time.perf_counter() - t1
    positions_nm = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    write_pdb(str(output_dir / "smoke_final.pdb"), evb_system.topology, positions_nm)
    smoke = {
        "ran": True,
        "forced_despite_failed_validation": bool(forced),
        "steps": int(smoke_steps),
        "platform": context.getPlatform().getName(),
        "context_creation_time_s": context_time,
        "md_wall_time_s": wall,
        "steps_per_s": smoke_steps / wall if wall else None,
        "ns_per_day": (smoke_steps * config.simulation.timestep_fs * 1e-6) / (wall / 86400.0) if wall else None,
        "parameters": asdict(params),
        "parameters_refit": ("fits" in str(config_path) or "fitted" in str(config_path) or bool(getattr(config, "q_region", {}) and config.q_region.get("calibration"))),
        "q_region_report": q_system.q_region_report,
        **metad.timing_report(),
    }
    write_json(output_dir / "smoke_benchmark.json", smoke)
    with (output_dir / "smoke_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# HG3.17 Q-Region Smoke Benchmark\n\n")
        handle.write(f"- ran: {smoke['ran']}\n")
        handle.write(f"- steps/s: {smoke['steps_per_s']}\n")
        handle.write(f"- ns/day: {smoke['ns_per_day']}\n")
        handle.write(f"- exactness_status: {q_system.q_region_report.get('exactness_status')}\n")
        handle.write(f"- duplicated_full_nonbonded: {q_system.q_region_report.get('duplicated_full_nonbonded')}\n")
    return smoke
