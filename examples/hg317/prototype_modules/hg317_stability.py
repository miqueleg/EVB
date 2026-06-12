from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import EVBConfig, load_config, resolve_native_gap_bias_settings
from .evb import EVBHamiltonian, EVBParameters
from .hg317_qregion import _builder
from .io import write_json
from .native_bias import NativeGapBiasTable1D, NativeWellTemperedGapMetadynamics1D
from .openmm_backend import write_pdb
from .q_region import QRegionSystemBuilder, q_region_spec_from_config, q_region_to_evb_openmm_system
from .simulation import create_integrator, ensure_output_dir


def run_hg317_qregion_stability_check(
    config_path: str | Path,
    output: str | Path,
    platform: str | None = None,
    *,
    plain_steps: int = 100_000,
    table_metad_steps: int = 100_000,
    quick: bool = False,
    report_interval: int | None = None,
    substrate_drift_threshold_nm: float = 1.0,
    force_explosion_threshold_kj_mol_nm: float = 50_000.0,
    energy_jump_threshold_kj_mol: float = 1.0e7,
    mixing_gap_tolerance_kj_mol: float = 250.0,
) -> dict[str, Any]:
    if quick:
        plain_steps = min(int(plain_steps), 10_000)
        table_metad_steps = min(int(table_metad_steps), 10_000)
    output_dir = ensure_output_dir(output)
    config_path = _resolve_stability_config_path(config_path)
    config = load_config(config_path)
    if platform:
        config.simulation.platform = platform
    report_stride = int(report_interval or max(1, min(1000, max(plain_steps, table_metad_steps) // 100)))
    runs = []
    if plain_steps > 0:
        runs.append(
            _run_stability_mode(
                config,
                config_path,
                output_dir / "plain_md",
                steps=int(plain_steps),
                mode="plain_md",
                report_interval=report_stride,
                substrate_drift_threshold_nm=substrate_drift_threshold_nm,
                force_explosion_threshold_kj_mol_nm=force_explosion_threshold_kj_mol_nm,
                energy_jump_threshold_kj_mol=energy_jump_threshold_kj_mol,
                mixing_gap_tolerance_kj_mol=mixing_gap_tolerance_kj_mol,
            )
        )
    if table_metad_steps > 0:
        runs.append(
            _run_stability_mode(
                config,
                config_path,
                output_dir / "table_metad",
                steps=int(table_metad_steps),
                mode="table_metad",
                report_interval=report_stride,
                substrate_drift_threshold_nm=substrate_drift_threshold_nm,
                force_explosion_threshold_kj_mol_nm=force_explosion_threshold_kj_mol_nm,
                energy_jump_threshold_kj_mol=energy_jump_threshold_kj_mol,
                mixing_gap_tolerance_kj_mol=mixing_gap_tolerance_kj_mol,
            )
        )
    summary = {
        "config": str(config_path),
        "output": str(output_dir),
        "platform": config.simulation.platform,
        "quick": bool(quick),
        "report_interval": report_stride,
        "parameters": {
            "delta_alpha_kj_mol": config.evb_parameters.delta_alpha,
            "h12_kj_mol": config.evb_parameters.h12,
        },
        "runs": runs,
        "ready_for_native_opes_development": all(run.get("stable") for run in runs) and any(run.get("sampling_promising") for run in runs),
        "opes_implemented": False,
    }
    write_json(output_dir / "stability_overall_summary.json", summary)
    _write_overall_md(output_dir / "stability_overall_summary.md", summary)
    return summary


def _resolve_stability_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.exists():
        return path
    if path.name == "fitted_config.yaml" and path.parent.name == "selected_candidate":
        root = path.parent.parent
        selected = root / "validation" / "selected_candidate.json"
        if selected.exists():
            with selected.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            fitted = Path(payload.get("fitted_config", ""))
            if fitted.exists():
                return fitted
    raise FileNotFoundError(f"Stability config not found: {config_path}")


def _run_stability_mode(
    config: EVBConfig,
    config_path: str | Path,
    output_dir: Path,
    *,
    steps: int,
    mode: str,
    report_interval: int,
    substrate_drift_threshold_nm: float,
    force_explosion_threshold_kj_mol_nm: float,
    energy_jump_threshold_kj_mol: float,
    mixing_gap_tolerance_kj_mol: float,
) -> dict[str, Any]:
    import openmm
    from openmm import unit

    output_dir = ensure_output_dir(output_dir)
    builder = _builder(config)
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    params = EVBParameters(config.evb_parameters.delta_alpha or 0.0, config.evb_parameters.h12 or 0.0)
    table = None
    metad = None
    if mode == "table_metad":
        native = resolve_native_gap_bias_settings(config)
        # Conservative default: cap height at 0.2 kJ/mol unless the config is already lower.
        table = NativeGapBiasTable1D(float(native.min_value), float(native.max_value), int(native.grid_width))
        metad = NativeWellTemperedGapMetadynamics1D(
            table,
            bias_width=float(native.bias_width),
            height_kj_mol=min(float(native.height_kj_mol), 0.2),
            bias_factor=float(native.bias_factor),
            temperature_k=float(native.temperature_k),
            frequency=max(1, int(native.frequency)),
            save_frequency=None,
            bias_dir=ensure_output_dir(output_dir / "native_bias"),
            restart=False,
        )
    t0 = time.perf_counter()
    q_system = QRegionSystemBuilder(q_region_spec_from_config(config)).build(
        state1,
        state2,
        params.delta_alpha,
        params.h12,
        native_gap_bias_table=table,
    )
    evb_system = q_region_to_evb_openmm_system(q_system)
    integrator = create_integrator(config.simulation.timestep_fs, config.simulation.temperature_k, config.simulation.friction_per_ps, config.simulation.integrator)
    platform_obj = openmm.Platform.getPlatformByName(config.simulation.platform) if config.simulation.platform else None
    context = openmm.Context(evb_system.system, integrator, platform_obj) if platform_obj else openmm.Context(evb_system.system, integrator)
    if evb_system.box_vectors_nm is not None:
        context.setPeriodicBoxVectors(*[v * unit.nanometer for v in evb_system.box_vectors_nm])
    context.setPositions(state1.positions_nm * unit.nanometer)
    context.setVelocitiesToTemperature(config.simulation.temperature_k * unit.kelvin, config.simulation.seed)
    context_creation_time = time.perf_counter() - t0

    initial_positions = state1.positions_nm.copy()
    initial_com = _substrate_com(initial_positions, config.reaction.substrate_atoms, state1.masses_amu)
    observable_rows: list[dict[str, Any]] = []
    nan_detected = False
    completed = False
    start = time.perf_counter()
    current = 0
    _append_observable_row(
        observable_rows,
        context,
        evb_system,
        config,
        params,
        table,
        step=0,
        initial_substrate_com_nm=initial_com,
        masses_amu=state1.masses_amu,
    )
    try:
        while current < steps:
            advance = min(report_interval, steps - current)
            integrator.step(advance)
            current += advance
            row = _append_observable_row(
                observable_rows,
                context,
                evb_system,
                config,
                params,
                table,
                step=current,
                initial_substrate_com_nm=initial_com,
                masses_amu=state1.masses_amu,
            )
            if metad is not None:
                metad.maybe_deposit(current, row["shifted_gap"], evb_system.evb_force, context)
            if any(not math.isfinite(float(value)) for value in _numeric_values(row)):
                nan_detected = True
                break
        completed = current >= steps and not nan_detected
    finally:
        wall = time.perf_counter() - start
    final_state = context.getState(getPositions=True)
    final_positions = final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    write_pdb(str(output_dir / "final.pdb"), evb_system.topology, final_positions)
    _write_observables(output_dir / "observables.csv", observable_rows)
    summary = _summarize_stability(
        mode,
        config,
        params,
        q_system.q_region_report,
        observable_rows,
        steps=current,
        requested_steps=steps,
        wall_time_s=wall,
        context_creation_time_s=context_creation_time,
        platform_name=context.getPlatform().getName(),
        completed=completed,
        nan_detected=nan_detected,
        substrate_drift_threshold_nm=substrate_drift_threshold_nm,
        force_explosion_threshold_kj_mol_nm=force_explosion_threshold_kj_mol_nm,
        energy_jump_threshold_kj_mol=energy_jump_threshold_kj_mol,
        mixing_gap_tolerance_kj_mol=mixing_gap_tolerance_kj_mol,
        metad=metad,
    )
    write_json(output_dir / "stability_summary.json", summary)
    _write_run_md(output_dir / "stability_summary.md", summary)
    return summary


def _append_observable_row(
    rows: list[dict[str, Any]],
    context: Any,
    evb_system: Any,
    config: EVBConfig,
    params: EVBParameters,
    table: NativeGapBiasTable1D | None,
    *,
    step: int,
    initial_substrate_com_nm: np.ndarray | None,
    masses_amu: np.ndarray,
) -> dict[str, Any]:
    from openmm import unit

    state = context.getState(getEnergy=True, getForces=True, getPositions=True, getVelocities=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    velocities = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    vals = [float(value) for value in evb_system.evb_force.getCollectiveVariableValues(context)]
    e1_q, e2_q = vals[0], vals[1]
    e_common = float(context.getState(getEnergy=True, groups={30}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    e1 = e_common + e1_q
    e2 = e_common + e2_q
    ham = EVBHamiltonian(params)
    evb_residual, w1, w2 = ham.lower_eigenvalue(e1_q, e2_q)
    eevb_unbiased = e_common + evb_residual
    shifted_gap = e1_q - e2_q - params.delta_alpha
    table_bias = 0.0 if table is None else table.evaluate(shifted_gap)
    max_force_norm = float(np.max(np.linalg.norm(forces, axis=1)))
    kinetic = _kinetic_energy_kj_mol(masses_amu, velocities)
    temperature = _temperature_k(kinetic, len(masses_amu))
    substrate_drift = _substrate_drift_nm(positions, config.reaction.substrate_atoms, masses_amu, initial_substrate_com_nm)
    row = {
        "step": int(step),
        "time_ps": float(step * config.simulation.timestep_fs * 1.0e-3),
        "E_common": e_common,
        "e1_Q": e1_q,
        "e2_Q": e2_q,
        "E1": e1,
        "E2": e2,
        "shifted_gap": shifted_gap,
        "Eevb_unbiased": eevb_unbiased,
        "Eevb_biased": eevb_unbiased + table_bias,
        "w1": float(w1),
        "w2": float(w2),
        "table_bias": table_bias,
        "temperature": temperature,
        "max_force_norm": max_force_norm,
        "substrate_com_displacement_nm": substrate_drift,
    }
    for distance in config.observables.distances:
        row[f"distance_{distance.name}_nm"] = _distance_nm(positions, distance.atom1, distance.atom2)
    rows.append(row)
    return row


def _numeric_values(row: dict[str, Any]) -> list[float]:
    values = []
    for value in row.values():
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _kinetic_energy_kj_mol(masses_amu: np.ndarray, velocities_nm_ps: np.ndarray) -> float:
    return float(0.5 * np.sum(masses_amu[:, None] * velocities_nm_ps * velocities_nm_ps))


def _temperature_k(kinetic_kj_mol: float, n_atoms: int) -> float:
    dof = max(1, 3 * int(n_atoms))
    return float(2.0 * kinetic_kj_mol / (dof * 0.00831446261815324))


def _substrate_com(positions_nm: np.ndarray, atoms: list[int], masses_amu: np.ndarray) -> np.ndarray | None:
    if not atoms:
        return None
    idx = np.asarray(atoms, dtype=int)
    weights = masses_amu[idx]
    return np.average(positions_nm[idx], axis=0, weights=weights)


def _substrate_drift_nm(positions_nm: np.ndarray, atoms: list[int], masses_amu: np.ndarray, initial_com: np.ndarray | None) -> float | None:
    if initial_com is None:
        return None
    current = _substrate_com(positions_nm, atoms, masses_amu)
    if current is None:
        return None
    return float(np.linalg.norm(current - initial_com))


def _distance_nm(positions_nm: np.ndarray, atom1: int, atom2: int) -> float:
    return float(np.linalg.norm(positions_nm[int(atom1)] - positions_nm[int(atom2)]))


def _write_observables(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_stability(
    mode: str,
    config: EVBConfig,
    params: EVBParameters,
    q_region_report: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    steps: int,
    requested_steps: int,
    wall_time_s: float,
    context_creation_time_s: float,
    platform_name: str,
    completed: bool,
    nan_detected: bool,
    substrate_drift_threshold_nm: float,
    force_explosion_threshold_kj_mol_nm: float,
    energy_jump_threshold_kj_mol: float,
    mixing_gap_tolerance_kj_mol: float,
    metad: NativeWellTemperedGapMetadynamics1D | None,
) -> dict[str, Any]:
    numeric = {key: np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=float) for key in rows[0] if isinstance(rows[0].get(key), (int, float))}
    gap = numeric["shifted_gap"]
    w2 = numeric["w2"]
    force = numeric["max_force_norm"]
    eevb = numeric["Eevb_biased"]
    substrate = numeric.get("substrate_com_displacement_nm", np.asarray([], dtype=float))
    max_energy_jump = float(np.max(np.abs(np.diff(eevb)))) if len(eevb) > 1 else 0.0
    max_substrate_drift = None if len(substrate) == 0 else float(np.nanmax(substrate))
    force_explosion = bool(float(np.max(force)) > force_explosion_threshold_kj_mol_nm)
    catastrophic_energy_jump = bool(max_energy_jump > energy_jump_threshold_kj_mol)
    substrate_ok = max_substrate_drift is None or max_substrate_drift <= substrate_drift_threshold_nm
    mixing_frames = int(np.sum((w2 >= 0.2) & (w2 <= 0.8)))
    gap_approached_zero = bool(np.min(np.abs(gap)) <= mixing_gap_tolerance_kj_mol)
    ts_visited = bool(mixing_frames > 0 or gap_approached_zero)
    stable = bool(completed and not nan_detected and not force_explosion and not catastrophic_energy_jump and substrate_ok)
    timing = {} if metad is None else metad.timing_report()
    return {
        "mode": mode,
        "requested_steps": int(requested_steps),
        "completed_steps": int(steps),
        "completed": bool(completed),
        "stable": stable,
        "sampling_promising": bool(stable and ts_visited),
        "ts_or_mixing_region_visited": ts_visited,
        "nan_detected": bool(nan_detected),
        "force_explosion_detected": force_explosion,
        "catastrophic_energy_jump_detected": catastrophic_energy_jump,
        "final_structure_written": True,
        "platform": platform_name,
        "context_creation_time_s": context_creation_time_s,
        "md_wall_time_s": wall_time_s,
        "steps_per_s": steps / wall_time_s if wall_time_s else None,
        "ns_per_day": (steps * config.simulation.timestep_fs * 1.0e-6) / (wall_time_s / 86400.0) if wall_time_s else None,
        "parameters": asdict(params),
        "max_absolute_energy_kj_mol": float(np.max(np.abs(eevb))),
        "max_energy_jump_kj_mol": max_energy_jump,
        "max_force_norm_kj_mol_nm": float(np.max(force)),
        "shifted_gap_min_kj_mol": float(np.min(gap)),
        "shifted_gap_max_kj_mol": float(np.max(gap)),
        "shifted_gap_mean_kj_mol": float(np.mean(gap)),
        "shifted_gap_std_kj_mol": float(np.std(gap)),
        "w2_min": float(np.min(w2)),
        "w2_max": float(np.max(w2)),
        "mixing_region_frame_count": mixing_frames,
        "gap_approached_zero": gap_approached_zero,
        "substrate_com_drift_max_nm": max_substrate_drift,
        "substrate_com_drift_threshold_nm": substrate_drift_threshold_nm,
        "q_region_report": q_region_report,
        "duplicated_full_nonbonded": q_region_report.get("duplicated_full_nonbonded"),
        "exactness_status": q_region_report.get("exactness_status"),
        "pme_approximation": q_region_report.get("pme_approximation"),
        "nonbonded_model_changed": q_region_report.get("nonbonded_model_changed"),
        **timing,
    }


def _write_run_md(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# HG3.17 Q-Region Stability: {summary['mode']}\n\n")
        for key in (
            "stable",
            "sampling_promising",
            "completed_steps",
            "steps_per_s",
            "ns_per_day",
            "shifted_gap_min_kj_mol",
            "shifted_gap_max_kj_mol",
            "w2_min",
            "w2_max",
            "mixing_region_frame_count",
            "duplicated_full_nonbonded",
            "exactness_status",
            "pme_approximation",
        ):
            handle.write(f"- {key}: {summary.get(key)}\n")


def _write_overall_md(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# HG3.17 Q-Region Stability Summary\n\n")
        handle.write(f"- ready_for_native_opes_development: {summary['ready_for_native_opes_development']}\n")
        handle.write(f"- opes_implemented: {summary['opes_implemented']}\n\n")
        handle.write("| mode | stable | sampling promising | steps/s | ns/day | w2 min | w2 max |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: |\n")
        for run in summary["runs"]:
            handle.write(
                f"| {run['mode']} | {run['stable']} | {run['sampling_promising']} | "
                f"{run.get('steps_per_s')} | {run.get('ns_per_day')} | {run.get('w2_min')} | {run.get('w2_max')} |\n"
            )
