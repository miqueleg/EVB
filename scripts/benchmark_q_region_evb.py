#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from kemp_evb.cli import load_or_calibrate_parameters, select_start_positions
from kemp_evb.config import load_config, resolve_native_gap_bias_settings
from kemp_evb.native_bias import NativeGapBiasTable1D, NativeWellTemperedGapMetadynamics1D
from kemp_evb.openmm_backend import AmberSystemLoader, EVBSystemBuilder
from kemp_evb.q_region import QRegionSystemBuilder, derive_q_region_spec, q_region_spec_from_config, q_region_to_evb_openmm_system, validate_q_region_against_legacy
from kemp_evb.simulation import create_integrator


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def builder(config):
    return EVBSystemBuilder(AmberSystemLoader(config.simulation.nonbonded_method, config.simulation.constraints))


def q_spec(config, state1, state2):
    if config.evb_representation == "q_region" and config.q_region:
        return q_region_spec_from_config(config)
    spec, _report = derive_q_region_spec(config, state1, state2, explicit_q_atoms=None, include_reaction_atoms=True)
    if not spec.q_atoms:
        spec.q_atoms = [0, 1] if state1.system.getNumParticles() >= 2 else [0]
        spec.correction_atoms = list(range(state1.system.getNumParticles()))
    return spec


def make_system(config, mode, state1, state2, parameters):
    b = builder(config)
    if mode == "legacy_full_state":
        return b.build_openmm_evb_system(state1, state2, parameters.delta_alpha, parameters.h12), None, None, {"q_region_build_time_s": None}
    if mode == "exact_decomposition":
        return b.build_openmm_evb_system_decomposed(state1, state2, parameters.delta_alpha, parameters.h12), None, None, {"q_region_build_time_s": None}
    spec = q_spec(config, state1, state2)
    table = None
    if mode == "q_region_table_metad":
        native = resolve_native_gap_bias_settings(config)
        table = NativeGapBiasTable1D(native.min_value or -10.0, native.max_value or 10.0, int(native.grid_width or 101))
    t_build = time.perf_counter()
    q_system = QRegionSystemBuilder(spec).build(state1, state2, parameters.delta_alpha, parameters.h12, native_gap_bias_table=table)
    build_time = time.perf_counter() - t_build
    return q_region_to_evb_openmm_system(q_system), q_system, table, {"q_region_build_time_s": build_time, "time_to_derive_bonded_mapping_s": build_time}


def run_repeat(config, mode: str, platform_name: str, steps: int, repeat: int, diagnose_bonded_mapping: bool = False) -> dict[str, Any]:
    import openmm
    from openmm import unit

    b = builder(config)
    state1, state2 = b.build_from_state_files(config.state1, config.state2)
    parameters = load_or_calibrate_parameters(config)
    legacy = b.build_openmm_evb_system(state1, state2, parameters.delta_alpha, parameters.h12)
    t0 = time.perf_counter()
    evb_system, q_system, table, build_timing = make_system(config, mode, state1, state2, parameters)
    integrator = create_integrator(config.simulation.timestep_fs, config.simulation.temperature_k, config.simulation.friction_per_ps, config.simulation.integrator)
    platform = openmm.Platform.getPlatformByName(platform_name) if platform_name else None
    context = openmm.Context(evb_system.system, integrator, platform) if platform else openmm.Context(evb_system.system, integrator)
    if evb_system.box_vectors_nm is not None:
        context.setPeriodicBoxVectors(*[v * unit.nanometer for v in evb_system.box_vectors_nm])
    context.setPositions(select_start_positions(config, state1.positions_nm, state2.positions_nm) * unit.nanometer)
    context.setVelocitiesToTemperature(config.simulation.temperature_k * unit.kelvin, config.simulation.seed)
    context_creation = time.perf_counter() - t0
    validation = None
    if q_system is not None:
        validation = validate_q_region_against_legacy(q_system, legacy, state1.positions_nm, parameters, platform_name="CPU")
    table_metad = None
    native = resolve_native_gap_bias_settings(config)
    if table is not None:
        table_metad = NativeWellTemperedGapMetadynamics1D(
            table,
            bias_width=float(native.bias_width or 0.5),
            height_kj_mol=float(native.height_kj_mol or 0.2),
            bias_factor=float(native.bias_factor or 5.0),
            temperature_k=float(native.temperature_k or config.simulation.temperature_k),
            frequency=int(native.frequency or 100),
            save_frequency=None,
            bias_dir=Path("/tmp") / f"q_region_table_metad_{repeat}",
            restart=False,
        )
    t1 = time.perf_counter()
    if table_metad is None:
        integrator.step(steps)
    else:
        current = 0
        while current < steps:
            advance = min(table_metad.frequency, steps - current)
            integrator.step(advance)
            current += advance
            if current % table_metad.frequency == 0:
                values = [float(v) for v in evb_system.evb_force.getCollectiveVariableValues(context)]
                gap = values[0] - values[1] - parameters.delta_alpha
                table_metad.maybe_deposit(current, gap, evb_system.evb_force, context)
    wall = time.perf_counter() - t1
    report = (evb_system.energy_decomposition_report or {}).get("q_region_report", {})
    timing = table_metad.timing_report() if table_metad else {"bias_update_time_s": None, "average_time_per_bias_update_s": None, "number_of_bias_updates": 0}
    bonded_mapping = report.get("common_force_summary", {}).get("bonded_mapping", {}) if report else {}
    bonded_diagnostics = {}
    if diagnose_bonded_mapping:
        bonded_diagnostics = {
            "n_common_bonded_terms": bonded_mapping.get("n_common_terms"),
            "n_state1_only_bonded_terms": bonded_mapping.get("n_state1_only_terms"),
            "n_state2_only_bonded_terms": bonded_mapping.get("n_state2_only_terms"),
            "n_changed_parameter_bonded_terms": bonded_mapping.get("n_changed_parameter_terms"),
            "n_ambiguous_bonded_terms": bonded_mapping.get("n_ambiguous_terms"),
            "n_failed_outside_q_terms": len(report.get("changed_atoms_not_in_q_region", [])) if report else None,
            "bonded_mapping_summary": bonded_mapping,
            **build_timing,
        }
    return {
        "mode": mode,
        "repeat": repeat,
        "platform": context.getPlatform().getName(),
        "atom_count": evb_system.system.getNumParticles(),
        "q_atom_count": len(report.get("q_atoms", [])),
        "correction_atom_count": len((q_spec(config, state1, state2).correction_atoms or [])) if q_system is not None else 0,
        "common_force_count": len(getattr(evb_system, "common_forces", []) or []),
        "q_state_force_count": (report.get("q_state_force_summary", {}).get("state1_force_count", 0) + report.get("q_state_force_summary", {}).get("state2_force_count", 0)) if report else 0,
        "duplicated_full_nonbonded": report.get("duplicated_full_nonbonded", (evb_system.energy_decomposition_report or {}).get("duplicated_full_nonbonded")),
        "pme_baseline_count": 1 if report.get("pme_status") in {"local_pme_approx", "identical_nonbonded_common"} else 0,
        "context_creation_time_s": context_creation,
        "md_wall_time_s": wall,
        "steps_per_s": steps / wall if wall else None,
        "ns_per_day": (steps * config.simulation.timestep_fs * 1e-6) / (wall / 86400.0) if wall else None,
        "exactness_status": report.get("exactness_status", "exact" if mode != "q_region_local_pme_approx" else "approximate"),
        "warnings": report.get("warnings", []),
        "validation": validation,
        "use_app_metadynamics": False,
        "use_bias_variable": False,
        **timing,
        **bonded_diagnostics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--platform", default="CPU")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", default="legacy_full_state,exact_decomposition,q_region_exact_direct,q_region_table_metad")
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnose-bonded-mapping", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config.simulation.platform = args.platform
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    records = [
        run_repeat(config, mode, args.platform, args.steps, repeat, diagnose_bonded_mapping=args.diagnose_bonded_mapping)
        for mode in modes
        for repeat in range(args.repeats)
    ]
    summary = {"git_sha": git_sha(), "records": records}
    for mode in modes:
        rows = [r for r in records if r["mode"] == mode]
        if rows:
            summary[mode] = {
                "mean_steps_per_s": float(np.mean([r["steps_per_s"] for r in rows])),
                "mean_ns_per_day": float(np.mean([r["ns_per_day"] for r in rows])),
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
