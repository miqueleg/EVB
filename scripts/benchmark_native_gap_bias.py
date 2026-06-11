#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from kemp_evb.config import load_config, resolve_native_gap_bias_settings
from kemp_evb.evb import EVBHamiltonian, EVBParameters
from kemp_evb.native_bias import NativeGapBiasTable1D, NativeWellTemperedGapMetadynamics1D
from kemp_evb.openmm_backend import AmberSystemLoader, EVBSystemBuilder, build_evb_gap_cv_force, evb_diabatic_energies
from kemp_evb.cli import load_or_calibrate_parameters, select_start_positions
from kemp_evb.simulation import create_integrator


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def make_builder(config):
    return EVBSystemBuilder(
        AmberSystemLoader(
            nonbonded_method=config.simulation.nonbonded_method,
            constraints=config.simulation.constraints,
        )
    )


def count_custom_cv(system) -> int:
    import openmm

    return sum(isinstance(system.getForce(i), openmm.CustomCVForce) for i in range(system.getNumForces()))


def build_system(config, mode: str, state1, state2, parameters):
    builder = make_builder(config)
    if mode == "table-metad":
        native = resolve_native_gap_bias_settings(config)
        table = NativeGapBiasTable1D(
            native.min_value,
            native.max_value,
            int(native.grid_width),
            out_of_grid=native.out_of_grid,
        )
        evb_system = builder.build_openmm_evb_system(
            state1,
            state2,
            parameters.delta_alpha,
            parameters.h12,
            energy_decomposition=config.energy_decomposition.enabled,
            energy_decomposition_mode=config.energy_decomposition.mode,
            fallback_to_legacy_for_unsupported_terms=config.energy_decomposition.fallback_to_legacy_for_unsupported_terms,
            report_energy_decomposition=True,
            common_force_placement=(config.energy_decomposition.common_force_placement if config.energy_decomposition.enabled else "cv_compatible"),
            native_gap_bias_table=table,
            native_gap_wall_force_constant=native.wall_force_constant_kj_mol2,
        )
        return evb_system, table
    evb_system = builder.build_openmm_evb_system(
        state1,
        state2,
        parameters.delta_alpha,
        parameters.h12,
        energy_decomposition=config.energy_decomposition.enabled,
        energy_decomposition_mode=config.energy_decomposition.mode,
        fallback_to_legacy_for_unsupported_terms=config.energy_decomposition.fallback_to_legacy_for_unsupported_terms,
        report_energy_decomposition=True,
    )
    return evb_system, None


def run_repeat(config, mode: str, platform_name: str, steps: int, repeat_index: int) -> dict[str, Any]:
    import openmm
    from openmm import app, unit

    builder = make_builder(config)
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    parameters = load_or_calibrate_parameters(config)
    native = resolve_native_gap_bias_settings(config)
    t0 = time.perf_counter()
    evb_system, table = build_system(config, mode, state1, state2, parameters)
    integrator = create_integrator(
        timestep_fs=config.simulation.timestep_fs,
        temperature_k=config.simulation.temperature_k,
        friction_per_ps=config.simulation.friction_per_ps,
        integrator_name=config.simulation.integrator,
    )
    platform = openmm.Platform.getPlatformByName(platform_name) if platform_name else None
    properties = {}
    if platform_name == "CUDA" and config.simulation.platform == "CUDA":
        properties = {}
    if mode == "app-metad":
        gap_cv = build_evb_gap_cv_force(
            state1.system,
            state2.system,
            parameters.delta_alpha,
            prefix=f"bench_gap_metad_{repeat_index}",
            energy_decomposition=config.energy_decomposition.enabled,
            energy_decomposition_mode=config.energy_decomposition.mode,
            fallback_to_legacy_for_unsupported_terms=config.energy_decomposition.fallback_to_legacy_for_unsupported_terms,
        )
        variable = app.BiasVariable(
            gap_cv,
            float(native.min_value),
            float(native.max_value),
            float(native.bias_width),
            periodic=False,
            gridWidth=native.grid_width,
        )
        metad = app.Metadynamics(
            evb_system.system,
            [variable],
            float(native.temperature_k) * unit.kelvin,
            float(native.bias_factor),
            float(native.height_kj_mol) * unit.kilojoule_per_mole,
            int(native.frequency),
            saveFrequency=None,
        )
        simulation = app.Simulation(evb_system.topology, evb_system.system, integrator, platform) if platform else app.Simulation(evb_system.topology, evb_system.system, integrator)
        context = simulation.context
    else:
        context = openmm.Context(evb_system.system, integrator, platform, properties) if platform else openmm.Context(evb_system.system, integrator)
        metad = None
    if evb_system.box_vectors_nm is not None:
        context.setPeriodicBoxVectors(*[vector * unit.nanometer for vector in evb_system.box_vectors_nm])
    context.setPositions(select_start_positions(config, state1.positions_nm, state2.positions_nm) * unit.nanometer)
    context.setVelocitiesToTemperature(config.simulation.temperature_k * unit.kelvin, config.simulation.seed)
    context_creation_s = time.perf_counter() - t0

    table_metad = None
    if mode == "table-metad":
        table_metad = NativeWellTemperedGapMetadynamics1D(
            table,
            bias_width=float(native.bias_width),
            height_kj_mol=float(native.height_kj_mol),
            bias_factor=float(native.bias_factor),
            temperature_k=float(native.temperature_k),
            frequency=int(native.frequency),
            save_frequency=None,
            bias_dir=Path("/tmp") / f"native_gap_bias_bench_{repeat_index}",
            restart=False,
        )
    t1 = time.perf_counter()
    if mode == "app-metad":
        metad.step(simulation, steps)
    elif mode == "table-metad":
        current = 0
        while current < steps:
            advance = min(int(native.frequency), steps - current)
            integrator.step(advance)
            current += advance
            if current % int(native.frequency) == 0:
                energy1, energy2 = evb_diabatic_energies(evb_system, context)
                gap = energy1 - energy2 - parameters.delta_alpha
                table_metad.maybe_deposit(current, gap, evb_system.evb_force, context)
    else:
        integrator.step(steps)
    md_wall_s = time.perf_counter() - t1
    energy1, energy2 = evb_diabatic_energies(evb_system, context)
    evb_energy, _, _ = EVBHamiltonian(EVBParameters(parameters.delta_alpha, parameters.h12)).lower_eigenvalue(energy1, energy2)
    report = evb_system.energy_decomposition_report or {}
    timing = table_metad.timing_report() if table_metad is not None else {
        "bias_update_time_s": None,
        "average_time_per_bias_update_s": None,
        "number_of_bias_updates": 0,
    }
    return {
        "mode": mode,
        "repeat": repeat_index,
        "platform": context.getPlatform().getName(),
        "atom_count": evb_system.system.getNumParticles(),
        "context_creation_time_s": context_creation_s,
        "md_wall_time_s": md_wall_s,
        "steps": steps,
        "steps_per_s": steps / md_wall_s if md_wall_s else None,
        "ns_per_day": (steps * config.simulation.timestep_fs * 1.0e-6) / (md_wall_s / 86400.0) if md_wall_s else None,
        "number_of_custom_cv_forces": count_custom_cv(evb_system.system),
        "number_of_inner_cv_contexts": report.get("custom_cv_inner_context_count"),
        "duplicated_full_nonbonded": report.get("duplicated_full_nonbonded"),
        "e_common_inside_custom_cv": report.get("e_common_inside_custom_cv"),
        "use_app_metadynamics": mode == "app-metad",
        "use_bias_variable": mode == "app-metad",
        "mean_evb_energy_kj_mol": evb_energy,
        "shifted_gap_kj_mol": energy1 - energy2 - parameters.delta_alpha,
        **timing,
    }


def summarize(records):
    summary = {"records": records}
    for mode in sorted({r["mode"] for r in records}):
        ok = [r for r in records if r["mode"] == mode and r.get("steps_per_s") is not None]
        if ok:
            summary[mode] = {
                "mean_steps_per_s": float(np.mean([r["steps_per_s"] for r in ok])),
                "std_steps_per_s": float(np.std([r["steps_per_s"] for r in ok])),
                "mean_ns_per_day": float(np.mean([r["ns_per_day"] for r in ok])),
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--platform", default="CPU")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", default="plain,app-metad,table-metad")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    native = config.sampling.native_gap_bias
    meta = config.sampling.metadynamics
    if native.min_value is None:
        native.min_value = meta.min_value if meta.min_value is not None else -10.0
    if native.max_value is None:
        native.max_value = meta.max_value if meta.max_value is not None else 10.0
    if native.grid_width is None:
        native.grid_width = meta.grid_width if meta.grid_width is not None else 101
    if native.bias_width is None:
        native.bias_width = meta.bias_width if meta.bias_width is not None else 0.5
    if native.height_kj_mol is None:
        native.height_kj_mol = meta.height_kj_mol
    if native.bias_factor is None:
        native.bias_factor = meta.bias_factor
    if native.frequency is None:
        native.frequency = meta.frequency
    if native.temperature_k is None:
        native.temperature_k = config.simulation.temperature_k
    config.simulation.platform = args.platform
    records = []
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        for repeat in range(args.repeats):
            records.append(run_repeat(config, mode, args.platform, args.steps, repeat))
    payload = summarize(records)
    payload["git_sha"] = git_sha()
    payload["config"] = args.config
    payload["platform_requested"] = args.platform
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
