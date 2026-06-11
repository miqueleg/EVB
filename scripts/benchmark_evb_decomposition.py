#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kemp_evb.cli import load_or_calibrate_parameters, select_start_positions
from kemp_evb.config import load_config
from kemp_evb.openmm_backend import AmberSystemLoader, EVBSystemBuilder, build_evb_gap_cv_force, evb_diabatic_energies
from kemp_evb.simulation import EVBSimulation, create_integrator

try:
    import openmm
    from openmm import app, unit
except ImportError:  # pragma: no cover
    openmm = None
    app = None
    unit = None


def parse_bool(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {text!r}.")


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def platform_available(name: str) -> bool:
    if openmm is None:
        return False
    try:
        openmm.Platform.getPlatformByName(name)
        return True
    except Exception:
        return False


def get_platform(name: str, require: bool):
    if openmm is None:
        raise ImportError("OpenMM is required for benchmarking.")
    try:
        return openmm.Platform.getPlatformByName(name)
    except Exception:
        if require:
            raise
        return openmm.Platform.getPlatformByName("CPU")


def platform_properties(args: argparse.Namespace) -> dict[str, str]:
    if args.platform == "CUDA" and args.cuda_precision:
        return {"Precision": args.cuda_precision}
    return {}


def decomposition_report(evb_system: Any, mode: str) -> dict[str, Any]:
    report = getattr(evb_system, "energy_decomposition_report", None)
    if report is not None:
        return dict(report)
    return {
        "enabled": mode == "decomposed",
        "mode": mode,
        "n_common_forces": 0,
        "n_state1_forces": 0,
        "n_state2_forces": 0,
        "n_common_terms": 0,
        "n_state1_terms": 0,
        "n_state2_terms": 0,
        "unsupported_forces": [],
        "warnings": [] if mode == "legacy" else ["Decomposed mode is not available in this checkout."],
    }


def _first_gap_window(config) -> tuple[float, float]:
    spec = config.sampling.windows.gap_umbrella
    centers = list(spec.centers_kj_mol)
    center = float(centers[len(centers) // 2]) if centers else 0.0
    k = float(spec.force_constant_kj_mol2) if spec.force_constant_kj_mol2 is not None else 0.0005
    return center, k


def _build_system(builder, state1, state2, params, config, args):
    enabled = args.mode == "decomposed"
    if args.workflow == "gap-umbrella":
        center, k = _first_gap_window(config)
        return builder.build_openmm_gap_umbrella_system(
            state1,
            state2,
            delta_alpha=params.delta_alpha,
            h12=params.h12,
            gap_center=center,
            gap_force_constant=k,
            energy_decomposition=enabled,
        )
    if enabled:
        return builder.build_openmm_evb_system_decomposed(state1, state2, params.delta_alpha, params.h12)
    return builder.build_openmm_evb_system(state1, state2, params.delta_alpha, params.h12)


def _snapshot(evb_system, context, params, get_forces: bool) -> tuple[float, float, float, float | None]:
    e1, e2 = evb_diabatic_energies(evb_system, context)
    gap = e1 - e2 - params.delta_alpha
    state = context.getState(getEnergy=True, getForces=get_forces)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    force_norm = None
    if get_forces:
        forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
        force_norm = float(np.mean(np.linalg.norm(forces, axis=1)))
    return float(energy), float(gap), float(e1), force_norm


def run_once(args: argparse.Namespace, repeat_index: int) -> dict[str, Any]:
    config = load_config(args.config)
    config.simulation.platform = args.platform
    config.sampling.md.platform = args.platform
    config.energy_decomposition.enabled = args.mode == "decomposed"
    builder = EVBSystemBuilder(
        AmberSystemLoader(
            nonbonded_method=config.simulation.nonbonded_method,
            constraints=config.simulation.constraints,
        )
    )
    t0 = time.perf_counter()
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    params = load_or_calibrate_parameters(config)
    evb_system = _build_system(builder, state1, state2, params, config, args)
    build_time = time.perf_counter() - t0

    integrator = create_integrator(
        timestep_fs=config.simulation.timestep_fs,
        temperature_k=config.simulation.temperature_k,
        friction_per_ps=config.simulation.friction_per_ps,
        integrator_name=config.simulation.integrator,
    )
    platform = get_platform(args.platform, require=args.require_inputs)
    properties = platform_properties(args)
    positions = select_start_positions(config, state1.positions_nm, state2.positions_nm)
    t_context0 = time.perf_counter()

    metadynamics = None
    if args.workflow == "gap-metad":
        if app is None:
            raise ImportError("OpenMM app module is required for metadynamics benchmarking.")
        meta = config.sampling.metadynamics
        gap_cv = build_evb_gap_cv_force(
            state1.system,
            state2.system,
            params.delta_alpha,
            prefix=f"bench_gap_metad_{repeat_index}",
            energy_decomposition=args.mode == "decomposed",
        )
        variable = app.BiasVariable(
            gap_cv,
            float(meta.min_value if meta.min_value is not None else -30000.0),
            float(meta.max_value if meta.max_value is not None else 30000.0),
            float(meta.bias_width if meta.bias_width is not None else 1000.0),
            periodic=False,
            gridWidth=meta.grid_width,
        )
        save_frequency = meta.save_frequency
        bias_dir = None
        if save_frequency is not None:
            bias_dir = Path(args.output).with_suffix("").parent / f"{Path(args.output).stem}_bias_rep{repeat_index}"
            bias_dir.mkdir(parents=True, exist_ok=True)
        metadynamics = app.Metadynamics(
            evb_system.system,
            [variable],
            config.simulation.temperature_k * unit.kelvin,
            float(meta.bias_factor),
            float(meta.height_kj_mol) * unit.kilojoule_per_mole,
            max(1, int(meta.frequency)),
            saveFrequency=save_frequency,
            biasDir=str(bias_dir) if bias_dir is not None else None,
        )
        app_sim = app.Simulation(evb_system.topology, evb_system.system, integrator, platform, properties)
        context = app_sim.context
    else:
        simulation = EVBSimulation(evb_system=evb_system, integrator=integrator, platform_name=platform.getName(), platform_properties=properties)
        context = simulation.context
        app_sim = None
    if evb_system.box_vectors_nm is not None:
        context.setPeriodicBoxVectors(*[vector * unit.nanometer for vector in evb_system.box_vectors_nm])
    context.setPositions(positions * unit.nanometer)
    context_time = time.perf_counter() - t_context0

    minimization_time = None
    if not args.skip_md and config.simulation.minimize_steps:
        t_min0 = time.perf_counter()
        openmm.LocalEnergyMinimizer.minimize(
            context,
            config.simulation.minimize_tolerance * unit.kilojoule_per_mole / unit.nanometer,
            config.simulation.minimize_steps,
        )
        minimization_time = time.perf_counter() - t_min0

    context.setVelocitiesToTemperature(config.simulation.temperature_k * unit.kelvin, config.simulation.seed + repeat_index)
    energies: list[float] = []
    gaps: list[float] = []
    force_norms: list[float] = []
    t_md0 = time.perf_counter()
    if not args.skip_md and args.steps > 0:
        report_interval = max(1, args.steps // 10)
        for start in range(0, args.steps, report_interval):
            advance = min(report_interval, args.steps - start)
            if metadynamics is not None:
                metadynamics.step(app_sim, advance)
            else:
                integrator.step(advance)
            energy, gap, _e1, force_norm = _snapshot(evb_system, context, params, get_forces=not args.no_forces)
            energies.append(energy)
            gaps.append(gap)
            if force_norm is not None:
                force_norms.append(force_norm)
    else:
        energy, gap, _e1, force_norm = _snapshot(evb_system, context, params, get_forces=not args.no_forces)
        energies.append(energy)
        gaps.append(gap)
        if force_norm is not None:
            force_norms.append(force_norm)
    md_wall_time = time.perf_counter() - t_md0
    steps_per_second = (args.steps / md_wall_time) if md_wall_time > 0 and not args.skip_md else None
    timestep_ps = config.simulation.timestep_fs * 1.0e-3
    ns_per_day = (steps_per_second * timestep_ps * 86400.0 / 1000.0) if steps_per_second is not None else None

    report = decomposition_report(evb_system, args.mode)
    return {
        "repeat": repeat_index,
        "status": "ok",
        "workflow": args.workflow,
        "build_time_seconds": build_time,
        "context_creation_time_seconds": context_time,
        "minimization_time_seconds": minimization_time,
        "md_wall_time_seconds": md_wall_time,
        "steps_per_second": steps_per_second,
        "ns_per_day": ns_per_day,
        "mean_evb_energy_kj_mol": float(np.mean(energies)),
        "mean_shifted_gap_kj_mol": float(np.mean(gaps)),
        "mean_absolute_force_norm_kj_mol_nm": float(np.mean(force_norms)) if force_norms else None,
        "number_of_atoms": int(evb_system.system.getNumParticles()),
        "number_of_forces": int(evb_system.system.getNumForces()),
        "number_of_common_forces": int(report.get("n_common_forces", 0)),
        "number_of_state_specific_forces": int(report.get("n_state1_forces", 0)) + int(report.get("n_state2_forces", 0)),
        "number_of_common_terms": int(report.get("n_common_terms", 0)),
        "number_of_state_specific_terms": int(report.get("n_state1_terms", 0)) + int(report.get("n_state2_terms", 0)),
        "warnings": list(report.get("warnings", [])),
        "decomposition_report": report,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [item for item in results if item.get("status") == "ok"]
    if not ok:
        return {}
    summary: dict[str, Any] = {}
    for key in (
        "context_creation_time_seconds",
        "minimization_time_seconds",
        "md_wall_time_seconds",
        "steps_per_second",
        "ns_per_day",
        "mean_evb_energy_kj_mol",
        "mean_shifted_gap_kj_mol",
        "mean_absolute_force_norm_kj_mol_nm",
    ):
        values = [item[key] for item in ok if item.get(key) is not None]
        if values:
            summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    for key in (
        "number_of_atoms",
        "number_of_forces",
        "number_of_common_forces",
        "number_of_state_specific_forces",
        "number_of_common_terms",
        "number_of_state_specific_terms",
    ):
        summary[key] = ok[0].get(key)
    warnings = []
    for item in ok:
        warnings.extend(item.get("warnings", []))
    summary["warnings"] = sorted(set(warnings))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--platform", choices=["CPU", "CUDA"], default="CPU")
    parser.add_argument("--cuda-precision", choices=["single", "mixed", "double"], default="mixed")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=["legacy", "decomposed"], default="legacy")
    parser.add_argument("--workflow", choices=["plain", "gap-umbrella", "gap-metad"], default="plain")
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-md", action="store_true")
    parser.add_argument("--no-forces", action="store_true")
    parser.add_argument("--require-inputs", type=parse_bool, default=False)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cuda_requested = args.platform == "CUDA"
    cuda_available = platform_available("CUDA")
    if cuda_requested and not cuda_available and args.require_inputs:
        raise RuntimeError("CUDA platform was requested but is not available.")

    results = [run_once(args, repeat) for repeat in range(args.repeats)]
    payload = {
        "config_path": args.config,
        "mode": args.mode,
        "workflow": args.workflow,
        "platform_requested": args.platform,
        "platform_properties": platform_properties(args),
        "cuda_available": cuda_available,
        "cuda_actually_used": cuda_requested and cuda_available,
        "steps": args.steps,
        "repeats": args.repeats,
        "skip_md": args.skip_md,
        "git_commit_sha": git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "openmm_version": None if openmm is None else openmm.version.full_version,
        "results": results,
        "summary": summarize(results),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
