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
from kemp_evb.openmm_backend import AmberSystemLoader, EVBSystemBuilder
from kemp_evb.simulation import EVBSimulation, create_integrator

try:
    import openmm
    from openmm import unit
except ImportError:  # pragma: no cover
    openmm = None
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


def run_once(args: argparse.Namespace, repeat_index: int) -> dict[str, Any]:
    if args.mode == "decomposed" and not hasattr(EVBSystemBuilder, "build_openmm_evb_system_decomposed"):
        return {
            "repeat": repeat_index,
            "status": "skipped",
            "warnings": ["Decomposed mode is not available in this checkout."],
        }

    config = load_config(args.config)
    config.simulation.platform = args.platform
    builder = EVBSystemBuilder(
        AmberSystemLoader(
            nonbonded_method=config.simulation.nonbonded_method,
            constraints=config.simulation.constraints,
        )
    )
    t0 = time.perf_counter()
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    parameters = load_or_calibrate_parameters(config)
    if args.mode == "decomposed":
        evb_system = builder.build_openmm_evb_system_decomposed(
            state1,
            state2,
            delta_alpha=parameters.delta_alpha,
            h12=parameters.h12,
        )
    else:
        evb_system = builder.build_openmm_evb_system(
            state1,
            state2,
            delta_alpha=parameters.delta_alpha,
            h12=parameters.h12,
        )
    build_time = time.perf_counter() - t0

    integrator = create_integrator(
        timestep_fs=config.simulation.timestep_fs,
        temperature_k=config.simulation.temperature_k,
        friction_per_ps=config.simulation.friction_per_ps,
        integrator_name=config.simulation.integrator,
    )
    platform = get_platform(args.platform, require=args.require_inputs)
    t_context0 = time.perf_counter()
    simulation = EVBSimulation(evb_system=evb_system, integrator=integrator, platform_name=platform.getName())
    simulation.set_positions(select_start_positions(config, state1.positions_nm, state2.positions_nm))
    context_time = time.perf_counter() - t_context0

    minimization_time = None
    if not args.skip_md and config.simulation.minimize_steps:
        t_min0 = time.perf_counter()
        simulation.minimize(
            tolerance_kjmol_per_mol_nm=config.simulation.minimize_tolerance,
            max_iterations=config.simulation.minimize_steps,
        )
        minimization_time = time.perf_counter() - t_min0

    simulation.set_velocities_to_temperature(config.simulation.temperature_k, seed=config.simulation.seed + repeat_index)
    energies = []
    gaps = []
    force_norms = []
    t_md0 = time.perf_counter()
    if not args.skip_md and args.steps > 0:
        report_interval = max(1, args.steps // 10)
        for start in range(0, args.steps, report_interval):
            simulation.integrator.step(min(report_interval, args.steps - start))
            result = simulation.single_point()
            energies.append(result.evb_energy)
            gaps.append(result.energy1 - result.energy2 - parameters.delta_alpha)
            force_norms.append(float(np.mean(np.linalg.norm(result.forces, axis=1))))
    else:
        result = simulation.single_point()
        energies.append(result.evb_energy)
        gaps.append(result.energy1 - result.energy2 - parameters.delta_alpha)
        force_norms.append(float(np.mean(np.linalg.norm(result.forces, axis=1))))
    md_wall_time = time.perf_counter() - t_md0
    steps_per_second = (args.steps / md_wall_time) if md_wall_time > 0 and not args.skip_md else None
    timestep_ps = config.simulation.timestep_fs * 1.0e-3
    ns_per_day = (steps_per_second * timestep_ps * 86400.0 / 1000.0) if steps_per_second is not None else None

    report = decomposition_report(evb_system, args.mode)
    force_count = evb_system.system.getNumForces()
    return {
        "repeat": repeat_index,
        "status": "ok",
        "build_time_seconds": build_time,
        "context_creation_time_seconds": context_time,
        "minimization_time_seconds": minimization_time,
        "md_wall_time_seconds": md_wall_time,
        "steps_per_second": steps_per_second,
        "ns_per_day": ns_per_day,
        "mean_evb_energy_kj_mol": float(np.mean(energies)),
        "mean_shifted_gap_kj_mol": float(np.mean(gaps)),
        "mean_absolute_force_norm_kj_mol_nm": float(np.mean(force_norms)),
        "number_of_atoms": int(evb_system.system.getNumParticles()),
        "number_of_forces": int(force_count),
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
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=["legacy", "decomposed"], default="legacy")
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-md", action="store_true")
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
        "platform_requested": args.platform,
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
