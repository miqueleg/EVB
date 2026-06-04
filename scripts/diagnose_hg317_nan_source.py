from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from kemp_evb.config import load_config
from kemp_evb.openmm_backend import EVBSystemBuilder, OpenMMStateEvaluator, load_positions_file
from kemp_evb.sampling.runners import MappingWindowRunner
from kemp_evb.sampling.windows import get_mapping_window

try:
    import openmm
    from openmm import unit
except ImportError as exc:  # pragma: no cover
    raise SystemExit("OpenMM is required for this diagnostic.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether HG3.17 NaNs come from diabatic systems, PBC, or EVB mapping.")
    parser.add_argument("--config", default="runs/hg317_evb_mapping/configs/rep01.yaml")
    parser.add_argument("--window", default="w020")
    parser.add_argument("--coords", default="prep/kemp_qm_openmm/05_templates/TS_solvated_template.pdb")
    parser.add_argument("--timesteps-fs", default="0.1,0.2,0.5,1.0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", default="outputs/hg317_nan_diagnostics/report.json")
    args = parser.parse_args()

    config = load_config(args.config)
    positions = load_positions_file(args.coords)
    builder = EVBSystemBuilder()
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    window = get_mapping_window(config, args.window)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "config": args.config,
        "window": args.window,
        "coords": args.coords,
        "timestep_fs": [float(value) for value in args.timesteps_fs.split(",") if value.strip()],
        "steps": args.steps,
        "state_particle_count": state1.system.getNumParticles(),
        "box_vectors_nm": None if state1.box_vectors_nm is None else np.asarray(state1.box_vectors_nm).tolist(),
        "coordinate_diagnostics": _coordinate_diagnostics(positions, state1.box_vectors_nm),
        "singlepoint": {},
        "dynamics": [],
    }

    for label, loaded_state in (("state1", state1), ("state2", state2)):
        report["singlepoint"][label] = _singlepoint_state(loaded_state, positions, platform_name=config.simulation.platform)
        report["singlepoint"][f"{label}_no_box"] = _singlepoint_state(
            _without_box(loaded_state),
            positions,
            platform_name=config.simulation.platform,
        )

    report["singlepoint"]["mapping"] = _singlepoint_mapping(config, window, state1, state2, positions, keep_box=True)
    report["singlepoint"]["mapping_no_box"] = _singlepoint_mapping(config, window, _without_box(state1), _without_box(state2), positions, keep_box=False)

    for timestep_fs in report["timestep_fs"]:
        for mode in ("state1", "state2", "mapping", "mapping_no_box"):
            result = _dynamics_check(config, window, state1, state2, positions, timestep_fs, args.steps, mode)
            report["dynamics"].append(result)
            print(f"dt={timestep_fs:g} fs mode={mode} ok={result['ok']} reason={result.get('error', '')[:100]}")

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


def _coordinate_diagnostics(positions_nm: np.ndarray, box_vectors_nm: np.ndarray | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "min_nm": np.min(positions_nm, axis=0).tolist(),
        "max_nm": np.max(positions_nm, axis=0).tolist(),
        "has_nan": bool(np.isnan(positions_nm).any()),
    }
    if box_vectors_nm is not None:
        lengths = np.asarray([box_vectors_nm[i][i] for i in range(3)], dtype=float)
        payload["box_lengths_nm"] = lengths.tolist()
        payload["atoms_below_zero"] = int(np.sum(np.any(positions_nm < 0.0, axis=1)))
        payload["atoms_above_box"] = int(np.sum(np.any(positions_nm > lengths, axis=1)))
    return payload


def _singlepoint_state(loaded_state, positions_nm: np.ndarray, platform_name: str | None) -> dict[str, Any]:
    try:
        evaluator = OpenMMStateEvaluator(loaded_state, platform_name=platform_name)
        energy, forces = evaluator.evaluate(positions_nm)
        return _energy_force_payload(energy, forces)
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _singlepoint_mapping(config, window, state1, state2, positions_nm: np.ndarray, keep_box: bool) -> dict[str, Any]:
    try:
        runner = MappingWindowRunner(config, window, state1=state1, state2=state2, initial_positions_nm=positions_nm)
        mapped = runner.builder.build_openmm_mapped_system(
            state1,
            state2,
            lambda_value=window.lambda_value,
            delta_alpha=runner.parameters.delta_alpha,
            far_field_restraint=None,
            unconstrained_atoms=set(),
        )
        if not keep_box:
            mapped.box_vectors_nm = None
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        platform = openmm.Platform.getPlatformByName(config.simulation.platform) if config.simulation.platform else None
        context = openmm.Context(mapped.system, integrator, platform) if platform is not None else openmm.Context(mapped.system, integrator)
        if keep_box and mapped.box_vectors_nm is not None:
            context.setPeriodicBoxVectors(*(vec * unit.nanometer for vec in mapped.box_vectors_nm))
        context.setPositions(positions_nm * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        return _energy_force_payload(float(energy), np.asarray(forces))
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _dynamics_check(config, window, state1, state2, positions_nm: np.ndarray, timestep_fs: float, steps: int, mode: str) -> dict[str, Any]:
    try:
        if mode in {"state1", "state2"}:
            loaded_state = state1 if mode == "state1" else state2
            system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(loaded_state.system))
            box_vectors_nm = loaded_state.box_vectors_nm
        else:
            s1 = state1 if mode == "mapping" else _without_box(state1)
            s2 = state2 if mode == "mapping" else _without_box(state2)
            runner = MappingWindowRunner(config, window, state1=s1, state2=s2, initial_positions_nm=positions_nm)
            mapped = runner.builder.build_openmm_mapped_system(
                s1,
                s2,
                lambda_value=window.lambda_value,
                delta_alpha=runner.parameters.delta_alpha,
                far_field_restraint=None,
                unconstrained_atoms=set(),
            )
            system = mapped.system
            box_vectors_nm = mapped.box_vectors_nm if mode == "mapping" else None
        integrator = openmm.LangevinMiddleIntegrator(
            config.simulation.temperature_k * unit.kelvin,
            config.simulation.friction_per_ps / unit.picosecond,
            timestep_fs * unit.femtoseconds,
        )
        platform = openmm.Platform.getPlatformByName(config.simulation.platform) if config.simulation.platform else None
        context = openmm.Context(system, integrator, platform) if platform is not None else openmm.Context(system, integrator)
        if box_vectors_nm is not None:
            context.setPeriodicBoxVectors(*(vec * unit.nanometer for vec in box_vectors_nm))
        context.setPositions(positions_nm * unit.nanometer)
        context.setVelocitiesToTemperature(config.simulation.temperature_k * unit.kelvin, config.simulation.seed)
        integrator.step(steps)
        state = context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        final_positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        return {"ok": bool(np.isfinite(energy) and np.isfinite(final_positions).all()), "mode": mode, "timestep_fs": timestep_fs, "energy_kj_mol": float(energy)}
    except Exception as exc:
        return {"ok": False, "mode": mode, "timestep_fs": timestep_fs, "error": repr(exc)}


def _energy_force_payload(energy: float, forces: np.ndarray) -> dict[str, Any]:
    finite_forces = np.asarray(forces, dtype=float)
    norms = np.linalg.norm(finite_forces, axis=1)
    return {
        "ok": bool(math.isfinite(energy) and np.isfinite(finite_forces).all()),
        "energy_kj_mol": float(energy),
        "max_force_kj_mol_nm": float(np.max(norms)),
        "mean_force_kj_mol_nm": float(np.mean(norms)),
        "max_force_atom": int(np.argmax(norms)),
    }


def _without_box(loaded_state):
    clone = copy.copy(loaded_state)
    clone.box_vectors_nm = None
    return clone


if __name__ == "__main__":
    main()
