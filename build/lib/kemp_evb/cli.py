from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import EVBConfig, load_config, validate_config
from .analysis import build_analysis_report, build_coupling_scan, write_coupling_scan_outputs, write_reaction_coordinate_plots
from .engine import validate_diabatic_states
from .evb import EVBParameters, calibrate_evb_parameters
from .fitting import fit_bootstrap_parameters, fit_ensemble_parameters
from .io import write_json
from .observables import compute_named_distances, compute_named_reaction_coordinates, make_gap_sample
from .openmm_backend import (
    AmberSystemLoader,
    EVBSystemBuilder,
    OpenMMStateEvaluator,
    create_dcd_writer,
    load_positions_file,
    write_dcd_frame,
    write_pdb,
)
from .simulation import EVBSimulation, create_integrator, ensure_output_dir
from .sampling import (
    run_gap_umbrella_series,
    run_gap_umbrella_window,
    run_mapping_series,
    run_mapping_window,
    run_proton_transfer_umbrella_series,
    run_proton_transfer_umbrella_window,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="evb")
    parser.add_argument(
        "command",
        choices=[
            "calibrate",
            "validate",
            "singlepoint",
            "minimize",
            "md",
            "run-md",
            "validate-states",
            "gap-eval",
            "sample-window",
            "sample-series",
            "analyze",
            "analyze-gap",
            "reconstruct-barrier",
            "report",
            "fit",
            "fit-bootstrap",
            "fit-ensemble",
            "scan-coupling",
            "plot-2d-rc",
            "plumed-md",
            "make-template",
        ],
    )
    parser.add_argument("--config", help="Path to YAML/JSON config file")
    parser.add_argument("--coords", help="Optional coordinates file for evaluation commands")
    parser.add_argument("--window", help="Window id for sample-window, e.g. w000")
    parser.add_argument("--h12-values", help="Comma-separated H12 values in kJ/mol for scan-coupling")
    parser.add_argument("--qm-guide", help="Optional QM geometry guide JSON for 2D reaction-coordinate plotting")
    parser.add_argument("--kind", choices=["solution", "enzyme", "toy"], help="Template kind for make-template")
    args = parser.parse_args()

    if args.command == "make-template":
        run_make_template(args.kind or "toy")
        return
    if not args.config:
        raise ValueError("--config is required for this command.")
    config = load_config(args.config)
    if args.command == "validate":
        run_validate(config)
    elif args.command == "calibrate":
        run_calibrate(config)
    elif args.command == "singlepoint":
        run_singlepoint(config)
    elif args.command == "minimize":
        run_minimize(config)
    elif args.command in {"md", "run-md"}:
        run_md(config)
    elif args.command == "validate-states":
        run_validate_states(config)
    elif args.command == "gap-eval":
        run_gap_eval(config, args.coords)
    elif args.command == "sample-window":
        if not args.window:
            raise ValueError("--window is required for sample-window.")
        run_sample_window(config, args.window)
    elif args.command == "sample-series":
        run_sample_series(config)
    elif args.command in {"analyze", "analyze-gap"}:
        run_analyze_gap(config)
    elif args.command == "reconstruct-barrier":
        run_reconstruct_barrier(config)
    elif args.command == "report":
        run_report(config)
    elif args.command == "fit":
        run_fit_ensemble(config)
    elif args.command == "fit-bootstrap":
        run_fit_bootstrap(config)
    elif args.command == "fit-ensemble":
        run_fit_ensemble(config)
    elif args.command == "scan-coupling":
        if not args.h12_values:
            raise ValueError("--h12-values is required for scan-coupling.")
        run_scan_coupling(config, args.h12_values)
    elif args.command == "plot-2d-rc":
        run_plot_2d_rc(config, args.qm_guide)
    elif args.command == "plumed-md":
        run_plumed_md(config, Path(args.config).parent)


def run_validate(config: EVBConfig) -> None:
    errors = validate_config(config)
    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    run_validate_states(config)


def run_calibrate(config: EVBConfig) -> EVBParameters:
    if config.calibration is None:
        raise ValueError("Calibration data is required for the calibrate command.")
    mm_values = resolve_mm_calibration_energies(config)
    params = calibrate_evb_parameters(
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
    output_dir = ensure_output_dir(config.output_dir)
    params.to_json(output_dir / "calibrated_evb_parameters.json")
    with (output_dir / "calibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"parameters": asdict(params), "mm_energies": mm_values}, handle, indent=2)
    return params


def run_singlepoint(config: EVBConfig) -> None:
    simulation = build_simulation(config, mode="singlepoint")
    result = simulation.single_point()
    output_dir = ensure_output_dir(config.output_dir)
    with (output_dir / "singlepoint.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "E1_kJmol": result.energy1,
                "E2_kJmol": result.energy2,
                "Eevb_kJmol": result.evb_energy,
                "weight1": result.weight1,
                "weight2": result.weight2,
                "cv": simulation.compute_cv(),
            },
            handle,
            indent=2,
        )
    write_pdb(str(output_dir / "singlepoint_positions.pdb"), simulation.topology, simulation.get_positions_nm())


def run_minimize(config: EVBConfig) -> None:
    simulation = build_simulation(config, mode="minimize")
    history = simulation.minimize(
        tolerance_kjmol_per_mol_nm=config.simulation.minimize_tolerance,
        max_iterations=config.simulation.minimize_steps,
    )
    output_dir = ensure_output_dir(config.output_dir)
    with (output_dir / "minimization_log.json").open("w", encoding="utf-8") as handle:
        json.dump([asdict(snapshot) for snapshot in history], handle, indent=2)
    write_pdb(str(output_dir / "minimized.pdb"), simulation.topology, simulation.get_positions_nm())


def run_md(config: EVBConfig) -> None:
    simulation = build_simulation(config, mode="md")
    simulation.set_velocities_to_temperature(config.simulation.temperature_k, seed=config.simulation.seed)
    output_dir = ensure_output_dir(config.output_dir)
    dcd_handle, dcd_writer = create_dcd_writer(
        str(output_dir / "evb_md.dcd"),
        simulation.topology,
        timestep_ps=config.simulation.timestep_fs * 1.0e-3,
    )
    try:
        simulation.run_md(
            steps=config.simulation.steps,
            report_interval=config.simulation.report_interval,
            trajectory_writer=lambda positions, step: write_dcd_frame(dcd_writer, positions, simulation.topology, step),
            log_path=str(output_dir / "evb_md_log.csv"),
        )
    finally:
        dcd_handle.close()
    write_pdb(str(output_dir / "evb_md_final.pdb"), simulation.topology, simulation.get_positions_nm())


def run_plumed_md(config: EVBConfig, base_dir: str | Path = ".") -> None:
    from .plumed import attach_plumed_force

    if not config.plumed.enabled:
        raise ValueError("plumed-md requires plumed.enabled: true in the config.")
    builder = EVBSystemBuilder(
        AmberSystemLoader(
            nonbonded_method=config.simulation.nonbonded_method,
            constraints=config.simulation.constraints,
        )
    )
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    parameters = load_or_calibrate_parameters(config)
    evb_system = builder.build_openmm_evb_system(
        state1,
        state2,
        delta_alpha=parameters.delta_alpha,
        h12=parameters.h12,
    )
    attach_plumed_force(evb_system.system, config.plumed, base_dir=base_dir)
    integrator = create_integrator(
        timestep_fs=config.simulation.timestep_fs,
        temperature_k=config.simulation.temperature_k,
        friction_per_ps=config.simulation.friction_per_ps,
        integrator_name=config.simulation.integrator,
    )
    simulation = EVBSimulation(evb_system=evb_system, integrator=integrator, platform_name=config.simulation.platform)
    simulation.set_positions(select_start_positions(config, state1.positions_nm, state2.positions_nm))
    simulation.set_velocities_to_temperature(config.simulation.temperature_k, seed=config.simulation.seed)
    output_dir = ensure_output_dir(config.output_dir)
    simulation.run_md(
        steps=config.simulation.steps,
        report_interval=config.simulation.report_interval,
        log_path=str(output_dir / "plumed_evb_md_log.csv"),
    )
    write_pdb(str(output_dir / "plumed_evb_md_final.pdb"), simulation.topology, simulation.get_positions_nm())


def run_validate_states(config: EVBConfig) -> None:
    loader = AmberSystemLoader(
        nonbonded_method=config.simulation.nonbonded_method,
        constraints=config.simulation.constraints,
    )
    state1 = EVBSystemBuilder(loader).load_state(config.state1)
    state2 = EVBSystemBuilder(loader).load_state(config.state2)
    report = validate_diabatic_states(state1, state2, config.reaction.atoms or config.cv)
    try:
        EVBSystemBuilder.validate_compatibility(state1, state2)
    except ValueError as exc:
        report.compatible = False
        report.notes.append(str(exc))
    output_dir = ensure_output_dir(config.output_dir)
    write_json(output_dir / "state_validation.json", report)


def run_gap_eval(config: EVBConfig, coords_path: str | None = None) -> None:
    simulation = build_simulation(config, mode="singlepoint")
    if coords_path:
        simulation.set_positions(load_positions_file(coords_path))
    result = simulation.single_point()
    parameters = simulation.parameters
    gap = make_gap_sample(
        result,
        frame=0,
        step=0,
        time_ps=0.0,
        delta_alpha_kj_mol=parameters.delta_alpha,
    )
    positions_nm = simulation.get_positions_nm()
    distances = compute_named_distances(positions_nm, config.observables.distances)
    reaction_coordinates = compute_named_reaction_coordinates(positions_nm, config.observables.reaction_coordinates)
    payload = {
        "gap": asdict(gap),
        "distances_nm": distances,
        "reaction_coordinates_nm": reaction_coordinates,
        "parameters": asdict(parameters),
    }
    output_dir = ensure_output_dir(config.output_dir)
    write_json(output_dir / "gap_eval.json", payload)


def run_sample_window(config: EVBConfig, window_id: str) -> None:
    if config.sampling.mode == "mapping":
        run_mapping_window(config, window_id)
        return
    if config.sampling.mode == "gap_umbrella":
        run_gap_umbrella_window(config, window_id)
        return
    if config.sampling.mode == "proton_transfer_umbrella":
        run_proton_transfer_umbrella_window(config, window_id)
        return
    raise ValueError(f"Unsupported sampling.mode {config.sampling.mode!r}.")


def run_sample_series(config: EVBConfig) -> None:
    if config.sampling.mode == "mapping":
        run_mapping_series(config)
        return
    if config.sampling.mode == "gap_umbrella":
        run_gap_umbrella_series(config)
        return
    if config.sampling.mode == "proton_transfer_umbrella":
        run_proton_transfer_umbrella_series(config)
        return
    raise ValueError(f"Unsupported sampling.mode {config.sampling.mode!r}.")


def run_analyze_gap(config: EVBConfig) -> None:
    build_analysis_report(config)


def run_reconstruct_barrier(config: EVBConfig) -> None:
    build_analysis_report(config)


def run_report(config: EVBConfig) -> None:
    report = build_analysis_report(config)
    output_dir = ensure_output_dir(config.output_dir)
    reports_dir = ensure_output_dir(output_dir / "reports")
    write_json(reports_dir / "summary.json", report)
    with (reports_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(f"# {config.project.name}\n\n")
        handle.write(f"- windows: {report['n_windows']}\n")
        handle.write(f"- frames: {report['n_frames_total']}\n")
        barrier = report["barrier_estimate"]
        handle.write(f"- forward barrier (kJ/mol): {barrier['barrier_forward_kj_mol']}\n")
        handle.write(f"- reaction free energy (kJ/mol): {barrier['reaction_free_energy_kj_mol']}\n")


def run_fit_bootstrap(config: EVBConfig) -> None:
    fit_bootstrap_parameters(config)


def run_fit_ensemble(config: EVBConfig) -> None:
    fit_ensemble_parameters(config)


def run_scan_coupling(config: EVBConfig, h12_values: str) -> None:
    values = [float(value.strip()) for value in h12_values.split(",") if value.strip()]
    curves = build_coupling_scan(config, values)
    output_dir = ensure_output_dir(Path(config.output_dir) / "analysis")
    write_coupling_scan_outputs(output_dir, curves)


def run_plot_2d_rc(config: EVBConfig, qm_guide: str | None) -> None:
    write_reaction_coordinate_plots(
        config.output_dir,
        qm_geometry_guide=qm_guide or str(Path("systems") / "KEMP_implicit" / "qm_geometry_guide.json"),
    )


def build_simulation(config: EVBConfig, mode: str) -> EVBSimulation:
    builder = EVBSystemBuilder(
        AmberSystemLoader(
            nonbonded_method=config.simulation.nonbonded_method,
            constraints=config.simulation.constraints,
        )
    )
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    parameters = load_or_calibrate_parameters(config)
    evb_system = builder.build_openmm_evb_system(
        state1,
        state2,
        delta_alpha=parameters.delta_alpha,
        h12=parameters.h12,
    )
    integrator_name = config.simulation.integrator if mode == "md" else "Verlet"
    integrator = create_integrator(
        timestep_fs=config.simulation.timestep_fs,
        temperature_k=config.simulation.temperature_k,
        friction_per_ps=config.simulation.friction_per_ps,
        integrator_name=integrator_name,
    )
    simulation = EVBSimulation(
        evb_system=evb_system,
        integrator=integrator,
        platform_name=config.simulation.platform,
        cv_atoms=None if config.cv is None else (config.cv.donor, config.cv.proton, config.cv.acceptor),
    )
    simulation.set_positions(select_start_positions(config, state1.positions_nm, state2.positions_nm))
    return simulation


def resolve_mm_calibration_energies(config: EVBConfig) -> dict[str, float]:
    if config.calibration is None:
        raise ValueError("Calibration section is required.")
    cal = config.calibration
    explicit = {
        "min1_state1": cal.e_mm_min1_state1,
        "min1_state2": cal.e_mm_min1_state2,
        "min2_state1": cal.e_mm_min2_state1,
        "min2_state2": cal.e_mm_min2_state2,
        "ts_state1": cal.e_mm_ts_state1,
        "ts_state2": cal.e_mm_ts_state2,
    }
    if all(value is not None for value in explicit.values()):
        return {key: float(value) for key, value in explicit.items()}
    builder = EVBSystemBuilder(
        AmberSystemLoader(
            nonbonded_method=config.simulation.nonbonded_method,
            constraints=config.simulation.constraints,
        )
    )
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    eval1 = OpenMMStateEvaluator(state1, platform_name=config.simulation.platform)
    eval2 = OpenMMStateEvaluator(state2, platform_name=config.simulation.platform)
    positions = {
        "min1": _resolve_calibration_positions(cal.coordinates.min1, state1.positions_nm),
        "min2": _resolve_calibration_positions(cal.coordinates.min2, state2.positions_nm),
        "ts": _resolve_calibration_positions(cal.coordinates.ts, None),
    }
    if positions["ts"] is None:
        raise ValueError("TS coordinates are required when TS MM energies are not supplied.")
    resolved: dict[str, float] = {}
    for label, coords in positions.items():
        energy1, _ = eval1.evaluate(coords)
        energy2, _ = eval2.evaluate(coords)
        resolved[f"{label}_state1"] = energy1
        resolved[f"{label}_state2"] = energy2
    return resolved


def _resolve_calibration_positions(path: str | None, default):
    if path:
        return load_positions_file(path)
    return default


def load_or_calibrate_parameters(config: EVBConfig) -> EVBParameters:
    delta_alpha = config.evb_parameters.delta_alpha
    h12 = config.evb_parameters.h12
    if delta_alpha is not None and h12 is not None:
        return EVBParameters(delta_alpha=delta_alpha, h12=h12)
    if config.calibration is None:
        raise ValueError("Either EVB parameters or calibration data must be provided.")
    return run_calibrate(config)


def select_start_positions(config: EVBConfig, state1_positions, state2_positions):
    if config.start_state == "state2":
        return state2_positions
    return state1_positions


def run_make_template(kind: str) -> None:
    from .plumed import PLUMED_TEMPLATES

    template_path = Path("examples") / f"{kind}_template.yaml"
    if kind == "toy":
        template_path = Path("examples") / "toy_evb.yaml"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "toy":
        content = _toy_template()
    elif kind == "solution":
        content = _solution_template()
    else:
        content = _enzyme_template()
    template_path.write_text(content, encoding="utf-8")
    if kind == "toy":
        from .openmm_backend import write_toy_evb_bundles

        write_toy_evb_bundles(Path("examples"))
        (Path("examples") / "plumed_opes_template.dat").write_text(PLUMED_TEMPLATES["opes_metad"], encoding="utf-8")


def _toy_template() -> str:
    return """project:
  name: toy-evb
  output_dir: outputs/toy
states:
  state1:
    format: openmm
    topology: examples/toy_state1/system.xml
    coordinates: examples/toy_state1/coordinates.pdb
  state2:
    format: openmm
    topology: examples/toy_state2/system.xml
    coordinates: examples/toy_state2/coordinates.pdb
evb:
  coupling_model:
    model: constant
    parameters:
      delta_alpha_kj_mol: 2.0
      h12_kj_mol: 5.0
sampling:
  mode: mapping
  windows:
    mapping:
      lambda_values: [0.0, 0.5, 1.0]
  md:
    production_steps: 100
    report_stride: 10
"""


def _solution_template() -> str:
    return _toy_template().replace("toy-evb", "solution-evb").replace("outputs/toy", "outputs/solution")


def _enzyme_template() -> str:
    return _toy_template().replace("toy-evb", "enzyme-evb").replace("outputs/toy", "outputs/enzyme")
