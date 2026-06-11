from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import EVBConfig, IRCSettings, ReferenceProfileSettings, load_config, validate_config
from .analysis import build_analysis_report, build_coupling_scan, write_coupling_scan_outputs, write_reaction_coordinate_plots
from .engine import validate_diabatic_states
from .evb import EVBHamiltonian, EVBParameters, calibrate_evb_parameters
from .evb_inputs import prepare_adiabatic_system_from_irc, prepare_evb_ready_inputs
from .fitting import fit_bootstrap_parameters, fit_ensemble_parameters
from .io import write_json
from .irc import read_irc_xyz, write_irc_outputs
from .irc_setup import setup_from_irc
from .hg317_prep import prepare_hg317_system
from .observables import compute_named_distances, compute_named_reaction_coordinates, make_gap_sample
from .openmm_backend import (
    AmberSystemLoader,
    EVBSystemBuilder,
    OpenMMStateEvaluator,
    build_evb_gap_cv_force,
    evb_diabatic_energies,
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
            "irc-analyze",
            "setup-from-irc",
            "prepare-evb-inputs",
            "prepare-adiabatic-system",
            "prepare-hg317-system",
            "plumed-md",
            "evb-metad",
            "evb-opes",
            "evb-gap-metad",
            "evb-gap-opes",
            "make-template",
        ],
    )
    parser.add_argument("--config", help="Path to YAML/JSON config file")
    parser.add_argument("--coords", help="Optional coordinates file for evaluation commands")
    parser.add_argument("--irc", help="Multi-frame XYZ IRC file for irc-analyze or setup-from-irc")
    parser.add_argument("--irc-order", choices=["rc_ts_prod", "prod_ts_rc", "auto"], help="IRC input order for setup-from-irc")
    parser.add_argument("--rc-frame", type=int, help="Original 0-based IRC frame index for RC")
    parser.add_argument("--ts-frame", type=int, help="Original 0-based IRC frame index for TS")
    parser.add_argument("--product-frame", type=int, help="Original 0-based IRC frame index for product")
    parser.add_argument("--reference-units", choices=["kJ/mol", "kj/mol", "kcal/mol"], help="Reference profile energy units")
    parser.add_argument("--g-rc", type=float, help="Reference RC free energy")
    parser.add_argument("--g-ts", type=float, help="Reference TS free energy")
    parser.add_argument("--g-product", type=float, help="Reference product free energy")
    parser.add_argument("--write-window-config", action="store_true", help="Write IRC-derived gap umbrella window proposal")
    parser.add_argument("--dry-run", action="store_true", help="Parse config and report planned setup without running MD")
    parser.add_argument("--execute", action="store_true", help="Execute external preparation tools when supported by the command")
    parser.add_argument("--no-sampling", action="store_true", help="Do not start sampling after setup-from-irc")
    parser.add_argument("--output", help="Output directory for commands that do not require --config")
    parser.add_argument("--reactive-atom", action="append", type=int, default=[], help="Extra 0-based OpenMM reactive atom index for prepare-evb-inputs")
    parser.add_argument("--reactive-pair", action="append", default=[], help="Extra 0-based reactive atom pair i,j for prepare-evb-inputs")
    parser.add_argument("--window", help="Window id for sample-window, e.g. w000")
    parser.add_argument("--h12-values", help="Comma-separated H12 values in kJ/mol for scan-coupling")
    parser.add_argument("--qm-guide", help="Optional QM geometry guide JSON for 2D reaction-coordinate plotting")
    parser.add_argument("--kind", choices=["solution", "enzyme", "toy"], help="Template kind for make-template")
    args = parser.parse_args()

    if args.command == "make-template":
        run_make_template(args.kind or "toy")
        return
    if args.command == "irc-analyze":
        if not args.irc:
            raise ValueError("--irc is required for irc-analyze.")
        run_irc_analyze(args.irc, args.output)
        return
    if not args.config:
        raise ValueError("--config is required for this command.")
    if args.command == "prepare-hg317-system":
        print(json.dumps(prepare_hg317_system(args.config, execute=args.execute), indent=2))
        return
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
    elif args.command == "setup-from-irc":
        run_setup_from_irc(config, args)
    elif args.command == "prepare-evb-inputs":
        run_prepare_evb_inputs(config, args)
    elif args.command == "prepare-adiabatic-system":
        run_prepare_adiabatic_system(config, args)
    elif args.command == "plumed-md":
        run_plumed_md(config, Path(args.config).parent)
    elif args.command == "evb-metad":
        config.plumed.mode = "metad"
        run_plumed_md(config, Path(args.config).parent, expected_mode="metad")
    elif args.command == "evb-opes":
        config.plumed.mode = "opes"
        run_plumed_md(config, Path(args.config).parent, expected_mode="opes")
    elif args.command == "evb-gap-metad":
        run_gap_metadynamics(config)
    elif args.command == "evb-gap-opes":
        raise ValueError(
            "Direct OPES on the EVB energy gap is not implemented. PLUMED cannot see the internal OpenMM EVB gap "
            "without a supported bridge. Use evb-gap-metad for native OpenMM gap metadynamics, or evb-opes for "
            "PLUMED OPES on geometrical CVs."
        )


def run_setup_from_irc(config: EVBConfig, args: argparse.Namespace) -> None:
    if config.irc is None:
        config.irc = IRCSettings()
    if args.irc:
        config.irc.path = args.irc
    if args.irc_order:
        config.irc.order = args.irc_order
    if args.rc_frame is not None:
        config.irc.rc_frame = args.rc_frame
    if args.ts_frame is not None:
        config.irc.ts_frame = args.ts_frame
    if args.product_frame is not None:
        config.irc.product_frame = args.product_frame
    if any(value is not None for value in (args.reference_units, args.g_rc, args.g_ts, args.g_product)):
        existing = config.reference_profile or ReferenceProfileSettings()
        config.reference_profile = ReferenceProfileSettings(
            units=args.reference_units or existing.units,
            rc=args.g_rc if args.g_rc is not None else existing.rc,
            ts=args.g_ts if args.g_ts is not None else existing.ts,
            product=args.g_product if args.g_product is not None else existing.product,
            source_label=existing.source_label,
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "irc_path": config.irc.path,
                    "irc_order": config.irc.order,
                    "reference_profile": None if config.reference_profile is None else asdict(config.reference_profile),
                    "sampling_will_run": False,
                    "note": "setup-from-irc only performs scan, fit, diagnostics, and optional window proposal; it never starts MD sampling.",
                },
                indent=2,
            )
        )
        return
    report = setup_from_irc(config, write_window_config=args.write_window_config)
    print(json.dumps(report, indent=2))


def run_prepare_evb_inputs(config: EVBConfig, args: argparse.Namespace) -> None:
    pairs = []
    for text in args.reactive_pair:
        left, right = text.replace(":", ",").split(",", 1)
        pairs.append((int(left), int(right)))
    report = prepare_evb_ready_inputs(
        config,
        config_path=args.config,
        output_dir=args.output,
        extra_reactive_atoms=args.reactive_atom,
        extra_reactive_pairs=pairs,
    )
    print(json.dumps(report, indent=2))


def run_prepare_adiabatic_system(config: EVBConfig, args: argparse.Namespace) -> None:
    pairs = []
    for text in args.reactive_pair:
        left, right = text.replace(":", ",").split(",", 1)
        pairs.append((int(left), int(right)))
    report = prepare_adiabatic_system_from_irc(
        config,
        config_path=args.config,
        output_dir=args.output,
        write_window_config=args.write_window_config,
        extra_reactive_atoms=args.reactive_atom,
        extra_reactive_pairs=pairs,
    )
    print(json.dumps(report, indent=2))


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
    output_dir = ensure_output_dir(config.output_dir).resolve()
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
                "energy_decomposition": simulation.evb_system.energy_decomposition_report,
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


def run_plumed_md(config: EVBConfig, base_dir: str | Path = ".", expected_mode: str | None = None) -> None:
    from .plumed import attach_plumed_force, load_plumed_script, validate_plumed_script

    if not config.plumed.enabled:
        raise ValueError("plumed-md requires plumed.enabled: true in the config.")
    script = load_plumed_script(config.plumed, base_dir=base_dir)
    mode = expected_mode or config.plumed.mode
    if mode == "metad" and "METAD" not in script.upper():
        raise ValueError("evb-metad requires a PLUMED script containing a METAD action.")
    if mode in {"opes", "opes_metad"} and "OPES_METAD" not in script.upper():
        raise ValueError("evb-opes requires a PLUMED script containing an OPES_METAD action.")
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
        energy_decomposition=config.energy_decomposition.enabled,
        energy_decomposition_mode=config.energy_decomposition.mode,
        fallback_to_legacy_for_unsupported_terms=config.energy_decomposition.fallback_to_legacy_for_unsupported_terms,
        report_energy_decomposition=config.energy_decomposition.report,
    )
    output_dir = ensure_output_dir(config.output_dir)
    write_json(
        output_dir / "plumed_evb_setup.json",
        {
            "mode": mode,
            "platform": config.simulation.platform,
            "timestep_fs": config.simulation.timestep_fs,
            "temperature_k": config.simulation.temperature_k,
            "friction_per_ps": config.simulation.friction_per_ps,
            "production_steps": config.simulation.steps,
            "report_interval": config.simulation.report_interval,
            "save_stride": config.sampling.md.save_stride,
            "warnings": validate_plumed_script(config.plumed, script),
            "scientific_scope": (
                "PLUMED biases geometrical CVs on the OpenMM EVB lower-surface system. "
                "The internal EVB energy gap is logged by OpenMM but is not directly visible to PLUMED."
            ),
        },
    )
    attach_plumed_force(evb_system.system, config.plumed, base_dir=base_dir)
    integrator = create_integrator(
        timestep_fs=config.simulation.timestep_fs,
        temperature_k=config.simulation.temperature_k,
        friction_per_ps=config.simulation.friction_per_ps,
        integrator_name=config.simulation.integrator,
    )
    old_cwd = Path.cwd()
    try:
        os.chdir(output_dir)
        simulation = EVBSimulation(evb_system=evb_system, integrator=integrator, platform_name=config.simulation.platform)
        simulation.set_positions(select_start_positions(config, state1.positions_nm, state2.positions_nm))
        if config.simulation.minimize_steps:
            simulation.minimize(
                tolerance_kjmol_per_mol_nm=config.simulation.minimize_tolerance,
                max_iterations=config.simulation.minimize_steps,
            )
        simulation.set_velocities_to_temperature(config.simulation.temperature_k, seed=config.simulation.seed)
        _run_plumed_evb_observable_md(simulation, config, output_dir)
    finally:
        os.chdir(old_cwd)
    write_pdb(str(output_dir / "plumed_evb_md_final.pdb"), simulation.topology, simulation.get_positions_nm())


def _run_plumed_evb_observable_md(simulation: EVBSimulation, config: EVBConfig, output_dir: Path) -> None:
    log_path = output_dir / "plumed_evb_observables.csv"
    save_stride = config.sampling.md.save_stride or config.simulation.report_interval
    dcd_handle, dcd_writer = create_dcd_writer(
        str(output_dir / "plumed_evb_md.dcd"),
        simulation.topology,
        timestep_ps=config.simulation.timestep_fs * 1.0e-3,
    )
    distance_names = [definition.name for definition in config.observables.distances]
    reaction_coordinate_names = [definition.name for definition in config.observables.reaction_coordinates]
    header = [
        "step",
        "time_ps",
        "E1_kJmol",
        "E2_kJmol",
        "gap_raw_kJmol",
        "gap_shifted_kJmol",
        "Eevb_kJmol",
        "w1",
        "w2",
    ]
    header += [f"distance_{name}_nm" for name in distance_names]
    header += [f"rc_{name}_nm" for name in reaction_coordinate_names]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        try:
            for start in range(0, config.simulation.steps, config.simulation.report_interval):
                advance = min(config.simulation.report_interval, config.simulation.steps - start)
                simulation.integrator.step(advance)
                step = start + advance
                result = simulation.single_point()
                positions_nm = simulation.get_positions_nm()
                distances = compute_named_distances(positions_nm, config.observables.distances)
                reaction_coordinates = compute_named_reaction_coordinates(
                    positions_nm, config.observables.reaction_coordinates
                )
                gap_raw = result.energy1 - result.energy2
                gap_shifted = gap_raw - simulation.parameters.delta_alpha
                row = [
                    step,
                    simulation._time_ps(),
                    result.energy1,
                    result.energy2,
                    gap_raw,
                    gap_shifted,
                    result.evb_energy,
                    result.weight1,
                    result.weight2,
                ]
                row += [distances[name] for name in distance_names]
                row += [reaction_coordinates[name] for name in reaction_coordinate_names]
                writer.writerow(row)
                if step % save_stride == 0:
                    write_dcd_frame(dcd_writer, positions_nm, simulation.topology, step)
        finally:
            dcd_handle.close()


def run_gap_metadynamics(config: EVBConfig) -> None:
    if config.plumed.enabled:
        raise ValueError(
            "evb-gap-metad uses native OpenMM metadynamics for the EVB energy gap. "
            "Disable plumed.enabled, or use evb-metad/evb-opes for geometrical PLUMED CVs."
        )
    errors = validate_config(config)
    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"- {error}" for error in errors))

    meta = config.sampling.metadynamics
    if meta.cv != "gap":
        raise ValueError("evb-gap-metad currently supports only sampling.metadynamics.cv: gap.")
    if meta.min_value is None or meta.max_value is None or meta.bias_width is None:
        raise ValueError("evb-gap-metad requires min_value, max_value, and bias_width in kJ/mol.")

    try:
        import openmm
        from openmm import app, unit
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenMM is required for evb-gap-metad.") from exc

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
        energy_decomposition=config.energy_decomposition.enabled,
        energy_decomposition_mode=config.energy_decomposition.mode,
        fallback_to_legacy_for_unsupported_terms=config.energy_decomposition.fallback_to_legacy_for_unsupported_terms,
        report_energy_decomposition=config.energy_decomposition.report,
    )
    if meta.wall_force_constant_kj_mol2 is not None:
        wall_cv = build_evb_gap_cv_force(
            state1.system,
            state2.system,
            parameters.delta_alpha,
            prefix="gap_wall",
            energy_decomposition=config.energy_decomposition.enabled,
            energy_decomposition_mode=config.energy_decomposition.mode,
            fallback_to_legacy_for_unsupported_terms=config.energy_decomposition.fallback_to_legacy_for_unsupported_terms,
        )
        wall_force = openmm.CustomCVForce(
            "0.5*k_gap_wall*(step(gap-gap_upper)*(gap-gap_upper)^2 + step(gap_lower-gap)*(gap-gap_lower)^2)"
        )
        wall_force.addCollectiveVariable("gap", wall_cv)
        wall_force.addGlobalParameter("k_gap_wall", float(meta.wall_force_constant_kj_mol2))
        wall_force.addGlobalParameter("gap_lower", float(meta.min_value))
        wall_force.addGlobalParameter("gap_upper", float(meta.max_value))
        evb_system.system.addForce(wall_force)
    gap_cv = build_evb_gap_cv_force(
        state1.system,
        state2.system,
        parameters.delta_alpha,
        prefix="gap_metad",
        energy_decomposition=config.energy_decomposition.enabled,
        energy_decomposition_mode=config.energy_decomposition.mode,
        fallback_to_legacy_for_unsupported_terms=config.energy_decomposition.fallback_to_legacy_for_unsupported_terms,
    )
    variable = app.BiasVariable(
        gap_cv,
        float(meta.min_value),
        float(meta.max_value),
        float(meta.bias_width),
        periodic=False,
        gridWidth=meta.grid_width,
    )

    output_dir = ensure_output_dir(config.output_dir).resolve()
    bias_dir = ensure_output_dir(output_dir / meta.bias_dir)
    metadynamics = app.Metadynamics(
        evb_system.system,
        [variable],
        config.simulation.temperature_k * unit.kelvin,
        float(meta.bias_factor),
        float(meta.height_kj_mol) * unit.kilojoule_per_mole,
        int(meta.frequency),
        saveFrequency=meta.save_frequency,
        biasDir=str(bias_dir),
    )
    integrator = create_integrator(
        timestep_fs=config.simulation.timestep_fs,
        temperature_k=config.simulation.temperature_k,
        friction_per_ps=config.simulation.friction_per_ps,
        integrator_name=config.simulation.integrator,
    )
    platform = openmm.Platform.getPlatformByName(config.simulation.platform) if config.simulation.platform else None
    if platform is None:
        app_simulation = app.Simulation(evb_system.topology, evb_system.system, integrator)
    else:
        app_simulation = app.Simulation(evb_system.topology, evb_system.system, integrator, platform)
    if evb_system.box_vectors_nm is not None:
        app_simulation.context.setPeriodicBoxVectors(*[vector * unit.nanometer for vector in evb_system.box_vectors_nm])
    app_simulation.context.setPositions(
        select_start_positions(config, state1.positions_nm, state2.positions_nm) * unit.nanometer
    )
    if config.simulation.minimize_steps:
        app_simulation.minimizeEnergy(
            tolerance=config.simulation.minimize_tolerance * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=config.simulation.minimize_steps,
        )
    app_simulation.context.setVelocitiesToTemperature(config.simulation.temperature_k * unit.kelvin, config.simulation.seed)

    _run_openmm_gap_metad_observable_md(app_simulation, evb_system, metadynamics, config, output_dir)
    positions_nm = app_simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(
        unit.nanometer
    )
    write_pdb(str(output_dir / "gap_metad_final.pdb"), evb_system.topology, positions_nm)
    write_json(
        output_dir / "gap_metad_setup.json",
        {
            "cv": "gap",
            "gap_units": "kJ/mol",
            "min_value_kj_mol": meta.min_value,
            "max_value_kj_mol": meta.max_value,
            "bias_width_kj_mol": meta.bias_width,
            "height_kj_mol": meta.height_kj_mol,
            "bias_factor": meta.bias_factor,
            "frequency_steps": meta.frequency,
            "wall_force_constant_kj_mol2": meta.wall_force_constant_kj_mol2,
            "bias_dir": str(bias_dir),
            "platform": app_simulation.context.getPlatform().getName(),
            "energy_decomposition": evb_system.energy_decomposition_report,
            "warning": (
                "This is native OpenMM metadynamics on the EVB energy gap, not PLUMED OPES. "
                "Use this for direct gap acceleration; use PLUMED only for geometry CVs until a gap bridge is implemented."
            ),
        },
    )


def _run_openmm_gap_metad_observable_md(app_simulation, evb_system, metadynamics, config: EVBConfig, output_dir: Path) -> None:
    from openmm import unit

    log_path = output_dir / "gap_metad_observables.csv"
    convergence_path = output_dir / "gap_metad_convergence.json"
    meta = config.sampling.metadynamics
    save_stride = config.sampling.md.save_stride or config.simulation.report_interval
    dcd_handle, dcd_writer = create_dcd_writer(
        str(output_dir / "gap_metad.dcd"),
        evb_system.topology,
        timestep_ps=config.simulation.timestep_fs * 1.0e-3,
    )
    distance_names = [definition.name for definition in config.observables.distances]
    reaction_coordinate_names = [definition.name for definition in config.observables.reaction_coordinates]
    header = [
        "step",
        "time_ps",
        "E1_kJmol",
        "E2_kJmol",
        "gap_raw_kJmol",
        "gap_shifted_kJmol",
        "Eevb_kJmol",
        "total_potential_kJmol",
        "w1",
        "w2",
    ]
    header += [f"distance_{name}_nm" for name in distance_names]
    header += [f"rc_{name}_nm" for name in reaction_coordinate_names]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        previous_fel: np.ndarray | None = None
        stable_checks = 0
        convergence_records: list[dict[str, float | int | bool | None]] = []
        stopped_by_convergence = False
        ts_sample_count = 0
        mixing_sample_count = 0
        max_weight2_seen = 0.0
        try:
            for start in range(0, config.simulation.steps, config.simulation.report_interval):
                advance = min(config.simulation.report_interval, config.simulation.steps - start)
                metadynamics.step(app_simulation, advance)
                step = start + advance
                context = app_simulation.context
                energy1, energy2 = evb_diabatic_energies(evb_system, context)
                parameters = EVBParameters(
                    delta_alpha=context.getParameter("delta_alpha"),
                    h12=context.getParameter("h12"),
                )
                eevb, w1, w2 = EVBHamiltonian(parameters).lower_eigenvalue(energy1, energy2)
                state = context.getState(getEnergy=True, getPositions=True)
                positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                distances = compute_named_distances(positions_nm, config.observables.distances)
                reaction_coordinates = compute_named_reaction_coordinates(
                    positions_nm, config.observables.reaction_coordinates
                )
                gap_raw = energy1 - energy2
                gap_shifted = gap_raw - parameters.delta_alpha
                row = [
                    step,
                    context.getTime().value_in_unit(unit.picoseconds),
                    energy1,
                    energy2,
                    gap_raw,
                    gap_shifted,
                    eevb,
                    state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                    w1,
                    w2,
                ]
                row += [distances[name] for name in distance_names]
                row += [reaction_coordinates[name] for name in reaction_coordinate_names]
                writer.writerow(row)
                max_weight2_seen = max(max_weight2_seen, float(w2))
                if (
                    meta.convergence_ts_window_kj_mol is not None
                    and abs(gap_shifted) <= meta.convergence_ts_window_kj_mol
                ):
                    ts_sample_count += 1
                    if meta.convergence_mixing_weight2_min <= w2 <= meta.convergence_mixing_weight2_max:
                        mixing_sample_count += 1
                if step % save_stride == 0:
                    write_dcd_frame(dcd_writer, positions_nm, evb_system.topology, step)
                should_check = (
                    meta.convergence_check_interval is not None
                    and meta.convergence_tolerance_kj_mol is not None
                    and step >= meta.convergence_min_steps
                    and step % meta.convergence_check_interval == 0
                )
                if should_check:
                    fel = metadynamics.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole)
                    fel_array = np.asarray(fel, dtype=float)
                    fel_array = fel_array - np.nanmin(fel_array)
                    rms_change = None
                    max_change = None
                    if previous_fel is not None:
                        delta = fel_array - previous_fel
                        rms_change = float(np.sqrt(np.nanmean(delta * delta)))
                        max_change = float(np.nanmax(np.abs(delta)))
                        mixing_ok = (
                            meta.convergence_min_weight2 is None
                            or max_weight2_seen >= meta.convergence_min_weight2
                        )
                        mixing_samples_ok = mixing_sample_count >= meta.convergence_min_mixing_samples
                        ts_sampling_ok = (
                            ts_sample_count >= meta.convergence_min_ts_samples
                        )
                        if (
                            rms_change <= meta.convergence_tolerance_kj_mol
                            and mixing_ok
                            and mixing_samples_ok
                            and ts_sampling_ok
                        ):
                            stable_checks += 1
                        else:
                            stable_checks = 0
                    previous_fel = fel_array
                    convergence_records.append(
                        {
                            "step": step,
                            "time_ps": context.getTime().value_in_unit(unit.picoseconds),
                            "rms_change_kj_mol": rms_change,
                            "max_change_kj_mol": max_change,
                            "stable_checks": stable_checks,
                            "required_stable_checks": meta.convergence_consecutive_checks,
                            "ts_sample_count": ts_sample_count,
                            "required_ts_samples": meta.convergence_min_ts_samples,
                            "mixing_sample_count": mixing_sample_count,
                            "required_mixing_samples": meta.convergence_min_mixing_samples,
                            "mixing_weight2_range": [
                                meta.convergence_mixing_weight2_min,
                                meta.convergence_mixing_weight2_max,
                            ],
                            "max_weight2_seen": max_weight2_seen,
                            "required_min_weight2": meta.convergence_min_weight2,
                            "converged": stable_checks >= meta.convergence_consecutive_checks,
                        }
                    )
                    write_json(
                        convergence_path,
                        {
                            "stopped_by_convergence": False,
                            "current_step": step,
                            "criteria": {
                                "rms_tolerance_kj_mol": meta.convergence_tolerance_kj_mol,
                                "consecutive_checks": meta.convergence_consecutive_checks,
                                "check_interval_steps": meta.convergence_check_interval,
                                "min_steps": meta.convergence_min_steps,
                                "ts_window_kj_mol": meta.convergence_ts_window_kj_mol,
                                "min_ts_samples": meta.convergence_min_ts_samples,
                                "min_weight2": meta.convergence_min_weight2,
                                "min_mixing_samples": meta.convergence_min_mixing_samples,
                                "mixing_weight2_min": meta.convergence_mixing_weight2_min,
                                "mixing_weight2_max": meta.convergence_mixing_weight2_max,
                            },
                            "records": convergence_records,
                        },
                    )
                    if stable_checks >= meta.convergence_consecutive_checks:
                        stopped_by_convergence = True
                        break
        finally:
            dcd_handle.close()
        write_json(
            convergence_path,
            {
                "stopped_by_convergence": stopped_by_convergence,
                "current_step": app_simulation.currentStep,
                "criteria": {
                    "rms_tolerance_kj_mol": meta.convergence_tolerance_kj_mol,
                    "consecutive_checks": meta.convergence_consecutive_checks,
                    "check_interval_steps": meta.convergence_check_interval,
                    "min_steps": meta.convergence_min_steps,
                    "ts_window_kj_mol": meta.convergence_ts_window_kj_mol,
                    "min_ts_samples": meta.convergence_min_ts_samples,
                    "min_weight2": meta.convergence_min_weight2,
                    "min_mixing_samples": meta.convergence_min_mixing_samples,
                    "mixing_weight2_min": meta.convergence_mixing_weight2_min,
                    "mixing_weight2_max": meta.convergence_mixing_weight2_max,
                },
                "ts_sample_count": ts_sample_count,
                "mixing_sample_count": mixing_sample_count,
                "max_weight2_seen": max_weight2_seen,
                "records": convergence_records,
            },
        )
        final_fel = np.asarray(metadynamics.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole), dtype=float)
        final_fel = final_fel - np.nanmin(final_fel)
        grid = np.linspace(float(meta.min_value), float(meta.max_value), final_fel.shape[0])
        with (output_dir / "gap_metad_fel.csv").open("w", newline="", encoding="utf-8") as fel_handle:
            fel_writer = csv.writer(fel_handle)
            fel_writer.writerow(["gap_kJmol", "free_energy_kJmol", "gap_kcalmol", "free_energy_kcalmol"])
            for gap_kj, free_kj in zip(grid, final_fel):
                fel_writer.writerow([gap_kj, float(free_kj), gap_kj / 4.184, float(free_kj) / 4.184])


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


def run_irc_analyze(irc_path: str, output_dir: str | None) -> None:
    frames = read_irc_xyz(irc_path)
    destination = output_dir or str(Path("outputs") / "irc" / Path(irc_path).stem)
    payload = write_irc_outputs(frames, destination, title=Path(irc_path).stem)
    print(json.dumps(payload, indent=2))


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
        energy_decomposition=config.energy_decomposition.enabled,
        energy_decomposition_mode=config.energy_decomposition.mode,
        fallback_to_legacy_for_unsupported_terms=config.energy_decomposition.fallback_to_legacy_for_unsupported_terms,
        report_energy_decomposition=config.energy_decomposition.report,
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
    if config.start_coordinates:
        return load_positions_file(config.start_coordinates)
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
