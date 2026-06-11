from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import EVBConfig
from ..evb import EVBHamiltonian, EVBParameters, EVBResult
from ..io import write_json
from ..observables import compute_named_distances, compute_named_reaction_coordinates, make_gap_sample
from ..openmm_backend import (
    AmberSystemLoader,
    EVBOpenMMSystem,
    EVBSystemBuilder,
    MappedOpenMMSystem,
    build_absolute_positional_restraint_force,
    evb_diabatic_energies,
    load_positions_file,
    write_pdb,
)
from ..simulation import create_integrator, ensure_output_dir
from ..types import FrameObservables
from .windows import (
    GapWindowSpec,
    MappingWindowSpec,
    ProtonTransferWindowSpec,
    build_gap_windows,
    build_mapping_windows,
    build_proton_transfer_windows,
    get_gap_window,
    get_mapping_window,
    get_proton_transfer_window,
)

try:
    import openmm
    from openmm import unit
except ImportError:  # pragma: no cover
    openmm = None
    unit = None


@dataclass(slots=True)
class WindowSummary:
    window_id: str
    lambda_value: float
    n_frames: int
    final_step: int
    final_time_ps: float
    proton_transfer_rc_min_nm: float | None
    proton_transfer_rc_max_nm: float | None
    proton_transfer_positive_frames: int
    reaction_coordinate_ranges_nm: dict[str, list[float | None]]
    output_dir: str


@dataclass(slots=True)
class UmbrellaWindowSummary:
    window_id: str
    gap_center_kj_mol: float
    force_constant_kj_mol2: float
    n_frames: int
    final_step: int
    final_time_ps: float
    proton_transfer_rc_min_nm: float | None
    proton_transfer_rc_max_nm: float | None
    proton_transfer_positive_frames: int
    reaction_coordinate_ranges_nm: dict[str, list[float | None]]
    output_dir: str


@dataclass(slots=True)
class ProtonTransferWindowSummary:
    window_id: str
    rc_center_nm: float
    force_constant_kj_mol_nm2: float
    n_frames: int
    final_step: int
    final_time_ps: float
    proton_transfer_rc_min_nm: float | None
    proton_transfer_rc_max_nm: float | None
    proton_transfer_positive_frames: int
    reaction_coordinate_ranges_nm: dict[str, list[float | None]]
    output_dir: str


class MappingWindowRunner:
    def __init__(self, config: EVBConfig, window: MappingWindowSpec, state1=None, state2=None, initial_positions_nm: np.ndarray | None = None):
        self.config = config
        self.window = window
        self.initial_positions_nm = initial_positions_nm
        self.parameters = EVBParameters(
            delta_alpha=config.evb_parameters.delta_alpha or 0.0,
            h12=config.evb_parameters.h12 or 0.0,
        )
        self.builder = EVBSystemBuilder(
            AmberSystemLoader(
                nonbonded_method=config.simulation.nonbonded_method,
                constraints=config.simulation.constraints,
            )
        )
        if state1 is not None and state2 is not None:
            self.builder.validate_compatibility(state1, state2)
            self.state1, self.state2 = state1, state2
        else:
            self.state1, self.state2 = self.builder.build_from_state_files(config.state1, config.state2)

    def run(self, output_root: str | Path) -> WindowSummary:
        mapped_system = self.builder.build_openmm_mapped_system(
            self.state1,
            self.state2,
            lambda_value=self.window.lambda_value,
            delta_alpha=self.parameters.delta_alpha,
            equilibration_restraint=self._resolve_equilibration_restraint(),
            substrate_com_restraint=self._resolve_substrate_com_restraint(),
            far_field_restraint=_resolve_far_field_restraint(self.config, self.state1.positions_nm),
            unconstrained_atoms=_reactive_constraint_exclusions(self.config),
        )
        integrator = create_integrator(
            timestep_fs=self.config.sampling.integrator.timestep_fs,
            temperature_k=self.config.sampling.md.temperature_k,
            friction_per_ps=self.config.sampling.integrator.friction_per_ps,
            integrator_name=self.config.sampling.integrator.name,
        )
        context = self._create_context(mapped_system, integrator)
        positions = self._select_start_positions()
        context.setPositions(positions * unit.nanometer)
        self._relax_seed_positions_if_needed(context, integrator, mapped_system)
        context.setVelocitiesToTemperature(
            self.config.sampling.md.temperature_k * unit.kelvin,
            self.config.sampling.integrator.seed,
        )
        if self.config.sampling.md.minimize_steps > 0:
            openmm.LocalEnergyMinimizer.minimize(
                context,
                self.config.sampling.md.minimize_tolerance * unit.kilojoule_per_mole / unit.nanometer,
                self.config.sampling.md.minimize_steps,
            )

        if self.config.sampling.md.equilibration_steps > 0:
            integrator.step(self.config.sampling.md.equilibration_steps)
        self._disable_equilibration_restraint(context)

        window_dir = ensure_output_dir(Path(output_root) / "windows" / self.window.window_id)
        write_json(window_dir / "window_spec.json", {"window_id": self.window.window_id, "lambda_value": self.window.lambda_value})

        frames = self._collect_production(context, integrator, mapped_system, window_dir)
        final_positions = self._get_positions_nm(context)
        write_pdb(str(window_dir / "final_state.pdb"), mapped_system.topology, final_positions)
        summary = WindowSummary(
            window_id=self.window.window_id,
            lambda_value=self.window.lambda_value,
            n_frames=len(frames),
            final_step=frames[-1].gap.step if frames else 0,
            final_time_ps=frames[-1].gap.time_ps if frames else 0.0,
            proton_transfer_rc_min_nm=min((frame.proton_transfer_rc_nm for frame in frames if frame.proton_transfer_rc_nm is not None), default=None),
            proton_transfer_rc_max_nm=max((frame.proton_transfer_rc_nm for frame in frames if frame.proton_transfer_rc_nm is not None), default=None),
            proton_transfer_positive_frames=sum(1 for frame in frames if frame.proton_transfer_event),
            reaction_coordinate_ranges_nm=_summarize_reaction_coordinate_ranges(frames),
            output_dir=str(window_dir),
        )
        write_json(window_dir / "summary.json", summary)
        return summary

    def _select_start_positions(self) -> np.ndarray:
        if self.initial_positions_nm is not None:
            return self.initial_positions_nm
        if self.config.start_state == "state2":
            return self.state2.positions_nm
        if self.config.start_state == "state1":
            if self.window.lambda_value > 0.5:
                return self.state2.positions_nm
            return self.state1.positions_nm
        raise ValueError(f"Unsupported start_state option: {self.config.start_state!r}")

    def _create_context(self, mapped_system: MappedOpenMMSystem, integrator):
        if self.config.sampling.md.platform:
            platform = openmm.Platform.getPlatformByName(self.config.sampling.md.platform)
            context = openmm.Context(mapped_system.system, integrator, platform)
        else:
            context = openmm.Context(mapped_system.system, integrator)
        if mapped_system.box_vectors_nm is not None:
            context.setPeriodicBoxVectors(*(vec * unit.nanometer for vec in mapped_system.box_vectors_nm))
        return context

    def _relax_seed_positions_if_needed(self, context, integrator, mapped_system: MappedOpenMMSystem) -> None:
        spec = self.config.sampling.seed_relaxation
        if not spec.enabled or self.initial_positions_nm is None:
            return
        restraint = build_absolute_positional_restraint_force(
            self.initial_positions_nm,
            force_constant_kj_mol_nm2=spec.restraint_force_constant_kj_mol_nm2,
            parameter_name="k_seed",
        )
        restraint_index = mapped_system.system.addForce(restraint)
        try:
            context.reinitialize(preserveState=True)
            for scale in spec.restraint_decay:
                context.setParameter("k_seed", spec.restraint_force_constant_kj_mol_nm2 * float(scale))
                if spec.minimization_steps > 0:
                    openmm.LocalEnergyMinimizer.minimize(
                        context,
                        self.config.sampling.md.minimize_tolerance * unit.kilojoule_per_mole / unit.nanometer,
                        spec.minimization_steps,
                    )
                if spec.equilibration_steps > 0:
                    context.setVelocitiesToTemperature(
                        (spec.temperature_k or self.config.sampling.md.temperature_k) * unit.kelvin,
                        self.config.sampling.integrator.seed,
                    )
                    integrator.step(spec.equilibration_steps)
        finally:
            mapped_system.system.removeForce(restraint_index)
            context.reinitialize(preserveState=True)

    def _collect_production(self, context, integrator, mapped_system: MappedOpenMMSystem, window_dir: Path) -> list[FrameObservables]:
        frames: list[FrameObservables] = []
        step_count = self.config.sampling.md.equilibration_steps
        csv_path = window_dir / "production_observables.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            headers = [
                "frame",
                "step",
                "time_ps",
                "window_id",
                "lambda",
                "E1_kj_mol",
                "E2_kj_mol",
                "delta_e_kj_mol",
                "delta_e_shifted_kj_mol",
                "Eevb_kj_mol",
                "w1",
                "w2",
                "proton_transfer_rc_nm",
                "proton_transfer_event",
            ] + [definition.name for definition in self.config.observables.reaction_coordinates] + [
                definition.name for definition in self.config.observables.distances
            ]
            writer.writerow(headers)
            stride = self.config.sampling.md.report_stride
            n_steps = self.config.sampling.md.production_steps
            for frame_index, start in enumerate(range(0, n_steps, stride)):
                advance = min(stride, n_steps - start)
                integrator.step(advance)
                step_count += advance
                observables = self._snapshot(context, mapped_system, frame_index, step_count)
                frames.append(observables)
                row = [
                    observables.gap.frame,
                    observables.gap.step,
                    observables.gap.time_ps,
                    self.window.window_id,
                    self.window.lambda_value,
                    observables.gap.energy1_kj_mol,
                    observables.gap.energy2_kj_mol,
                    observables.gap.delta_e_kj_mol,
                    observables.gap.delta_e_shifted_kj_mol,
                    observables.gap.evb_energy_kj_mol,
                    observables.gap.weight1,
                    observables.gap.weight2,
                    observables.proton_transfer_rc_nm,
                    int(observables.proton_transfer_event),
                ] + [observables.reaction_coordinates_nm[definition.name] for definition in self.config.observables.reaction_coordinates] + [
                    observables.distances_nm[definition.name] for definition in self.config.observables.distances
                ]
                writer.writerow(row)
        return frames

    def _snapshot(self, context, mapped_system: MappedOpenMMSystem, frame_index: int, step_count: int) -> FrameObservables:
        state = context.getState(getEnergy=True, getPositions=True)
        positions_nm = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        values = mapped_system.mapping_force.getCollectiveVariableValues(context)
        energy1 = float(values[0])
        energy2 = float(values[1])
        evb_energy, weight1, weight2 = EVBHamiltonian(self.parameters).lower_eigenvalue(energy1, energy2)
        result = EVBResult(
            energy1=energy1,
            energy2=energy2,
            e2_shifted=energy2 + self.parameters.delta_alpha,
            evb_energy=evb_energy,
            weight1=weight1,
            weight2=weight2,
            forces=np.zeros_like(positions_nm),
        )
        gap = make_gap_sample(
            result,
            frame=frame_index,
            step=step_count,
            time_ps=step_count * self.config.sampling.integrator.timestep_fs * 1.0e-3,
            delta_alpha_kj_mol=self.parameters.delta_alpha,
        )
        distances = compute_named_distances(positions_nm, self.config.observables.distances)
        reaction_coordinates = compute_named_reaction_coordinates(positions_nm, self.config.observables.reaction_coordinates)
        rc_nm, event = _compute_proton_transfer_monitor(positions_nm, self.config, reaction_coordinates)
        return FrameObservables(
            gap=gap,
            distances_nm=distances,
            reaction_coordinates_nm=reaction_coordinates,
            proton_transfer_rc_nm=rc_nm,
            proton_transfer_event=event,
        )

    @staticmethod
    def _get_positions_nm(context) -> np.ndarray:
        state = context.getState(getPositions=True)
        return np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))

    def _resolve_equilibration_restraint(self) -> tuple[int, int, float, float] | None:
        return _resolve_equilibration_restraint(self.config, self.state1.positions_nm)

    def _resolve_substrate_com_restraint(self) -> tuple[list[int], float] | None:
        return _resolve_substrate_com_restraint(self.config)

    @staticmethod
    def _disable_equilibration_restraint(context) -> None:
        if hasattr(context, "setParameter"):
            try:
                context.setParameter("k_rest", 0.0)
            except Exception:
                pass

def run_mapping_window(config: EVBConfig, window_id: str) -> WindowSummary:
    window = get_mapping_window(config, window_id)
    runner = MappingWindowRunner(config, window, initial_positions_nm=_resolve_seed_positions(config, window_id))
    return runner.run(config.output_dir)


def run_mapping_series(config: EVBConfig) -> list[WindowSummary]:
    output_root = ensure_output_dir(config.output_dir)
    summaries: list[WindowSummary] = []
    previous_positions_nm: np.ndarray | None = None
    for window in build_mapping_windows(config):
        seed_positions_nm = _resolve_seed_positions(config, window.window_id)
        runner = MappingWindowRunner(
            config, window, initial_positions_nm=seed_positions_nm if seed_positions_nm is not None else previous_positions_nm
        )
        summary = runner.run(output_root)
        summaries.append(summary)
        previous_positions_nm = load_pdb_positions(summary.output_dir + "/final_state.pdb")
    ensure_output_dir(output_root / "series")
    write_json(output_root / "series" / "window_index.json", [asdict(summary) for summary in summaries])
    return summaries


def load_pdb_positions(path: str) -> np.ndarray:
    positions: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = line[:6].strip()
            if record in {"ATOM", "HETATM"}:
                positions.append(
                    [
                        float(line[30:38]) * 0.1,
                        float(line[38:46]) * 0.1,
                        float(line[46:54]) * 0.1,
                    ]
                )
    if not positions:
        raise ValueError(f"No ATOM/HETATM records found in {path}")
    return np.asarray(positions, dtype=float)


class GapUmbrellaWindowRunner:
    def __init__(self, config: EVBConfig, window: GapWindowSpec, state1=None, state2=None, initial_positions_nm: np.ndarray | None = None):
        self.config = config
        self.window = window
        self.initial_positions_nm = initial_positions_nm
        self.parameters = EVBParameters(
            delta_alpha=config.evb_parameters.delta_alpha or 0.0,
            h12=config.evb_parameters.h12 or 0.0,
        )
        self.builder = EVBSystemBuilder(
            AmberSystemLoader(
                nonbonded_method=config.simulation.nonbonded_method,
                constraints=config.simulation.constraints,
            )
        )
        if state1 is not None and state2 is not None:
            self.builder.validate_compatibility(state1, state2)
            self.state1, self.state2 = state1, state2
        else:
            self.state1, self.state2 = self.builder.build_from_state_files(config.state1, config.state2)

    def run(self, output_root: str | Path) -> UmbrellaWindowSummary:
        umbrella_system = self.builder.build_openmm_gap_umbrella_system(
            self.state1,
            self.state2,
            delta_alpha=self.parameters.delta_alpha,
            h12=self.parameters.h12,
            gap_center=self.window.center_kj_mol,
            gap_force_constant=self.window.force_constant_kj_mol2,
            equilibration_restraint=self._resolve_equilibration_restraint(),
            substrate_com_restraint=self._resolve_substrate_com_restraint(),
            far_field_restraint=_resolve_far_field_restraint(self.config, self.state1.positions_nm),
            unconstrained_atoms=_reactive_constraint_exclusions(self.config),
            energy_decomposition=self.config.energy_decomposition.enabled,
        )
        integrator = create_integrator(
            timestep_fs=self.config.sampling.integrator.timestep_fs,
            temperature_k=self.config.sampling.md.temperature_k,
            friction_per_ps=self.config.sampling.integrator.friction_per_ps,
            integrator_name=self.config.sampling.integrator.name,
        )
        context = self._create_context(umbrella_system, integrator)
        positions = self._select_start_positions()
        context.setPositions(positions * unit.nanometer)
        self._relax_seed_positions_if_needed(context, integrator, umbrella_system)
        context.setVelocitiesToTemperature(
            self.config.sampling.md.temperature_k * unit.kelvin,
            self.config.sampling.integrator.seed,
        )
        if self.config.sampling.md.minimize_steps > 0:
            openmm.LocalEnergyMinimizer.minimize(
                context,
                self.config.sampling.md.minimize_tolerance * unit.kilojoule_per_mole / unit.nanometer,
                self.config.sampling.md.minimize_steps,
            )
        self._equilibrate_with_umbrella_ramp(context, integrator)
        self._disable_equilibration_restraint(context)

        window_dir = ensure_output_dir(Path(output_root) / "windows" / self.window.window_id)
        write_json(
            window_dir / "window_spec.json",
            {
                "window_id": self.window.window_id,
                "gap_center_kj_mol": self.window.center_kj_mol,
                "force_constant_kj_mol2": self.window.force_constant_kj_mol2,
            },
        )
        frames = self._collect_production(context, integrator, umbrella_system, window_dir)
        final_positions = self._get_positions_nm(context)
        write_pdb(str(window_dir / "final_state.pdb"), umbrella_system.topology, final_positions)
        summary = UmbrellaWindowSummary(
            window_id=self.window.window_id,
            gap_center_kj_mol=self.window.center_kj_mol,
            force_constant_kj_mol2=self.window.force_constant_kj_mol2,
            n_frames=len(frames),
            final_step=frames[-1].gap.step if frames else 0,
            final_time_ps=frames[-1].gap.time_ps if frames else 0.0,
            proton_transfer_rc_min_nm=min((frame.proton_transfer_rc_nm for frame in frames if frame.proton_transfer_rc_nm is not None), default=None),
            proton_transfer_rc_max_nm=max((frame.proton_transfer_rc_nm for frame in frames if frame.proton_transfer_rc_nm is not None), default=None),
            proton_transfer_positive_frames=sum(1 for frame in frames if frame.proton_transfer_event),
            reaction_coordinate_ranges_nm=_summarize_reaction_coordinate_ranges(frames),
            output_dir=str(window_dir),
        )
        write_json(window_dir / "summary.json", summary)
        return summary

    def _select_start_positions(self) -> np.ndarray:
        if self.initial_positions_nm is not None:
            return self.initial_positions_nm
        return self.state1.positions_nm

    def _create_context(self, umbrella_system: EVBOpenMMSystem, integrator):
        if self.config.sampling.md.platform:
            platform = openmm.Platform.getPlatformByName(self.config.sampling.md.platform)
            context = openmm.Context(umbrella_system.system, integrator, platform)
        else:
            context = openmm.Context(umbrella_system.system, integrator)
        if umbrella_system.box_vectors_nm is not None:
            context.setPeriodicBoxVectors(*(vec * unit.nanometer for vec in umbrella_system.box_vectors_nm))
        return context

    def _relax_seed_positions_if_needed(self, context, integrator, umbrella_system: EVBOpenMMSystem) -> None:
        spec = self.config.sampling.seed_relaxation
        if not spec.enabled or self.initial_positions_nm is None:
            return
        full_k_gap = context.getParameter("k_gap") if "k_gap" in [umbrella_system.evb_force.getGlobalParameterName(i) for i in range(umbrella_system.evb_force.getNumGlobalParameters())] else None
        if full_k_gap is not None:
            context.setParameter("k_gap", 0.0)
        restraint = build_absolute_positional_restraint_force(
            self.initial_positions_nm,
            force_constant_kj_mol_nm2=spec.restraint_force_constant_kj_mol_nm2,
            parameter_name="k_seed",
        )
        restraint_index = umbrella_system.system.addForce(restraint)
        try:
            context.reinitialize(preserveState=True)
            for scale in spec.restraint_decay:
                context.setParameter("k_seed", spec.restraint_force_constant_kj_mol_nm2 * float(scale))
                if spec.minimization_steps > 0:
                    openmm.LocalEnergyMinimizer.minimize(
                        context,
                        self.config.sampling.md.minimize_tolerance * unit.kilojoule_per_mole / unit.nanometer,
                        spec.minimization_steps,
                    )
                if spec.equilibration_steps > 0:
                    context.setVelocitiesToTemperature(
                        (spec.temperature_k or self.config.sampling.md.temperature_k) * unit.kelvin,
                        self.config.sampling.integrator.seed,
                    )
                    integrator.step(spec.equilibration_steps)
        finally:
            umbrella_system.system.removeForce(restraint_index)
            context.reinitialize(preserveState=True)
            if full_k_gap is not None:
                context.setParameter("k_gap", full_k_gap)

    def _collect_production(self, context, integrator, umbrella_system: EVBOpenMMSystem, window_dir: Path) -> list[FrameObservables]:
        frames: list[FrameObservables] = []
        step_count = self.config.sampling.md.equilibration_steps
        csv_path = window_dir / "production_observables.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            headers = [
                "frame",
                "step",
                "time_ps",
                "window_id",
                "gap_center_kj_mol",
                "E1_kj_mol",
                "E2_kj_mol",
                "delta_e_kj_mol",
                "delta_e_shifted_kj_mol",
                "Eevb_kj_mol",
                "w1",
                "w2",
                "proton_transfer_rc_nm",
                "proton_transfer_event",
            ] + [definition.name for definition in self.config.observables.reaction_coordinates] + [
                definition.name for definition in self.config.observables.distances
            ]
            writer.writerow(headers)
            stride = self.config.sampling.md.report_stride
            n_steps = self.config.sampling.md.production_steps
            for frame_index, start in enumerate(range(0, n_steps, stride)):
                advance = min(stride, n_steps - start)
                integrator.step(advance)
                step_count += advance
                observables = self._snapshot(context, umbrella_system, frame_index, step_count)
                frames.append(observables)
                row = [
                    observables.gap.frame,
                    observables.gap.step,
                    observables.gap.time_ps,
                    self.window.window_id,
                    self.window.center_kj_mol,
                    observables.gap.energy1_kj_mol,
                    observables.gap.energy2_kj_mol,
                    observables.gap.delta_e_kj_mol,
                    observables.gap.delta_e_shifted_kj_mol,
                    observables.gap.evb_energy_kj_mol,
                    observables.gap.weight1,
                    observables.gap.weight2,
                    observables.proton_transfer_rc_nm,
                    int(observables.proton_transfer_event),
                ] + [observables.reaction_coordinates_nm[definition.name] for definition in self.config.observables.reaction_coordinates] + [
                    observables.distances_nm[definition.name] for definition in self.config.observables.distances
                ]
                writer.writerow(row)
        return frames

    def _snapshot(self, context, umbrella_system: EVBOpenMMSystem, frame_index: int, step_count: int) -> FrameObservables:
        state = context.getState(getEnergy=True, getPositions=True)
        positions_nm = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        energy1, energy2 = evb_diabatic_energies(umbrella_system, context)
        evb_energy, weight1, weight2 = EVBHamiltonian(self.parameters).lower_eigenvalue(energy1, energy2)
        result = EVBResult(
            energy1=energy1,
            energy2=energy2,
            e2_shifted=energy2 + self.parameters.delta_alpha,
            evb_energy=evb_energy,
            weight1=weight1,
            weight2=weight2,
            forces=np.zeros_like(positions_nm),
        )
        gap = make_gap_sample(
            result,
            frame=frame_index,
            step=step_count,
            time_ps=step_count * self.config.sampling.integrator.timestep_fs * 1.0e-3,
            delta_alpha_kj_mol=self.parameters.delta_alpha,
        )
        distances = compute_named_distances(positions_nm, self.config.observables.distances)
        reaction_coordinates = compute_named_reaction_coordinates(positions_nm, self.config.observables.reaction_coordinates)
        rc_nm, event = _compute_proton_transfer_monitor(positions_nm, self.config, reaction_coordinates)
        return FrameObservables(
            gap=gap,
            distances_nm=distances,
            reaction_coordinates_nm=reaction_coordinates,
            proton_transfer_rc_nm=rc_nm,
            proton_transfer_event=event,
        )

    @staticmethod
    def _get_positions_nm(context) -> np.ndarray:
        state = context.getState(getPositions=True)
        return np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))

    def _resolve_equilibration_restraint(self) -> tuple[int, int, float, float] | None:
        return _resolve_equilibration_restraint(self.config, self.state1.positions_nm)

    def _resolve_substrate_com_restraint(self) -> tuple[list[int], float] | None:
        return _resolve_substrate_com_restraint(self.config)

    def _equilibrate_with_umbrella_ramp(self, context, integrator) -> None:
        steps = self.config.sampling.md.equilibration_steps
        if steps <= 0:
            return
        full_k = self.window.force_constant_kj_mol2
        ramp = self.config.sampling.umbrella_ramp
        if not ramp.enabled:
            integrator.step(steps)
            return
        fractions = [float(value) for value in ramp.fractions if float(value) > 0.0]
        if not fractions:
            integrator.step(steps)
            return
        chunk = max(steps // len(fractions), 1)
        completed = 0
        for fraction in fractions:
            context.setParameter("k_gap", full_k * min(fraction, 1.0))
            advance = min(chunk, steps - completed)
            if advance > 0:
                integrator.step(advance)
                completed += advance
            if completed >= steps:
                break
        if completed < steps:
            context.setParameter("k_gap", full_k)
            integrator.step(steps - completed)
        context.setParameter("k_gap", full_k)

    @staticmethod
    def _disable_equilibration_restraint(context) -> None:
        if hasattr(context, "setParameter"):
            try:
                context.setParameter("k_rest", 0.0)
            except Exception:
                pass

def run_gap_umbrella_window(config: EVBConfig, window_id: str) -> UmbrellaWindowSummary:
    window = get_gap_window(config, window_id)
    runner = GapUmbrellaWindowRunner(config, window, initial_positions_nm=_resolve_seed_positions(config, window_id))
    return runner.run(config.output_dir)


def run_gap_umbrella_series(config: EVBConfig) -> list[UmbrellaWindowSummary]:
    output_root = ensure_output_dir(config.output_dir)
    summaries: list[UmbrellaWindowSummary] = []
    previous_positions_nm: np.ndarray | None = None
    for window in build_gap_windows(config):
        seed_positions_nm = _resolve_seed_positions(config, window.window_id)
        runner = GapUmbrellaWindowRunner(
            config, window, initial_positions_nm=seed_positions_nm if seed_positions_nm is not None else previous_positions_nm
        )
        summary = runner.run(output_root)
        summaries.append(summary)
        previous_positions_nm = load_pdb_positions(summary.output_dir + "/final_state.pdb")
    ensure_output_dir(output_root / "series")
    write_json(output_root / "series" / "window_index.json", [asdict(summary) for summary in summaries])
    return summaries


class ProtonTransferUmbrellaWindowRunner:
    def __init__(self, config: EVBConfig, window: ProtonTransferWindowSpec, state1=None, state2=None, initial_positions_nm: np.ndarray | None = None):
        self.config = config
        self.window = window
        self.initial_positions_nm = initial_positions_nm
        self.parameters = EVBParameters(
            delta_alpha=config.evb_parameters.delta_alpha or 0.0,
            h12=config.evb_parameters.h12 or 0.0,
        )
        self.builder = EVBSystemBuilder(
            AmberSystemLoader(
                nonbonded_method=config.simulation.nonbonded_method,
                constraints=config.simulation.constraints,
            )
        )
        if state1 is not None and state2 is not None:
            self.builder.validate_compatibility(state1, state2)
            self.state1, self.state2 = state1, state2
        else:
            self.state1, self.state2 = self.builder.build_from_state_files(config.state1, config.state2)

    def run(self, output_root: str | Path) -> ProtonTransferWindowSummary:
        atoms = self.config.reaction.atoms or self.config.cv
        if atoms is None:
            raise ValueError("proton_transfer_umbrella sampling requires reaction atoms (donor/proton/acceptor) in the config.")
        system = self.builder.build_openmm_proton_transfer_umbrella_system(
            self.state1,
            self.state2,
            delta_alpha=self.parameters.delta_alpha,
            h12=self.parameters.h12,
            donor_index=atoms.donor,
            proton_index=atoms.proton,
            acceptor_index=atoms.acceptor,
            rc_center_nm=self.window.center_nm,
            rc_force_constant_kj_mol_nm2=self.window.force_constant_kj_mol_nm2,
            equilibration_restraint=self._resolve_equilibration_restraint(),
            substrate_com_restraint=self._resolve_substrate_com_restraint(),
            far_field_restraint=_resolve_far_field_restraint(self.config, self.state1.positions_nm),
            unconstrained_atoms=_reactive_constraint_exclusions(self.config),
        )
        integrator = create_integrator(
            timestep_fs=self.config.sampling.integrator.timestep_fs,
            temperature_k=self.config.sampling.md.temperature_k,
            friction_per_ps=self.config.sampling.integrator.friction_per_ps,
            integrator_name=self.config.sampling.integrator.name,
        )
        context = self._create_context(system, integrator)
        positions = self._select_start_positions()
        context.setPositions(positions * unit.nanometer)
        context.setVelocitiesToTemperature(
            self.config.sampling.md.temperature_k * unit.kelvin,
            self.config.sampling.integrator.seed,
        )
        if self.config.sampling.md.minimize_steps > 0:
            openmm.LocalEnergyMinimizer.minimize(
                context,
                self.config.sampling.md.minimize_tolerance * unit.kilojoule_per_mole / unit.nanometer,
                self.config.sampling.md.minimize_steps,
            )
        if self.config.sampling.md.equilibration_steps > 0:
            integrator.step(self.config.sampling.md.equilibration_steps)
        self._disable_equilibration_restraint(context)

        window_dir = ensure_output_dir(Path(output_root) / "windows" / self.window.window_id)
        write_json(
            window_dir / "window_spec.json",
            {
                "window_id": self.window.window_id,
                "rc_center_nm": self.window.center_nm,
                "force_constant_kj_mol_nm2": self.window.force_constant_kj_mol_nm2,
            },
        )
        frames = self._collect_production(context, integrator, system, window_dir)
        final_positions = self._get_positions_nm(context)
        write_pdb(str(window_dir / "final_state.pdb"), system.topology, final_positions)
        summary = ProtonTransferWindowSummary(
            window_id=self.window.window_id,
            rc_center_nm=self.window.center_nm,
            force_constant_kj_mol_nm2=self.window.force_constant_kj_mol_nm2,
            n_frames=len(frames),
            final_step=frames[-1].gap.step if frames else 0,
            final_time_ps=frames[-1].gap.time_ps if frames else 0.0,
            proton_transfer_rc_min_nm=min((frame.proton_transfer_rc_nm for frame in frames if frame.proton_transfer_rc_nm is not None), default=None),
            proton_transfer_rc_max_nm=max((frame.proton_transfer_rc_nm for frame in frames if frame.proton_transfer_rc_nm is not None), default=None),
            proton_transfer_positive_frames=sum(1 for frame in frames if frame.proton_transfer_event),
            reaction_coordinate_ranges_nm=_summarize_reaction_coordinate_ranges(frames),
            output_dir=str(window_dir),
        )
        write_json(window_dir / "summary.json", summary)
        return summary

    def _select_start_positions(self) -> np.ndarray:
        if self.initial_positions_nm is not None:
            return self.initial_positions_nm
        return self.state1.positions_nm

    def _create_context(self, umbrella_system: EVBOpenMMSystem, integrator):
        if self.config.sampling.md.platform:
            platform = openmm.Platform.getPlatformByName(self.config.sampling.md.platform)
            context = openmm.Context(umbrella_system.system, integrator, platform)
        else:
            context = openmm.Context(umbrella_system.system, integrator)
        if umbrella_system.box_vectors_nm is not None:
            context.setPeriodicBoxVectors(*(vec * unit.nanometer for vec in umbrella_system.box_vectors_nm))
        return context

    def _collect_production(self, context, integrator, umbrella_system: EVBOpenMMSystem, window_dir: Path) -> list[FrameObservables]:
        frames: list[FrameObservables] = []
        step_count = self.config.sampling.md.equilibration_steps
        csv_path = window_dir / "production_observables.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            headers = [
                "frame", "step", "time_ps", "window_id", "rc_center_nm",
                "E1_kj_mol", "E2_kj_mol", "delta_e_kj_mol", "delta_e_shifted_kj_mol",
                "Eevb_kj_mol", "w1", "w2", "proton_transfer_rc_nm", "proton_transfer_event",
            ] + [definition.name for definition in self.config.observables.reaction_coordinates] + [
                definition.name for definition in self.config.observables.distances
            ]
            writer.writerow(headers)
            stride = self.config.sampling.md.report_stride
            n_steps = self.config.sampling.md.production_steps
            for frame_index, start in enumerate(range(0, n_steps, stride)):
                advance = min(stride, n_steps - start)
                integrator.step(advance)
                step_count += advance
                observables = self._snapshot(context, umbrella_system, frame_index, step_count)
                frames.append(observables)
                row = [
                    observables.gap.frame, observables.gap.step, observables.gap.time_ps,
                    self.window.window_id, self.window.center_nm,
                    observables.gap.energy1_kj_mol, observables.gap.energy2_kj_mol,
                    observables.gap.delta_e_kj_mol, observables.gap.delta_e_shifted_kj_mol,
                    observables.gap.evb_energy_kj_mol, observables.gap.weight1, observables.gap.weight2,
                    observables.proton_transfer_rc_nm, int(observables.proton_transfer_event),
                ] + [observables.reaction_coordinates_nm[definition.name] for definition in self.config.observables.reaction_coordinates] + [
                    observables.distances_nm[definition.name] for definition in self.config.observables.distances
                ]
                writer.writerow(row)
        return frames

    def _snapshot(self, context, umbrella_system: EVBOpenMMSystem, frame_index: int, step_count: int) -> FrameObservables:
        state = context.getState(getEnergy=True, getPositions=True)
        positions_nm = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        energy1, energy2 = evb_diabatic_energies(umbrella_system, context)
        evb_energy, weight1, weight2 = EVBHamiltonian(self.parameters).lower_eigenvalue(energy1, energy2)
        result = EVBResult(
            energy1=energy1, energy2=energy2, e2_shifted=energy2 + self.parameters.delta_alpha,
            evb_energy=evb_energy, weight1=weight1, weight2=weight2, forces=np.zeros_like(positions_nm),
        )
        gap = make_gap_sample(result, frame=frame_index, step=step_count, time_ps=step_count * self.config.sampling.integrator.timestep_fs * 1.0e-3, delta_alpha_kj_mol=self.parameters.delta_alpha)
        distances = compute_named_distances(positions_nm, self.config.observables.distances)
        reaction_coordinates = compute_named_reaction_coordinates(positions_nm, self.config.observables.reaction_coordinates)
        rc_nm, event = _compute_proton_transfer_monitor(positions_nm, self.config, reaction_coordinates)
        return FrameObservables(
            gap=gap,
            distances_nm=distances,
            reaction_coordinates_nm=reaction_coordinates,
            proton_transfer_rc_nm=rc_nm,
            proton_transfer_event=event,
        )

    @staticmethod
    def _get_positions_nm(context) -> np.ndarray:
        state = context.getState(getPositions=True)
        return np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))

    def _resolve_equilibration_restraint(self) -> tuple[int, int, float, float] | None:
        return _resolve_equilibration_restraint(self.config, self.state1.positions_nm)

    def _resolve_substrate_com_restraint(self) -> tuple[list[int], float] | None:
        return _resolve_substrate_com_restraint(self.config)

    @staticmethod
    def _disable_equilibration_restraint(context) -> None:
        if hasattr(context, "setParameter"):
            try:
                context.setParameter("k_rest", 0.0)
            except Exception:
                pass


def run_proton_transfer_umbrella_window(config: EVBConfig, window_id: str) -> ProtonTransferWindowSummary:
    window = get_proton_transfer_window(config, window_id)
    runner = ProtonTransferUmbrellaWindowRunner(config, window, initial_positions_nm=_resolve_seed_positions(config, window_id))
    return runner.run(config.output_dir)


def run_proton_transfer_umbrella_series(config: EVBConfig) -> list[ProtonTransferWindowSummary]:
    if config.sampling.bidirectional:
        manifests = run_proton_transfer_umbrella_bidirectional_series(config)
        return manifests["forward"]
    output_root = ensure_output_dir(config.output_dir)
    summaries: list[ProtonTransferWindowSummary] = []
    previous_positions_nm: np.ndarray | None = None
    for window in build_proton_transfer_windows(config):
        seed_positions_nm = _resolve_seed_positions(config, window.window_id)
        runner = ProtonTransferUmbrellaWindowRunner(
            config, window, initial_positions_nm=seed_positions_nm if seed_positions_nm is not None else previous_positions_nm
        )
        summary = runner.run(output_root)
        summaries.append(summary)
        previous_positions_nm = load_pdb_positions(summary.output_dir + "/final_state.pdb")
    ensure_output_dir(output_root / "series")
    write_json(output_root / "series" / "window_index.json", [asdict(summary) for summary in summaries])
    return summaries


def run_proton_transfer_umbrella_bidirectional_series(config: EVBConfig) -> dict[str, list[ProtonTransferWindowSummary]]:
    output_root = ensure_output_dir(config.output_dir)
    forward_root = ensure_output_dir(output_root / "branches" / "forward")
    reverse_root = ensure_output_dir(output_root / "branches" / "reverse")

    base_windows = build_proton_transfer_windows(config)
    builder = EVBSystemBuilder(
        AmberSystemLoader(
            nonbonded_method=config.simulation.nonbonded_method,
            constraints=config.simulation.constraints,
        )
    )
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)

    forward = _run_proton_transfer_branch(
        config=config,
        windows=base_windows,
        output_root=forward_root,
        state1=state1,
        state2=state2,
        initial_positions_nm=state1.positions_nm,
    )
    reverse = _run_proton_transfer_branch(
        config=config,
        windows=list(reversed(base_windows)),
        output_root=reverse_root,
        state1=state1,
        state2=state2,
        initial_positions_nm=state2.positions_nm,
    )

    ensure_output_dir(output_root / "series")
    manifest = {
        "bidirectional": True,
        "forward_root": str(forward_root),
        "reverse_root": str(reverse_root),
        "forward": [asdict(summary) for summary in forward],
        "reverse": [asdict(summary) for summary in reverse],
    }
    write_json(output_root / "series" / "bidirectional_index.json", manifest)
    return {"forward": forward, "reverse": reverse}


def _run_proton_transfer_branch(
    config: EVBConfig,
    windows: list[ProtonTransferWindowSpec],
    output_root: Path,
    state1,
    state2,
    initial_positions_nm: np.ndarray,
) -> list[ProtonTransferWindowSummary]:
    summaries: list[ProtonTransferWindowSummary] = []
    previous_positions_nm: np.ndarray | None = initial_positions_nm
    for window in windows:
        seed_positions_nm = _resolve_seed_positions(config, window.window_id, branch=_infer_branch_name(output_root))
        runner = ProtonTransferUmbrellaWindowRunner(
            config,
            window,
            state1=state1,
            state2=state2,
            initial_positions_nm=seed_positions_nm if seed_positions_nm is not None else previous_positions_nm,
        )
        summary = runner.run(output_root)
        summaries.append(summary)
        previous_positions_nm = load_pdb_positions(summary.output_dir + "/final_state.pdb")
    ensure_output_dir(output_root / "series")
    write_json(output_root / "series" / "window_index.json", [asdict(summary) for summary in summaries])
    return summaries


def _compute_proton_transfer_monitor(
    positions_nm: np.ndarray, config: EVBConfig, reaction_coordinates: dict[str, float]
) -> tuple[float | None, bool]:
    if "proton_transfer_rc" in reaction_coordinates:
        rc_nm = reaction_coordinates["proton_transfer_rc"]
        threshold = 0.0
        for definition in config.observables.reaction_coordinates:
            if definition.name == "proton_transfer_rc" and definition.event_threshold_nm is not None:
                threshold = definition.event_threshold_nm
                break
        return rc_nm, rc_nm > threshold
    atoms = config.reaction.atoms or config.cv
    if atoms is None:
        return None, False
    donor_h = float(np.linalg.norm(positions_nm[atoms.donor] - positions_nm[atoms.proton]))
    h_acceptor = float(np.linalg.norm(positions_nm[atoms.proton] - positions_nm[atoms.acceptor]))
    rc_nm = donor_h - h_acceptor
    return rc_nm, rc_nm > 0.0


def _summarize_reaction_coordinate_ranges(frames: list[FrameObservables]) -> dict[str, list[float | None]]:
    ranges: dict[str, list[float | None]] = {}
    names = {name for frame in frames for name in frame.reaction_coordinates_nm}
    for name in sorted(names):
        values = [frame.reaction_coordinates_nm[name] for frame in frames if name in frame.reaction_coordinates_nm]
        ranges[name] = [min(values), max(values)] if values else [None, None]
    return ranges


def _resolve_seed_positions(config: EVBConfig, window_id: str, branch: str | None = None) -> np.ndarray | None:
    for definition in config.sampling.seed_windows:
        if definition.window_id != window_id:
            continue
        if definition.branch is not None and definition.branch != branch:
            continue
        return load_positions_file(definition.coordinates)
    return None


def _infer_branch_name(output_root: Path) -> str | None:
    name = output_root.name
    if name in {"forward", "reverse"}:
        return name
    return None


def _resolve_equilibration_restraint(config: EVBConfig, reference_positions_nm: np.ndarray) -> tuple[int, int, float, float] | None:
    spec = config.sampling.equilibration_restraint
    if not spec.enabled:
        return None
    atom1 = spec.atom1
    atom2 = spec.atom2
    atoms = config.reaction.atoms or config.cv
    if atom1 is None or atom2 is None:
        if atoms is None:
            raise ValueError("equilibration_restraint.enabled requires atom1/atom2 or reaction atoms in config.")
        atom1 = atoms.donor
        atom2 = atoms.acceptor
    target = spec.target_distance_nm
    if target is None:
        target = float(np.linalg.norm(reference_positions_nm[atom1] - reference_positions_nm[atom2]))
    return atom1, atom2, target, spec.force_constant_kj_mol_nm2


def _resolve_substrate_com_restraint(config: EVBConfig) -> tuple[list[int], float] | None:
    spec = config.sampling.production_restraint
    if not spec.enabled:
        return None
    atom_indices = spec.substrate_com_atoms or config.reaction.substrate_atoms
    if not atom_indices:
        raise ValueError("production_restraint.enabled requires substrate_com_atoms or reaction.substrate_atoms.")
    if spec.substrate_com_force_constant_kj_mol_nm2 is None:
        raise ValueError("production_restraint.enabled requires substrate_com_force_constant_kj_mol_nm2.")
    return list(atom_indices), spec.substrate_com_force_constant_kj_mol_nm2


def _resolve_far_field_restraint(config: EVBConfig, reference_positions_nm: np.ndarray) -> tuple[list[int], np.ndarray, float] | None:
    spec = config.sampling.far_field_restraint
    if not spec.enabled:
        return None
    if spec.restrained_atoms:
        restrained_atoms = sorted(set(spec.restrained_atoms))
    else:
        active_atoms = _active_atoms_for_far_field(config)
        if not active_atoms:
            raise ValueError("far_field_restraint.enabled requires active_atoms, reaction atoms, or reaction.substrate_atoms.")
        active_positions = reference_positions_nm[active_atoms]
        restrained_atoms = []
        for atom_index, position in enumerate(reference_positions_nm):
            distances = np.linalg.norm(active_positions - position, axis=1)
            if float(np.min(distances)) > spec.radius_nm:
                restrained_atoms.append(atom_index)
    active_set = set(_active_atoms_for_far_field(config))
    restrained_atoms = [index for index in restrained_atoms if index not in active_set]
    if not restrained_atoms:
        return None
    return restrained_atoms, reference_positions_nm, spec.force_constant_kj_mol_nm2


def _active_atoms_for_far_field(config: EVBConfig) -> list[int]:
    active_atoms = set(config.sampling.far_field_restraint.active_atoms)
    active_atoms.update(config.reaction.substrate_atoms)
    atoms = config.reaction.atoms or config.cv
    if atoms is not None:
        active_atoms.update([atoms.donor, atoms.proton, atoms.acceptor])
    return sorted(active_atoms)


def _reactive_constraint_exclusions(config: EVBConfig) -> set[int]:
    atoms = config.reaction.atoms or config.cv
    if atoms is None:
        return set()
    return {atoms.donor, atoms.proton, atoms.acceptor}
