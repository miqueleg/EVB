from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import openmm
    from openmm import unit
    from openmm.app import AmberInpcrdFile, AmberPrmtopFile, DCDFile, PDBFile
except ImportError as exc:  # pragma: no cover - exercised only when OpenMM unavailable
    openmm = None
    unit = None
    AmberInpcrdFile = AmberPrmtopFile = DCDFile = PDBFile = None
    OPENMM_IMPORT_ERROR = exc
else:  # pragma: no cover - import tested indirectly when OpenMM available
    OPENMM_IMPORT_ERROR = None


_NONBONDED_METHODS = {
    "NoCutoff": None,
    "CutoffNonPeriodic": None,
    "CutoffPeriodic": None,
    "Ewald": None,
    "PME": None,
    "LJPME": None,
}

_CONSTRAINTS = {
    "None": None,
    "HBonds": None,
    "AllBonds": None,
    "HAngles": None,
}

if openmm is not None:  # pragma: no branch
    from openmm.app import AllBonds, CutoffNonPeriodic, CutoffPeriodic, Ewald, HBonds, HAngles, LJPME, NoCutoff, PME

    _NONBONDED_METHODS.update(
        {
            "NoCutoff": NoCutoff,
            "CutoffNonPeriodic": CutoffNonPeriodic,
            "CutoffPeriodic": CutoffPeriodic,
            "Ewald": Ewald,
            "PME": PME,
            "LJPME": LJPME,
        }
    )
    _CONSTRAINTS.update(
        {
            "None": None,
            "HBonds": HBonds,
            "AllBonds": AllBonds,
            "HAngles": HAngles,
        }
    )


@dataclass(slots=True)
class LoadedAmberState:
    prmtop_path: str
    inpcrd_path: str
    topology: Any
    system: Any
    positions_nm: np.ndarray
    box_vectors_nm: np.ndarray | None
    atom_labels: list[tuple[str, str, str, int]]
    atom_names: list[str]
    masses_amu: np.ndarray


@dataclass(slots=True)
class EVBOpenMMSystem:
    system: Any
    topology: Any
    positions_nm: np.ndarray
    box_vectors_nm: np.ndarray | None
    masses_amu: np.ndarray
    evb_force: Any
    state1_force: Any
    state2_force: Any
    umbrella_force: Any | None = None
    equilibration_restraint_force: Any | None = None
    production_restraint_force: Any | None = None
    far_field_restraint_force: Any | None = None
    energy_decomposition_report: dict[str, Any] | None = None
    native_gap_bias: Any | None = None
    table_bias_function_index: int | None = None
    common_forces: list[Any] | None = None
    common_force_group: int | None = None
    force_groups: dict[str, int] | None = None
    bias_report: dict[str, Any] | None = None


@dataclass(slots=True)
class MappedOpenMMSystem:
    system: Any
    topology: Any
    positions_nm: np.ndarray
    box_vectors_nm: np.ndarray | None
    masses_amu: np.ndarray
    mapping_force: Any
    state1_force: Any
    state2_force: Any
    equilibration_restraint_force: Any | None = None
    production_restraint_force: Any | None = None
    far_field_restraint_force: Any | None = None


class OpenMMStateEvaluator:
    def __init__(self, loaded_state: LoadedAmberState, platform_name: str | None = None):
        _require_openmm()
        integrator = openmm.VerletIntegrator(0.001)
        if platform_name:
            platform = openmm.Platform.getPlatformByName(platform_name)
            self.context = openmm.Context(loaded_state.system, integrator, platform)
        else:
            self.context = openmm.Context(loaded_state.system, integrator)
        self.loaded_state = loaded_state
        if loaded_state.box_vectors_nm is not None:
            self.context.setPeriodicBoxVectors(*_to_box_vectors(loaded_state.box_vectors_nm))

    def evaluate(self, positions_nm: np.ndarray) -> tuple[float, np.ndarray]:
        self.context.setPositions(_to_openmm_positions(positions_nm))
        state = self.context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        return float(energy), np.asarray(forces)


class AmberSystemLoader:
    def __init__(self, nonbonded_method: str = "PME", constraints: str = "HBonds"):
        _require_openmm()
        self.nonbonded_method = nonbonded_method
        self.constraints = constraints

    def load(self, prmtop_path: str, inpcrd_path: str) -> LoadedAmberState:
        prmtop = AmberPrmtopFile(prmtop_path)
        positions_nm, box_vectors_nm = _load_positions_and_box(inpcrd_path)
        system = prmtop.createSystem(
            nonbondedMethod=_NONBONDED_METHODS[self.nonbonded_method],
            constraints=_CONSTRAINTS[self.constraints],
        )
        atom_labels = []
        atom_names = []
        masses_amu = []
        for atom in prmtop.topology.atoms():
            atom_labels.append((atom.residue.chain.id, atom.residue.name, atom.name, atom.index))
            atom_names.append(atom.name)
        for index in range(system.getNumParticles()):
            masses_amu.append(system.getParticleMass(index).value_in_unit(unit.amu))
        return LoadedAmberState(
            prmtop_path=prmtop_path,
            inpcrd_path=inpcrd_path,
            topology=prmtop.topology,
            system=system,
            positions_nm=np.asarray(positions_nm),
            box_vectors_nm=box_vectors_nm,
            atom_labels=atom_labels,
            atom_names=atom_names,
            masses_amu=np.asarray(masses_amu),
        )


class OpenMMBundleLoader:
    def load(self, system_xml_path: str, coordinates_path: str) -> LoadedAmberState:
        _require_openmm()
        system_xml = Path(system_xml_path).read_text(encoding="utf-8")
        system = openmm.XmlSerializer.deserialize(system_xml)
        positions_nm, box_vectors_nm, topology = _load_pdb_positions_box_and_topology(coordinates_path)
        atom_labels = []
        atom_names = []
        masses_amu = []
        for atom in topology.atoms():
            atom_labels.append((atom.residue.chain.id, atom.residue.name, atom.name, atom.index))
            atom_names.append(atom.name)
        for index in range(system.getNumParticles()):
            masses_amu.append(system.getParticleMass(index).value_in_unit(unit.amu))
        return LoadedAmberState(
            prmtop_path=system_xml_path,
            inpcrd_path=coordinates_path,
            topology=topology,
            system=system,
            positions_nm=np.asarray(positions_nm),
            box_vectors_nm=box_vectors_nm,
            atom_labels=atom_labels,
            atom_names=atom_names,
            masses_amu=np.asarray(masses_amu),
        )


class EVBSystemBuilder:
    def __init__(self, loader: AmberSystemLoader | None = None, openmm_loader: OpenMMBundleLoader | None = None):
        self.loader = loader or AmberSystemLoader()
        self.openmm_loader = openmm_loader or OpenMMBundleLoader()

    def build(self, state1_prmtop: str, state1_inpcrd: str, state2_prmtop: str, state2_inpcrd: str) -> tuple[LoadedAmberState, LoadedAmberState]:
        state1 = self.loader.load(state1_prmtop, state1_inpcrd)
        state2 = self.loader.load(state2_prmtop, state2_inpcrd)
        self.validate_compatibility(state1, state2)
        return state1, state2

    def load_state(self, state_files) -> LoadedAmberState:
        if getattr(state_files, "format", "amber") == "openmm":
            return self.openmm_loader.load(state_files.prmtop, state_files.inpcrd)
        return self.loader.load(state_files.prmtop, state_files.inpcrd)

    def build_from_state_files(self, state1_files, state2_files) -> tuple[LoadedAmberState, LoadedAmberState]:
        state1 = self.load_state(state1_files)
        state2 = self.load_state(state2_files)
        self.validate_compatibility(state1, state2)
        return state1, state2

    @staticmethod
    def validate_compatibility(state1: LoadedAmberState, state2: LoadedAmberState) -> None:
        if state1.system.getNumParticles() != state2.system.getNumParticles():
            raise ValueError("State 1 and state 2 have different atom counts.")
        if state1.positions_nm.shape != state2.positions_nm.shape:
            raise ValueError("State 1 and state 2 coordinates have different shapes.")
        if not np.allclose(state1.masses_amu, state2.masses_amu, atol=1.0e-6):
            raise ValueError("State 1 and state 2 particle masses differ.")
        if state1.atom_names != state2.atom_names:
            raise ValueError("State 1 and state 2 atom names/order differ.")
        if (state1.box_vectors_nm is None) != (state2.box_vectors_nm is None):
            raise ValueError("Only one state defines periodic box vectors.")
        if state1.box_vectors_nm is not None and state2.box_vectors_nm is not None:
            if not np.allclose(state1.box_vectors_nm, state2.box_vectors_nm, atol=1.0e-8):
                raise ValueError("State 1 and state 2 periodic box vectors differ.")
        _validate_constraints(state1.system, state2.system)
        _validate_virtual_sites(state1.system, state2.system)

    def build_openmm_evb_system(
        self,
        state1: LoadedAmberState,
        state2: LoadedAmberState,
        delta_alpha: float,
        h12: float,
        equilibration_restraint: tuple[int, int, float, float] | None = None,
        substrate_com_restraint: tuple[list[int], float] | None = None,
        far_field_restraint: tuple[list[int], np.ndarray, float] | None = None,
        unconstrained_atoms: set[int] | None = None,
        add_cmmotion_remover: bool = True,
        energy_decomposition: bool = False,
        energy_decomposition_mode: str = "exact",
        fallback_to_legacy_for_unsupported_terms: bool = True,
        report_energy_decomposition: bool = True,
        common_force_placement: str = "cv_compatible",
        native_gap_bias_table: Any | None = None,
        native_gap_wall_force_constant: float | None = None,
    ) -> EVBOpenMMSystem:
        if native_gap_wall_force_constant is not None and native_gap_bias_table is None:
            raise ValueError("native_gap_wall_force_constant requires native_gap_bias_table so grid bounds are defined.")
        if energy_decomposition and energy_decomposition_mode == "legacy":
            energy_decomposition = False
        if energy_decomposition:
            return self.build_openmm_evb_system_decomposed(
                state1,
                state2,
                delta_alpha,
                h12,
                equilibration_restraint=equilibration_restraint,
                substrate_com_restraint=substrate_com_restraint,
                far_field_restraint=far_field_restraint,
                unconstrained_atoms=unconstrained_atoms,
                add_cmmotion_remover=add_cmmotion_remover,
                fallback_to_legacy_for_unsupported_terms=fallback_to_legacy_for_unsupported_terms,
                report_energy_decomposition=report_energy_decomposition,
                common_force_placement=common_force_placement,
                native_gap_bias_table=native_gap_bias_table,
                native_gap_wall_force_constant=native_gap_wall_force_constant,
            )
        if energy_decomposition_mode not in {"exact", "legacy"}:
            raise ValueError(f"Unsupported EVB energy decomposition mode: {energy_decomposition_mode!r}.")
        self.validate_compatibility(state1, state2)
        system = openmm.System()
        for index in range(state1.system.getNumParticles()):
            system.addParticle(state1.system.getParticleMass(index))
        _copy_constraints(state1.system, system, skip_atoms=unconstrained_atoms)
        _copy_virtual_sites(state1.system, system)
        if state1.box_vectors_nm is not None:
            system.setDefaultPeriodicBoxVectors(*_to_box_vectors(state1.box_vectors_nm))

        state1_force = _build_state_energy_force(state1.system, "s1")
        state2_force = _build_state_energy_force(state2.system, "s2")
        evb_force = openmm.CustomCVForce(
            _evb_lower_surface_expression(
                include_native_bias=native_gap_bias_table is not None,
                wall_force_constant=native_gap_wall_force_constant,
            )
        )
        evb_force.addCollectiveVariable("e1", state1_force)
        evb_force.addCollectiveVariable("e2", state2_force)
        evb_force.addGlobalParameter("delta_alpha", delta_alpha)
        evb_force.addGlobalParameter("h12", h12)
        if native_gap_wall_force_constant is not None:
            evb_force.addGlobalParameter("k_gap_wall", float(native_gap_wall_force_constant))
            evb_force.addGlobalParameter("gap_lower", float(native_gap_bias_table.grid_min))
            evb_force.addGlobalParameter("gap_upper", float(native_gap_bias_table.grid_max))
        table_bias_function_index = None
        if native_gap_bias_table is not None:
            table_bias_function_index = native_gap_bias_table.add_to_force(evb_force)
        system.addForce(evb_force)
        restraint_force = None
        if equilibration_restraint is not None:
            restraint_force = _build_distance_restraint_force(*equilibration_restraint)
            system.addForce(restraint_force)
        production_restraint_force = None
        if substrate_com_restraint is not None:
            production_restraint_force = _build_centroid_restraint_force(
                substrate_com_restraint[0],
                state1.positions_nm,
                state1.masses_amu,
                substrate_com_restraint[1],
            )
            system.addForce(production_restraint_force)
        far_field_restraint_force = None
        if far_field_restraint is not None:
            far_field_restraint_force = _build_positional_restraint_force(*far_field_restraint)
            system.addForce(far_field_restraint_force)

        if add_cmmotion_remover and _has_cmmotion_remover(state1.system):
            system.addForce(openmm.CMMotionRemover())

        return EVBOpenMMSystem(
            system=system,
            topology=state1.topology,
            positions_nm=state1.positions_nm.copy(),
            box_vectors_nm=state1.box_vectors_nm,
            masses_amu=state1.masses_amu.copy(),
            evb_force=evb_force,
            state1_force=state1_force,
            state2_force=state2_force,
            umbrella_force=None,
            equilibration_restraint_force=restraint_force,
            production_restraint_force=production_restraint_force,
            far_field_restraint_force=far_field_restraint_force,
            energy_decomposition_report=_add_runtime_diagnostics(
                _legacy_decomposition_report() if report_energy_decomposition else None,
                common_force_placement="legacy",
                e_common_inside_custom_cv=False,
                native_gap_bias_table=native_gap_bias_table,
            ),
            native_gap_bias=native_gap_bias_table,
            table_bias_function_index=table_bias_function_index,
            common_forces=[],
            common_force_group=None,
            force_groups={"evb": evb_force.getForceGroup()},
            bias_report=_native_bias_report(native_gap_bias_table),
        )

    def build_openmm_evb_system_decomposed(
        self,
        state1: LoadedAmberState,
        state2: LoadedAmberState,
        delta_alpha: float,
        h12: float,
        equilibration_restraint: tuple[int, int, float, float] | None = None,
        substrate_com_restraint: tuple[list[int], float] | None = None,
        far_field_restraint: tuple[list[int], np.ndarray, float] | None = None,
        unconstrained_atoms: set[int] | None = None,
        add_cmmotion_remover: bool = True,
        fallback_to_legacy_for_unsupported_terms: bool = True,
        report_energy_decomposition: bool = True,
        common_force_placement: str = "cv_compatible",
        native_gap_bias_table: Any | None = None,
        native_gap_wall_force_constant: float | None = None,
    ) -> EVBOpenMMSystem:
        if common_force_placement not in {"outer_system", "cv_compatible"}:
            raise ValueError("common_force_placement must be 'outer_system' or 'cv_compatible'.")
        if native_gap_wall_force_constant is not None and native_gap_bias_table is None:
            raise ValueError("native_gap_wall_force_constant requires native_gap_bias_table so grid bounds are defined.")
        self.validate_compatibility(state1, state2)
        system = openmm.System()
        for index in range(state1.system.getNumParticles()):
            system.addParticle(state1.system.getParticleMass(index))
        _copy_constraints(state1.system, system, skip_atoms=unconstrained_atoms)
        _copy_virtual_sites(state1.system, system)
        if state1.box_vectors_nm is not None:
            system.setDefaultPeriodicBoxVectors(*_to_box_vectors(state1.box_vectors_nm))

        decomposition = _decompose_system_forces(
            state1.system,
            state2.system,
            fallback_to_legacy_for_unsupported_terms=fallback_to_legacy_for_unsupported_terms,
            report_energy_decomposition=report_energy_decomposition,
        )
        common_outer_forces: list[Any] = []
        common_force_group = None
        e_common_inside_custom_cv = common_force_placement == "cv_compatible"
        if common_force_placement == "outer_system":
            common_force_group = 30
            common_outer_forces = _add_common_forces_to_outer_system(
                system,
                decomposition["common_forces"],
                common_force_group,
            )
            common_force = None
        else:
            common_force = _aggregate_energy_forces(decomposition["common_forces"], "common")
        state1_force = _aggregate_energy_forces(decomposition["state1_forces"], "s1")
        state2_force = _aggregate_energy_forces(decomposition["state2_forces"], "s2")
        evb_force = openmm.CustomCVForce(
            _evb_lower_surface_expression(
                include_common=e_common_inside_custom_cv,
                include_native_bias=native_gap_bias_table is not None,
                wall_force_constant=native_gap_wall_force_constant,
            )
        )
        if e_common_inside_custom_cv:
            evb_force.addCollectiveVariable("e_common", common_force)
        evb_force.addCollectiveVariable("e1", state1_force)
        evb_force.addCollectiveVariable("e2", state2_force)
        evb_force.addGlobalParameter("delta_alpha", delta_alpha)
        evb_force.addGlobalParameter("h12", h12)
        if native_gap_wall_force_constant is not None:
            evb_force.addGlobalParameter("k_gap_wall", float(native_gap_wall_force_constant))
            evb_force.addGlobalParameter("gap_lower", float(native_gap_bias_table.grid_min))
            evb_force.addGlobalParameter("gap_upper", float(native_gap_bias_table.grid_max))
        table_bias_function_index = None
        if native_gap_bias_table is not None:
            table_bias_function_index = native_gap_bias_table.add_to_force(evb_force)
        system.addForce(evb_force)
        restraint_force = None
        if equilibration_restraint is not None:
            restraint_force = _build_distance_restraint_force(*equilibration_restraint)
            system.addForce(restraint_force)
        production_restraint_force = None
        if substrate_com_restraint is not None:
            production_restraint_force = _build_centroid_restraint_force(
                substrate_com_restraint[0],
                state1.positions_nm,
                state1.masses_amu,
                substrate_com_restraint[1],
            )
            system.addForce(production_restraint_force)
        far_field_restraint_force = None
        if far_field_restraint is not None:
            far_field_restraint_force = _build_positional_restraint_force(*far_field_restraint)
            system.addForce(far_field_restraint_force)

        if add_cmmotion_remover and _has_cmmotion_remover(state1.system):
            system.addForce(openmm.CMMotionRemover())

        return EVBOpenMMSystem(
            system=system,
            topology=state1.topology,
            positions_nm=state1.positions_nm.copy(),
            box_vectors_nm=state1.box_vectors_nm,
            masses_amu=state1.masses_amu.copy(),
            evb_force=evb_force,
            state1_force=state1_force,
            state2_force=state2_force,
            umbrella_force=None,
            equilibration_restraint_force=restraint_force,
            production_restraint_force=production_restraint_force,
            far_field_restraint_force=far_field_restraint_force,
            energy_decomposition_report=_add_runtime_diagnostics(
                decomposition["report"],
                common_force_placement=common_force_placement,
                e_common_inside_custom_cv=e_common_inside_custom_cv,
                native_gap_bias_table=native_gap_bias_table,
            ),
            native_gap_bias=native_gap_bias_table,
            table_bias_function_index=table_bias_function_index,
            common_forces=common_outer_forces,
            common_force_group=common_force_group,
            force_groups={"evb": evb_force.getForceGroup(), "common": common_force_group},
            bias_report=_native_bias_report(native_gap_bias_table),
        )

    def build_openmm_gap_umbrella_system(
        self,
        state1: LoadedAmberState,
        state2: LoadedAmberState,
        delta_alpha: float,
        h12: float,
        gap_center: float,
        gap_force_constant: float,
        equilibration_restraint: tuple[int, int, float, float] | None = None,
        substrate_com_restraint: tuple[list[int], float] | None = None,
        far_field_restraint: tuple[list[int], np.ndarray, float] | None = None,
        unconstrained_atoms: set[int] | None = None,
        add_cmmotion_remover: bool = True,
        energy_decomposition: bool = False,
        energy_decomposition_mode: str = "exact",
        fallback_to_legacy_for_unsupported_terms: bool = True,
        report_energy_decomposition: bool = True,
        common_force_placement: str = "cv_compatible",
    ) -> EVBOpenMMSystem:
        if common_force_placement not in {"outer_system", "cv_compatible"}:
            raise ValueError("common_force_placement must be 'outer_system' or 'cv_compatible'.")
        self.validate_compatibility(state1, state2)
        system = openmm.System()
        for index in range(state1.system.getNumParticles()):
            system.addParticle(state1.system.getParticleMass(index))
        _copy_constraints(state1.system, system, skip_atoms=unconstrained_atoms)
        _copy_virtual_sites(state1.system, system)
        if state1.box_vectors_nm is not None:
            system.setDefaultPeriodicBoxVectors(*_to_box_vectors(state1.box_vectors_nm))

        decomposition_report = _legacy_decomposition_report() if report_energy_decomposition else None
        common_outer_forces: list[Any] = []
        common_force_group = None
        e_common_inside_custom_cv = False
        if energy_decomposition and energy_decomposition_mode == "legacy":
            energy_decomposition = False
        if energy_decomposition:
            decomposition = _decompose_system_forces(
                state1.system,
                state2.system,
                fallback_to_legacy_for_unsupported_terms=fallback_to_legacy_for_unsupported_terms,
                report_energy_decomposition=report_energy_decomposition,
            )
            e_common_inside_custom_cv = common_force_placement == "cv_compatible"
            if common_force_placement == "outer_system":
                common_force_group = 30
                common_outer_forces = _add_common_forces_to_outer_system(
                    system,
                    decomposition["common_forces"],
                    common_force_group,
                )
                common_force = None
            else:
                common_force = _aggregate_energy_forces(decomposition["common_forces"], "common")
            state1_force = _aggregate_energy_forces(decomposition["state1_forces"], "s1")
            state2_force = _aggregate_energy_forces(decomposition["state2_forces"], "s2")
            evb_force = openmm.CustomCVForce(
                _evb_lower_surface_expression(include_common=e_common_inside_custom_cv)
                + " + 0.5*k_gap*((e1 - e2 - delta_alpha)-gap_center)^2"
            )
            if e_common_inside_custom_cv:
                evb_force.addCollectiveVariable("e_common", common_force)
            decomposition_report = decomposition["report"]
        else:
            if energy_decomposition_mode not in {"exact", "legacy"}:
                raise ValueError(f"Unsupported EVB energy decomposition mode: {energy_decomposition_mode!r}.")
            state1_force = _build_state_energy_force(state1.system, "s1")
            state2_force = _build_state_energy_force(state2.system, "s2")
            evb_force = openmm.CustomCVForce(
                _evb_lower_surface_expression()
                + " + 0.5*k_gap*((e1 - e2 - delta_alpha)-gap_center)^2"
            )
        evb_force.addCollectiveVariable("e1", state1_force)
        evb_force.addCollectiveVariable("e2", state2_force)
        evb_force.addGlobalParameter("delta_alpha", delta_alpha)
        evb_force.addGlobalParameter("h12", h12)
        evb_force.addGlobalParameter("k_gap", gap_force_constant)
        evb_force.addGlobalParameter("gap_center", gap_center)
        system.addForce(evb_force)
        restraint_force = None
        if equilibration_restraint is not None:
            restraint_force = _build_distance_restraint_force(*equilibration_restraint)
            system.addForce(restraint_force)
        production_restraint_force = None
        if substrate_com_restraint is not None:
            production_restraint_force = _build_centroid_restraint_force(
                substrate_com_restraint[0],
                state1.positions_nm,
                state1.masses_amu,
                substrate_com_restraint[1],
            )
            system.addForce(production_restraint_force)
        far_field_restraint_force = None
        if far_field_restraint is not None:
            far_field_restraint_force = _build_positional_restraint_force(*far_field_restraint)
            system.addForce(far_field_restraint_force)

        if add_cmmotion_remover and _has_cmmotion_remover(state1.system):
            system.addForce(openmm.CMMotionRemover())

        return EVBOpenMMSystem(
            system=system,
            topology=state1.topology,
            positions_nm=state1.positions_nm.copy(),
            box_vectors_nm=state1.box_vectors_nm,
            masses_amu=state1.masses_amu.copy(),
            evb_force=evb_force,
            state1_force=state1_force,
            state2_force=state2_force,
            umbrella_force=evb_force,
            equilibration_restraint_force=restraint_force,
            production_restraint_force=production_restraint_force,
            far_field_restraint_force=far_field_restraint_force,
            energy_decomposition_report=_add_runtime_diagnostics(
                decomposition_report,
                common_force_placement=common_force_placement if energy_decomposition else "legacy",
                e_common_inside_custom_cv=e_common_inside_custom_cv,
                native_gap_bias_table=None,
            ),
            native_gap_bias=None,
            table_bias_function_index=None,
            common_forces=common_outer_forces,
            common_force_group=common_force_group,
            force_groups={"evb": evb_force.getForceGroup(), "common": common_force_group},
            bias_report=None,
        )

    def build_openmm_proton_transfer_umbrella_system(
        self,
        state1: LoadedAmberState,
        state2: LoadedAmberState,
        delta_alpha: float,
        h12: float,
        donor_index: int,
        proton_index: int,
        acceptor_index: int,
        rc_center_nm: float,
        rc_force_constant_kj_mol_nm2: float,
        equilibration_restraint: tuple[int, int, float, float] | None = None,
        substrate_com_restraint: tuple[list[int], float] | None = None,
        far_field_restraint: tuple[list[int], np.ndarray, float] | None = None,
        unconstrained_atoms: set[int] | None = None,
        add_cmmotion_remover: bool = True,
    ) -> EVBOpenMMSystem:
        self.validate_compatibility(state1, state2)
        system = openmm.System()
        for index in range(state1.system.getNumParticles()):
            system.addParticle(state1.system.getParticleMass(index))
        _copy_constraints(state1.system, system, skip_atoms=unconstrained_atoms)
        _copy_virtual_sites(state1.system, system)
        if state1.box_vectors_nm is not None:
            system.setDefaultPeriodicBoxVectors(*_to_box_vectors(state1.box_vectors_nm))

        state1_force = _build_state_energy_force(state1.system, "s1")
        state2_force = _build_state_energy_force(state2.system, "s2")
        evb_force = openmm.CustomCVForce(
            "0.5*(e1 + e2 + delta_alpha) - sqrt(0.25*(e1 - e2 - delta_alpha)^2 + h12^2)"
        )
        evb_force.addCollectiveVariable("e1", state1_force)
        evb_force.addCollectiveVariable("e2", state2_force)
        evb_force.addGlobalParameter("delta_alpha", delta_alpha)
        evb_force.addGlobalParameter("h12", h12)
        system.addForce(evb_force)

        proton_transfer_cv = build_proton_transfer_cv_force(donor_index, proton_index, acceptor_index)
        umbrella_force = openmm.CustomCVForce("0.5*k_pt*(rc_pt-rc_center)^2")
        umbrella_force.addCollectiveVariable("rc_pt", proton_transfer_cv)
        umbrella_force.addGlobalParameter("k_pt", rc_force_constant_kj_mol_nm2)
        umbrella_force.addGlobalParameter("rc_center", rc_center_nm)
        system.addForce(umbrella_force)

        restraint_force = None
        if equilibration_restraint is not None:
            restraint_force = _build_distance_restraint_force(*equilibration_restraint)
            system.addForce(restraint_force)
        production_restraint_force = None
        if substrate_com_restraint is not None:
            production_restraint_force = _build_centroid_restraint_force(
                substrate_com_restraint[0],
                state1.positions_nm,
                state1.masses_amu,
                substrate_com_restraint[1],
            )
            system.addForce(production_restraint_force)
        far_field_restraint_force = None
        if far_field_restraint is not None:
            far_field_restraint_force = _build_positional_restraint_force(*far_field_restraint)
            system.addForce(far_field_restraint_force)

        if add_cmmotion_remover and _has_cmmotion_remover(state1.system):
            system.addForce(openmm.CMMotionRemover())

        return EVBOpenMMSystem(
            system=system,
            topology=state1.topology,
            positions_nm=state1.positions_nm.copy(),
            box_vectors_nm=state1.box_vectors_nm,
            masses_amu=state1.masses_amu.copy(),
            evb_force=evb_force,
            state1_force=state1_force,
            state2_force=state2_force,
            umbrella_force=umbrella_force,
            equilibration_restraint_force=restraint_force,
            production_restraint_force=production_restraint_force,
            far_field_restraint_force=far_field_restraint_force,
        )

    def build_openmm_mapped_system(
        self,
        state1: LoadedAmberState,
        state2: LoadedAmberState,
        lambda_value: float,
        delta_alpha: float,
        equilibration_restraint: tuple[int, int, float, float] | None = None,
        substrate_com_restraint: tuple[list[int], float] | None = None,
        far_field_restraint: tuple[list[int], np.ndarray, float] | None = None,
        unconstrained_atoms: set[int] | None = None,
        add_cmmotion_remover: bool = True,
    ) -> MappedOpenMMSystem:
        self.validate_compatibility(state1, state2)
        system = openmm.System()
        for index in range(state1.system.getNumParticles()):
            system.addParticle(state1.system.getParticleMass(index))
        _copy_constraints(state1.system, system, skip_atoms=unconstrained_atoms)
        _copy_virtual_sites(state1.system, system)
        if state1.box_vectors_nm is not None:
            system.setDefaultPeriodicBoxVectors(*_to_box_vectors(state1.box_vectors_nm))

        state1_force = _build_state_energy_force(state1.system, "s1")
        state2_force = _build_state_energy_force(state2.system, "s2")
        mapping_force = openmm.CustomCVForce("(1-lambda_map)*e1 + lambda_map*(e2 + delta_alpha)")
        mapping_force.addCollectiveVariable("e1", state1_force)
        mapping_force.addCollectiveVariable("e2", state2_force)
        mapping_force.addGlobalParameter("lambda_map", lambda_value)
        mapping_force.addGlobalParameter("delta_alpha", delta_alpha)
        system.addForce(mapping_force)
        restraint_force = None
        if equilibration_restraint is not None:
            restraint_force = _build_distance_restraint_force(*equilibration_restraint)
            system.addForce(restraint_force)
        production_restraint_force = None
        if substrate_com_restraint is not None:
            production_restraint_force = _build_centroid_restraint_force(
                substrate_com_restraint[0],
                state1.positions_nm,
                state1.masses_amu,
                substrate_com_restraint[1],
            )
            system.addForce(production_restraint_force)
        far_field_restraint_force = None
        if far_field_restraint is not None:
            far_field_restraint_force = _build_positional_restraint_force(*far_field_restraint)
            system.addForce(far_field_restraint_force)

        if add_cmmotion_remover and _has_cmmotion_remover(state1.system):
            system.addForce(openmm.CMMotionRemover())

        return MappedOpenMMSystem(
            system=system,
            topology=state1.topology,
            positions_nm=state1.positions_nm.copy(),
            box_vectors_nm=state1.box_vectors_nm,
            masses_amu=state1.masses_amu.copy(),
            mapping_force=mapping_force,
            state1_force=state1_force,
            state2_force=state2_force,
            equilibration_restraint_force=restraint_force,
            production_restraint_force=production_restraint_force,
            far_field_restraint_force=far_field_restraint_force,
        )


def build_proton_transfer_cv_force(donor_index: int, proton_index: int, acceptor_index: int):
    _require_openmm()
    cv_force = openmm.CustomCompoundBondForce(3, "distance(p1,p2) - distance(p2,p3)")
    cv_force.addBond([donor_index, proton_index, acceptor_index], [])
    return cv_force


def build_evb_gap_cv_force(
    state1_system: Any,
    state2_system: Any,
    delta_alpha: float,
    prefix: str = "gap",
    energy_decomposition: bool = False,
    energy_decomposition_mode: str = "exact",
    fallback_to_legacy_for_unsupported_terms: bool = True,
):
    """Build a CustomCVForce for gap = E1 - E2 - delta_alpha in kJ/mol."""
    _require_openmm()
    if energy_decomposition and energy_decomposition_mode == "legacy":
        energy_decomposition = False
    if energy_decomposition:
        decomposition = _decompose_system_forces(
            state1_system,
            state2_system,
            fallback_to_legacy_for_unsupported_terms=fallback_to_legacy_for_unsupported_terms,
            report_energy_decomposition=False,
        )
        state1_force = _aggregate_energy_forces(decomposition["state1_forces"], f"{prefix}_s1")
        state2_force = _aggregate_energy_forces(decomposition["state2_forces"], f"{prefix}_s2")
    else:
        if energy_decomposition_mode not in {"exact", "legacy"}:
            raise ValueError(f"Unsupported EVB energy decomposition mode: {energy_decomposition_mode!r}.")
        state1_force = _build_state_energy_force(state1_system, f"{prefix}_s1")
        state2_force = _build_state_energy_force(state2_system, f"{prefix}_s2")
    gap_force = openmm.CustomCVForce("e1 - e2 - delta_alpha")
    gap_force.addCollectiveVariable("e1", state1_force)
    gap_force.addCollectiveVariable("e2", state2_force)
    gap_force.addGlobalParameter("delta_alpha", delta_alpha)
    return gap_force


def evb_diabatic_energies(evb_system: EVBOpenMMSystem, context: Any) -> tuple[float, float]:
    values = [float(value) for value in evb_system.evb_force.getCollectiveVariableValues(context)]
    cv_names = [
        evb_system.evb_force.getCollectiveVariableName(index)
        for index in range(evb_system.evb_force.getNumCollectiveVariables())
    ]
    if cv_names and cv_names[0] == "e_common":
        e_common, e1, e2 = values[:3]
        return e_common + e1, e_common + e2
    if evb_system.common_force_group is not None:
        e_common = _context_group_energy(context, evb_system.common_force_group)
        return e_common + values[0], e_common + values[1]
    return values[0], values[1]


def evb_common_energy(evb_system: EVBOpenMMSystem, context: Any) -> float | None:
    values = [float(value) for value in evb_system.evb_force.getCollectiveVariableValues(context)]
    cv_names = [
        evb_system.evb_force.getCollectiveVariableName(index)
        for index in range(evb_system.evb_force.getNumCollectiveVariables())
    ]
    if cv_names and cv_names[0] == "e_common":
        return values[0]
    if evb_system.common_force_group is not None:
        return _context_group_energy(context, evb_system.common_force_group)
    return None


def evb_residual_energies(evb_system: EVBOpenMMSystem, context: Any) -> tuple[float, float]:
    values = [float(value) for value in evb_system.evb_force.getCollectiveVariableValues(context)]
    cv_names = [
        evb_system.evb_force.getCollectiveVariableName(index)
        for index in range(evb_system.evb_force.getNumCollectiveVariables())
    ]
    if cv_names and cv_names[0] == "e_common":
        return values[1], values[2]
    return values[0], values[1]


def _context_group_energy(context: Any, group: int) -> float:
    state = context.getState(getEnergy=True, groups={int(group)})
    return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def _evb_lower_surface_expression(
    *,
    include_common: bool = False,
    include_native_bias: bool = False,
    wall_force_constant: float | None = None,
) -> str:
    expression = "0.5*(e1 + e2 + delta_alpha) - sqrt(0.25*(e1 - e2 - delta_alpha)^2 + h12^2)"
    if include_common:
        expression = "e_common + " + expression
    if include_native_bias:
        expression += " + gap_bias(e1 - e2 - delta_alpha)"
    if wall_force_constant is not None:
        expression += " + 0.5*k_gap_wall*(step((e1 - e2 - delta_alpha)-gap_upper)*((e1 - e2 - delta_alpha)-gap_upper)^2 + step(gap_lower-(e1 - e2 - delta_alpha))*(gap_lower-(e1 - e2 - delta_alpha))^2)"
    return expression


def _add_common_forces_to_outer_system(system: Any, common_forces: list[Any], force_group: int) -> list[Any]:
    outer_forces: list[Any] = []
    for index, force in enumerate(common_forces):
        cloned = _clone_openmm_object(force)
        _rename_force_global_parameters(cloned, f"common_outer_{index}")
        cloned.setForceGroup(int(force_group))
        system.addForce(cloned)
        outer_forces.append(cloned)
    return outer_forces


def _add_runtime_diagnostics(
    report: dict[str, Any] | None,
    *,
    common_force_placement: str,
    e_common_inside_custom_cv: bool,
    native_gap_bias_table: Any | None,
) -> dict[str, Any] | None:
    if report is None:
        return None
    report = dict(report)
    report["common_force_placement"] = common_force_placement
    report["e_common_inside_custom_cv"] = bool(e_common_inside_custom_cv)
    base_inner_contexts = 1
    if e_common_inside_custom_cv:
        base_inner_contexts += 1
    report["custom_cv_inner_context_count"] = base_inner_contexts
    report["native_gap_bias_uses_app_metadynamics"] = False if native_gap_bias_table is not None else None
    report["native_gap_bias_uses_bias_variable"] = False if native_gap_bias_table is not None else None
    return report


def _native_bias_report(native_gap_bias_table: Any | None) -> dict[str, Any] | None:
    if native_gap_bias_table is None:
        return None
    return {
        "enabled": True,
        "type": "NativeGapBiasTable1D",
        "function_name": getattr(native_gap_bias_table, "function_name", "gap_bias"),
        "function_index": native_gap_bias_table.function_index,
        "grid_min_kj_mol": native_gap_bias_table.grid_min,
        "grid_max_kj_mol": native_gap_bias_table.grid_max,
        "grid_width": native_gap_bias_table.grid_width,
        "uses_app_metadynamics": False,
        "uses_bias_variable": False,
    }


def _build_distance_restraint_force(atom1: int, atom2: int, target_distance_nm: float, force_constant_kj_mol_nm2: float):
    force = openmm.CustomBondForce("0.5*k_rest*(r-r0)^2")
    force.addGlobalParameter("k_rest", force_constant_kj_mol_nm2)
    force.addPerBondParameter("r0")
    force.addBond(atom1, atom2, [target_distance_nm])
    return force


def _build_centroid_restraint_force(
    atom_indices: list[int],
    reference_positions_nm: np.ndarray,
    masses_amu: np.ndarray,
    force_constant_kj_mol_nm2: float,
):
    if not atom_indices:
        raise ValueError("Centroid restraint requires at least one atom index.")
    weights = [float(masses_amu[index]) for index in atom_indices]
    reference_com = np.average(reference_positions_nm[atom_indices], axis=0, weights=weights)
    force = openmm.CustomCentroidBondForce(1, "0.5*k_com*((x1-x0)^2 + (y1-y0)^2 + (z1-z0)^2)")
    force.addGlobalParameter("k_com", force_constant_kj_mol_nm2)
    for name in ("x0", "y0", "z0"):
        force.addPerBondParameter(name)
    group_index = force.addGroup(atom_indices, weights)
    force.addBond([group_index], [float(reference_com[0]), float(reference_com[1]), float(reference_com[2])])
    return force


def _build_positional_restraint_force(
    atom_indices: list[int],
    reference_positions_nm: np.ndarray,
    force_constant_kj_mol_nm2: float,
    parameter_name: str = "k_pos",
):
    if not atom_indices:
        raise ValueError("Positional restraint requires at least one atom index.")
    force = openmm.CustomExternalForce(f"0.5*{parameter_name}*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter(parameter_name, force_constant_kj_mol_nm2)
    for parameter in ("x0", "y0", "z0"):
        force.addPerParticleParameter(parameter)
    for atom_index in atom_indices:
        position = reference_positions_nm[atom_index]
        force.addParticle(atom_index, [float(position[0]), float(position[1]), float(position[2])])
    return force


def build_absolute_positional_restraint_force(
    reference_positions_nm: np.ndarray,
    atom_indices: list[int] | None = None,
    force_constant_kj_mol_nm2: float = 250.0,
    parameter_name: str = "k_pos",
):
    _require_openmm()
    indices = list(range(len(reference_positions_nm))) if atom_indices is None else list(atom_indices)
    return _build_positional_restraint_force(indices, reference_positions_nm, force_constant_kj_mol_nm2, parameter_name=parameter_name)


def load_positions_file(path: str) -> np.ndarray:
    _require_openmm()
    positions_nm, _ = _load_positions_and_box(path)
    return positions_nm


def write_openmm_bundle(path: str | Path, system: Any, topology: Any, positions_nm: np.ndarray, box_vectors_nm: np.ndarray | None = None) -> None:
    _require_openmm()
    bundle_dir = Path(path)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "system.xml").write_text(openmm.XmlSerializer.serialize(system), encoding="utf-8")
    with (bundle_dir / "coordinates.pdb").open("w", encoding="utf-8") as handle:
        if box_vectors_nm is not None:
            topology.setPeriodicBoxVectors(tuple(openmm.Vec3(*box_vectors_nm[i]) for i in range(3)))
        PDBFile.writeFile(topology, _to_openmm_positions(positions_nm), handle)


def write_toy_evb_bundles(base_dir: str | Path) -> None:
    """Write a two-state harmonic-bond toy system for examples and tests."""
    _require_openmm()
    from openmm.app import Topology, element

    base_path = Path(base_dir)

    def make_topology():
        topology = Topology()
        chain = topology.addChain("A")
        residue = topology.addResidue("MOL", chain)
        topology.addAtom("A1", element.carbon, residue)
        topology.addAtom("A2", element.carbon, residue)
        return topology

    def make_system(r0_nm: float):
        system = openmm.System()
        for _ in range(2):
            system.addParticle(12.0 * unit.amu)
        bond_force = openmm.HarmonicBondForce()
        bond_force.addBond(
            0,
            1,
            r0_nm * unit.nanometer,
            1000.0 * unit.kilojoule_per_mole / unit.nanometer**2,
        )
        system.addForce(bond_force)
        return system

    positions_nm = np.asarray([[0.0, 0.0, 0.0], [0.18, 0.0, 0.0]], dtype=float)
    write_openmm_bundle(base_path / "toy_state1", make_system(0.12), make_topology(), positions_nm)
    write_openmm_bundle(base_path / "toy_state2", make_system(0.14), make_topology(), positions_nm)


def write_pdb(path: str, topology: Any, positions_nm: np.ndarray) -> None:
    _require_openmm()
    with open(path, "w", encoding="utf-8") as handle:
        PDBFile.writeFile(topology, _to_openmm_positions(positions_nm), handle)


def write_dcd_frame(dcd: Any, positions_nm: np.ndarray, topology: Any, step: int) -> None:
    del topology, step
    dcd.writeModel(_to_openmm_positions(positions_nm))


def create_dcd_writer(path: str, topology: Any, timestep_ps: float) -> Any:
    _require_openmm()
    handle = open(path, "wb")
    return handle, DCDFile(handle, topology, dt=timestep_ps * unit.picoseconds)



def _legacy_decomposition_report() -> dict[str, Any]:
    empty_partition = _partition_diagnostics([], 0)
    return {
        "enabled": False,
        "mode": "legacy",
        "n_common_forces": 0,
        "n_state1_forces": 0,
        "n_state2_forces": 0,
        "n_common_terms": 0,
        "n_state1_terms": 0,
        "n_state2_terms": 0,
        "unsupported_forces": [],
        "warnings": [],
        "partitions": {
            "common": empty_partition,
            "state1": empty_partition,
            "state2": empty_partition,
        },
        "duplicated_full_nonbonded": False,
        "performance_warnings": [],
    }


def _decompose_system_forces(
    system1: Any,
    system2: Any,
    *,
    fallback_to_legacy_for_unsupported_terms: bool = True,
    report_energy_decomposition: bool = True,
) -> dict[str, Any]:
    common_forces: list[Any] = []
    state1_forces: list[Any] = []
    state2_forces: list[Any] = []
    unsupported: list[str] = []
    warnings: list[str] = []
    counts = {"n_common_terms": 0, "n_state1_terms": 0, "n_state2_terms": 0}
    max_forces = max(system1.getNumForces(), system2.getNumForces())
    for force_index in range(max_forces):
        if force_index >= system1.getNumForces():
            force2 = system2.getForce(force_index)
            if not _should_skip_subforce(force2):
                state2_forces.append(_clone_openmm_object(force2))
                counts["n_state2_terms"] += _force_term_count(force2)
            continue
        if force_index >= system2.getNumForces():
            force1 = system1.getForce(force_index)
            if not _should_skip_subforce(force1):
                state1_forces.append(_clone_openmm_object(force1))
                counts["n_state1_terms"] += _force_term_count(force1)
            continue
        force1 = system1.getForce(force_index)
        force2 = system2.getForce(force_index)
        if _should_skip_subforce(force1) and _should_skip_subforce(force2):
            continue
        if _force_xml(force1) == _force_xml(force2):
            common_forces.append(_clone_openmm_object(force1))
            counts["n_common_terms"] += _force_term_count(force1)
            continue
        decomposed = _decompose_supported_force(force1, force2)
        if decomposed is None:
            name = type(force1).__name__ if type(force1) is type(force2) else f"{type(force1).__name__}/{type(force2).__name__}"
            unsupported.append(name)
            if isinstance(force1, openmm.NonbondedForce) or isinstance(force2, openmm.NonbondedForce):
                message = (
                    "NonbondedForce differs between states and cannot be decomposed exactly in PME/local "
                    "nonbonded form; full state-specific NonbondedForce objects are required unless fallback is disabled."
                )
            else:
                message = f"{name} differs between states and cannot be decomposed exactly by the current implementation."
            if not fallback_to_legacy_for_unsupported_terms:
                raise ValueError(
                    f"Exact EVB energy decomposition failed for {name}: {message} "
                    "Set evb.energy_decomposition.fallback_to_legacy_for_unsupported_terms: true to keep the full force in e1/e2."
                )
            warnings.append(message)
            state1_forces.append(_clone_openmm_object(force1))
            state2_forces.append(_clone_openmm_object(force2))
            counts["n_state1_terms"] += _force_term_count(force1)
            counts["n_state2_terms"] += _force_term_count(force2)
            continue
        for key, dest in (("common", common_forces), ("state1", state1_forces), ("state2", state2_forces)):
            if decomposed[key] is not None:
                dest.append(decomposed[key])
        counts["n_common_terms"] += decomposed["n_common_terms"]
        counts["n_state1_terms"] += decomposed["n_state1_terms"]
        counts["n_state2_terms"] += decomposed["n_state2_terms"]

    report = None
    if report_energy_decomposition:
        common_diag = _partition_diagnostics(common_forces, system1.getNumParticles())
        state1_diag = _partition_diagnostics(state1_forces, system1.getNumParticles())
        state2_diag = _partition_diagnostics(state2_forces, system2.getNumParticles())
        duplicated_full_nonbonded = _has_full_nonbonded(state1_diag) and _has_full_nonbonded(state2_diag)
        performance_warnings: list[str] = []
        if duplicated_full_nonbonded:
            performance_warnings.append(
                "Exact decomposition still evaluates full state-specific NonbondedForce objects for both states; PME/nonbonded cost is duplicated and speedup should not be expected."
            )
        report_warnings = sorted(set(warnings + performance_warnings))
        report = {
            "enabled": True,
            "mode": "exact",
            "n_common_forces": len(common_forces),
            "n_state1_forces": len(state1_forces),
            "n_state2_forces": len(state2_forces),
            "n_common_terms": counts["n_common_terms"],
            "n_state1_terms": counts["n_state1_terms"],
            "n_state2_terms": counts["n_state2_terms"],
            "unsupported_forces": unsupported,
            "warnings": report_warnings,
            "partitions": {
                "common": common_diag,
                "state1": state1_diag,
                "state2": state2_diag,
            },
            "duplicated_full_nonbonded": duplicated_full_nonbonded,
            "performance_warnings": performance_warnings,
        }
    return {"common_forces": common_forces, "state1_forces": state1_forces, "state2_forces": state2_forces, "report": report}

def _force_xml(force: Any) -> str:
    return openmm.XmlSerializer.serialize(force)


def _aggregate_energy_forces(forces: list[Any], prefix: str):
    prepared = []
    for index, force in enumerate(forces):
        cloned = _clone_openmm_object(force)
        _rename_force_global_parameters(cloned, f"{prefix}_{index}")
        prepared.append(cloned)
    if not prepared:
        return openmm.CustomCVForce("0")
    if len(prepared) == 1:
        return prepared[0]
    variable_names = [f"{prefix}_{index}" for index in range(len(prepared))]
    aggregate = openmm.CustomCVForce(" + ".join(variable_names))
    for variable_name, force in zip(variable_names, prepared):
        aggregate.addCollectiveVariable(variable_name, force)
    return aggregate


def _force_term_count(force: Any) -> int:
    if isinstance(force, openmm.NonbondedForce):
        return int(force.getNumParticles() + force.getNumExceptions())
    for method in ("getNumBonds", "getNumAngles", "getNumTorsions"):
        if hasattr(force, method):
            return int(getattr(force, method)())
    return 1


def _partition_diagnostics(forces: list[Any], system_particles: int) -> dict[str, Any]:
    forces_by_class: dict[str, int] = {}
    nonbonded_forces = []
    for force in forces:
        class_name = type(force).__name__
        forces_by_class[class_name] = forces_by_class.get(class_name, 0) + 1
        if isinstance(force, openmm.NonbondedForce):
            full_system = force.getNumParticles() == system_particles
            nonbonded_forces.append(
                {
                    "class": class_name,
                    "force_group": int(force.getForceGroup()),
                    "particles": int(force.getNumParticles()),
                    "exceptions": int(force.getNumExceptions()),
                    "nonbonded_method": _nonbonded_method_name(force),
                    "uses_periodic_boundary_conditions": bool(force.usesPeriodicBoundaryConditions()),
                    "full_system": bool(full_system),
                    "reduced_or_correction_only": not bool(full_system),
                    "performance_relevance": "dominant_full_system" if full_system else "reduced_or_correction_only",
                }
            )
    return {
        "n_forces": len(forces),
        "forces_by_class": forces_by_class,
        "has_nonbonded_force": bool(nonbonded_forces),
        "nonbonded_forces": nonbonded_forces,
    }


def _has_full_nonbonded(partition: dict[str, Any]) -> bool:
    return any(item.get("full_system") for item in partition.get("nonbonded_forces", []))


def _nonbonded_method_name(force: Any) -> str:
    method = force.getNonbondedMethod()
    names = {
        openmm.NonbondedForce.NoCutoff: "NoCutoff",
        openmm.NonbondedForce.CutoffNonPeriodic: "CutoffNonPeriodic",
        openmm.NonbondedForce.CutoffPeriodic: "CutoffPeriodic",
        openmm.NonbondedForce.Ewald: "Ewald",
        openmm.NonbondedForce.PME: "PME",
        openmm.NonbondedForce.LJPME: "LJPME",
    }
    return names.get(method, str(method))


def _copy_force_metadata(source: Any, target: Any) -> None:
    target.setForceGroup(source.getForceGroup())
    if hasattr(source, "usesPeriodicBoundaryConditions") and hasattr(target, "setUsesPeriodicBoundaryConditions"):
        try:
            target.setUsesPeriodicBoundaryConditions(source.usesPeriodicBoundaryConditions())
        except Exception:
            pass


def _decompose_supported_force(force1: Any, force2: Any) -> dict[str, Any] | None:
    if type(force1) is not type(force2):
        return None
    for cls, kind in ((openmm.HarmonicBondForce, "harmonic_bond"), (openmm.HarmonicAngleForce, "harmonic_angle"), (openmm.PeriodicTorsionForce, "periodic_torsion"), (openmm.RBTorsionForce, "rb_torsion"), (openmm.CustomBondForce, "custom_bond"), (openmm.CustomAngleForce, "custom_angle"), (openmm.CustomTorsionForce, "custom_torsion")):
        if isinstance(force1, cls):
            return _decompose_term_force(force1, force2, kind)
    return None


def _decompose_term_force(force1: Any, force2: Any, kind: str) -> dict[str, Any] | None:
    if kind.startswith("custom") and not _custom_force_compatible(force1, force2, kind):
        return None
    common = _make_empty_term_force(force1, kind)
    state1 = _make_empty_term_force(force1, kind)
    state2 = _make_empty_term_force(force2, kind)
    terms1 = [_term_signature(force1, kind, i) for i in range(_force_term_count(force1))]
    terms2 = [_term_signature(force2, kind, i) for i in range(_force_term_count(force2))]
    n_common = n_state1 = n_state2 = 0
    used2: set[int] = set()
    for i, term1 in enumerate(terms1):
        match = next((j for j, term2 in enumerate(terms2) if j not in used2 and term1 == term2), None)
        if match is None:
            _add_term(state1, kind, force1, i)
            n_state1 += 1
        else:
            _add_term(common, kind, force1, i)
            used2.add(match)
            n_common += 1
    for j in range(len(terms2)):
        if j not in used2:
            _add_term(state2, kind, force2, j)
            n_state2 += 1
    return {"common": common if _force_term_count(common) else None, "state1": state1 if _force_term_count(state1) else None, "state2": state2 if _force_term_count(state2) else None, "n_common_terms": n_common, "n_state1_terms": n_state1, "n_state2_terms": n_state2}


def _custom_force_compatible(force1: Any, force2: Any, kind: str) -> bool:
    count_method, name_method, _add_method = _custom_parameter_methods(kind)
    if (
        force1.getEnergyFunction() != force2.getEnergyFunction()
        or force1.getNumGlobalParameters() != force2.getNumGlobalParameters()
        or getattr(force1, count_method)() != getattr(force2, count_method)()
    ):
        return False
    for i in range(force1.getNumGlobalParameters()):
        if force1.getGlobalParameterName(i) != force2.getGlobalParameterName(i) or force1.getGlobalParameterDefaultValue(i) != force2.getGlobalParameterDefaultValue(i):
            return False
    for i in range(getattr(force1, count_method)()):
        if getattr(force1, name_method)(i) != getattr(force2, name_method)(i):
            return False
    return True


def _custom_parameter_methods(kind: str) -> tuple[str, str, str]:
    if kind == "custom_bond":
        return "getNumPerBondParameters", "getPerBondParameterName", "addPerBondParameter"
    if kind == "custom_angle":
        return "getNumPerAngleParameters", "getPerAngleParameterName", "addPerAngleParameter"
    if kind == "custom_torsion":
        return "getNumPerTorsionParameters", "getPerTorsionParameterName", "addPerTorsionParameter"
    raise ValueError(f"Unsupported custom force kind: {kind}")


def _make_empty_term_force(source: Any, kind: str):
    if kind == "harmonic_bond":
        force = openmm.HarmonicBondForce()
    elif kind == "harmonic_angle":
        force = openmm.HarmonicAngleForce()
    elif kind == "periodic_torsion":
        force = openmm.PeriodicTorsionForce()
    elif kind == "rb_torsion":
        force = openmm.RBTorsionForce()
    elif kind == "custom_bond":
        force = openmm.CustomBondForce(source.getEnergyFunction())
        _copy_custom_parameters(source, force, kind)
    elif kind == "custom_angle":
        force = openmm.CustomAngleForce(source.getEnergyFunction())
        _copy_custom_parameters(source, force, kind)
    elif kind == "custom_torsion":
        force = openmm.CustomTorsionForce(source.getEnergyFunction())
        _copy_custom_parameters(source, force, kind)
    else:
        raise ValueError(f"Unsupported term force kind: {kind}")
    _copy_force_metadata(source, force)
    return force


def _copy_custom_parameters(source: Any, target: Any, kind: str) -> None:
    count_method, name_method, add_method = _custom_parameter_methods(kind)
    for i in range(source.getNumGlobalParameters()):
        target.addGlobalParameter(source.getGlobalParameterName(i), source.getGlobalParameterDefaultValue(i))
    for i in range(getattr(source, count_method)()):
        getattr(target, add_method)(getattr(source, name_method)(i))


def _term_signature(force: Any, kind: str, index: int) -> str:
    holder = _make_empty_term_force(force, kind)
    _add_term(holder, kind, force, index)
    return _force_xml(holder)


def _add_term(target: Any, kind: str, source: Any, index: int) -> None:
    if kind == "harmonic_bond":
        target.addBond(*source.getBondParameters(index))
    elif kind == "harmonic_angle":
        target.addAngle(*source.getAngleParameters(index))
    elif kind in {"periodic_torsion", "rb_torsion"}:
        target.addTorsion(*source.getTorsionParameters(index))
    elif kind == "custom_bond":
        p1, p2, params = source.getBondParameters(index)
        target.addBond(p1, p2, params)
    elif kind == "custom_angle":
        p1, p2, p3, params = source.getAngleParameters(index)
        target.addAngle(p1, p2, p3, params)
    elif kind == "custom_torsion":
        p1, p2, p3, p4, params = source.getTorsionParameters(index)
        target.addTorsion(p1, p2, p3, p4, params)
    else:
        raise ValueError(f"Unsupported term force kind: {kind}")

def _build_state_energy_force(source_system: Any, prefix: str):
    energy_forces = []
    for force_index in range(source_system.getNumForces()):
        source_force = source_system.getForce(force_index)
        if _should_skip_subforce(source_force):
            continue
        cloned_force = _clone_openmm_object(source_force)
        _rename_force_global_parameters(cloned_force, f"{prefix}_{force_index}")
        energy_forces.append(cloned_force)
    if not energy_forces:
        return openmm.CustomCVForce("0")
    if len(energy_forces) == 1:
        return energy_forces[0]
    variable_names = [f"{prefix}_{index}" for index in range(len(energy_forces))]
    aggregate = openmm.CustomCVForce(" + ".join(variable_names))
    for variable_name, force in zip(variable_names, energy_forces):
        aggregate.addCollectiveVariable(variable_name, force)
    return aggregate


def _should_skip_subforce(force: Any) -> bool:
    return isinstance(force, openmm.CMMotionRemover)


def _clone_openmm_object(obj: Any):
    return openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(obj))


def _rename_force_global_parameters(force: Any, prefix: str) -> None:
    if not hasattr(force, "getNumGlobalParameters"):
        return
    if not hasattr(force, "getGlobalParameterName") or not hasattr(force, "setGlobalParameterName"):
        return
    for parameter_index in range(force.getNumGlobalParameters()):
        parameter_name = force.getGlobalParameterName(parameter_index)
        force.setGlobalParameterName(parameter_index, f"{prefix}_{parameter_name}")


def _has_cmmotion_remover(system: Any) -> bool:
    for index in range(system.getNumForces()):
        if isinstance(system.getForce(index), openmm.CMMotionRemover):
            return True
    return False


def _validate_constraints(system1: Any, system2: Any) -> None:
    if system1.getNumConstraints() != system2.getNumConstraints():
        raise ValueError("State 1 and state 2 constraints differ.")
    for index in range(system1.getNumConstraints()):
        p1_a, p1_b, dist1 = system1.getConstraintParameters(index)
        p2_a, p2_b, dist2 = system2.getConstraintParameters(index)
        if (p1_a, p1_b) != (p2_a, p2_b):
            raise ValueError("State 1 and state 2 constraint atom ordering differs.")
        if abs(dist1.value_in_unit(unit.nanometer) - dist2.value_in_unit(unit.nanometer)) > 1.0e-8:
            raise ValueError("State 1 and state 2 constraint distances differ.")


def _copy_constraints(source: Any, target: Any, skip_atoms: set[int] | None = None) -> None:
    skip_atoms = skip_atoms or set()
    for index in range(source.getNumConstraints()):
        particle1, particle2, distance = source.getConstraintParameters(index)
        if particle1 in skip_atoms or particle2 in skip_atoms:
            continue
        target.addConstraint(particle1, particle2, distance)


def _validate_virtual_sites(system1: Any, system2: Any) -> None:
    for index in range(system1.getNumParticles()):
        vs1 = system1.getVirtualSite(index) if system1.isVirtualSite(index) else None
        vs2 = system2.getVirtualSite(index) if system2.isVirtualSite(index) else None
        if (vs1 is None) != (vs2 is None):
            raise ValueError("State 1 and state 2 virtual-site layout differs.")
        if vs1 is None:
            continue
        xml1 = openmm.XmlSerializer.serialize(vs1)
        xml2 = openmm.XmlSerializer.serialize(vs2)
        if xml1 != xml2:
            raise ValueError("State 1 and state 2 virtual-site definitions differ.")


def _copy_virtual_sites(source: Any, target: Any) -> None:
    for index in range(source.getNumParticles()):
        virtual_site = source.getVirtualSite(index) if source.isVirtualSite(index) else None
        if virtual_site is not None:
            target.setVirtualSite(index, _clone_openmm_object(virtual_site))


def _require_openmm() -> None:
    if OPENMM_IMPORT_ERROR is not None:
        raise ImportError("OpenMM is required for this operation.") from OPENMM_IMPORT_ERROR


def _to_openmm_positions(positions_nm: np.ndarray):
    return positions_nm * unit.nanometer


def _to_box_vectors(box_vectors_nm: np.ndarray):
    return tuple(box_vectors_nm[i] * unit.nanometer for i in range(3))


def _load_positions_and_box(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    suffix = Path(path).suffix.lower()
    if suffix in {".inpcrd", ".rst7"}:
        inpcrd = AmberInpcrdFile(path)
        positions_nm = np.asarray(inpcrd.positions.value_in_unit(unit.nanometer))
        box_vectors_nm = None
        if inpcrd.boxVectors is not None:
            box_vectors_nm = np.asarray(inpcrd.boxVectors.value_in_unit(unit.nanometer))
        return positions_nm, box_vectors_nm
    if suffix == ".pdb":
        return _load_pdb_positions_and_box(path)
    raise ValueError(f"Unsupported coordinate format: {path}")


def _load_pdb_positions_box_and_topology(path: str) -> tuple[np.ndarray, np.ndarray | None, Any]:
    pdb = PDBFile(path)
    positions_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
    box_vectors_nm = None
    box_vectors = pdb.topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        box_vectors_nm = np.asarray(box_vectors.value_in_unit(unit.nanometer))
    return positions_nm, box_vectors_nm, pdb.topology


def _load_pdb_positions_and_box(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    coordinates_angstrom: list[list[float]] = []
    box_vectors_nm = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "CRYST1":
                a = float(line[6:15]) * 0.1
                b = float(line[15:24]) * 0.1
                c = float(line[24:33]) * 0.1
                alpha = float(line[33:40])
                beta = float(line[40:47])
                gamma = float(line[47:54])
                if any(abs(angle - 90.0) > 1.0e-3 for angle in (alpha, beta, gamma)):
                    raise ValueError(f"Only orthorhombic CRYST1 boxes are currently supported in PDB input: {path}")
                box_vectors_nm = np.asarray(
                    [
                        [a, 0.0, 0.0],
                        [0.0, b, 0.0],
                        [0.0, 0.0, c],
                    ],
                    dtype=float,
                )
            elif record in {"ATOM", "HETATM"}:
                coordinates_angstrom.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
    if not coordinates_angstrom:
        raise ValueError(f"No ATOM/HETATM records were found in PDB file: {path}")
    return np.asarray(coordinates_angstrom, dtype=float) * 0.1, box_vectors_nm
