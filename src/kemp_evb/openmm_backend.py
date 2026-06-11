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
    ) -> EVBOpenMMSystem:
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
            )
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
            energy_decomposition_report=_legacy_decomposition_report(),
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
    ) -> EVBOpenMMSystem:
        self.validate_compatibility(state1, state2)
        system = openmm.System()
        for index in range(state1.system.getNumParticles()):
            system.addParticle(state1.system.getParticleMass(index))
        _copy_constraints(state1.system, system, skip_atoms=unconstrained_atoms)
        _copy_virtual_sites(state1.system, system)
        if state1.box_vectors_nm is not None:
            system.setDefaultPeriodicBoxVectors(*_to_box_vectors(state1.box_vectors_nm))

        decomposition = _decompose_system_forces(state1.system, state2.system)
        common_force = _aggregate_energy_forces(decomposition["common_forces"], "common")
        state1_force = _aggregate_energy_forces(decomposition["state1_forces"], "s1")
        state2_force = _aggregate_energy_forces(decomposition["state2_forces"], "s2")
        evb_force = openmm.CustomCVForce(
            "e_common + 0.5*(e1 + e2 + delta_alpha) - sqrt(0.25*(e1 - e2 - delta_alpha)^2 + h12^2)"
        )
        evb_force.addCollectiveVariable("e_common", common_force)
        evb_force.addCollectiveVariable("e1", state1_force)
        evb_force.addCollectiveVariable("e2", state2_force)
        evb_force.addGlobalParameter("delta_alpha", delta_alpha)
        evb_force.addGlobalParameter("h12", h12)
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
            energy_decomposition_report=decomposition["report"],
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
    ) -> EVBOpenMMSystem:
        self.validate_compatibility(state1, state2)
        system = openmm.System()
        for index in range(state1.system.getNumParticles()):
            system.addParticle(state1.system.getParticleMass(index))
        _copy_constraints(state1.system, system, skip_atoms=unconstrained_atoms)
        _copy_virtual_sites(state1.system, system)
        if state1.box_vectors_nm is not None:
            system.setDefaultPeriodicBoxVectors(*_to_box_vectors(state1.box_vectors_nm))

        decomposition_report = _legacy_decomposition_report()
        if energy_decomposition:
            decomposition = _decompose_system_forces(state1.system, state2.system)
            common_force = _aggregate_energy_forces(decomposition["common_forces"], "common")
            state1_force = _aggregate_energy_forces(decomposition["state1_forces"], "s1")
            state2_force = _aggregate_energy_forces(decomposition["state2_forces"], "s2")
            evb_force = openmm.CustomCVForce(
                "e_common + 0.5*(e1 + e2 + delta_alpha) - sqrt(0.25*(e1 - e2 - delta_alpha)^2 + h12^2)"
                " + 0.5*k_gap*((e1 - e2 - delta_alpha)-gap_center)^2"
            )
            evb_force.addCollectiveVariable("e_common", common_force)
            decomposition_report = decomposition["report"]
        else:
            state1_force = _build_state_energy_force(state1.system, "s1")
            state2_force = _build_state_energy_force(state2.system, "s2")
            evb_force = openmm.CustomCVForce(
                "0.5*(e1 + e2 + delta_alpha) - sqrt(0.25*(e1 - e2 - delta_alpha)^2 + h12^2)"
                " + 0.5*k_gap*((e1 - e2 - delta_alpha)-gap_center)^2"
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
            energy_decomposition_report=decomposition_report,
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
):
    """Build a CustomCVForce for gap = E1 - E2 - delta_alpha in kJ/mol."""
    _require_openmm()
    if energy_decomposition:
        decomposition = _decompose_system_forces(state1_system, state2_system)
        state1_force = _aggregate_energy_forces(decomposition["state1_forces"], f"{prefix}_s1")
        state2_force = _aggregate_energy_forces(decomposition["state2_forces"], f"{prefix}_s2")
    else:
        state1_force = _build_state_energy_force(state1_system, f"{prefix}_s1")
        state2_force = _build_state_energy_force(state2_system, f"{prefix}_s2")
    gap_force = openmm.CustomCVForce("e1 - e2 - delta_alpha")
    gap_force.addCollectiveVariable("e1", state1_force)
    gap_force.addCollectiveVariable("e2", state2_force)
    gap_force.addGlobalParameter("delta_alpha", delta_alpha)
    return gap_force


def evb_diabatic_energies(evb_system: EVBOpenMMSystem, context: Any) -> tuple[float, float]:
    values = [float(value) for value in evb_system.evb_force.getCollectiveVariableValues(context)]
    report = evb_system.energy_decomposition_report or {}
    if report.get("enabled") and len(values) >= 3:
        e_common, e1, e2 = values[:3]
        return e_common + e1, e_common + e2
    return values[0], values[1]


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
    }


def _decompose_system_forces(system1: Any, system2: Any) -> dict[str, Any]:
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
                warnings.append("NonbondedForce differs between states and was kept in full state-specific EVB evaluation; exact local decomposition for PME is not implemented.")
            else:
                warnings.append(f"{name} differs between states and was kept in full state-specific EVB evaluation.")
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
        "warnings": sorted(set(warnings)),
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
    for method in ("getNumBonds", "getNumAngles", "getNumTorsions"):
        if hasattr(force, method):
            return int(getattr(force, method)())
    return 1


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
    if kind.startswith("custom") and not _custom_force_compatible(force1, force2):
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


def _custom_force_compatible(force1: Any, force2: Any) -> bool:
    if force1.getEnergyFunction() != force2.getEnergyFunction() or force1.getNumGlobalParameters() != force2.getNumGlobalParameters() or force1.getNumPerBondParameters() != force2.getNumPerBondParameters():
        return False
    for i in range(force1.getNumGlobalParameters()):
        if force1.getGlobalParameterName(i) != force2.getGlobalParameterName(i) or force1.getGlobalParameterDefaultValue(i) != force2.getGlobalParameterDefaultValue(i):
            return False
    for i in range(force1.getNumPerBondParameters()):
        if force1.getPerBondParameterName(i) != force2.getPerBondParameterName(i):
            return False
    return True


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
        _copy_custom_parameters(source, force)
    elif kind == "custom_angle":
        force = openmm.CustomAngleForce(source.getEnergyFunction())
        _copy_custom_parameters(source, force)
    elif kind == "custom_torsion":
        force = openmm.CustomTorsionForce(source.getEnergyFunction())
        _copy_custom_parameters(source, force)
    else:
        raise ValueError(f"Unsupported term force kind: {kind}")
    _copy_force_metadata(source, force)
    return force


def _copy_custom_parameters(source: Any, target: Any) -> None:
    for i in range(source.getNumGlobalParameters()):
        target.addGlobalParameter(source.getGlobalParameterName(i), source.getGlobalParameterDefaultValue(i))
    for i in range(source.getNumPerBondParameters()):
        target.addPerBondParameter(source.getPerBondParameterName(i))


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
