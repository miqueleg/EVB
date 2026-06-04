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
