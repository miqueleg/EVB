from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import openmm
    from openmm import unit
except ImportError:  # pragma: no cover
    openmm = None
    unit = None

from .evb import EVBHamiltonian, EVBParameters
from .native_bias import NativeGapBiasTable1D
from .openmm_backend import (
    EVBOpenMMSystem,
    LoadedAmberState,
    _clone_openmm_object,
    _copy_constraints,
    _copy_virtual_sites,
    _force_xml,
    _has_cmmotion_remover,
    _rename_force_global_parameters,
    _require_openmm,
    _to_box_vectors,
    _to_openmm_positions,
)

_COULOMB = 138.935456


@dataclass(slots=True)
class QRegionBondedPolicy:
    derive_from_state_differences: bool = True
    include_bonds: bool = True
    include_angles: bool = True
    include_torsions: bool = True
    include_impropers: bool = True


@dataclass(slots=True)
class QRegionNonbondedPolicy:
    mode: str = "exact_direct_or_local_pme_approx"
    exact_for_non_pme: bool = True
    pme_policy: str = "fail_exact_or_use_local_approx"
    local_approx_enabled: bool = False
    correction_atoms: list[int] | str | None = "auto"
    correction_cutoff_nm: float = 1.2
    include_q_q: bool = True
    include_q_environment: bool = True
    include_q_water: bool = True
    include_exceptions: bool = True


@dataclass(slots=True)
class QRegionValidationSettings:
    compare_to_legacy: bool = True
    frames: list[str] = field(default_factory=list)
    max_energy_error_kj_mol_exact: float = 1.0e-6
    max_gap_error_kj_mol_exact: float = 1.0e-6
    max_force_rmsd_exact: float = 1.0e-6
    max_energy_error_kj_mol_local_approx: float = 2.0
    max_gap_error_kj_mol_local_approx: float = 2.0
    max_force_rmsd_local_approx: float = 50.0


@dataclass(slots=True)
class QRegionSpec:
    q_atoms: list[int]
    environment_atoms: list[int] = field(default_factory=list)
    correction_atoms: list[int] = field(default_factory=list)
    baseline_state: str = "state1"
    changed_atom_policy: str = "require_subset"
    common_force_placement: str = "outer_system"
    bonded: QRegionBondedPolicy = field(default_factory=QRegionBondedPolicy)
    nonbonded: QRegionNonbondedPolicy = field(default_factory=QRegionNonbondedPolicy)
    validation: QRegionValidationSettings = field(default_factory=QRegionValidationSettings)


@dataclass(slots=True)
class QRegionDerivationReport:
    q_atoms: list[int]
    proposed_q_atoms: list[int]
    changed_nonbonded_atoms: list[int]
    changed_bonded_terms: list[dict[str, Any]]
    changed_exceptions: list[tuple[int, int]]
    changed_atoms_not_in_q_region: list[int]
    common_force_summary: dict[str, Any]
    q_state_force_summary: dict[str, Any]
    exactness_status: str
    pme_status: str
    warnings: list[str]
    recommendations: list[str]


@dataclass(slots=True)
class QRegionEVBSystem:
    system: Any
    topology: Any
    positions_nm: np.ndarray
    box_vectors_nm: np.ndarray | None
    masses_amu: np.ndarray
    common_forces: list[Any]
    q_state1_force: Any
    q_state2_force: Any
    evb_force: Any
    native_gap_bias: NativeGapBiasTable1D | None
    table_bias_function_index: int | None
    q_region_report: dict[str, Any]
    validation_report: dict[str, Any] | None = None


def q_region_to_evb_openmm_system(q_system: QRegionEVBSystem) -> EVBOpenMMSystem:
    return EVBOpenMMSystem(
        system=q_system.system,
        topology=q_system.topology,
        positions_nm=q_system.positions_nm,
        box_vectors_nm=q_system.box_vectors_nm,
        masses_amu=q_system.masses_amu,
        evb_force=q_system.evb_force,
        state1_force=q_system.q_state1_force,
        state2_force=q_system.q_state2_force,
        native_gap_bias=q_system.native_gap_bias,
        table_bias_function_index=q_system.table_bias_function_index,
        common_forces=q_system.common_forces,
        common_force_group=30,
        force_groups={"common": 30, "evb": q_system.evb_force.getForceGroup()},
        bias_report=None if q_system.native_gap_bias is None else {
            "enabled": True,
            "uses_app_metadynamics": False,
            "uses_bias_variable": False,
            "function_index": q_system.table_bias_function_index,
        },
        energy_decomposition_report={
            "enabled": True,
            "mode": "q_region",
            "common_force_placement": "outer_system",
            "e_common_inside_custom_cv": False,
            "duplicated_full_nonbonded": bool(q_system.q_region_report.get("duplicated_full_nonbonded", False)),
            "q_region_report": q_system.q_region_report,
            "native_gap_bias_uses_app_metadynamics": False if q_system.native_gap_bias is not None else None,
            "native_gap_bias_uses_bias_variable": False if q_system.native_gap_bias is not None else None,
        },
    )


class QRegionSystemBuilder:
    def __init__(self, spec: QRegionSpec):
        _require_openmm()
        self.spec = spec
        self.q_atoms = sorted(set(int(i) for i in spec.q_atoms))
        if not self.q_atoms:
            raise ValueError("Q-region mode requires at least one q_atom.")

    def build(
        self,
        state1: LoadedAmberState,
        state2: LoadedAmberState,
        delta_alpha: float,
        h12: float,
        native_gap_bias_table: NativeGapBiasTable1D | None = None,
        add_cmmotion_remover: bool = True,
    ) -> QRegionEVBSystem:
        baseline = state1 if self.spec.baseline_state == "state1" else state2
        other = state2 if baseline is state1 else state1
        if baseline.system.getNumParticles() != other.system.getNumParticles():
            raise ValueError("Q-region states must have the same atom count.")
        system = openmm.System()
        for index in range(baseline.system.getNumParticles()):
            system.addParticle(baseline.system.getParticleMass(index))
        _copy_constraints(baseline.system, system)
        _copy_virtual_sites(baseline.system, system)
        if baseline.box_vectors_nm is not None:
            system.setDefaultPeriodicBoxVectors(*_to_box_vectors(baseline.box_vectors_nm))

        common_forces: list[Any] = []
        state1_residuals: list[Any] = []
        state2_residuals: list[Any] = []
        warnings: list[str] = []
        changed_bonded_terms: list[dict[str, Any]] = []
        changed_nb_atoms: set[int] = set()
        changed_exceptions: list[tuple[int, int]] = []
        pme_status = "none"
        exactness_status = "exact"
        common_nb_count = 0
        q_nb_count = 0

        for force_index in range(baseline.system.getNumForces()):
            force1 = state1.system.getForce(force_index)
            force2 = state2.system.getForce(force_index)
            if isinstance(force1, openmm.CMMotionRemover):
                continue
            if type(force1) is not type(force2):
                raise ValueError(f"Q-region cannot compare differing force classes at index {force_index}.")
            if isinstance(force1, openmm.NonbondedForce):
                result = self._handle_nonbonded(force1, force2, force_index)
                common_forces.extend(result["common"])
                state1_residuals.extend(result["state1"])
                state2_residuals.extend(result["state2"])
                changed_nb_atoms.update(result["changed_atoms"])
                changed_exceptions.extend(result["changed_exceptions"])
                warnings.extend(result["warnings"])
                pme_status = result["pme_status"]
                exactness_status = result["exactness_status"]
                common_nb_count += result["common_nonbonded_count"]
                q_nb_count += result["q_nonbonded_count"]
                continue
            if _is_supported_bonded_force(force1):
                common, s1, s2, changed = self._split_bonded_force(force1, force2, force_index)
                if common is not None:
                    common_forces.append(common)
                if s1 is not None:
                    state1_residuals.append(s1)
                if s2 is not None:
                    state2_residuals.append(s2)
                changed_bonded_terms.extend(changed)
                continue
            if _force_xml(force1) == _force_xml(force2):
                common_forces.append(_clone_openmm_object(force1))
            else:
                raise ValueError(f"Q-region changed force {type(force1).__name__} is not supported in exact mode.")

        changed_atoms = set(changed_nb_atoms)
        for term in changed_bonded_terms:
            changed_atoms.update(term.get("atoms", []))
        changed_atoms_not_in_q = sorted(changed_atoms.difference(self.q_atoms).difference(self.spec.correction_atoms))
        if changed_atoms_not_in_q and self.spec.changed_atom_policy == "require_subset":
            raise ValueError(
                "Q-region changed atoms are outside q_atoms/correction_atoms: "
                f"{changed_atoms_not_in_q}. Add them to q_atoms or correction_atoms."
            )

        for force in common_forces:
            force.setForceGroup(30)
            system.addForce(force)
        q_state1_force = _aggregate_residual_forces(state1_residuals, "q_s1")
        q_state2_force = _aggregate_residual_forces(state2_residuals, "q_s2")
        expression = "0.5*(e1_Q + e2_Q + delta_alpha) - sqrt(0.25*(e1_Q - e2_Q - delta_alpha)^2 + h12^2)"
        if native_gap_bias_table is not None:
            expression += " + gap_bias(e1_Q - e2_Q - delta_alpha)"
        evb_force = openmm.CustomCVForce(expression)
        evb_force.addCollectiveVariable("e1_Q", q_state1_force)
        evb_force.addCollectiveVariable("e2_Q", q_state2_force)
        evb_force.addGlobalParameter("delta_alpha", float(delta_alpha))
        evb_force.addGlobalParameter("h12", float(h12))
        table_index = None
        if native_gap_bias_table is not None:
            table_index = native_gap_bias_table.add_to_force(evb_force)
        system.addForce(evb_force)
        if add_cmmotion_remover and _has_cmmotion_remover(baseline.system):
            system.addForce(openmm.CMMotionRemover())

        report = QRegionDerivationReport(
            q_atoms=self.q_atoms,
            proposed_q_atoms=self.q_atoms,
            changed_nonbonded_atoms=sorted(changed_nb_atoms),
            changed_bonded_terms=changed_bonded_terms,
            changed_exceptions=sorted(set(changed_exceptions)),
            changed_atoms_not_in_q_region=changed_atoms_not_in_q,
            common_force_summary={
                "n_forces": len(common_forces),
                "nonbonded_force_count": common_nb_count,
                "full_duplicated_nonbonded": False,
            },
            q_state_force_summary={
                "state1_force_count": len(state1_residuals),
                "state2_force_count": len(state2_residuals),
                "q_nonbonded_force_count": q_nb_count,
            },
            exactness_status=exactness_status,
            pme_status=pme_status,
            warnings=warnings,
            recommendations=[
                "Validate Q-region energies, gaps, and forces against legacy before production use.",
                "Use local_pme_approx only as an explicitly validated approximation.",
            ],
        )
        report_dict = asdict(report)
        report_dict["duplicated_full_nonbonded"] = False
        report_dict["pme_approximation"] = exactness_status == "approximate"
        report_dict["reciprocal_pme_difference_ignored_or_approximated"] = exactness_status == "approximate"
        return QRegionEVBSystem(
            system=system,
            topology=baseline.topology,
            positions_nm=baseline.positions_nm.copy(),
            box_vectors_nm=baseline.box_vectors_nm,
            masses_amu=baseline.masses_amu.copy(),
            common_forces=common_forces,
            q_state1_force=q_state1_force,
            q_state2_force=q_state2_force,
            evb_force=evb_force,
            native_gap_bias=native_gap_bias_table,
            table_bias_function_index=table_index,
            q_region_report=report_dict,
        )

    def _split_bonded_force(self, force1: Any, force2: Any, force_index: int):
        kind = _bonded_kind(force1)
        common = _empty_like(force1, kind)
        s1 = _empty_like(force1, kind)
        s2 = _empty_like(force2, kind)
        changed: list[dict[str, Any]] = []
        count = _term_count(force1, kind)
        if count != _term_count(force2, kind):
            raise ValueError(f"Q-region changed {type(force1).__name__} term count differs; this is not supported yet.")
        for i in range(count):
            term1 = _get_term(force1, kind, i)
            term2 = _get_term(force2, kind, i)
            if _term_equal(term1, term2):
                _add_term(common, kind, term1)
                continue
            atoms = list(term1[0])
            if not any(atom in self.q_atoms for atom in atoms):
                raise ValueError(f"Changed bonded term outside Q region at force {force_index}, term {i}: {atoms}")
            _add_term(s1, kind, term1)
            _add_term(s2, kind, term2)
            changed.append({"force_index": force_index, "term_index": i, "kind": kind, "atoms": atoms})
        return _nonempty(common, kind), _nonempty(s1, kind), _nonempty(s2, kind), changed

    def _handle_nonbonded(self, force1: Any, force2: Any, force_index: int) -> dict[str, Any]:
        del force_index
        if _force_xml(force1) == _force_xml(force2):
            return {
                "common": [_clone_openmm_object(force1)],
                "state1": [],
                "state2": [],
                "changed_atoms": set(),
                "changed_exceptions": [],
                "warnings": [],
                "pme_status": "identical_nonbonded_common",
                "exactness_status": "exact",
                "common_nonbonded_count": 1,
                "q_nonbonded_count": 0,
            }
        method = determine_nonbonded_method(force1)
        changed_atoms = set(find_changed_nonbonded_particles(force1, force2))
        changed_exceptions = find_changed_exceptions(force1, force2)
        if method in {"PME", "Ewald", "LJPME"}:
            if not self.spec.nonbonded.local_approx_enabled:
                raise ValueError(
                    "Q-region exact PME decomposition is not implemented; use full-state exact mode or explicitly enable local_pme_approx with validation."
                )
            baseline = _clone_openmm_object(force1 if self.spec.baseline_state == "state1" else force2)
            correction = _build_direct_nonbonded_correction(force1, force2, changed_atoms or set(self.q_atoms), self.spec.correction_atoms, approximate=True)
            warnings = [
                "local_pme_approx is enabled: reciprocal PME differences are ignored or approximated by local direct-space corrections."
            ]
            return {
                "common": [baseline],
                "state1": [] if self.spec.baseline_state == "state1" else [correction["state1"]],
                "state2": [correction["state2"]] if self.spec.baseline_state == "state1" else [],
                "changed_atoms": changed_atoms,
                "changed_exceptions": changed_exceptions,
                "warnings": warnings,
                "pme_status": "local_pme_approx",
                "exactness_status": "approximate",
                "common_nonbonded_count": 1,
                "q_nonbonded_count": 1,
            }
        if method not in {"NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic"}:
            raise ValueError(f"Q-region exact direct nonbonded does not support method {method}.")
        baseline = _clone_openmm_object(force1 if self.spec.baseline_state == "state1" else force2)
        correction = _build_direct_nonbonded_correction(force1, force2, changed_atoms or set(self.q_atoms), self.spec.correction_atoms, approximate=False)
        if self.spec.baseline_state == "state1":
            s1 = []
            s2 = [correction["state2"]]
        else:
            s1 = [correction["state1"]]
            s2 = []
        return {
            "common": [baseline],
            "state1": s1,
            "state2": s2,
            "changed_atoms": changed_atoms,
            "changed_exceptions": changed_exceptions,
            "warnings": [],
            "pme_status": "exact_direct_nonbonded",
            "exactness_status": "exact",
            "common_nonbonded_count": 1,
            "q_nonbonded_count": len(s1) + len(s2),
        }


def q_region_spec_from_config(config: Any) -> QRegionSpec:
    payload = dict(getattr(config, "q_region", {}) or {})
    q_atoms = list(payload.get("q_atoms") or [])
    if payload.get("q_atoms_from_reaction", False):
        q_atoms.extend(getattr(config.reaction, "substrate_atoms", []) or [])
        if config.reaction.atoms is not None:
            q_atoms.extend([config.reaction.atoms.donor, config.reaction.atoms.proton, config.reaction.atoms.acceptor])
    bonded_payload = dict(payload.get("bonded", {}) or {})
    nonbonded_payload = dict(payload.get("nonbonded", {}) or {})
    validation_payload = dict(payload.get("validation", {}) or {})
    return QRegionSpec(
        q_atoms=sorted(set(int(atom) for atom in q_atoms)),
        environment_atoms=list(payload.get("environment_atoms") or []),
        correction_atoms=list(payload.get("correction_atoms") or []),
        baseline_state=payload.get("baseline_state", "state1"),
        changed_atom_policy=payload.get("changed_atom_policy", "require_subset"),
        common_force_placement=payload.get("common_force_placement", "outer_system"),
        bonded=QRegionBondedPolicy(**bonded_payload),
        nonbonded=QRegionNonbondedPolicy(**nonbonded_payload),
        validation=QRegionValidationSettings(**validation_payload),
    )


def extract_nonbonded_parameters(force: Any) -> dict[str, Any]:
    particles = []
    for i in range(force.getNumParticles()):
        q, sigma, epsilon = force.getParticleParameters(i)
        particles.append((
            float(q.value_in_unit(unit.elementary_charge)),
            float(sigma.value_in_unit(unit.nanometer)),
            float(epsilon.value_in_unit(unit.kilojoule_per_mole)),
        ))
    exceptions = {}
    for i in range(force.getNumExceptions()):
        a, b, chargeprod, sigma, epsilon = force.getExceptionParameters(i)
        key = tuple(sorted((int(a), int(b))))
        exceptions[key] = (
            float(chargeprod.value_in_unit(unit.elementary_charge**2)),
            float(sigma.value_in_unit(unit.nanometer)),
            float(epsilon.value_in_unit(unit.kilojoule_per_mole)),
        )
    return {"particles": particles, "exceptions": exceptions, "summary": summarize_nonbonded_force(force)}


def compare_nonbonded_forces(force1: Any, force2: Any) -> dict[str, Any]:
    return {
        "identical_xml": _force_xml(force1) == _force_xml(force2),
        "changed_particles": find_changed_nonbonded_particles(force1, force2),
        "changed_exceptions": find_changed_exceptions(force1, force2),
        "method1": determine_nonbonded_method(force1),
        "method2": determine_nonbonded_method(force2),
    }


def find_changed_nonbonded_particles(force1: Any, force2: Any) -> list[int]:
    changed = []
    for i in range(force1.getNumParticles()):
        p1 = force1.getParticleParameters(i)
        p2 = force2.getParticleParameters(i)
        vals1 = [float(p1[0].value_in_unit(unit.elementary_charge)), float(p1[1].value_in_unit(unit.nanometer)), float(p1[2].value_in_unit(unit.kilojoule_per_mole))]
        vals2 = [float(p2[0].value_in_unit(unit.elementary_charge)), float(p2[1].value_in_unit(unit.nanometer)), float(p2[2].value_in_unit(unit.kilojoule_per_mole))]
        if not np.allclose(vals1, vals2, atol=1.0e-12, rtol=0.0):
            changed.append(i)
    return changed


def find_changed_exceptions(force1: Any, force2: Any) -> list[tuple[int, int]]:
    p1 = extract_nonbonded_parameters(force1)["exceptions"]
    p2 = extract_nonbonded_parameters(force2)["exceptions"]
    keys = set(p1) | set(p2)
    return sorted(key for key in keys if key not in p1 or key not in p2 or not np.allclose(p1[key], p2[key], atol=1.0e-12, rtol=0.0))


def determine_nonbonded_method(force: Any) -> str:
    names = {
        openmm.NonbondedForce.NoCutoff: "NoCutoff",
        openmm.NonbondedForce.CutoffNonPeriodic: "CutoffNonPeriodic",
        openmm.NonbondedForce.CutoffPeriodic: "CutoffPeriodic",
        openmm.NonbondedForce.Ewald: "Ewald",
        openmm.NonbondedForce.PME: "PME",
        openmm.NonbondedForce.LJPME: "LJPME",
    }
    return names.get(force.getNonbondedMethod(), str(force.getNonbondedMethod()))


def summarize_nonbonded_force(force: Any) -> dict[str, Any]:
    cutoff = None
    try:
        cutoff = float(force.getCutoffDistance().value_in_unit(unit.nanometer))
    except Exception:
        pass
    pme = None
    try:
        alpha, nx, ny, nz = force.getPMEParameters()
        pme = {"alpha": float(alpha.value_in_unit(1 / unit.nanometer)), "grid": [int(nx), int(ny), int(nz)]}
    except Exception:
        pass
    return {
        "particles": int(force.getNumParticles()),
        "exceptions": int(force.getNumExceptions()),
        "method": determine_nonbonded_method(force),
        "cutoff_nm": cutoff,
        "pme_parameters": pme,
        "use_dispersion_correction": bool(force.getUseDispersionCorrection()),
        "use_switching_function": bool(force.getUseSwitchingFunction()),
    }


def derive_q_region_spec(config: Any, state1: LoadedAmberState, state2: LoadedAmberState, explicit_q_atoms: list[int] | None = None, include_reaction_atoms: bool = False) -> tuple[QRegionSpec, QRegionDerivationReport]:
    q_atoms = set(explicit_q_atoms or [])
    if include_reaction_atoms:
        q_atoms.update(getattr(config.reaction, "substrate_atoms", []) or [])
        if config.reaction.atoms is not None:
            q_atoms.update([config.reaction.atoms.donor, config.reaction.atoms.proton, config.reaction.atoms.acceptor])
    changed_nb: set[int] = set()
    changed_terms: list[dict[str, Any]] = []
    changed_exceptions: list[tuple[int, int]] = []
    for i in range(state1.system.getNumForces()):
        f1 = state1.system.getForce(i)
        f2 = state2.system.getForce(i)
        if isinstance(f1, openmm.NonbondedForce) and isinstance(f2, openmm.NonbondedForce):
            changed_nb.update(find_changed_nonbonded_particles(f1, f2))
            changed_exceptions.extend(find_changed_exceptions(f1, f2))
        elif _is_supported_bonded_force(f1) and type(f1) is type(f2):
            kind = _bonded_kind(f1)
            n1 = _term_count(f1, kind)
            n2 = _term_count(f2, kind)
            for j in range(min(n1, n2)):
                t1 = _get_term(f1, kind, j)
                t2 = _get_term(f2, kind, j)
                if not _term_equal(t1, t2):
                    changed_terms.append({"force_index": i, "term_index": j, "kind": kind, "atoms": list(t1[0])})
                    q_atoms.update(t1[0])
            if n1 != n2:
                changed_terms.append({"force_index": i, "term_index": None, "kind": kind, "atoms": [], "term_count_state1": n1, "term_count_state2": n2})
    q_atoms.update(changed_nb)
    spec = QRegionSpec(q_atoms=sorted(q_atoms), correction_atoms=sorted(q_atoms))
    report = QRegionDerivationReport(
        q_atoms=spec.q_atoms,
        proposed_q_atoms=spec.q_atoms,
        changed_nonbonded_atoms=sorted(changed_nb),
        changed_bonded_terms=changed_terms,
        changed_exceptions=sorted(set(changed_exceptions)),
        changed_atoms_not_in_q_region=[],
        common_force_summary={},
        q_state_force_summary={},
        exactness_status="proposed",
        pme_status="requires_validation",
        warnings=["Derived Q atoms are a proposal and require user review."],
        recommendations=["Inspect q_region_derivation_report.json before production use."],
    )
    return spec, report


def validate_q_region_against_legacy(q_system: QRegionEVBSystem, legacy: EVBOpenMMSystem, positions_nm: np.ndarray, parameters: EVBParameters, platform_name: str = "CPU") -> dict[str, Any]:
    integrator1 = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    integrator2 = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name) if platform_name else None
    q_context = openmm.Context(q_system.system, integrator1, platform) if platform else openmm.Context(q_system.system, integrator1)
    legacy_context = openmm.Context(legacy.system, integrator2, platform) if platform else openmm.Context(legacy.system, integrator2)
    if q_system.box_vectors_nm is not None:
        q_context.setPeriodicBoxVectors(*_to_box_vectors(q_system.box_vectors_nm))
        legacy_context.setPeriodicBoxVectors(*_to_box_vectors(q_system.box_vectors_nm))
    q_context.setPositions(_to_openmm_positions(positions_nm))
    legacy_context.setPositions(_to_openmm_positions(positions_nm))
    q_state = q_context.getState(getEnergy=True, getForces=True)
    l_state = legacy_context.getState(getEnergy=True, getForces=True)
    q_energy = float(q_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    l_energy = float(l_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    q_forces = np.asarray(q_state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    l_forces = np.asarray(l_state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    q_vals = [float(v) for v in q_system.evb_force.getCollectiveVariableValues(q_context)]
    l_vals = [float(v) for v in legacy.evb_force.getCollectiveVariableValues(legacy_context)]
    q_common = float(q_context.getState(getEnergy=True, groups={30}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    q_e1 = q_common + q_vals[0]
    q_e2 = q_common + q_vals[1]
    l_e1, l_e2 = l_vals[0], l_vals[1]
    q_gap = q_e1 - q_e2 - parameters.delta_alpha
    l_gap = l_e1 - l_e2 - parameters.delta_alpha
    return {
        "legacy_energy_kj_mol": l_energy,
        "q_region_energy_kj_mol": q_energy,
        "energy_error_kj_mol": q_energy - l_energy,
        "legacy_gap_kj_mol": l_gap,
        "q_region_gap_kj_mol": q_gap,
        "gap_error_kj_mol": q_gap - l_gap,
        "force_rmsd_kj_mol_nm": float(np.sqrt(np.mean((q_forces - l_forces) ** 2))),
        "force_max_abs_kj_mol_nm": float(np.max(np.abs(q_forces - l_forces))),
        "exactness_status": q_system.q_region_report.get("exactness_status"),
    }


def write_q_region_config_fragment(spec: QRegionSpec, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evb": {
            "representation": "q_region",
            "q_region": asdict(spec),
        }
    }
    path.write_text(_simple_yaml(payload), encoding="utf-8")


def _build_direct_nonbonded_correction(force1: Any, force2: Any, changed_atoms: set[int], correction_atoms: list[int], approximate: bool) -> dict[str, Any]:
    del approximate
    all_atoms = set(range(force1.getNumParticles()))
    interaction_atoms = set(correction_atoms or []) or all_atoms
    interaction_atoms.update(changed_atoms)

    def make_force(state_force: Any, baseline_force: Any, name: str):
        expr = f"{_COULOMB}*(q_state1*q_state2-q_base1*q_base2)/r + 4*(epsilon_state-sqrt(epsilon_base1*epsilon_base2))*((sigma_state/r)^12-(sigma_state/r)^6); sigma_state=0.5*(sigma_state1+sigma_state2); epsilon_state=sqrt(epsilon_state1*epsilon_state2)"
        cf = openmm.CustomNonbondedForce(expr)
        for param in ("q_state", "sigma_state", "epsilon_state", "q_base", "sigma_base", "epsilon_base"):
            cf.addPerParticleParameter(param)
        cf.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
        for i in range(state_force.getNumParticles()):
            qs, ss, es = state_force.getParticleParameters(i)
            qb, sb, eb = baseline_force.getParticleParameters(i)
            cf.addParticle([
                qs.value_in_unit(unit.elementary_charge),
                ss.value_in_unit(unit.nanometer),
                es.value_in_unit(unit.kilojoule_per_mole),
                qb.value_in_unit(unit.elementary_charge),
                sb.value_in_unit(unit.nanometer),
                eb.value_in_unit(unit.kilojoule_per_mole),
            ])
        for i in range(state_force.getNumExceptions()):
            a, b, *_ = state_force.getExceptionParameters(i)
            cf.addExclusion(int(a), int(b))
        cf.addInteractionGroup(sorted(changed_atoms), sorted(interaction_atoms))
        _rename_force_global_parameters(cf, name)
        return cf

    if not changed_atoms:
        return {"state1": openmm.CustomCVForce("0"), "state2": openmm.CustomCVForce("0")}
    if True:
        baseline = force1
        return {
            "state1": openmm.CustomCVForce("0"),
            "state2": make_force(force2, baseline, "q_nb_s2"),
        }


def _aggregate_residual_forces(forces: list[Any], prefix: str) -> Any:
    if not forces:
        return openmm.CustomCVForce("0")
    prepared = []
    for i, force in enumerate(forces):
        cloned = _clone_openmm_object(force)
        _rename_force_global_parameters(cloned, f"{prefix}_{i}")
        prepared.append(cloned)
    if len(prepared) == 1:
        return prepared[0]
    cv = openmm.CustomCVForce(" + ".join(f"f{i}" for i in range(len(prepared))))
    for i, force in enumerate(prepared):
        cv.addCollectiveVariable(f"f{i}", force)
    return cv


def _is_supported_bonded_force(force: Any) -> bool:
    return isinstance(force, (openmm.HarmonicBondForce, openmm.HarmonicAngleForce, openmm.PeriodicTorsionForce, openmm.RBTorsionForce, openmm.CustomBondForce, openmm.CustomAngleForce, openmm.CustomTorsionForce))


def _bonded_kind(force: Any) -> str:
    if isinstance(force, openmm.HarmonicBondForce):
        return "harmonic_bond"
    if isinstance(force, openmm.HarmonicAngleForce):
        return "harmonic_angle"
    if isinstance(force, openmm.PeriodicTorsionForce):
        return "periodic_torsion"
    if isinstance(force, openmm.RBTorsionForce):
        return "rb_torsion"
    if isinstance(force, openmm.CustomBondForce):
        return "custom_bond"
    if isinstance(force, openmm.CustomAngleForce):
        return "custom_angle"
    if isinstance(force, openmm.CustomTorsionForce):
        return "custom_torsion"
    raise ValueError(type(force).__name__)


def _empty_like(source: Any, kind: str) -> Any:
    if kind == "harmonic_bond":
        return openmm.HarmonicBondForce()
    if kind == "harmonic_angle":
        return openmm.HarmonicAngleForce()
    if kind == "periodic_torsion":
        return openmm.PeriodicTorsionForce()
    if kind == "rb_torsion":
        return openmm.RBTorsionForce()
    if kind == "custom_bond":
        target = openmm.CustomBondForce(source.getEnergyFunction())
        _copy_custom_params(source, target, "Bond")
        return target
    if kind == "custom_angle":
        target = openmm.CustomAngleForce(source.getEnergyFunction())
        _copy_custom_params(source, target, "Angle")
        return target
    if kind == "custom_torsion":
        target = openmm.CustomTorsionForce(source.getEnergyFunction())
        _copy_custom_params(source, target, "Torsion")
        return target
    raise ValueError(kind)


def _copy_custom_params(source: Any, target: Any, label: str) -> None:
    for i in range(source.getNumGlobalParameters()):
        target.addGlobalParameter(source.getGlobalParameterName(i), source.getGlobalParameterDefaultValue(i))
    count = getattr(source, f"getNumPer{label}Parameters")()
    for i in range(count):
        getattr(target, f"addPer{label}Parameter")(getattr(source, f"getPer{label}ParameterName")(i))


def _term_count(force: Any, kind: str) -> int:
    if kind == "harmonic_bond" or kind == "custom_bond":
        return force.getNumBonds()
    if kind == "harmonic_angle" or kind == "custom_angle":
        return force.getNumAngles()
    return force.getNumTorsions()


def _get_term(force: Any, kind: str, i: int):
    if kind == "harmonic_bond":
        a, b, length, k = force.getBondParameters(i)
        return ((int(a), int(b)), (length, k))
    if kind == "harmonic_angle":
        a, b, c, angle, k = force.getAngleParameters(i)
        return ((int(a), int(b), int(c)), (angle, k))
    if kind in {"periodic_torsion", "rb_torsion"}:
        a, b, c, d, *params = force.getTorsionParameters(i)
        return ((int(a), int(b), int(c), int(d)), tuple(params))
    if kind == "custom_bond":
        a, b, params = force.getBondParameters(i)
        return ((int(a), int(b)), tuple(params))
    if kind == "custom_angle":
        a, b, c, params = force.getAngleParameters(i)
        return ((int(a), int(b), int(c)), tuple(params))
    if kind == "custom_torsion":
        a, b, c, d, params = force.getTorsionParameters(i)
        return ((int(a), int(b), int(c), int(d)), tuple(params))
    raise ValueError(kind)


def _add_term(force: Any, kind: str, term) -> None:
    atoms, params = term
    if kind == "harmonic_bond":
        force.addBond(atoms[0], atoms[1], *params)
    elif kind == "harmonic_angle":
        force.addAngle(atoms[0], atoms[1], atoms[2], *params)
    elif kind in {"periodic_torsion", "rb_torsion"}:
        force.addTorsion(atoms[0], atoms[1], atoms[2], atoms[3], *params)
    elif kind == "custom_bond":
        force.addBond(atoms[0], atoms[1], list(params))
    elif kind == "custom_angle":
        force.addAngle(atoms[0], atoms[1], atoms[2], list(params))
    elif kind == "custom_torsion":
        force.addTorsion(atoms[0], atoms[1], atoms[2], atoms[3], list(params))
    else:
        raise ValueError(kind)


def _term_equal(term1, term2) -> bool:
    if term1[0] != term2[0]:
        return False
    return _param_values(term1[1]) == _param_values(term2[1])


def _param_values(params) -> tuple[float, ...]:
    values = []
    for param in params:
        if hasattr(param, "value_in_unit_system"):
            values.append(float(param.value_in_unit_system(unit.md_unit_system)))
        else:
            values.append(float(param))
    return tuple(values)


def _nonempty(force: Any, kind: str) -> Any | None:
    return force if _term_count(force, kind) else None


def _simple_yaml(payload: dict[str, Any], indent: int = 0) -> str:
    lines = []
    for key, value in payload.items():
        prefix = " " * indent + f"{key}:"
        if isinstance(value, dict):
            lines.append(prefix)
            lines.append(_simple_yaml(value, indent + 2).rstrip())
        elif isinstance(value, list):
            lines.append(prefix + " [" + ", ".join(str(v) for v in value) + "]")
        else:
            lines.append(prefix + f" {json.dumps(value)}")
    return "\n".join(lines) + "\n"
