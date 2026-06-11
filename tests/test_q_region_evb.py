from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmm")

import openmm as mm
from openmm import unit
from openmm.app import Topology, element

from kemp_evb.cli import run_gap_table_metadynamics
from kemp_evb.evb import EVBParameters
from kemp_evb.config import load_config
from kemp_evb.native_bias import NativeGapBiasTable1D
from kemp_evb.openmm_backend import AmberSystemLoader, EVBSystemBuilder, LoadedAmberState
from kemp_evb.q_region import QRegionNonbondedPolicy, QRegionSpec, QRegionSystemBuilder, derive_q_region_spec, q_region_to_evb_openmm_system
from kemp_evb.simulation import EVBSimulation, create_integrator

PARAMETERS = EVBParameters(delta_alpha=2.0, h12=5.0)


def _topology(n_atoms: int):
    topology = Topology()
    chain = topology.addChain("A")
    residue = topology.addResidue("MOL", chain)
    for i in range(n_atoms):
        topology.addAtom(f"A{i}", element.carbon, residue)
    return topology


def _state(system, positions, box=None):
    return LoadedAmberState(
        prmtop_path="toy",
        inpcrd_path="toy",
        topology=_topology(len(positions)),
        system=system,
        positions_nm=np.asarray(positions, dtype=float),
        box_vectors_nm=box,
        atom_labels=[("A", "MOL", f"A{i}", i) for i in range(len(positions))],
        atom_names=[f"A{i}" for i in range(len(positions))],
        masses_amu=np.full(len(positions), 12.0),
    )


def _bonded_state(q_r0: float):
    system = mm.System()
    for _ in range(4):
        system.addParticle(12.0 * unit.amu)
    q_bond = mm.HarmonicBondForce()
    q_bond.addBond(0, 1, q_r0 * unit.nanometer, 1000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    q_bond.addBond(2, 3, 0.16 * unit.nanometer, 500.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    system.addForce(q_bond)
    return _state(system, [[0, 0, 0], [0.18, 0, 0], [0.4, 0, 0], [0.56, 0, 0]])



def _reactive_bond_state(include_bond: bool, r0: float = 0.12, reversed_order: bool = False):
    system = mm.System()
    for _ in range(4):
        system.addParticle(12.0 * unit.amu)
    bond = mm.HarmonicBondForce()
    if include_bond:
        atoms = (1, 0) if reversed_order else (0, 1)
        bond.addBond(*atoms, r0 * unit.nanometer, 900.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    bond.addBond(2, 3, 0.16 * unit.nanometer, 500.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    system.addForce(bond)
    return _state(system, [[0, 0, 0], [0.18, 0, 0], [0.4, 0, 0], [0.56, 0, 0]])


def _reactive_angle_state(include_angle: bool, theta0: float = 1.8, reversed_order: bool = False):
    system = mm.System()
    for _ in range(4):
        system.addParticle(12.0 * unit.amu)
    angle = mm.HarmonicAngleForce()
    if include_angle:
        atoms = (2, 1, 0) if reversed_order else (0, 1, 2)
        angle.addAngle(*atoms, theta0 * unit.radian, 120.0 * unit.kilojoule_per_mole / unit.radian**2)
    angle.addAngle(1, 2, 3, 1.9 * unit.radian, 80.0 * unit.kilojoule_per_mole / unit.radian**2)
    system.addForce(angle)
    return _state(system, [[0, 0, 0], [0.18, 0, 0], [0.25, 0.15, 0], [0.45, 0.16, 0]])


def _reactive_torsion_state(terms, reversed_order: bool = False):
    system = mm.System()
    for _ in range(5):
        system.addParticle(12.0 * unit.amu)
    torsion = mm.PeriodicTorsionForce()
    atoms = (3, 2, 1, 0) if reversed_order else (0, 1, 2, 3)
    for periodicity, phase, k in terms:
        torsion.addTorsion(
            *atoms,
            int(periodicity),
            phase * unit.radian,
            k * unit.kilojoule_per_mole,
        )
    torsion.addTorsion(1, 2, 3, 4, 2, 0.4 * unit.radian, 0.6 * unit.kilojoule_per_mole)
    system.addForce(torsion)
    return _state(system, [[0, 0, 0], [0.16, 0, 0], [0.25, 0.14, 0], [0.35, 0.16, 0.12], [0.50, 0.16, 0.13]])

def _nonbonded_state(charges, method="NoCutoff"):
    system = mm.System()
    for _ in charges:
        system.addParticle(12.0 * unit.amu)
    nb = mm.NonbondedForce()
    if method == "PME":
        nb.setNonbondedMethod(mm.NonbondedForce.PME)
        nb.setCutoffDistance(0.9 * unit.nanometer)
    elif method == "CutoffNonPeriodic":
        nb.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
        nb.setCutoffDistance(1.0 * unit.nanometer)
    else:
        nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for q in charges:
        nb.addParticle(q * unit.elementary_charge, 0.3 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    system.addForce(nb)
    box = np.eye(3) * 3.0 if method == "PME" else None
    if box is not None:
        system.setDefaultPeriodicBoxVectors(*(mm.Vec3(*box[i]) * unit.nanometer for i in range(3)))
    return _state(system, [[0, 0, 0], [0.33, 0, 0], [0.0, 0.41, 0]], box=box)


def _single(evb_system, positions):
    sim = EVBSimulation(evb_system, create_integrator(1.0, integrator_name="Verlet"), platform_name="CPU")
    return sim.single_point(positions)


def _compare_q_to_legacy(state1, state2, spec, tol_energy=1e-6, tol_gap=1e-6, tol_force=1e-6):
    legacy = EVBSystemBuilder().build_openmm_evb_system(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12)
    q_system = QRegionSystemBuilder(spec).build(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12)
    q_evb = q_region_to_evb_openmm_system(q_system)
    positions = state1.positions_nm
    legacy_result = _single(legacy, positions)
    q_result = _single(q_evb, positions)
    legacy_gap = legacy_result.energy1 - legacy_result.energy2 - PARAMETERS.delta_alpha
    q_gap = q_result.energy1 - q_result.energy2 - PARAMETERS.delta_alpha
    force_rmsd = float(np.sqrt(np.mean((legacy_result.forces - q_result.forces) ** 2)))
    assert abs(q_result.evb_energy - legacy_result.evb_energy) <= tol_energy
    assert abs(q_gap - legacy_gap) <= tol_gap
    assert force_rmsd <= tol_force
    return q_system, q_result, legacy_result


def test_q_region_bonded_exact_toy_matches_legacy():
    state1 = _bonded_state(0.12)
    state2 = _bonded_state(0.14)
    q_system, q_result, legacy_result = _compare_q_to_legacy(state1, state2, QRegionSpec(q_atoms=[0, 1]))
    assert q_system.q_region_report["duplicated_full_nonbonded"] is False
    assert abs(q_result.energy1 - legacy_result.energy1) <= 1.0e-6
    assert abs(q_result.energy2 - legacy_result.energy2) <= 1.0e-6



def test_q_region_supports_state1_only_bonded_term():
    state1 = _reactive_bond_state(True)
    state2 = _reactive_bond_state(False)
    q_system, _q_result, _legacy_result = _compare_q_to_legacy(state1, state2, QRegionSpec(q_atoms=[0, 1]))
    mapping = q_system.q_region_report["common_force_summary"]["bonded_mapping"]
    assert mapping["n_state1_only_terms"] == 1
    assert mapping["n_state2_only_terms"] == 0


def test_q_region_supports_state2_only_bonded_term():
    state1 = _reactive_bond_state(False)
    state2 = _reactive_bond_state(True)
    q_system, _q_result, _legacy_result = _compare_q_to_legacy(state1, state2, QRegionSpec(q_atoms=[0, 1]))
    mapping = q_system.q_region_report["common_force_summary"]["bonded_mapping"]
    assert mapping["n_state1_only_terms"] == 0
    assert mapping["n_state2_only_terms"] == 1


def test_q_region_supports_angle_appearing_and_disappearing():
    state1 = _reactive_angle_state(True)
    state2 = _reactive_angle_state(False)
    q_system, _q_result, _legacy_result = _compare_q_to_legacy(state1, state2, QRegionSpec(q_atoms=[0, 1, 2]))
    assert q_system.q_region_report["common_force_summary"]["bonded_mapping"]["n_state1_only_terms"] == 1


def test_q_region_supports_torsion_appearing_and_disappearing():
    state1 = _reactive_torsion_state([(1, 0.2, 1.1)])
    state2 = _reactive_torsion_state([])
    q_system, _q_result, _legacy_result = _compare_q_to_legacy(state1, state2, QRegionSpec(q_atoms=[0, 1, 2, 3]))
    assert q_system.q_region_report["common_force_summary"]["bonded_mapping"]["n_state1_only_terms"] == 1


def test_q_region_changed_bond_parameters_are_state_specific():
    state1 = _reactive_bond_state(True, r0=0.12)
    state2 = _reactive_bond_state(True, r0=0.14)
    q_system, _q_result, _legacy_result = _compare_q_to_legacy(state1, state2, QRegionSpec(q_atoms=[0, 1]))
    mapping = q_system.q_region_report["common_force_summary"]["bonded_mapping"]
    assert mapping["n_changed_parameter_terms"] == 1
    assert mapping["n_common_terms"] == 1


def test_q_region_matches_multiple_torsions_by_multiset():
    state1 = _reactive_torsion_state([(1, 0.2, 1.1), (2, 0.5, 0.7)])
    state2 = _reactive_torsion_state([(2, 0.5, 0.7), (3, 0.8, 0.4)])
    q_system, _q_result, _legacy_result = _compare_q_to_legacy(state1, state2, QRegionSpec(q_atoms=[0, 1, 2, 3]))
    mapping = q_system.q_region_report["common_force_summary"]["bonded_mapping"]
    assert mapping["n_common_terms"] == 2
    assert mapping["n_changed_parameter_terms"] == 1


def test_q_region_recognizes_reversed_angle_and_torsion_order():
    angle1 = _reactive_angle_state(True, reversed_order=False)
    angle2 = _reactive_angle_state(True, reversed_order=True)
    _compare_q_to_legacy(angle1, angle2, QRegionSpec(q_atoms=[0, 1, 2]))
    torsion1 = _reactive_torsion_state([(1, 0.2, 1.1)], reversed_order=False)
    torsion2 = _reactive_torsion_state([(1, 0.2, 1.1)], reversed_order=True)
    _compare_q_to_legacy(torsion1, torsion2, QRegionSpec(q_atoms=[0, 1, 2, 3]))


def test_q_region_state_only_bonded_term_outside_q_region_fails():
    state1 = _reactive_bond_state(True)
    state2 = _reactive_bond_state(False)
    with pytest.raises(ValueError, match="state1-only bonded term outside Q region"):
        QRegionSystemBuilder(QRegionSpec(q_atoms=[2, 3])).build(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12)


def test_q_region_changed_parameter_bonded_term_outside_q_region_fails():
    state1 = _reactive_bond_state(True, r0=0.12)
    state2 = _reactive_bond_state(True, r0=0.14)
    with pytest.raises(ValueError, match="changed-parameter bonded term outside Q region"):
        QRegionSystemBuilder(QRegionSpec(q_atoms=[2, 3])).build(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12)

def test_q_region_exact_direct_nonbonded_toy_matches_legacy():
    state1 = _nonbonded_state([0.2, -0.1, 0.05])
    state2 = _nonbonded_state([0.3, -0.1, 0.05])
    q_system, _q_result, _legacy_result = _compare_q_to_legacy(state1, state2, QRegionSpec(q_atoms=[0], correction_atoms=[0, 1, 2]), tol_force=1e-5)
    assert q_system.q_region_report["pme_status"] == "exact_direct_nonbonded"
    assert q_system.q_region_report["duplicated_full_nonbonded"] is False


def test_q_region_changed_atoms_outside_q_region_fails():
    state1 = _nonbonded_state([0.2, -0.1, 0.05])
    state2 = _nonbonded_state([0.2, -0.2, 0.05])
    with pytest.raises(ValueError, match="outside q_atoms"):
        QRegionSystemBuilder(QRegionSpec(q_atoms=[0])).build(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12)


def test_q_region_identical_nonbonded_is_common_baseline():
    state1 = _nonbonded_state([0.2, -0.1, 0.05])
    state2 = _nonbonded_state([0.2, -0.1, 0.05])
    q_system, _q_result, _legacy_result = _compare_q_to_legacy(state1, state2, QRegionSpec(q_atoms=[0]))
    assert q_system.q_region_report["pme_status"] == "identical_nonbonded_common"
    assert q_system.q_region_report["q_state_force_summary"]["q_nonbonded_force_count"] == 0


def test_q_region_differing_pme_fails_without_local_approx():
    state1 = _nonbonded_state([0.2, -0.1, 0.05], method="PME")
    state2 = _nonbonded_state([0.3, -0.1, 0.05], method="PME")
    with pytest.raises(ValueError, match="Q-region exact PME decomposition is not implemented"):
        QRegionSystemBuilder(QRegionSpec(q_atoms=[0], correction_atoms=[0, 1, 2])).build(
            state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12
        )


def test_q_region_pme_local_approx_is_marked_approximate():
    state1 = _nonbonded_state([0.2, -0.1, 0.05], method="PME")
    state2 = _nonbonded_state([0.3, -0.1, 0.05], method="PME")
    spec = QRegionSpec(
        q_atoms=[0],
        correction_atoms=[0, 1, 2],
        nonbonded=QRegionNonbondedPolicy(local_approx_enabled=True),
    )
    q_system = QRegionSystemBuilder(spec).build(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12)
    assert q_system.q_region_report["exactness_status"] == "approximate"
    assert q_system.q_region_report["pme_approximation"] is True
    assert q_system.q_region_report["reciprocal_pme_difference_ignored_or_approximated"] is True


def test_q_region_native_table_bias_zero_and_constant():
    state1 = _bonded_state(0.12)
    state2 = _bonded_state(0.14)
    spec = QRegionSpec(q_atoms=[0, 1])
    zero = NativeGapBiasTable1D(-10.0, 10.0, 101)
    const = NativeGapBiasTable1D(-10.0, 10.0, 101, values_kj_mol=np.full(101, 3.0))
    unbiased = q_region_to_evb_openmm_system(QRegionSystemBuilder(spec).build(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12))
    zero_biased = q_region_to_evb_openmm_system(QRegionSystemBuilder(spec).build(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12, native_gap_bias_table=zero))
    const_biased = q_region_to_evb_openmm_system(QRegionSystemBuilder(spec).build(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12, native_gap_bias_table=const))
    a = _single(unbiased, state1.positions_nm)
    b = _single(zero_biased, state1.positions_nm)
    c = _single(const_biased, state1.positions_nm)
    assert abs(a.evb_energy - b.evb_energy) <= 1.0e-6
    assert float(np.sqrt(np.mean((a.forces - b.forces) ** 2))) <= 1.0e-6
    assert abs((c.evb_energy - a.evb_energy) - 3.0) <= 1.0e-6
    assert float(np.sqrt(np.mean((a.forces - c.forces) ** 2))) <= 1.0e-6


def test_q_region_table_metad_does_not_use_app_metadynamics_or_biasvariable():
    source = inspect.getsource(run_gap_table_metadynamics)
    assert "app.Metadynamics" not in source
    assert "BiasVariable" not in source


def test_hg317_derivation_smoke_reports_bonded_mapping_if_files_available():
    config_path = Path("examples/hg317_evb_gap_metad.yaml")
    if not config_path.exists():
        pytest.skip("HG3.17 example config is absent")
    config = load_config(config_path)
    required = [
        Path(config.state1.prmtop),
        Path(config.state1.inpcrd),
        Path(config.state2.prmtop),
        Path(config.state2.inpcrd),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip("HG3.17 prepared files are absent: " + ", ".join(missing))
    builder = EVBSystemBuilder(AmberSystemLoader(config.simulation.nonbonded_method, config.simulation.constraints))
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    _spec, report = derive_q_region_spec(config, state1, state2, include_reaction_atoms=True)
    payload = json.dumps(report.changed_bonded_terms)
    assert "term count differs" not in payload
    assert all("mapping" in row for row in report.changed_bonded_terms)
