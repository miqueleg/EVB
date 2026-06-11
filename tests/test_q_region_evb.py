from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("openmm")

import openmm as mm
from openmm import unit
from openmm.app import Topology, element

from kemp_evb.cli import run_gap_table_metadynamics
from kemp_evb.evb import EVBParameters
from kemp_evb.native_bias import NativeGapBiasTable1D
from kemp_evb.openmm_backend import EVBSystemBuilder, LoadedAmberState
from kemp_evb.q_region import QRegionNonbondedPolicy, QRegionSpec, QRegionSystemBuilder, q_region_to_evb_openmm_system
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
