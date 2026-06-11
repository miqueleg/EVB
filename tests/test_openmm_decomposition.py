from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openmm")

import openmm as mm
from openmm import unit
from openmm.app import Topology, element

from kemp_evb.evb import EVBParameters
from kemp_evb.openmm_backend import EVBSystemBuilder, LoadedAmberState
from kemp_evb.simulation import EVBSimulation, create_integrator


POSITIONS = np.array(
    [
        [0.00, 0.00, 0.00],
        [0.15, 0.00, 0.00],
        [0.15, 0.16, 0.00],
        [0.25, 0.16, 0.12],
    ],
    dtype=float,
)
PARAMETERS = EVBParameters(delta_alpha=1.25, h12=3.5)


def _topology(n_atoms: int) -> Topology:
    topology = Topology()
    chain = topology.addChain("A")
    residue = topology.addResidue("MOL", chain)
    for index in range(n_atoms):
        topology.addAtom(f"A{index + 1}", element.carbon, residue)
    return topology


def _system_with_force(force, n_atoms: int = 4, box: bool = False) -> mm.System:
    system = mm.System()
    for _ in range(n_atoms):
        system.addParticle(12.0 * unit.amu)
    if box:
        system.setDefaultPeriodicBoxVectors(
            mm.Vec3(3.0, 0.0, 0.0) * unit.nanometer,
            mm.Vec3(0.0, 3.0, 0.0) * unit.nanometer,
            mm.Vec3(0.0, 0.0, 3.0) * unit.nanometer,
        )
    system.addForce(force)
    return system


def _state(system: mm.System, box: bool = False) -> LoadedAmberState:
    return LoadedAmberState(
        prmtop_path="toy",
        inpcrd_path="toy",
        topology=_topology(system.getNumParticles()),
        system=system,
        positions_nm=POSITIONS[: system.getNumParticles()].copy(),
        box_vectors_nm=(np.diag([3.0, 3.0, 3.0]) if box else None),
        atom_labels=[("A", "MOL", f"A{i + 1}", i) for i in range(system.getNumParticles())],
        atom_names=[f"A{i + 1}" for i in range(system.getNumParticles())],
        masses_amu=np.full(system.getNumParticles(), 12.0),
    )


def _compare_legacy_decomposed(
    state1: LoadedAmberState,
    state2: LoadedAmberState,
    *,
    minimize: bool = True,
    force_rmsd_tolerance: float = 1.0e-6,
    force_max_tolerance: float = 1.0e-6,
):
    builder = EVBSystemBuilder()
    legacy = builder.build_openmm_evb_system(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12)
    decomposed = builder.build_openmm_evb_system_decomposed(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12)
    legacy_sim = EVBSimulation(legacy, create_integrator(1.0, integrator_name="Verlet"), platform_name="CPU")
    decomposed_sim = EVBSimulation(decomposed, create_integrator(1.0, integrator_name="Verlet"), platform_name="CPU")
    legacy_result = legacy_sim.single_point(state1.positions_nm)
    decomposed_result = decomposed_sim.single_point(state1.positions_nm)

    legacy_gap = legacy_result.energy1 - legacy_result.energy2 - PARAMETERS.delta_alpha
    decomposed_gap = decomposed_result.energy1 - decomposed_result.energy2 - PARAMETERS.delta_alpha
    force_delta = decomposed_result.forces - legacy_result.forces

    assert abs(decomposed_result.evb_energy - legacy_result.evb_energy) <= 1.0e-6
    assert abs(decomposed_gap - legacy_gap) <= 1.0e-6
    assert np.sqrt(np.mean(force_delta**2)) <= force_rmsd_tolerance
    assert np.max(np.abs(force_delta)) <= force_max_tolerance

    if minimize:
        legacy_history = legacy_sim.minimize(max_iterations=5)
        decomposed_history = decomposed_sim.minimize(max_iterations=5)
        assert abs(decomposed_history[-1].evb_energy - legacy_history[-1].evb_energy) <= 1.0e-6
    return decomposed.energy_decomposition_report


def _assert_one_common_one_residual(report: dict):
    assert report["n_common_terms"] == 1
    assert report["n_state1_terms"] == 1
    assert report["n_state2_terms"] == 1
    assert report["duplicated_full_nonbonded"] is False
    assert report["warnings"] == []


def _harmonic_bond_states():
    f1 = mm.HarmonicBondForce()
    f2 = mm.HarmonicBondForce()
    f1.addBond(0, 1, 0.14 * unit.nanometer, 100.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    f2.addBond(0, 1, 0.14 * unit.nanometer, 100.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    f1.addBond(2, 3, 0.12 * unit.nanometer, 80.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    f2.addBond(2, 3, 0.13 * unit.nanometer, 80.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    return _state(_system_with_force(f1)), _state(_system_with_force(f2))


def _harmonic_angle_states():
    f1 = mm.HarmonicAngleForce()
    f2 = mm.HarmonicAngleForce()
    f1.addAngle(0, 1, 2, 1.5 * unit.radian, 10.0 * unit.kilojoule_per_mole / unit.radian**2)
    f2.addAngle(0, 1, 2, 1.5 * unit.radian, 10.0 * unit.kilojoule_per_mole / unit.radian**2)
    f1.addAngle(1, 2, 3, 1.2 * unit.radian, 12.0 * unit.kilojoule_per_mole / unit.radian**2)
    f2.addAngle(1, 2, 3, 1.3 * unit.radian, 12.0 * unit.kilojoule_per_mole / unit.radian**2)
    return _state(_system_with_force(f1)), _state(_system_with_force(f2))


def _periodic_torsion_states():
    f1 = mm.PeriodicTorsionForce()
    f2 = mm.PeriodicTorsionForce()
    f1.addTorsion(0, 1, 2, 3, 1, 0.0 * unit.radian, 1.0 * unit.kilojoule_per_mole)
    f2.addTorsion(0, 1, 2, 3, 1, 0.0 * unit.radian, 1.0 * unit.kilojoule_per_mole)
    f1.addTorsion(0, 1, 2, 3, 2, 0.5 * unit.radian, 0.7 * unit.kilojoule_per_mole)
    f2.addTorsion(0, 1, 2, 3, 2, 0.6 * unit.radian, 0.7 * unit.kilojoule_per_mole)
    return _state(_system_with_force(f1)), _state(_system_with_force(f2))


def _rb_torsion_states():
    f1 = mm.RBTorsionForce()
    f2 = mm.RBTorsionForce()
    f1.addTorsion(0, 1, 2, 3, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0)
    f2.addTorsion(0, 1, 2, 3, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0)
    f1.addTorsion(3, 2, 1, 0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0)
    f2.addTorsion(3, 2, 1, 0, 0.25, 0.1, 0.0, 0.0, 0.0, 0.0)
    return _state(_system_with_force(f1)), _state(_system_with_force(f2))


def _custom_bond_states():
    f1 = mm.CustomBondForce("0.5*k*(r-r0)^2")
    f2 = mm.CustomBondForce("0.5*k*(r-r0)^2")
    for force in (f1, f2):
        force.addPerBondParameter("r0")
        force.addPerBondParameter("k")
    f1.addBond(0, 1, [0.14, 100.0])
    f2.addBond(0, 1, [0.14, 100.0])
    f1.addBond(2, 3, [0.12, 80.0])
    f2.addBond(2, 3, [0.13, 80.0])
    return _state(_system_with_force(f1)), _state(_system_with_force(f2))


def _custom_angle_states():
    f1 = mm.CustomAngleForce("0.5*k*(theta-theta0)^2")
    f2 = mm.CustomAngleForce("0.5*k*(theta-theta0)^2")
    for force in (f1, f2):
        force.addPerAngleParameter("theta0")
        force.addPerAngleParameter("k")
    f1.addAngle(0, 1, 2, [1.5, 10.0])
    f2.addAngle(0, 1, 2, [1.5, 10.0])
    f1.addAngle(1, 2, 3, [1.2, 12.0])
    f2.addAngle(1, 2, 3, [1.3, 12.0])
    return _state(_system_with_force(f1)), _state(_system_with_force(f2))


def _custom_torsion_states():
    f1 = mm.CustomTorsionForce("k*(1+cos(theta-theta0))")
    f2 = mm.CustomTorsionForce("k*(1+cos(theta-theta0))")
    for force in (f1, f2):
        force.addPerTorsionParameter("theta0")
        force.addPerTorsionParameter("k")
    f1.addTorsion(0, 1, 2, 3, [0.0, 1.0])
    f2.addTorsion(0, 1, 2, 3, [0.0, 1.0])
    f1.addTorsion(3, 2, 1, 0, [0.5, 0.7])
    f2.addTorsion(3, 2, 1, 0, [0.6, 0.7])
    return _state(_system_with_force(f1)), _state(_system_with_force(f2))


@pytest.mark.parametrize(
    "factory, class_name",
    [
        (_harmonic_bond_states, "HarmonicBondForce"),
        (_harmonic_angle_states, "HarmonicAngleForce"),
        (_periodic_torsion_states, "PeriodicTorsionForce"),
        (_rb_torsion_states, "RBTorsionForce"),
        (_custom_bond_states, "CustomBondForce"),
        (_custom_angle_states, "CustomAngleForce"),
        (_custom_torsion_states, "CustomTorsionForce"),
    ],
)
def test_exact_decomposes_supported_force_terms(factory, class_name):
    state1, state2 = factory()
    report = _compare_legacy_decomposed(state1, state2)
    _assert_one_common_one_residual(report)
    assert report["partitions"]["common"]["forces_by_class"] == {class_name: 1}
    assert report["partitions"]["state1"]["forces_by_class"] == {class_name: 1}
    assert report["partitions"]["state2"]["forces_by_class"] == {class_name: 1}


def _nonbonded_force(charge_delta: float = 0.0) -> mm.NonbondedForce:
    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.PME)
    force.setCutoffDistance(0.9 * unit.nanometer)
    charges = [0.1 + charge_delta, -0.1, 0.2, -0.2]
    for charge in charges:
        force.addParticle(charge * unit.elementary_charge, 0.30 * unit.nanometer, 0.20 * unit.kilojoule_per_mole)
    force.addException(0, 1, 0.0 * unit.elementary_charge**2, 0.30 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    return force


def test_identical_nonbonded_force_moves_to_common():
    state1 = _state(_system_with_force(_nonbonded_force(), box=True), box=True)
    state2 = _state(_system_with_force(_nonbonded_force(), box=True), box=True)
    # PME forces can differ at the last reduction bits when moved inside a different
    # CustomCVForce tree; energies and gaps still use the tight exact tolerance.
    report = _compare_legacy_decomposed(
        state1,
        state2,
        minimize=False,
        force_rmsd_tolerance=1.0e-4,
        force_max_tolerance=2.0e-4,
    )
    assert report["n_common_forces"] == 1
    assert report["n_common_terms"] == 5
    assert report["n_state1_terms"] == 0
    assert report["n_state2_terms"] == 0
    assert report["duplicated_full_nonbonded"] is False
    common_nb = report["partitions"]["common"]["nonbonded_forces"][0]
    assert common_nb["particles"] == 4
    assert common_nb["exceptions"] == 1
    assert common_nb["nonbonded_method"] == "PME"
    assert common_nb["full_system"] is True


def test_differing_pme_nonbonded_falls_back_and_reports_full_duplication():
    state1 = _state(_system_with_force(_nonbonded_force(), box=True), box=True)
    state2 = _state(_system_with_force(_nonbonded_force(charge_delta=0.01), box=True), box=True)
    # PME force reductions differ at the CustomCV/PME summation level on CPU even though
    # the same full state-specific NonbondedForce objects are evaluated. Energies and gaps
    # remain at the tighter exact-mode tolerance above.
    report = _compare_legacy_decomposed(
        state1,
        state2,
        minimize=False,
        force_rmsd_tolerance=1.0e-4,
        force_max_tolerance=1.0e-4,
    )
    assert report["n_common_terms"] == 0
    assert report["n_state1_terms"] == 5
    assert report["n_state2_terms"] == 5
    assert report["duplicated_full_nonbonded"] is True
    assert report["partitions"]["state1"]["nonbonded_forces"][0]["full_system"] is True
    assert report["partitions"]["state2"]["nonbonded_forces"][0]["full_system"] is True
    assert any("speedup should not be expected" in warning for warning in report["warnings"])


def test_differing_pme_nonbonded_fails_when_fallback_disabled():
    state1 = _state(_system_with_force(_nonbonded_force(), box=True), box=True)
    state2 = _state(_system_with_force(_nonbonded_force(charge_delta=0.01), box=True), box=True)
    with pytest.raises(ValueError, match="fallback_to_legacy_for_unsupported_terms"):
        EVBSystemBuilder().build_openmm_evb_system_decomposed(
            state1,
            state2,
            PARAMETERS.delta_alpha,
            PARAMETERS.h12,
            fallback_to_legacy_for_unsupported_terms=False,
        )
