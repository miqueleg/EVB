from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openmm")

import openmm as mm
from openmm import unit
from openmm.app import Topology, element

from kemp_evb.evb import EVBParameters, EVBHamiltonian
from kemp_evb.openmm_backend import LoadedAmberState, EVBSystemBuilder
from kemp_evb.simulation import EVBSimulation, create_integrator



def _make_topology():
    topology = Topology()
    chain = topology.addChain("A")
    residue = topology.addResidue("MOL", chain)
    topology.addAtom("A1", element.carbon, residue)
    topology.addAtom("A2", element.carbon, residue)
    return topology



def _make_state(r0_nm: float, distance_nm: float) -> LoadedAmberState:
    system = mm.System()
    for _ in range(2):
        system.addParticle(12.0 * unit.amu)
    bond_force = mm.HarmonicBondForce()
    bond_force.addBond(0, 1, r0_nm * unit.nanometer, 1000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    system.addForce(bond_force)
    return LoadedAmberState(
        prmtop_path="toy",
        inpcrd_path="toy",
        topology=_make_topology(),
        system=system,
        positions_nm=np.array([[0.0, 0.0, 0.0], [distance_nm, 0.0, 0.0]]),
        box_vectors_nm=None,
        atom_labels=[("A", "MOL", "A1", 0), ("A", "MOL", "A2", 1)],
        atom_names=["A1", "A2"],
        masses_amu=np.array([12.0, 12.0]),
    )



def test_native_openmm_evb_toy_system_matches_analytical_energy_and_minimizes():
    state1 = _make_state(r0_nm=0.12, distance_nm=0.18)
    state2 = _make_state(r0_nm=0.14, distance_nm=0.18)
    parameters = EVBParameters(delta_alpha=2.0, h12=5.0)

    builder = EVBSystemBuilder()
    evb_system = builder.build_openmm_evb_system(state1, state2, parameters.delta_alpha, parameters.h12)
    simulation = EVBSimulation(
        evb_system=evb_system,
        integrator=create_integrator(1.0, integrator_name="Verlet"),
        platform_name="CPU",
    )

    result = simulation.single_point()
    analytical_energy, weight1, weight2 = EVBHamiltonian(parameters).lower_eigenvalue(result.energy1, result.energy2)
    assert np.isclose(result.evb_energy, analytical_energy)
    assert np.isclose(result.weight1, weight1)
    assert np.isclose(result.weight2, weight2)

    history = simulation.minimize(max_iterations=50)
    assert history[-1].evb_energy <= history[0].evb_energy


def test_native_openmm_evb_weights_match_delta_alpha_finite_difference():
    state1 = _make_state(r0_nm=0.12, distance_nm=0.18)
    state2 = _make_state(r0_nm=0.14, distance_nm=0.18)
    parameters = EVBParameters(delta_alpha=2.0, h12=5.0)

    evb_system = EVBSystemBuilder().build_openmm_evb_system(
        state1,
        state2,
        parameters.delta_alpha,
        parameters.h12,
    )
    simulation = EVBSimulation(
        evb_system=evb_system,
        integrator=create_integrator(1.0, integrator_name="Verlet"),
        platform_name="CPU",
    )

    result = simulation.single_point()
    eps = 1.0e-4
    simulation.context.setParameter("delta_alpha", parameters.delta_alpha + eps)
    e_plus = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    simulation.context.setParameter("delta_alpha", parameters.delta_alpha - eps)
    e_minus = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    finite_difference = (e_plus - e_minus) / (2.0 * eps)

    assert np.isclose(finite_difference, result.weight2, rtol=1.0e-5, atol=1.0e-5)


def test_native_openmm_evb_force_matches_position_finite_difference():
    state1 = _make_state(r0_nm=0.12, distance_nm=0.18)
    state2 = _make_state(r0_nm=0.14, distance_nm=0.18)
    parameters = EVBParameters(delta_alpha=2.0, h12=5.0)

    evb_system = EVBSystemBuilder().build_openmm_evb_system(
        state1,
        state2,
        parameters.delta_alpha,
        parameters.h12,
    )
    simulation = EVBSimulation(
        evb_system=evb_system,
        integrator=create_integrator(1.0, integrator_name="Verlet"),
        platform_name="CPU",
    )
    positions = state1.positions_nm.copy()
    result = simulation.single_point(positions)

    eps = 1.0e-5
    displaced = positions.copy()
    displaced[1, 0] += eps
    e_plus = simulation.single_point(displaced).evb_energy
    displaced[1, 0] -= 2.0 * eps
    e_minus = simulation.single_point(displaced).evb_energy
    finite_difference_force = -(e_plus - e_minus) / (2.0 * eps)

    assert np.isclose(result.forces[1, 0], finite_difference_force, rtol=1.0e-4, atol=1.0e-4)


def test_native_openmm_evb_toy_system_runs_on_cuda_when_available():
    try:
        mm.Platform.getPlatformByName("CUDA")
    except Exception:
        pytest.skip("CUDA platform is not available in this OpenMM installation.")

    state1 = _make_state(r0_nm=0.12, distance_nm=0.18)
    state2 = _make_state(r0_nm=0.14, distance_nm=0.18)
    parameters = EVBParameters(delta_alpha=2.0, h12=5.0)
    builder = EVBSystemBuilder()
    evb_system = builder.build_openmm_evb_system(state1, state2, parameters.delta_alpha, parameters.h12)

    try:
        simulation = EVBSimulation(
            evb_system=evb_system,
            integrator=create_integrator(1.0, integrator_name="Verlet"),
            platform_name="CUDA",
        )
    except Exception as exc:
        pytest.skip(f"CUDA platform is present but context initialization failed: {exc}")

    result = simulation.single_point()
    analytical_energy, _, _ = EVBHamiltonian(parameters).lower_eigenvalue(result.energy1, result.energy2)
    assert np.isclose(result.evb_energy, analytical_energy)



def test_exact_decomposed_evb_matches_legacy_energy_gap_forces_and_minimization():
    state1 = _make_state(r0_nm=0.12, distance_nm=0.18)
    state2 = _make_state(r0_nm=0.14, distance_nm=0.18)
    parameters = EVBParameters(delta_alpha=2.0, h12=5.0)
    builder = EVBSystemBuilder()
    legacy = builder.build_openmm_evb_system(state1, state2, parameters.delta_alpha, parameters.h12)
    decomposed = builder.build_openmm_evb_system_decomposed(state1, state2, parameters.delta_alpha, parameters.h12)

    legacy_sim = EVBSimulation(legacy, create_integrator(1.0, integrator_name="Verlet"), platform_name="CPU")
    decomposed_sim = EVBSimulation(decomposed, create_integrator(1.0, integrator_name="Verlet"), platform_name="CPU")
    positions = state1.positions_nm.copy()
    legacy_result = legacy_sim.single_point(positions)
    decomposed_result = decomposed_sim.single_point(positions)

    legacy_gap = legacy_result.energy1 - legacy_result.energy2 - parameters.delta_alpha
    decomposed_gap = decomposed_result.energy1 - decomposed_result.energy2 - parameters.delta_alpha
    assert abs(decomposed_result.evb_energy - legacy_result.evb_energy) <= 1.0e-6
    assert abs(decomposed_gap - legacy_gap) <= 1.0e-6
    force_delta = decomposed_result.forces - legacy_result.forces
    assert np.sqrt(np.mean(force_delta**2)) <= 1.0e-6
    assert np.max(np.abs(force_delta)) <= 1.0e-6

    legacy_history = legacy_sim.minimize(max_iterations=50)
    decomposed_history = decomposed_sim.minimize(max_iterations=50)
    assert abs(decomposed_history[-1].evb_energy - legacy_history[-1].evb_energy) <= 1.0e-6
    assert decomposed.energy_decomposition_report["enabled"] is True
    assert decomposed.energy_decomposition_report["n_state1_terms"] == 1
    assert decomposed.energy_decomposition_report["n_state2_terms"] == 1


def test_exact_decomposition_reports_identical_common_force():
    state1 = _make_state(r0_nm=0.12, distance_nm=0.18)
    state2 = _make_state(r0_nm=0.12, distance_nm=0.18)
    parameters = EVBParameters(delta_alpha=2.0, h12=5.0)
    decomposed = EVBSystemBuilder().build_openmm_evb_system_decomposed(
        state1,
        state2,
        parameters.delta_alpha,
        parameters.h12,
    )
    report = decomposed.energy_decomposition_report
    assert report["n_common_forces"] == 1
    assert report["n_common_terms"] == 1
    assert report["n_state1_terms"] == 0
    assert report["n_state2_terms"] == 0
    assert report["warnings"] == []
