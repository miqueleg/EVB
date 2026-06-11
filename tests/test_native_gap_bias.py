from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("openmm")

import openmm as mm
from openmm import unit
from openmm.app import Topology, element

from kemp_evb.cli import run_gap_table_metadynamics
from kemp_evb.config import load_config
from kemp_evb.native_bias import NativeGapBiasTable1D, NativeWellTemperedGapMetadynamics1D
from kemp_evb.openmm_backend import EVBSystemBuilder, LoadedAmberState, evb_diabatic_energies
from kemp_evb.simulation import EVBSimulation, create_integrator


PARAMETERS = dict(delta_alpha=2.0, h12=5.0)


def _make_topology():
    topology = Topology()
    chain = topology.addChain("A")
    residue = topology.addResidue("MOL", chain)
    topology.addAtom("A1", element.carbon, residue)
    topology.addAtom("A2", element.carbon, residue)
    return topology


def _make_state(r0_nm: float, distance_nm: float = 0.18) -> LoadedAmberState:
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


def _systems(table: NativeGapBiasTable1D):
    state1 = _make_state(0.12)
    state2 = _make_state(0.14)
    builder = EVBSystemBuilder()
    legacy = builder.build_openmm_evb_system(state1, state2, **PARAMETERS)
    native = builder.build_openmm_evb_system(state1, state2, **PARAMETERS, native_gap_bias_table=table)
    return state1, state2, legacy, native


def _single(evb_system, positions):
    sim = EVBSimulation(evb_system, create_integrator(1.0, integrator_name="Verlet"), platform_name="CPU")
    return sim.single_point(positions), sim


def _force_rmsd(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_zero_bias_table_matches_unbiased_evb_energy_gap_and_forces():
    table = NativeGapBiasTable1D(-10.0, 10.0, 101)
    state1, _state2, legacy, native = _systems(table)
    legacy_result, _ = _single(legacy, state1.positions_nm)
    native_result, _ = _single(native, state1.positions_nm)
    legacy_gap = legacy_result.energy1 - legacy_result.energy2 - PARAMETERS["delta_alpha"]
    native_gap = native_result.energy1 - native_result.energy2 - PARAMETERS["delta_alpha"]

    assert abs(native_result.evb_energy - legacy_result.evb_energy) <= 1.0e-6
    assert abs(native_gap - legacy_gap) <= 1.0e-6
    assert _force_rmsd(native_result.forces, legacy_result.forces) <= 1.0e-6


def test_constant_bias_table_shifts_energy_without_changing_forces():
    table = NativeGapBiasTable1D(-10.0, 10.0, 101, values_kj_mol=np.full(101, 3.0))
    state1, _state2, legacy, native = _systems(table)
    legacy_result, _ = _single(legacy, state1.positions_nm)
    native_result, _ = _single(native, state1.positions_nm)

    assert abs((native_result.evb_energy - legacy_result.evb_energy) - 3.0) <= 1.0e-6
    assert _force_rmsd(native_result.forces, legacy_result.forces) <= 1.0e-6


def test_harmonic_table_bias_matches_native_gap_umbrella():
    state1 = _make_state(0.12)
    state2 = _make_state(0.14)
    center = -1.0
    k_gap = 0.2
    table = NativeGapBiasTable1D(-5.0, 5.0, 10001)
    table.set_values(0.5 * k_gap * (table.grid - center) ** 2)
    builder = EVBSystemBuilder()
    table_system = builder.build_openmm_evb_system(state1, state2, **PARAMETERS, native_gap_bias_table=table)
    umbrella = builder.build_openmm_gap_umbrella_system(state1, state2, **PARAMETERS, gap_center=center, gap_force_constant=k_gap)
    table_result, _ = _single(table_system, state1.positions_nm)
    umbrella_result, _ = _single(umbrella, state1.positions_nm)
    table_gap = table_result.energy1 - table_result.energy2 - PARAMETERS["delta_alpha"]
    umbrella_gap = umbrella_result.energy1 - umbrella_result.energy2 - PARAMETERS["delta_alpha"]

    assert abs(table_result.evb_energy - umbrella_result.evb_energy) <= 1.0e-5
    assert abs(table_gap - umbrella_gap) <= 1.0e-6
    assert _force_rmsd(table_result.forces, umbrella_result.forces) <= 1.0e-4


def test_gaussian_deposition_writes_restart_and_reloads(tmp_path):
    table = NativeGapBiasTable1D(-10.0, 10.0, 101)
    state1, _state2, _legacy, native = _systems(table)
    _result, sim = _single(native, state1.positions_nm)
    energy1, energy2 = evb_diabatic_energies(native, sim.context)
    gap = energy1 - energy2 - PARAMETERS["delta_alpha"]
    metad = NativeWellTemperedGapMetadynamics1D(
        table,
        bias_width=0.5,
        height_kj_mol=1.0,
        bias_factor=10.0,
        temperature_k=300.0,
        frequency=1,
        save_frequency=1,
        bias_dir=tmp_path,
        restart=True,
    )
    before = table.values_kj_mol.copy()
    assert metad.maybe_deposit(1, gap, native.evb_force, sim.context)
    assert np.max(np.abs(table.values_kj_mol - before)) > 0.0
    metad.save_state()
    reloaded = NativeGapBiasTable1D.from_restart(tmp_path / "native_gap_bias_state.json")
    assert abs(reloaded.evaluate(gap) - table.evaluate(gap)) <= 1.0e-12


def test_gap_table_metadynamics_does_not_use_app_metadynamics_or_biasvariable():
    source = inspect.getsource(run_gap_table_metadynamics)
    assert "app.Metadynamics" not in source
    assert "BiasVariable" not in source
    table = NativeGapBiasTable1D(-10.0, 10.0, 11)
    state1, state2, _legacy, _native = _systems(table)
    evb_system = EVBSystemBuilder().build_openmm_evb_system(state1, state2, **PARAMETERS, native_gap_bias_table=table)
    assert evb_system.native_gap_bias is table
    assert evb_system.table_bias_function_index is not None
    assert evb_system.bias_report["uses_app_metadynamics"] is False
    assert evb_system.bias_report["uses_bias_variable"] is False


def test_decomposed_outer_common_matches_cv_compatible_and_reports_e1_e2():
    state1 = _make_state(0.12)
    state2 = _make_state(0.12)
    builder = EVBSystemBuilder()
    cv_system = builder.build_openmm_evb_system_decomposed(state1, state2, **PARAMETERS, common_force_placement="cv_compatible")
    outer_system = builder.build_openmm_evb_system_decomposed(state1, state2, **PARAMETERS, common_force_placement="outer_system")
    cv_result, _ = _single(cv_system, state1.positions_nm)
    outer_result, _ = _single(outer_system, state1.positions_nm)

    assert abs(outer_result.evb_energy - cv_result.evb_energy) <= 1.0e-6
    assert abs(outer_result.energy1 - cv_result.energy1) <= 1.0e-6
    assert abs(outer_result.energy2 - cv_result.energy2) <= 1.0e-6
    assert _force_rmsd(outer_result.forces, cv_result.forces) <= 1.0e-6
    report = outer_system.energy_decomposition_report
    assert report["common_force_placement"] == "outer_system"
    assert report["e_common_inside_custom_cv"] is False


def test_gap_table_metadynamics_cli_runs_toy(tmp_path):
    config = load_config("examples/toy_evb.yaml")
    config.output_dir = str(tmp_path / "toy_table")
    config.sampling.mode = "gap_table_metadynamics"
    config.sampling.md.production_steps = 4
    config.sampling.md.report_stride = 2
    config.sampling.md.platform = "CPU"
    config.sampling.native_gap_bias.min_value = -10.0
    config.sampling.native_gap_bias.max_value = 10.0
    config.sampling.native_gap_bias.grid_width = 41
    config.sampling.native_gap_bias.bias_width = 0.5
    config.sampling.native_gap_bias.height_kj_mol = 0.2
    config.sampling.native_gap_bias.bias_factor = 5.0
    config.sampling.native_gap_bias.frequency = 2
    config.sampling.native_gap_bias.save_frequency = 2
    config.sampling.native_gap_bias.bias_dir = "native_bias"
    run_gap_table_metadynamics(config)
    assert (tmp_path / "toy_table" / "gap_table_metad_setup.json").exists()
    assert (tmp_path / "toy_table" / "native_bias" / "native_gap_bias_state.json").exists()
