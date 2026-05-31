from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openmm")

import openmm as mm
from openmm import unit
from openmm.app import Topology, element

from kemp_evb.config import StateFiles
from kemp_evb.openmm_backend import EVBSystemBuilder, LoadedAmberState, write_openmm_bundle


def _make_topology():
    topology = Topology()
    chain = topology.addChain("A")
    residue = topology.addResidue("MOL", chain)
    topology.addAtom("A1", element.carbon, residue)
    topology.addAtom("A2", element.carbon, residue)
    topology.setPeriodicBoxVectors(
        (
            mm.Vec3(2.0, 0.0, 0.0) * unit.nanometer,
            mm.Vec3(0.0, 2.0, 0.0) * unit.nanometer,
            mm.Vec3(0.0, 0.0, 2.0) * unit.nanometer,
        )
    )
    return topology


def _make_system():
    system = mm.System()
    for _ in range(2):
        system.addParticle(12.0 * unit.amu)
    bond_force = mm.HarmonicBondForce()
    bond_force.addBond(0, 1, 0.14 * unit.nanometer, 1000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    system.addForce(bond_force)
    return system


def test_openmm_bundle_loader_roundtrip(tmp_path):
    positions_nm = np.array([[0.0, 0.0, 0.0], [0.18, 0.0, 0.0]], dtype=float)
    topology = _make_topology()
    system = _make_system()
    bundle1 = tmp_path / "state1"
    bundle2 = tmp_path / "state2"
    write_openmm_bundle(bundle1, system, topology, positions_nm, np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]))
    write_openmm_bundle(bundle2, system, topology, positions_nm, np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]))

    builder = EVBSystemBuilder()
    state1, state2 = builder.build_from_state_files(
        StateFiles(prmtop=str(bundle1 / "system.xml"), inpcrd=str(bundle1 / "coordinates.pdb"), format="openmm"),
        StateFiles(prmtop=str(bundle2 / "system.xml"), inpcrd=str(bundle2 / "coordinates.pdb"), format="openmm"),
    )
    assert isinstance(state1, LoadedAmberState)
    assert state1.system.getNumParticles() == 2
    assert np.allclose(state1.positions_nm, positions_nm)
    assert state1.atom_names == ["A1", "A2"]
    builder.validate_compatibility(state1, state2)
