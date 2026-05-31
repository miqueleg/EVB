from __future__ import annotations

from pathlib import Path

import pytest

from kemp_evb.config import (
    DistanceCVDefinition,
    EVBConfig,
    EVBParameterConfig,
    GapUmbrellaWindows,
    MappingWindows,
    MDRunSettings,
    ObservableSettings,
    ReactionCoordinateDefinition,
    ReactionSettings,
    CVDefinition,
    SamplingSettings,
    SamplingWindows,
    SimulationSettings,
    StateFiles,
)
from kemp_evb.sampling.runners import GapUmbrellaWindowRunner, MappingWindowRunner
from kemp_evb.sampling.windows import GapWindowSpec, MappingWindowSpec, build_gap_windows, build_mapping_windows

pytest.importorskip("openmm")

import numpy as np
import openmm as mm
from openmm import unit
from openmm.app import Topology, element

from kemp_evb.openmm_backend import LoadedAmberState


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


def test_build_mapping_windows_from_config():
    config = EVBConfig(
        state1=StateFiles(prmtop="a", inpcrd="b"),
        state2=StateFiles(prmtop="c", inpcrd="d"),
        sampling=SamplingSettings(windows=SamplingWindows(mapping=MappingWindows(lambda_values=[0.0, 0.25, 1.0]))),
    )
    windows = build_mapping_windows(config)
    assert [window.window_id for window in windows] == ["w000", "w001", "w002"]
    assert [window.lambda_value for window in windows] == [0.0, 0.25, 1.0]


def test_build_gap_windows_from_config():
    config = EVBConfig(
        state1=StateFiles(prmtop="a", inpcrd="b"),
        state2=StateFiles(prmtop="c", inpcrd="d"),
        sampling=SamplingSettings(
            mode="gap_umbrella",
            windows=SamplingWindows(gap_umbrella=GapUmbrellaWindows(centers_kj_mol=[-10.0, 0.0, 10.0], force_constant_kj_mol2=25.0)),
        ),
    )
    windows = build_gap_windows(config)
    assert [window.window_id for window in windows] == ["u000", "u001", "u002"]
    assert [window.center_kj_mol for window in windows] == [-10.0, 0.0, 10.0]
    assert all(window.force_constant_kj_mol2 == 25.0 for window in windows)


def test_mapping_window_runner_writes_observable_log(tmp_path: Path):
    state1 = _make_state(r0_nm=0.12, distance_nm=0.18)
    state2 = _make_state(r0_nm=0.14, distance_nm=0.18)
    config = EVBConfig(
        state1=StateFiles(prmtop="unused", inpcrd="unused"),
        state2=StateFiles(prmtop="unused", inpcrd="unused"),
        evb_parameters=EVBParameterConfig(delta_alpha=2.0, h12=5.0),
        reaction=ReactionSettings(atoms=CVDefinition(donor=0, proton=1, acceptor=1)),
        observables=ObservableSettings(
            reaction_coordinates=[ReactionCoordinateDefinition(name="bond", kind="distance", atom1=0, atom2=1)],
            distances=[DistanceCVDefinition(name="ab", atom1=0, atom2=1)],
        ),
        sampling=SamplingSettings(
            md=MDRunSettings(
                equilibration_steps=0,
                production_steps=4,
                report_stride=2,
                temperature_k=300.0,
                minimize_steps=0,
            )
        ),
        simulation=SimulationSettings(platform="CPU"),
        output_dir=str(tmp_path),
    )
    runner = MappingWindowRunner(config, MappingWindowSpec(window_id="w000", lambda_value=0.5), state1=state1, state2=state2)
    summary = runner.run(tmp_path)
    assert summary.n_frames == 2
    csv_path = tmp_path / "windows" / "w000" / "production_observables.csv"
    text = csv_path.read_text(encoding="utf-8")
    assert "window_id,lambda,E1_kj_mol,E2_kj_mol,delta_e_kj_mol" in text
    assert "bond" in text
    assert "ab" in text


def test_gap_umbrella_window_runner_writes_observable_log(tmp_path: Path):
    state1 = _make_state(r0_nm=0.12, distance_nm=0.18)
    state2 = _make_state(r0_nm=0.14, distance_nm=0.18)
    config = EVBConfig(
        state1=StateFiles(prmtop="unused", inpcrd="unused"),
        state2=StateFiles(prmtop="unused", inpcrd="unused"),
        evb_parameters=EVBParameterConfig(delta_alpha=2.0, h12=5.0),
        reaction=ReactionSettings(atoms=CVDefinition(donor=0, proton=1, acceptor=1)),
        observables=ObservableSettings(
            reaction_coordinates=[ReactionCoordinateDefinition(name="bond", kind="distance", atom1=0, atom2=1)],
            distances=[DistanceCVDefinition(name="ab", atom1=0, atom2=1)],
        ),
        sampling=SamplingSettings(
            mode="gap_umbrella",
            md=MDRunSettings(
                equilibration_steps=0,
                production_steps=4,
                report_stride=2,
                temperature_k=300.0,
                minimize_steps=0,
            ),
            windows=SamplingWindows(gap_umbrella=GapUmbrellaWindows(centers_kj_mol=[0.0], force_constant_kj_mol2=25.0)),
        ),
        simulation=SimulationSettings(platform="CPU"),
        output_dir=str(tmp_path),
    )
    runner = GapUmbrellaWindowRunner(
        config,
        GapWindowSpec(window_id="u000", center_kj_mol=0.0, force_constant_kj_mol2=25.0),
        state1=state1,
        state2=state2,
    )
    summary = runner.run(tmp_path)
    assert summary.n_frames == 2
    csv_path = tmp_path / "windows" / "u000" / "production_observables.csv"
    text = csv_path.read_text(encoding="utf-8")
    assert "window_id,gap_center_kj_mol,E1_kj_mol,E2_kj_mol,delta_e_kj_mol" in text
    assert "bond" in text
    assert "ab" in text
