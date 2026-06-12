from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmm")

import openmm as mm
from openmm import unit
from openmm.app import Topology, element

from kemp_evb.config import load_config
from kemp_evb.evb import EVBParameters
from kemp_evb.hg317_qregion import make_hg317_qregion_candidates, validate_hg317_qregion_candidates
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


def _nb_state(params, method="NoCutoff", exception=None):
    system = mm.System()
    for _ in params:
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
    for charge, sigma, epsilon in params:
        nb.addParticle(charge * unit.elementary_charge, sigma * unit.nanometer, epsilon * unit.kilojoule_per_mole)
    if exception is not None:
        a, b, chargeprod, sigma, epsilon = exception
        nb.addException(a, b, chargeprod * unit.elementary_charge**2, sigma * unit.nanometer, epsilon * unit.kilojoule_per_mole)
    system.addForce(nb)
    box = np.eye(3) * 3.0 if method == "PME" else None
    if box is not None:
        system.setDefaultPeriodicBoxVectors(*(mm.Vec3(*box[i]) * unit.nanometer for i in range(3)))
    return _state(system, [[0, 0, 0], [0.33, 0, 0], [0.0, 0.41, 0]], box=box)


def _single(evb_system, positions):
    sim = EVBSimulation(evb_system, create_integrator(1.0, integrator_name="Verlet"), platform_name="CPU")
    return sim.single_point(positions)


def _compare_q_to_legacy(state1, state2, spec, tol=1e-6):
    legacy = EVBSystemBuilder().build_openmm_evb_system(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12)
    q_system = QRegionSystemBuilder(spec).build(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12)
    q_evb = q_region_to_evb_openmm_system(q_system)
    legacy_result = _single(legacy, state1.positions_nm)
    q_result = _single(q_evb, state1.positions_nm)
    force_rmsd = float(np.sqrt(np.mean((legacy_result.forces - q_result.forces) ** 2)))
    assert abs(q_result.evb_energy - legacy_result.evb_energy) <= tol
    assert abs((q_result.energy1 - q_result.energy2) - (legacy_result.energy1 - legacy_result.energy2)) <= tol
    assert force_rmsd <= max(tol, 1e-5)
    return q_system


def test_shared_nonbonded_model_builds_differing_pme():
    state1 = _nb_state([(0.2, 0.3, 0.1), (-0.1, 0.3, 0.1), (0.05, 0.3, 0.1)], method="PME")
    state2 = _nb_state([(0.3, 0.3, 0.1), (-0.1, 0.3, 0.1), (0.05, 0.3, 0.1)], method="PME")
    with pytest.raises(ValueError, match="exact PME decomposition is not implemented"):
        QRegionSystemBuilder(QRegionSpec(q_atoms=[0])).build(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12)
    spec = QRegionSpec(q_atoms=[0], nonbonded=QRegionNonbondedPolicy(mode="shared_nonbonded_model"))
    q_system = QRegionSystemBuilder(spec).build(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12)
    report = q_system.q_region_report
    assert report["duplicated_full_nonbonded"] is False
    assert report["exactness_status"] == "exact_for_shared_nonbonded_model"
    assert report["nonbonded_model_changed"] is True
    assert report["legacy_equivalence"] == "false"


def test_local_pme_approx_builds_differing_pme():
    state1 = _nb_state([(0.2, 0.3, 0.1), (-0.1, 0.3, 0.1), (0.05, 0.3, 0.1)], method="PME")
    state2 = _nb_state([(0.3, 0.3, 0.1), (-0.1, 0.3, 0.1), (0.05, 0.3, 0.1)], method="PME")
    spec = QRegionSpec(
        q_atoms=[0],
        nonbonded=QRegionNonbondedPolicy(mode="local_pme_approx", local_approx_enabled=True, correction_atoms="all_atoms"),
    )
    q_system = QRegionSystemBuilder(spec).build(state1, state2, PARAMETERS.delta_alpha, PARAMETERS.h12)
    report = q_system.q_region_report
    assert report["exactness_status"] == "approximate"
    assert report["pme_approximation"] is True
    assert report["reciprocal_pme_difference_ignored_or_approximated"] is True
    assert report["duplicated_full_nonbonded"] is False


@pytest.mark.parametrize(
    "params2",
    [
        [(0.3, 0.3, 0.1), (-0.1, 0.3, 0.1), (0.05, 0.3, 0.1)],
        [(0.2, 0.35, 0.1), (-0.1, 0.3, 0.1), (0.05, 0.3, 0.1)],
        [(0.2, 0.3, 0.2), (-0.1, 0.3, 0.1), (0.05, 0.3, 0.1)],
    ],
)
def test_local_direct_residual_particle_formula_matches_legacy(params2):
    params1 = [(0.2, 0.3, 0.1), (-0.1, 0.3, 0.1), (0.05, 0.3, 0.1)]
    state1 = _nb_state(params1)
    state2 = _nb_state(params2)
    _compare_q_to_legacy(state1, state2, QRegionSpec(q_atoms=[0], correction_atoms=[0, 1, 2]))


def test_local_direct_residual_exception_formula_matches_legacy():
    params = [(0.2, 0.3, 0.1), (-0.1, 0.3, 0.1), (0.05, 0.3, 0.1)]
    state1 = _nb_state(params, exception=(0, 1, -0.01, 0.3, 0.01))
    state2 = _nb_state(params, exception=(0, 1, -0.02, 0.32, 0.02))
    _compare_q_to_legacy(state1, state2, QRegionSpec(q_atoms=[0, 1], correction_atoms=[0, 1, 2]))


def test_baseline_state2_residual_sign_matches_legacy():
    state1 = _nb_state([(0.2, 0.3, 0.1), (-0.1, 0.3, 0.1), (0.05, 0.3, 0.1)])
    state2 = _nb_state([(0.3, 0.3, 0.1), (-0.1, 0.3, 0.1), (0.05, 0.3, 0.1)])
    _compare_q_to_legacy(state1, state2, QRegionSpec(q_atoms=[0], correction_atoms=[0, 1, 2], baseline_state="state2"))


def test_candidate_generation_writes_all_configs(tmp_path):
    config_path = Path("examples/toy_evb.yaml")
    config = load_config(config_path)
    summary = make_hg317_qregion_candidates(config, config_path, tmp_path / "candidates", include_reaction_atoms=True)
    names = {row["name"] for row in summary["candidates"]}
    assert "shared_nonbonded_state1" in names
    assert "local_pme_all_atoms_cutoff_1.2" in names
    assert len(names) == 10
    for row in summary["candidates"]:
        assert Path(row["path"]).exists()


def test_validation_summary_and_smoke_guard(tmp_path):
    config_path = Path("examples/toy_evb.yaml")
    config = load_config(config_path)
    root = tmp_path / "candidates"
    make_hg317_qregion_candidates(config, config_path, root, include_reaction_atoms=True)
    summary = validate_hg317_qregion_candidates(root, root / "validation", "CPU", run_smoke=True, smoke_steps=5)
    assert (root / "validation" / "candidate_validation_summary.json").exists()
    assert (root / "validation" / "candidate_validation_summary.csv").exists()
    assert (root / "validation" / "candidate_validation_summary.md").exists()
    assert summary["smoke"]["ran"] is False
    assert "No candidate passed" in summary["smoke"]["reason"]
