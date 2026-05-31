from __future__ import annotations

import numpy as np

from kemp_evb.config import DistanceCVDefinition, ReactionCoordinateDefinition
from kemp_evb.evb import EVBResult
from kemp_evb.observables import compute_energy_gap, compute_named_distances, compute_named_reaction_coordinates, make_gap_sample


def test_gap_observables_report_raw_and_shifted_gaps():
    delta_e, shifted = compute_energy_gap(10.0, 6.0, 2.5)
    assert delta_e == 4.0
    assert shifted == 1.5


def test_gap_sample_carries_evb_frame_data():
    result = EVBResult(
        energy1=10.0,
        energy2=6.0,
        e2_shifted=8.5,
        evb_energy=7.2,
        weight1=0.7,
        weight2=0.3,
        forces=np.zeros((1, 3)),
    )
    sample = make_gap_sample(result, frame=1, step=100, time_ps=0.2, delta_alpha_kj_mol=2.5)
    assert sample.delta_e_kj_mol == 4.0
    assert sample.delta_e_shifted_kj_mol == 1.5
    assert sample.evb_energy_kj_mol == 7.2


def test_named_distances_are_computed_from_positions():
    positions = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.4, 0.0]])
    distances = compute_named_distances(
        positions,
        [
            DistanceCVDefinition(name="ab", atom1=0, atom2=1),
            DistanceCVDefinition(name="bc", atom1=1, atom2=2),
        ],
    )
    assert np.isclose(distances["ab"], 0.3)
    assert np.isclose(distances["bc"], 0.4)


def test_named_reaction_coordinates_support_distance_and_difference():
    positions = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [0.25, 0.0, 0.0]])
    coords = compute_named_reaction_coordinates(
        positions,
        [
            ReactionCoordinateDefinition(name="pt_rc", kind="difference_of_distances", atom1=0, atom2=1, atom3=2),
            ReactionCoordinateDefinition(name="bond_break", kind="distance", atom1=0, atom2=2),
        ],
    )
    assert np.isclose(coords["pt_rc"], 0.10 - 0.15)
    assert np.isclose(coords["bond_break"], 0.25)
