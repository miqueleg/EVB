from __future__ import annotations

import numpy as np

from kemp_evb.evb import EVBParameters, EVBHamiltonian, calibrate_evb_parameters
from kemp_evb.evb_core import diabatic_gap, evb_energy, evb_weights
from kemp_evb.cv import proton_transfer_coordinate


def test_evb_combination_returns_consistent_weights_and_energy():
    ham = EVBHamiltonian(EVBParameters(delta_alpha=2.0, h12=3.0))
    result = ham.combine(10.0, np.array([[1.0, 0.0, 0.0]]), 6.0, np.array([[0.0, 1.0, 0.0]]))
    assert result.evb_energy < min(result.energy1, result.e2_shifted)
    assert np.isclose(result.weight1 + result.weight2, 1.0)
    assert np.allclose(result.forces, np.array([[result.weight1, result.weight2, 0.0]]))


def test_general_evb_core_helpers_and_finite_difference_weights():
    e1 = 12.0
    e2 = 8.0
    delta_alpha = 1.5
    h12 = 3.0
    eps = 1.0e-5
    weight1, weight2 = evb_weights(e1, e2, delta_alpha, h12)
    d_de1 = (evb_energy(e1 + eps, e2, delta_alpha, h12) - evb_energy(e1 - eps, e2, delta_alpha, h12)) / (2 * eps)
    d_de2 = (evb_energy(e1, e2 + eps, delta_alpha, h12) - evb_energy(e1, e2 - eps, delta_alpha, h12)) / (2 * eps)
    assert np.isclose(diabatic_gap(e1, e2, delta_alpha), 2.5)
    assert np.isclose(weight1, d_de1)
    assert np.isclose(weight2, d_de2)


def test_calibration_produces_barrier_consistent_parameters():
    params = calibrate_evb_parameters(
        e_mm_min1_state1=0.0,
        e_mm_min1_state2=20.0,
        e_mm_min2_state1=22.0,
        e_mm_min2_state2=0.0,
        e_mm_ts_state1=12.0,
        e_mm_ts_state2=10.0,
        e_qmmm_min1=0.0,
        e_qmmm_min2=-5.0,
        e_qmmm_ts=4.0,
    )
    ham = EVBHamiltonian(params)
    evb_min1, _, _ = ham.lower_eigenvalue(0.0, 20.0)
    evb_min2, _, _ = ham.lower_eigenvalue(22.0, 0.0)
    evb_ts, _, _ = ham.lower_eigenvalue(12.0, 10.0)
    assert np.isclose(evb_min2 - evb_min1, -5.0, atol=0.25)
    assert np.isclose(evb_ts - evb_min1, 4.0, atol=0.05)


def test_proton_transfer_cv():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ]
    )
    value = proton_transfer_coordinate(positions, 0, 1, 2)
    assert np.isclose(value, -0.2)
