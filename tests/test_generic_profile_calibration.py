from __future__ import annotations

from kemp_evb.evb import EVBHamiltonian, EVBParameters
from kemp_evb.profile_calibration import CalibrationFrameEnergy, fit_profile_parameters
from kemp_evb.reference_profile import ReferencePoint, ReferenceProfile


def test_profile_calibration_recovers_synthetic_surface():
    true = EVBParameters(delta_alpha=25.0, h12=12.0)
    frames = [
        CalibrationFrameEnergy("reactant", -20.0, 10.0),
        CalibrationFrameEnergy("transition_state", 5.0, -5.0),
        CalibrationFrameEnergy("product", 20.0, -25.0),
    ]
    energies = {f.label: EVBHamiltonian(true).lower_eigenvalue(f.e1_kj_mol, f.e2_kj_mol)[0] for f in frames}
    zero = energies["reactant"]
    profile = ReferenceProfile("synthetic", "reactant", {k: ReferencePoint(k, v - zero) for k, v in energies.items()}, list(energies))

    fit = fit_profile_parameters(frames, profile, initial_delta_alpha=0.0, initial_h12=10.0)

    assert fit.rms_residual_kj_mol < 3.0
    assert fit.max_residual_kj_mol < 4.1


def test_barrier_only_fit_is_marked_limited():
    frames = [CalibrationFrameEnergy("reactant", -10.0, 0.0), CalibrationFrameEnergy("transition_state", 0.0, -5.0)]
    profile = ReferenceProfile("barrier", "reactant", {"reactant": ReferencePoint("reactant", 0.0), "transition_state": ReferencePoint("transition_state", 20.0)}, ["reactant", "transition_state"])

    fit = fit_profile_parameters(frames, profile, mode="barrier_only_fit")

    assert fit.limited_fit is True
