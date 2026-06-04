from __future__ import annotations

import numpy as np

from kemp_evb.fitting.irc_fit import evb_relative_profile, fit_evb_to_irc_profile


def test_fit_evb_to_irc_profile_recovers_synthetic_parameters():
    e1 = np.linspace(0.0, 120.0, 9)
    e2 = np.linspace(150.0, -40.0, 9)
    target = evb_relative_profile(e1, e2, delta_alpha_kj_mol=35.0, h12_kj_mol=22.0)

    result, fitted = fit_evb_to_irc_profile(
        e1,
        e2,
        target,
        delta_alpha_initial_kj_mol=35.0,
        h12_initial_kj_mol=22.0,
        levels=3,
        samples_per_axis=31,
    )

    assert result.objective_rmse_kj_mol < 1.0
    assert np.sqrt(np.mean((fitted - target) ** 2)) < 1.0
