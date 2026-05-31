from __future__ import annotations

import csv
from pathlib import Path

from kemp_evb.config import (
    AnalysisSettings,
    BarrierSettings,
    EVBConfig,
    FitScanSettings,
    FitSettings,
    FitTargets,
    HistogramSettings,
    PMFSettings,
    ProjectSettings,
    StateFiles,
)
from kemp_evb.fitting import fit_ensemble_parameters


def _write_window_csv(root: Path, window_id: str, lambda_value: float, rows: list[tuple[float, float]]):
    window_dir = root / "windows" / window_id
    window_dir.mkdir(parents=True, exist_ok=True)
    with (window_dir / "production_observables.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "step",
                "time_ps",
                "window_id",
                "lambda",
                "E1_kj_mol",
                "E2_kj_mol",
                "delta_e_kj_mol",
                "delta_e_shifted_kj_mol",
                "Eevb_kj_mol",
                "w1",
                "w2",
            ]
        )
        for idx, (e1, e2) in enumerate(rows):
            delta = e1 - e2
            writer.writerow([idx, idx, 0.01 * idx, window_id, lambda_value, e1, e2, delta, delta, min(e1, e2), 0.5, 0.5])


def test_ensemble_fit_writes_outputs(tmp_path: Path):
    _write_window_csv(tmp_path, "w000", 0.0, [(-5.0, 8.0), (-4.0, 7.0), (-3.5, 6.5), (-2.0, 4.0)])
    _write_window_csv(tmp_path, "w001", 0.5, [(-1.0, 1.0), (0.0, 0.0), (1.0, -1.0), (2.0, -2.0)])
    _write_window_csv(tmp_path, "w002", 1.0, [(4.0, -2.0), (6.5, -3.5), (7.0, -4.0), (8.0, -5.0)])

    config = EVBConfig(
        state1=StateFiles(prmtop="a", inpcrd="b"),
        state2=StateFiles(prmtop="c", inpcrd="d"),
        output_dir=str(tmp_path),
        project=ProjectSettings(name="fit-test", output_dir=str(tmp_path)),
        analysis=AnalysisSettings(
            histogram=HistogramSettings(bin_min_kj_mol=-20.0, bin_max_kj_mol=20.0, n_bins=40),
            pmf=PMFSettings(temperature_k=300.0, zero_mode="reactant_min"),
            barrier=BarrierSettings(reactant_region=(-20.0, -3.0), product_region=(3.0, 20.0)),
        ),
        fit=FitSettings(
            ensemble_targets=FitTargets(barrier_kj_mol=2.0, reaction_free_energy_kj_mol=0.0),
            scan=FitScanSettings(
                delta_alpha_min_kj_mol=-2.0,
                delta_alpha_max_kj_mol=2.0,
                delta_alpha_samples=5,
                h12_min_kj_mol=0.0,
                h12_max_kj_mol=4.0,
                h12_samples=3,
            ),
        ),
    )
    result = fit_ensemble_parameters(config)
    assert result.n_candidates == 15
    assert (tmp_path / "fitting" / "ensemble_fit.json").exists()
    assert (tmp_path / "fitting" / "fit_scan.csv").exists()
    assert result.objective_value >= 0.0
