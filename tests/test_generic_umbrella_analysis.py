from __future__ import annotations

from pathlib import Path

from kemp_evb.umbrella_analysis import analyze_umbrella_outputs, calculate_barrier


def test_barrier_calculation_from_synthetic_pmf():
    result = calculate_barrier([-10.0, 0.0, 10.0], [0.0, 5.0, 1.0])

    assert result["barrier_from_left_kcal"] == 5.0
    assert result["barrier_from_right_kcal"] == 4.0


def test_analyze_umbrella_observables(tmp_path: Path):
    obs = tmp_path / "observables.csv"
    obs.write_text("step,shifted_gap\n0,-10\n1,0\n2,10\n3,0\n", encoding="utf-8")

    summary = analyze_umbrella_outputs([obs], tmp_path / "analysis")

    assert summary["n_samples"] == 4
    assert (tmp_path / "analysis" / "pmf.csv").exists()
