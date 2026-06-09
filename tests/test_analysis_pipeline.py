from __future__ import annotations

import csv
import json
from pathlib import Path

from kemp_evb.analysis import build_analysis_report, build_gap_histograms, build_gap_pmf, compute_window_overlap_matrix, estimate_barrier, load_window_observables
from kemp_evb.config import (
    AnalysisSettings,
    BarrierSettings,
    EVBConfig,
    GapUmbrellaWindows,
    HistogramSettings,
    ObservableSettings,
    PMFSettings,
    ProjectSettings,
    SamplingSettings,
    SamplingWindows,
    SimulationSettings,
    StateFiles,
)


def _write_window_csv(root: Path, window_id: str, lambda_value: float | None, shifted_values: list[float], gap_center_kj_mol: float | None = None) -> None:
    window_dir = root / "windows" / window_id
    window_dir.mkdir(parents=True, exist_ok=True)
    with (window_dir / "production_observables.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["frame", "step", "time_ps", "window_id"]
        if lambda_value is not None:
            header.append("lambda")
        if gap_center_kj_mol is not None:
            header.append("gap_center_kj_mol")
        header.extend(["E1_kj_mol", "E2_kj_mol", "delta_e_kj_mol", "delta_e_shifted_kj_mol", "Eevb_kj_mol", "w1", "w2"])
        writer.writerow(header)
        for frame, value in enumerate(shifted_values):
            row = [frame, frame * 10, frame * 0.01, window_id]
            if lambda_value is not None:
                row.append(lambda_value)
            if gap_center_kj_mol is not None:
                row.append(gap_center_kj_mol)
            row.extend([0.0, 0.0, value, value, 0.0, 0.5, 0.5])
            writer.writerow(row)


def test_analysis_pipeline_builds_report_and_files(tmp_path: Path):
    _write_window_csv(tmp_path, "w000", 0.0, [-120.0, -100.0, -90.0, -40.0])
    _write_window_csv(tmp_path, "w001", 0.5, [-20.0, 0.0, 20.0, 40.0])
    _write_window_csv(tmp_path, "w002", 1.0, [60.0, 80.0, 100.0, 120.0])

    config = EVBConfig(
        state1=StateFiles(prmtop="a", inpcrd="b"),
        state2=StateFiles(prmtop="c", inpcrd="d"),
        output_dir=str(tmp_path),
        project=ProjectSettings(name="analysis-test", output_dir=str(tmp_path)),
        observables=ObservableSettings(),
        simulation=SimulationSettings(),
        analysis=AnalysisSettings(
            histogram=HistogramSettings(bin_min_kj_mol=-150.0, bin_max_kj_mol=150.0, n_bins=15),
            pmf=PMFSettings(temperature_k=300.0, zero_mode="reactant_min"),
            barrier=BarrierSettings(reactant_region=(-150.0, -25.0), product_region=(25.0, 150.0)),
        ),
    )

    windows = load_window_observables(tmp_path, shifted=True)
    _, hist = build_gap_histograms(config, windows)
    pmf = build_gap_pmf(config, windows)
    overlap = compute_window_overlap_matrix(config, windows)
    barrier = estimate_barrier(config, pmf)
    report = build_analysis_report(config)

    assert len(windows) == 3
    assert len(hist) == 3
    assert len(overlap) == 3
    assert barrier.reactant_gap_kj_mol is not None
    assert barrier.product_gap_kj_mol is not None
    assert report["n_windows"] == 3
    assert (tmp_path / "analysis" / "analysis_report.json").exists()
    assert (tmp_path / "analysis" / "gap_histograms.json").exists()
    assert (tmp_path / "analysis" / "barrier_estimate.json").exists()
    assert (tmp_path / "analysis" / "pmf_gap.csv").exists()


def test_gap_umbrella_analysis_builds_wham_pmf(tmp_path: Path):
    _write_window_csv(tmp_path, "u000", None, [-40.0, -35.0, -30.0, -25.0], gap_center_kj_mol=-35.0)
    _write_window_csv(tmp_path, "u001", None, [-25.0, -20.0, -15.0, -10.0], gap_center_kj_mol=-20.0)
    _write_window_csv(tmp_path, "u002", None, [-10.0, -5.0, 0.0, 5.0], gap_center_kj_mol=-5.0)

    config = EVBConfig(
        state1=StateFiles(prmtop="a", inpcrd="b"),
        state2=StateFiles(prmtop="c", inpcrd="d"),
        output_dir=str(tmp_path),
        project=ProjectSettings(name="analysis-umbrella-test", output_dir=str(tmp_path)),
        observables=ObservableSettings(),
        simulation=SimulationSettings(),
        sampling=SamplingSettings(
            mode="gap_umbrella",
            windows=SamplingWindows(gap_umbrella=GapUmbrellaWindows(centers_kj_mol=[-35.0, -20.0, -5.0], force_constant_kj_mol2=0.05)),
        ),
        analysis=AnalysisSettings(
            histogram=HistogramSettings(bin_min_kj_mol=-50.0, bin_max_kj_mol=10.0, n_bins=12),
            pmf=PMFSettings(temperature_k=300.0, zero_mode="reactant_min"),
            barrier=BarrierSettings(reactant_region=(-50.0, -20.0), product_region=(-10.0, 10.0)),
        ),
    )

    windows = load_window_observables(tmp_path, shifted=True)
    pmf = build_gap_pmf(config, windows)
    finite = [point for point in pmf if point.free_energy_kj_mol is not None]
    assert finite
    reactant = [point.free_energy_kj_mol for point in finite if -50.0 <= point.gap_kj_mol <= -20.0]
    assert reactant
    assert min(reactant) == 0.0


def test_mapping_analysis_builds_reweighted_pmf(tmp_path: Path):
    _write_window_csv(tmp_path, "w000", 0.0, [-120.0, -110.0, -105.0, -95.0])
    _write_window_csv(tmp_path, "w001", 0.5, [-20.0, -5.0, 5.0, 20.0])
    _write_window_csv(tmp_path, "w002", 1.0, [90.0, 100.0, 110.0, 120.0])

    config = EVBConfig(
        state1=StateFiles(prmtop="a", inpcrd="b"),
        state2=StateFiles(prmtop="c", inpcrd="d"),
        output_dir=str(tmp_path),
        project=ProjectSettings(name="analysis-mapping-test", output_dir=str(tmp_path)),
        observables=ObservableSettings(),
        simulation=SimulationSettings(),
        sampling=SamplingSettings(mode="mapping"),
        analysis=AnalysisSettings(
            histogram=HistogramSettings(bin_min_kj_mol=-150.0, bin_max_kj_mol=150.0, n_bins=15),
            pmf=PMFSettings(temperature_k=300.0, zero_mode="reactant_min"),
            barrier=BarrierSettings(reactant_region=(-150.0, -50.0), product_region=(50.0, 150.0)),
        ),
    )

    windows = load_window_observables(tmp_path, shifted=True)
    pmf = build_gap_pmf(config, windows)
    finite = [point for point in pmf if point.free_energy_kj_mol is not None]
    assert finite
    assert abs(sum(point.probability for point in pmf) - 1.0) < 1.0e-8


def test_analysis_derives_barrier_regions_from_irc_report(tmp_path: Path):
    _write_window_csv(tmp_path, "u000", None, [-102.0, -100.0, -98.0], gap_center_kj_mol=-100.0)
    _write_window_csv(tmp_path, "u001", None, [98.0, 100.0, 102.0], gap_center_kj_mol=100.0)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "evb_reference_fit_from_irc.json").write_text(
        json.dumps(
            {
                "role_energies": {
                    "RC": {"gap_shifted_kj_mol": -100.0},
                    "PROD": {"gap_shifted_kj_mol": 100.0},
                }
            }
        ),
        encoding="utf-8",
    )
    config = EVBConfig(
        state1=StateFiles(prmtop="a", inpcrd="b"),
        state2=StateFiles(prmtop="c", inpcrd="d"),
        output_dir=str(tmp_path),
        sampling=SamplingSettings(
            mode="gap_umbrella",
            windows=SamplingWindows(gap_umbrella=GapUmbrellaWindows(centers_kj_mol=[-100.0, 100.0], force_constant_kj_mol2=0.01)),
        ),
        analysis=AnalysisSettings(
            histogram=HistogramSettings(bin_min_kj_mol=-150.0, bin_max_kj_mol=150.0, n_bins=30),
            pmf=PMFSettings(temperature_k=300.0, zero_mode="reactant_min"),
            barrier=BarrierSettings(),
        ),
    )

    report = build_analysis_report(config)

    assert report["barrier_region_source"] == "derived_from_irc_fit_report"
    assert config.analysis.barrier.reactant_region is not None
    assert config.analysis.barrier.product_region is not None


def test_analysis_without_regions_does_not_use_sign_fallback(tmp_path: Path):
    _write_window_csv(tmp_path, "u000", None, [-20.0, -10.0, 10.0, 20.0], gap_center_kj_mol=0.0)
    config = EVBConfig(
        state1=StateFiles(prmtop="a", inpcrd="b"),
        state2=StateFiles(prmtop="c", inpcrd="d"),
        output_dir=str(tmp_path),
        sampling=SamplingSettings(
            mode="gap_umbrella",
            windows=SamplingWindows(gap_umbrella=GapUmbrellaWindows(centers_kj_mol=[0.0], force_constant_kj_mol2=0.01)),
        ),
        analysis=AnalysisSettings(
            histogram=HistogramSettings(bin_min_kj_mol=-30.0, bin_max_kj_mol=30.0, n_bins=12),
            barrier=BarrierSettings(),
        ),
    )

    report = build_analysis_report(config)

    assert report["barrier_estimate"]["barrier_forward_kj_mol"] is None
    assert report["barrier_region_source"] == "undefined"


def test_pooled_diagnostic_mode_does_not_report_production_barrier(tmp_path: Path):
    _write_window_csv(tmp_path, "p000", None, [-0.2, 0.0, 0.2], gap_center_kj_mol=None)
    config = EVBConfig(
        state1=StateFiles(prmtop="a", inpcrd="b"),
        state2=StateFiles(prmtop="c", inpcrd="d"),
        output_dir=str(tmp_path),
        sampling=SamplingSettings(mode="proton_transfer_umbrella"),
        analysis=AnalysisSettings(
            histogram=HistogramSettings(bin_min_kj_mol=-1.0, bin_max_kj_mol=1.0, n_bins=10),
            barrier=BarrierSettings(reactant_region=(-1.0, -0.1), product_region=(0.1, 1.0)),
        ),
    )

    report = build_analysis_report(config)

    assert report["pmf_method"] == "diagnostic_pooled_histogram"
    assert report["barrier_estimate"]["barrier_forward_kj_mol"] is None
