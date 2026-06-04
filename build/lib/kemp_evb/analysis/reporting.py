from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..config import EVBConfig
from ..io import write_json
from .barrier import estimate_barrier
from .histograms import build_gap_histograms, load_window_observables
from .overlap import compute_window_overlap_matrix
from .pmf import build_gap_pmf
from .statistics import compute_window_statistics


def build_analysis_report(config: EVBConfig) -> dict:
    window_data = load_window_observables(config.output_dir, shifted=config.observables.gap.shifted)
    edges, histograms = build_gap_histograms(config, window_data)
    pmf = build_gap_pmf(config, window_data)
    overlap = compute_window_overlap_matrix(config, window_data)
    statistics = compute_window_statistics(config, window_data)
    barrier = estimate_barrier(config, pmf)
    report = {
        "n_windows": len(window_data),
        "n_frames_total": sum(window.n_frames for window in window_data),
        "gap_shifted": config.observables.gap.shifted,
        "tracked_reaction_coordinates": [definition.name for definition in config.observables.reaction_coordinates],
        "histogram_edges_kj_mol": edges.tolist(),
        "sampling_mode": config.sampling.mode,
        "pmf_method": _pmf_method_name(config),
        "pmf_limitations": _pmf_limitations(config),
        "windows": [
            {
                "window_id": window.window_id,
                "lambda_value": window.lambda_value,
                "gap_center_kj_mol": window.gap_center_kj_mol,
                "n_frames": window.n_frames,
            }
            for window in window_data
        ],
        "barrier_estimate": asdict(barrier),
        "barrier_uncertainty_kj_mol": None,
    }
    output_dir = Path(config.output_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "analysis_report.json", report)
    write_json(output_dir / "gap_histograms.json", [asdict(item) for item in histograms])
    write_json(output_dir / "window_overlap.json", [asdict(item) for item in overlap])
    write_json(output_dir / "window_statistics.json", [asdict(item) for item in statistics])
    write_json(output_dir / "barrier_estimate.json", asdict(barrier))
    _write_pmf_csv(output_dir / "pmf_gap.csv", pmf)
    return report


def _write_pmf_csv(path: Path, pmf_points) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("gap_kj_mol,free_energy_kj_mol,probability,counts\n")
        for point in pmf_points:
            handle.write(
                f"{point.gap_kj_mol},{'' if point.free_energy_kj_mol is None else point.free_energy_kj_mol},"
                f"{point.probability},{point.counts}\n"
            )


def _pmf_method_name(config: EVBConfig) -> str:
    if config.sampling.mode == "mapping":
        return "mapping_mbar_like_reweighting"
    if config.sampling.mode == "gap_umbrella":
        return "wham_like_histogram_reconstruction"
    return "diagnostic_pooled_histogram"


def _pmf_limitations(config: EVBConfig) -> str:
    if config.sampling.mode in {"mapping", "gap_umbrella"}:
        return "Current implementation is lightweight and should be validated with overlap, blocking, and replicates before reporting barriers."
    return "Non-rigorous diagnostic histogram only."
