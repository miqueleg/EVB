from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..config import EVBConfig
from ..io import write_json
from .barrier import BarrierEstimate, estimate_barrier
from .histograms import build_gap_histograms, load_window_observables
from .overlap import compute_window_overlap_matrix
from .pmf import build_gap_pmf
from .standard_plots import write_standard_analysis_plots
from .statistics import compute_window_statistics


def build_analysis_report(config: EVBConfig) -> dict:
    window_data = load_window_observables(config.output_dir, shifted=config.observables.gap.shifted)
    region_info = _resolve_barrier_regions(config)
    edges, histograms = build_gap_histograms(config, window_data)
    pmf = build_gap_pmf(config, window_data)
    overlap = compute_window_overlap_matrix(config, window_data)
    statistics = compute_window_statistics(config, window_data)
    production_pmf = config.sampling.mode in {"mapping", "gap_umbrella"}
    barrier = estimate_barrier(config, pmf) if production_pmf else BarrierEstimate(None, None, None, None, None, None, None, None)
    barrier_warnings = list(region_info["warnings"])
    if not production_pmf:
        barrier_warnings.append("Sampling mode uses a pooled diagnostic histogram; no production EVB barrier was estimated.")
    if barrier.barrier_forward_kj_mol is None:
        barrier_warnings.append("No production barrier estimate was written because reactant/product regions were not safely defined or sampled.")
    report = {
        "n_windows": len(window_data),
        "n_frames_total": sum(window.n_frames for window in window_data),
        "gap_shifted": config.observables.gap.shifted,
        "tracked_reaction_coordinates": [definition.name for definition in config.observables.reaction_coordinates],
        "histogram_edges_kj_mol": edges.tolist(),
        "sampling_mode": config.sampling.mode,
        "pmf_method": _pmf_method_name(config),
        "pmf_limitations": _pmf_limitations(config),
        "barrier_region_source": region_info["source"],
        "barrier_regions_used": {
            "reactant_region": config.analysis.barrier.reactant_region,
            "product_region": config.analysis.barrier.product_region,
        },
        "barrier_warnings": barrier_warnings,
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
    report["standard_plots"] = write_standard_analysis_plots(config, output_dir)
    write_json(output_dir / "analysis_report.json", report)
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


def _resolve_barrier_regions(config: EVBConfig) -> dict:
    if config.analysis.barrier.reactant_region is not None and config.analysis.barrier.product_region is not None:
        return {"source": "explicit_config", "warnings": []}
    report_path = Path(config.output_dir) / "analysis" / "evb_reference_fit_from_irc.json"
    if config.analysis.barrier.derive_regions_from_irc or report_path.exists():
        if report_path.exists():
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                role_energies = payload.get("role_energies", {})
                rc_gap = float(role_energies["RC"]["gap_shifted_kj_mol"])
                product_gap = float(role_energies["PROD"]["gap_shifted_kj_mol"])
                width = _endpoint_region_width(config, rc_gap, product_gap)
                config.analysis.barrier.reactant_region = (rc_gap - width, rc_gap + width)
                config.analysis.barrier.product_region = (product_gap - width, product_gap + width)
                return {
                    "source": "derived_from_irc_fit_report",
                    "warnings": [
                        "Barrier regions were derived from canonical RC/PROD shifted gaps in evb_reference_fit_from_irc.json; verify overlap before production use."
                    ],
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return {
                    "source": "failed_irc_derivation",
                    "warnings": [f"Could not derive barrier regions from IRC fit report: {exc}"],
                }
        if config.analysis.barrier.derive_regions_from_irc:
            return {
                "source": "missing_irc_fit_report",
                "warnings": ["analysis.barrier.derive_regions_from_irc is true, but no IRC fit report was found."],
            }
    if config.analysis.barrier.allow_sign_fallback:
        return {
            "source": "sign_based_user_allowed",
            "warnings": ["Using sign-based gap regions because analysis.barrier.allow_sign_fallback is true; this is unsafe for production EVB reports."],
        }
    return {
        "source": "undefined",
        "warnings": [
            "No explicit barrier regions were configured and sign-based gap regions are disabled. Set analysis.barrier.reactant_region/product_region or derive_regions_from_irc."
        ],
    }


def _endpoint_region_width(config: EVBConfig, rc_gap: float, product_gap: float) -> float:
    histogram = config.analysis.histogram
    bin_width = abs(histogram.bin_max_kj_mol - histogram.bin_min_kj_mol) / max(histogram.n_bins, 1)
    endpoint_span = abs(product_gap - rc_gap)
    return max(2.0 * bin_width, min(50.0, 0.1 * endpoint_span if endpoint_span > 0.0 else 50.0))
