from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EVBConfig
from .histograms import WindowObservableData, histogram_centers, histogram_edges

_R_KJ_MOL_K = 0.008314462618


@dataclass(slots=True)
class PMFPoint:
    gap_kj_mol: float
    free_energy_kj_mol: float | None
    probability: float
    counts: int


def build_gap_pmf(config: EVBConfig, window_data: list[WindowObservableData]) -> list[PMFPoint]:
    if config.sampling.mode == "gap_umbrella":
        return _build_gap_pmf_wham(config, window_data)
    if config.sampling.mode == "mapping":
        return _build_gap_pmf_mapping_mbar(config, window_data)
    return _build_gap_pmf_pooled(config, window_data)


def _build_gap_pmf_pooled(config: EVBConfig, window_data: list[WindowObservableData]) -> list[PMFPoint]:
    edges = histogram_edges(config)
    centers = histogram_centers(config)
    all_values = np.concatenate([window.values_kj_mol for window in window_data])
    counts, _ = np.histogram(all_values, bins=edges)
    free_energy, probabilities = _free_energy_from_counts(config, centers, counts)
    return [
        PMFPoint(
            gap_kj_mol=float(center),
            free_energy_kj_mol=None if np.isnan(fe) else float(fe),
            probability=float(prob),
            counts=int(count),
        )
        for center, fe, prob, count in zip(centers, free_energy, probabilities, counts)
    ]


def _build_gap_pmf_wham(config: EVBConfig, window_data: list[WindowObservableData]) -> list[PMFPoint]:
    edges = histogram_edges(config)
    centers = histogram_centers(config)
    beta = 1.0 / (_R_KJ_MOL_K * config.analysis.pmf.temperature_k)
    force_constant = config.sampling.windows.gap_umbrella.force_constant_kj_mol2
    if force_constant is None:
        raise ValueError("Gap umbrella PMF reconstruction requires config.sampling.windows.gap_umbrella.force_constant_kj_mol2.")
    centers_by_window: list[float] = []
    histogram_rows: list[np.ndarray] = []
    sample_counts: list[int] = []
    for window in window_data:
        if window.gap_center_kj_mol is None:
            raise ValueError("Gap umbrella PMF reconstruction requires gap_center_kj_mol in the window observable data.")
        counts, _ = np.histogram(window.values_kj_mol, bins=edges)
        centers_by_window.append(window.gap_center_kj_mol)
        histogram_rows.append(counts.astype(float))
        sample_counts.append(window.n_frames)

    counts_matrix = np.asarray(histogram_rows, dtype=float)
    n_per_window = np.asarray(sample_counts, dtype=float)
    bias_matrix = np.asarray(
        [
            0.5 * force_constant * (centers - window_center) ** 2
            for window_center in centers_by_window
        ],
        dtype=float,
    )
    total_counts_per_bin = counts_matrix.sum(axis=0)
    free_offsets = np.zeros(len(window_data), dtype=float)
    probabilities = np.full(len(centers), 1.0 / max(len(centers), 1), dtype=float)
    probabilities /= probabilities.sum()

    for _ in range(10000):
        denominator = np.zeros(len(centers), dtype=float)
        for i in range(len(window_data)):
            denominator += n_per_window[i] * np.exp(beta * free_offsets[i] - beta * bias_matrix[i])
        mask = denominator > 0.0
        new_probabilities = np.zeros_like(probabilities)
        new_probabilities[mask] = total_counts_per_bin[mask] / denominator[mask]
        total_probability = new_probabilities.sum()
        if total_probability > 0.0:
            new_probabilities /= total_probability

        new_offsets = np.zeros_like(free_offsets)
        for i in range(len(window_data)):
            weighted = np.sum(new_probabilities * np.exp(-beta * bias_matrix[i]))
            if weighted > 0.0:
                new_offsets[i] = -(1.0 / beta) * np.log(weighted)
        if np.max(np.abs(new_offsets - free_offsets)) < 1.0e-6 and np.max(np.abs(new_probabilities - probabilities)) < 1.0e-10:
            probabilities = new_probabilities
            free_offsets = new_offsets
            break
        probabilities = new_probabilities
        free_offsets = new_offsets

    counts = total_counts_per_bin.astype(int)
    total = counts.sum()
    if total > 0:
        probabilities = probabilities * (counts > 0)
        probabilities /= probabilities.sum()
    free_energy = _free_energy_from_probabilities(config, centers, probabilities)
    mask = probabilities > 0.0
    return [
        PMFPoint(
            gap_kj_mol=float(center),
            free_energy_kj_mol=None if np.isnan(fe) else float(fe),
            probability=float(prob),
            counts=int(count),
        )
        for center, fe, prob, count in zip(centers, free_energy, probabilities, counts)
    ]


def _build_gap_pmf_mapping_mbar(config: EVBConfig, window_data: list[WindowObservableData]) -> list[PMFPoint]:
    if any(window.lambda_value is None for window in window_data):
        return _build_gap_pmf_pooled(config, window_data)
    if any(window.energy1_kj_mol is None or window.energy2_kj_mol is None or window.evb_energy_kj_mol is None for window in window_data):
        return _build_gap_pmf_pooled(config, window_data)

    edges = histogram_edges(config)
    centers = histogram_centers(config)
    beta = 1.0 / (_R_KJ_MOL_K * config.analysis.pmf.temperature_k)
    delta_alpha = config.evb_parameters.delta_alpha or 0.0

    lambda_values = np.asarray([float(window.lambda_value) for window in window_data], dtype=float)
    sample_counts = np.asarray([window.n_frames for window in window_data], dtype=float)
    gaps = np.concatenate([window.values_kj_mol for window in window_data])
    energy1 = np.concatenate([window.energy1_kj_mol for window in window_data if window.energy1_kj_mol is not None])
    energy2 = np.concatenate([window.energy2_kj_mol for window in window_data if window.energy2_kj_mol is not None])
    evb_energy = np.concatenate([window.evb_energy_kj_mol for window in window_data if window.evb_energy_kj_mol is not None])

    mapped_energies = np.asarray(
        [
            (1.0 - lam) * energy1 + lam * (energy2 + delta_alpha)
            for lam in lambda_values
        ],
        dtype=float,
    )
    reduced_potentials = beta * mapped_energies
    target_reduced = beta * evb_energy

    free_offsets = np.zeros(len(window_data), dtype=float)
    for _ in range(20000):
        log_terms = np.log(sample_counts)[:, None] + free_offsets[:, None] - reduced_potentials
        log_denominator = _logsumexp(log_terms, axis=0)
        new_offsets = -_logsumexp(-reduced_potentials - log_denominator[None, :], axis=1)
        new_offsets -= new_offsets[0]
        if np.max(np.abs(new_offsets - free_offsets)) < 1.0e-10:
            free_offsets = new_offsets
            break
        free_offsets = new_offsets

    log_terms = np.log(sample_counts)[:, None] + free_offsets[:, None] - reduced_potentials
    log_denominator = _logsumexp(log_terms, axis=0)
    log_weights = -target_reduced - log_denominator
    log_weights -= _logsumexp(log_weights)
    weights = np.exp(log_weights)

    counts, _ = np.histogram(gaps, bins=edges)
    weighted_counts, _ = np.histogram(gaps, bins=edges, weights=weights)
    probabilities = np.asarray(weighted_counts, dtype=float)
    total = probabilities.sum()
    if total > 0.0:
        probabilities /= total
    free_energy = _free_energy_from_probabilities(config, centers, probabilities)
    return [
        PMFPoint(
            gap_kj_mol=float(center),
            free_energy_kj_mol=None if np.isnan(fe) else float(fe),
            probability=float(prob),
            counts=int(count),
        )
        for center, fe, prob, count in zip(centers, free_energy, probabilities, counts)
    ]


def _free_energy_from_counts(config: EVBConfig, centers: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = counts.sum()
    probabilities = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
    free_energy = _free_energy_from_probabilities(config, centers, probabilities)
    return free_energy, probabilities


def _free_energy_from_probabilities(config: EVBConfig, centers: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    temperature = config.analysis.pmf.temperature_k
    rt = _R_KJ_MOL_K * temperature
    free_energy = np.full_like(probabilities, np.nan, dtype=float)
    mask = probabilities > 0.0
    free_energy[mask] = -rt * np.log(probabilities[mask])
    if np.any(mask):
        if config.analysis.pmf.zero_mode == "reactant_min" and config.analysis.barrier.reactant_region is not None:
            lo, hi = config.analysis.barrier.reactant_region
            region_mask = mask & (centers >= lo) & (centers <= hi)
            zero_value = np.nanmin(free_energy[region_mask]) if np.any(region_mask) else np.nanmin(free_energy[mask])
        else:
            zero_value = np.nanmin(free_energy[mask])
        free_energy[mask] -= zero_value
    return free_energy


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    if axis is None:
        maximum = float(np.max(values))
        return maximum + np.log(np.sum(np.exp(values - maximum)))
    maxima = np.max(values, axis=axis, keepdims=True)
    shifted = values - maxima
    return np.squeeze(maxima, axis=axis) + np.log(np.sum(np.exp(shifted), axis=axis))
