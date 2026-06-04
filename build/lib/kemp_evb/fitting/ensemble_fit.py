from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..analysis.barrier import BarrierEstimate, estimate_barrier
from ..analysis.pmf import PMFPoint
from ..config import EVBConfig, FitTargets
from ..evb import EVBHamiltonian, EVBParameters
from ..io import write_json

_R_KJ_MOL_K = 0.008314462618


@dataclass(slots=True)
class FitEvaluation:
    delta_alpha_kj_mol: float
    h12_kj_mol: float
    objective_value: float
    barrier_forward_kj_mol: float | None
    reaction_free_energy_kj_mol: float | None


@dataclass(slots=True)
class EnsembleFitResult:
    parameters: EVBParameters
    objective_value: float
    barrier_forward_kj_mol: float | None
    reaction_free_energy_kj_mol: float | None
    n_candidates: int


def fit_ensemble_parameters(config: EVBConfig) -> EnsembleFitResult:
    targets = config.fit.ensemble_targets
    frames = load_frame_data(config.output_dir)
    delta_grid = np.linspace(
        config.fit.scan.delta_alpha_min_kj_mol,
        config.fit.scan.delta_alpha_max_kj_mol,
        config.fit.scan.delta_alpha_samples,
    )
    h12_grid = np.linspace(
        config.fit.scan.h12_min_kj_mol,
        config.fit.scan.h12_max_kj_mol,
        config.fit.scan.h12_samples,
    )
    evaluations: list[FitEvaluation] = []
    best: FitEvaluation | None = None
    for delta_alpha in delta_grid:
        for h12 in h12_grid:
            barrier = evaluate_candidate_barrier(config, frames, float(delta_alpha), float(h12))
            objective = objective_value(barrier, targets)
            evaluation = FitEvaluation(
                delta_alpha_kj_mol=float(delta_alpha),
                h12_kj_mol=float(h12),
                objective_value=objective,
                barrier_forward_kj_mol=barrier.barrier_forward_kj_mol,
                reaction_free_energy_kj_mol=barrier.reaction_free_energy_kj_mol,
            )
            evaluations.append(evaluation)
            if best is None or evaluation.objective_value < best.objective_value:
                best = evaluation
    if best is None:
        raise ValueError("No ensemble-fit candidates were evaluated.")

    result = EnsembleFitResult(
        parameters=EVBParameters(delta_alpha=best.delta_alpha_kj_mol, h12=best.h12_kj_mol),
        objective_value=best.objective_value,
        barrier_forward_kj_mol=best.barrier_forward_kj_mol,
        reaction_free_energy_kj_mol=best.reaction_free_energy_kj_mol,
        n_candidates=len(evaluations),
    )
    output_dir = Path(config.output_dir) / "fitting"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "ensemble_fit.json", result)
    _write_scan_csv(output_dir / "fit_scan.csv", evaluations)
    return result


def load_frame_data(output_dir: str | Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, float]] = []
    for csv_path in sorted((Path(output_dir) / "windows").glob("w*/production_observables.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "lambda": float(row["lambda"]),
                        "E1": float(row["E1_kj_mol"]),
                        "E2": float(row["E2_kj_mol"]),
                    }
                )
    if not rows:
        raise ValueError(f"No window observables found under {Path(output_dir) / 'windows'}")
    return {key: np.asarray([row[key] for row in rows], dtype=float) for key in ("lambda", "E1", "E2")}


def evaluate_candidate_barrier(config: EVBConfig, frames: dict[str, np.ndarray], delta_alpha: float, h12: float) -> BarrierEstimate:
    params = EVBParameters(delta_alpha=delta_alpha, h12=h12)
    ham = EVBHamiltonian(params)
    source_delta = config.evb_parameters.delta_alpha or 0.0
    temperature = config.analysis.pmf.temperature_k
    beta_rt = _R_KJ_MOL_K * temperature
    source_energy = (1.0 - frames["lambda"]) * frames["E1"] + frames["lambda"] * (frames["E2"] + source_delta)
    evb_energies = np.empty_like(frames["E1"])
    shifted_gap = frames["E1"] - (frames["E2"] + delta_alpha)
    for index, (e1, e2) in enumerate(zip(frames["E1"], frames["E2"])):
        evb_energies[index], _, _ = ham.lower_eigenvalue(float(e1), float(e2))
    weights = np.exp(-(evb_energies - source_energy) / beta_rt)
    pmf = _build_weighted_pmf(config, shifted_gap, weights)
    return estimate_barrier(config, pmf)


def objective_value(barrier: BarrierEstimate, targets: FitTargets) -> float:
    value = 0.0
    count = 0
    if targets.barrier_kj_mol is not None:
        estimate = barrier.barrier_forward_kj_mol
        if estimate is None:
            return float("inf")
        value += (estimate - targets.barrier_kj_mol) ** 2
        count += 1
    if targets.reaction_free_energy_kj_mol is not None:
        estimate = barrier.reaction_free_energy_kj_mol
        if estimate is None:
            return float("inf")
        value += (estimate - targets.reaction_free_energy_kj_mol) ** 2
        count += 1
    if count == 0:
        return abs(barrier.barrier_forward_kj_mol or 0.0) + abs(barrier.reaction_free_energy_kj_mol or 0.0)
    return value / count


def _build_weighted_pmf(config: EVBConfig, shifted_gap: np.ndarray, weights: np.ndarray) -> list[PMFPoint]:
    edges = np.linspace(
        config.analysis.histogram.bin_min_kj_mol,
        config.analysis.histogram.bin_max_kj_mol,
        config.analysis.histogram.n_bins + 1,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    weighted_counts, _ = np.histogram(shifted_gap, bins=edges, weights=weights)
    raw_counts, _ = np.histogram(shifted_gap, bins=edges)
    total = weighted_counts.sum()
    probabilities = weighted_counts / total if total > 0 else np.zeros_like(weighted_counts, dtype=float)
    free_energy = np.full_like(probabilities, np.nan, dtype=float)
    mask = probabilities > 0.0
    beta_rt = _R_KJ_MOL_K * config.analysis.pmf.temperature_k
    free_energy[mask] = -beta_rt * np.log(probabilities[mask])
    if np.any(mask):
        if config.analysis.barrier.reactant_region is not None:
            lo, hi = config.analysis.barrier.reactant_region
            zero_mask = mask & (centers >= lo) & (centers <= hi)
            zero = np.nanmin(free_energy[zero_mask]) if np.any(zero_mask) else np.nanmin(free_energy[mask])
        else:
            zero = np.nanmin(free_energy[mask])
        free_energy[mask] -= zero
    return [
        PMFPoint(
            gap_kj_mol=float(center),
            free_energy_kj_mol=None if np.isnan(fe) else float(fe),
            probability=float(prob),
            counts=int(count),
        )
        for center, fe, prob, count in zip(centers, free_energy, probabilities, raw_counts)
    ]


def _write_scan_csv(path: Path, evaluations: list[FitEvaluation]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "delta_alpha_kj_mol",
                "h12_kj_mol",
                "objective_value",
                "barrier_forward_kj_mol",
                "reaction_free_energy_kj_mol",
            ]
        )
        for evaluation in evaluations:
            writer.writerow(
                [
                    evaluation.delta_alpha_kj_mol,
                    evaluation.h12_kj_mol,
                    evaluation.objective_value,
                    evaluation.barrier_forward_kj_mol,
                    evaluation.reaction_free_energy_kj_mol,
                ]
            )
