from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class EVBParameters:
    delta_alpha: float = 0.0
    h12: float = 0.0

    def to_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)


@dataclass(slots=True)
class EVBResult:
    energy1: float
    energy2: float
    e2_shifted: float
    evb_energy: float
    weight1: float
    weight2: float
    forces: np.ndarray


@dataclass(slots=True)
class EVBReferenceFit:
    parameters: EVBParameters
    objective_value: float
    fitted_reaction_free_energy: float
    fitted_barrier: float
    ts_weight2: float
    rc_evb_energy: float
    prod_evb_energy: float
    ts_evb_energy: float


class EVBHamiltonian:
    def __init__(self, parameters: EVBParameters):
        self.parameters = parameters

    def gap(self, energy1: float, energy2: float) -> float:
        return float(energy1 - energy2 - self.parameters.delta_alpha)

    def lower_eigenvalue(self, energy1: float, energy2: float) -> tuple[float, float, float]:
        shifted2 = energy2 + self.parameters.delta_alpha
        gap = self.gap(energy1, energy2)
        root = np.sqrt(0.25 * gap * gap + self.parameters.h12 * self.parameters.h12)
        evb_energy = 0.5 * (energy1 + shifted2) - root
        if root == 0.0:
            return evb_energy, 0.5, 0.5
        d_ev_d_e1 = 0.5 - gap / (4.0 * root)
        d_ev_d_e2 = 0.5 + gap / (4.0 * root)
        return evb_energy, d_ev_d_e1, d_ev_d_e2

    def combine(self, energy1: float, forces1: np.ndarray, energy2: float, forces2: np.ndarray) -> EVBResult:
        evb_energy, weight1, weight2 = self.lower_eigenvalue(energy1, energy2)
        forces = weight1 * forces1 + weight2 * forces2
        return EVBResult(
            energy1=energy1,
            energy2=energy2,
            e2_shifted=energy2 + self.parameters.delta_alpha,
            evb_energy=evb_energy,
            weight1=weight1,
            weight2=weight2,
            forces=forces,
        )


def solve_delta_alpha(
    e11: float,
    e21: float,
    e12: float,
    e22: float,
    target_difference: float,
    h12: float = 0.0,
    grid_min: float = -400.0,
    grid_max: float = 400.0,
    samples: int = 4001,
) -> float:
    grid = np.linspace(grid_min, grid_max, samples)
    best_delta = 0.0
    best_error = float("inf")
    for delta_alpha in grid:
        ham = EVBHamiltonian(EVBParameters(delta_alpha=delta_alpha, h12=h12))
        evb1, _, _ = ham.lower_eigenvalue(e11, e21)
        evb2, _, _ = ham.lower_eigenvalue(e12, e22)
        error = abs((evb2 - evb1) - target_difference)
        if error < best_error:
            best_error = error
            best_delta = float(delta_alpha)
    return best_delta


def solve_h12_from_relative_barrier(
    e1_min1: float,
    e2_min1: float,
    e1_ts: float,
    e2_ts: float,
    delta_alpha: float,
    target_barrier: float,
    grid_min: float = 0.0,
    grid_max: float = 400.0,
    samples: int = 4001,
) -> float:
    grid = np.linspace(grid_min, grid_max, samples)
    best_h12 = 0.0
    best_error = float("inf")
    for h12 in grid:
        ham = EVBHamiltonian(EVBParameters(delta_alpha=delta_alpha, h12=float(h12)))
        evb_min1, _, _ = ham.lower_eigenvalue(e1_min1, e2_min1)
        evb_ts, _, _ = ham.lower_eigenvalue(e1_ts, e2_ts)
        error = abs((evb_ts - evb_min1) - target_barrier)
        if error < best_error:
            best_error = error
            best_h12 = float(h12)
    return best_h12


def calibrate_evb_parameters(
    e_mm_min1_state1: float,
    e_mm_min1_state2: float,
    e_mm_min2_state1: float,
    e_mm_min2_state2: float,
    e_mm_ts_state1: float,
    e_mm_ts_state2: float,
    e_qmmm_min1: float,
    e_qmmm_min2: float,
    e_qmmm_ts: float,
) -> EVBParameters:
    fit = fit_evb_reference_profile(
        e_mm_min1_state1=e_mm_min1_state1,
        e_mm_min1_state2=e_mm_min1_state2,
        e_mm_min2_state1=e_mm_min2_state1,
        e_mm_min2_state2=e_mm_min2_state2,
        e_mm_ts_state1=e_mm_ts_state1,
        e_mm_ts_state2=e_mm_ts_state2,
        e_qmmm_min1=e_qmmm_min1,
        e_qmmm_min2=e_qmmm_min2,
        e_qmmm_ts=e_qmmm_ts,
    )
    return fit.parameters


def fit_evb_reference_profile(
    e_mm_min1_state1: float,
    e_mm_min1_state2: float,
    e_mm_min2_state1: float,
    e_mm_min2_state2: float,
    e_mm_ts_state1: float,
    e_mm_ts_state2: float,
    e_qmmm_min1: float,
    e_qmmm_min2: float,
    e_qmmm_ts: float,
    ts_mixing_weight: float = 1.0,
    levels: int = 5,
    samples_per_axis: int = 81,
) -> EVBReferenceFit:
    target_difference = e_qmmm_min2 - e_qmmm_min1
    target_barrier = e_qmmm_ts - e_qmmm_min1
    gaps = np.asarray(
        [
            e_mm_min1_state1 - e_mm_min1_state2,
            e_mm_min2_state1 - e_mm_min2_state2,
            e_mm_ts_state1 - e_mm_ts_state2,
        ],
        dtype=float,
    )
    delta_center = float(gaps[2])
    delta_span = max(float(np.max(np.abs(gaps))), abs(target_difference), abs(target_barrier), 1000.0)
    h12_span = max(0.5 * float(np.max(np.abs(gaps))), abs(target_barrier), 1000.0)
    delta_lo = delta_center - delta_span
    delta_hi = delta_center + delta_span
    h12_lo = 0.0
    h12_hi = h12_span

    best: EVBReferenceFit | None = None
    for _ in range(levels):
        delta_grid = np.linspace(delta_lo, delta_hi, samples_per_axis)
        h12_grid = np.linspace(h12_lo, h12_hi, samples_per_axis)
        for delta_alpha in delta_grid:
            for h12 in h12_grid:
                candidate = _evaluate_reference_fit(
                    float(delta_alpha),
                    float(h12),
                    e_mm_min1_state1,
                    e_mm_min1_state2,
                    e_mm_min2_state1,
                    e_mm_min2_state2,
                    e_mm_ts_state1,
                    e_mm_ts_state2,
                    target_difference,
                    target_barrier,
                    ts_mixing_weight,
                )
                if best is None or candidate.objective_value < best.objective_value:
                    best = candidate
        assert best is not None
        delta_width = max((delta_hi - delta_lo) / 8.0, 1.0)
        h12_width = max((h12_hi - h12_lo) / 8.0, 1.0)
        delta_lo = best.parameters.delta_alpha - delta_width
        delta_hi = best.parameters.delta_alpha + delta_width
        h12_lo = max(0.0, best.parameters.h12 - h12_width)
        h12_hi = best.parameters.h12 + h12_width
    return best


def _evaluate_reference_fit(
    delta_alpha: float,
    h12: float,
    e_mm_min1_state1: float,
    e_mm_min1_state2: float,
    e_mm_min2_state1: float,
    e_mm_min2_state2: float,
    e_mm_ts_state1: float,
    e_mm_ts_state2: float,
    target_difference: float,
    target_barrier: float,
    ts_mixing_weight: float,
) -> EVBReferenceFit:
    ham = EVBHamiltonian(EVBParameters(delta_alpha=delta_alpha, h12=h12))
    rc_evb, _, _ = ham.lower_eigenvalue(e_mm_min1_state1, e_mm_min1_state2)
    prod_evb, _, _ = ham.lower_eigenvalue(e_mm_min2_state1, e_mm_min2_state2)
    ts_evb, _, ts_w2 = ham.lower_eigenvalue(e_mm_ts_state1, e_mm_ts_state2)
    fitted_difference = prod_evb - rc_evb
    fitted_barrier = ts_evb - rc_evb
    mixing_penalty = (ts_w2 - 0.5) ** 2
    objective = (
        (fitted_difference - target_difference) ** 2
        + (fitted_barrier - target_barrier) ** 2
        + ts_mixing_weight * mixing_penalty
    )
    return EVBReferenceFit(
        parameters=EVBParameters(delta_alpha=delta_alpha, h12=h12),
        objective_value=float(objective),
        fitted_reaction_free_energy=float(fitted_difference),
        fitted_barrier=float(fitted_barrier),
        ts_weight2=float(ts_w2),
        rc_evb_energy=float(rc_evb),
        prod_evb_energy=float(prod_evb),
        ts_evb_energy=float(ts_evb),
    )
