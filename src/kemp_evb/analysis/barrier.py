from __future__ import annotations

from dataclasses import dataclass

from ..config import EVBConfig
from .pmf import PMFPoint


@dataclass(slots=True)
class BarrierEstimate:
    reactant_gap_kj_mol: float | None
    reactant_free_energy_kj_mol: float | None
    product_gap_kj_mol: float | None
    product_free_energy_kj_mol: float | None
    ts_gap_kj_mol: float | None
    ts_free_energy_kj_mol: float | None
    barrier_forward_kj_mol: float | None
    reaction_free_energy_kj_mol: float | None


def estimate_barrier(config: EVBConfig, pmf_points: list[PMFPoint]) -> BarrierEstimate:
    finite_points = [point for point in pmf_points if point.free_energy_kj_mol is not None]
    if not finite_points:
        return BarrierEstimate(None, None, None, None, None, None, None, None)

    reactant_region = config.analysis.barrier.reactant_region
    product_region = config.analysis.barrier.product_region
    reactant_candidates = _region_points(finite_points, reactant_region) if reactant_region else [point for point in finite_points if point.gap_kj_mol <= 0.0]
    product_candidates = _region_points(finite_points, product_region) if product_region else [point for point in finite_points if point.gap_kj_mol >= 0.0]
    if not reactant_candidates:
        reactant_candidates = finite_points
    if not product_candidates:
        product_candidates = finite_points

    reactant = min(reactant_candidates, key=lambda point: point.free_energy_kj_mol)
    product = min(product_candidates, key=lambda point: point.free_energy_kj_mol)
    lower_gap, upper_gap = sorted((reactant.gap_kj_mol, product.gap_kj_mol))
    between = [point for point in finite_points if lower_gap <= point.gap_kj_mol <= upper_gap]
    ts = max(between, key=lambda point: point.free_energy_kj_mol) if between else max(finite_points, key=lambda point: point.free_energy_kj_mol)
    return BarrierEstimate(
        reactant_gap_kj_mol=reactant.gap_kj_mol,
        reactant_free_energy_kj_mol=reactant.free_energy_kj_mol,
        product_gap_kj_mol=product.gap_kj_mol,
        product_free_energy_kj_mol=product.free_energy_kj_mol,
        ts_gap_kj_mol=ts.gap_kj_mol,
        ts_free_energy_kj_mol=ts.free_energy_kj_mol,
        barrier_forward_kj_mol=None if ts.free_energy_kj_mol is None or reactant.free_energy_kj_mol is None else ts.free_energy_kj_mol - reactant.free_energy_kj_mol,
        reaction_free_energy_kj_mol=None if product.free_energy_kj_mol is None or reactant.free_energy_kj_mol is None else product.free_energy_kj_mol - reactant.free_energy_kj_mol,
    )


def _region_points(points: list[PMFPoint], region: tuple[float, float]) -> list[PMFPoint]:
    lo, hi = region
    return [point for point in points if lo <= point.gap_kj_mol <= hi]
