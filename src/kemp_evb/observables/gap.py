from __future__ import annotations

from ..evb import EVBResult
from ..types import GapSample


def compute_energy_gap(energy1_kj_mol: float, energy2_kj_mol: float, delta_alpha_kj_mol: float = 0.0) -> tuple[float, float]:
    delta_e = energy1_kj_mol - energy2_kj_mol
    shifted = energy1_kj_mol - (energy2_kj_mol + delta_alpha_kj_mol)
    return float(delta_e), float(shifted)


def make_gap_sample(result: EVBResult, *, frame: int, step: int, time_ps: float, delta_alpha_kj_mol: float) -> GapSample:
    delta_e, shifted = compute_energy_gap(result.energy1, result.energy2, delta_alpha_kj_mol)
    return GapSample(
        frame=frame,
        step=step,
        time_ps=time_ps,
        energy1_kj_mol=result.energy1,
        energy2_kj_mol=result.energy2,
        delta_e_kj_mol=delta_e,
        delta_e_shifted_kj_mol=shifted,
        evb_energy_kj_mol=result.evb_energy,
        weight1=result.weight1,
        weight2=result.weight2,
    )
