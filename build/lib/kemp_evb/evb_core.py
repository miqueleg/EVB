from __future__ import annotations

from .evb import EVBHamiltonian, EVBParameters, EVBReferenceFit, EVBResult

KJMOL = "kJ/mol"
NM = "nm"
PS = "ps"


def evb_energy(energy1_kj_mol: float, energy2_kj_mol: float, delta_alpha_kj_mol: float, h12_kj_mol: float) -> float:
    energy, _, _ = EVBHamiltonian(EVBParameters(delta_alpha_kj_mol, h12_kj_mol)).lower_eigenvalue(
        energy1_kj_mol,
        energy2_kj_mol,
    )
    return float(energy)


def evb_weights(energy1_kj_mol: float, energy2_kj_mol: float, delta_alpha_kj_mol: float, h12_kj_mol: float) -> tuple[float, float]:
    _, weight1, weight2 = EVBHamiltonian(EVBParameters(delta_alpha_kj_mol, h12_kj_mol)).lower_eigenvalue(
        energy1_kj_mol,
        energy2_kj_mol,
    )
    return float(weight1), float(weight2)


def diabatic_gap(energy1_kj_mol: float, energy2_kj_mol: float, delta_alpha_kj_mol: float) -> float:
    return float(energy1_kj_mol - energy2_kj_mol - delta_alpha_kj_mol)


__all__ = [
    "EVBHamiltonian",
    "EVBParameters",
    "EVBReferenceFit",
    "EVBResult",
    "KJMOL",
    "NM",
    "PS",
    "diabatic_gap",
    "evb_energy",
    "evb_weights",
]
