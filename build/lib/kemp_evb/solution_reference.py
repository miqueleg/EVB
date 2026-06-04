from __future__ import annotations

from dataclasses import asdict, dataclass

REACTANT_COMPLEX_SMILES = (
    "[O:1]=[N+:2]([O-:3])[c:4]1[cH:5][cH:6][c:7]2[o:8][n:9][cH:10][c:11]2[cH:12]1."
    "[CH3:13][C:14](=[O:15])[O-:16]"
)

PRODUCT_COMPLEX_SMILES = (
    "[O:1]=[N+:2]([O-:3])[c:4]1[cH:5][cH:6][c:7]([O-:8])[c:11]([C:10]#[N:9])[cH:12]1."
    "[CH3:13][C:14](=[O:15])[OH:16]"
)


@dataclass(slots=True)
class SolutionReferenceTargets:
    substrate: str
    solvent: str
    base: str
    barrier_kcal_mol: float
    barrier_source: str
    barrier_note: str
    secondary_barrier_kcal_mol: float | None = None
    secondary_source: str | None = None
    secondary_note: str | None = None
    rate_constant_m_inv_s_inv: float | None = None
    rate_source: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


REFERENCE_TARGETS = SolutionReferenceTargets(
    substrate="5-nitro-1,2-benzisoxazole",
    solvent="water",
    base="acetate",
    barrier_kcal_mol=23.8,
    barrier_source="J. Phys. Chem. A 2020, search-result snippet for acetate + 5-nitrobenzisoxazole in water",
    barrier_note=(
        "Use this as the primary aqueous reference for 5-nitrobenzisoxazole with acetate. "
        "This is the same substrate requested here."
    ),
    secondary_barrier_kcal_mol=21.2,
    secondary_source="JACS 2008 / PMC2680199 Table 1 reference reaction with water cage (Asp/Glu surrogate)",
    secondary_note=(
        "Related aqueous EVB-style reference used in Kemp enzyme benchmarking, but not the same "
        "acetate/5-nitrobenzisoxazole pairing."
    ),
    rate_constant_m_inv_s_inv=5.8e-5,
    rate_source="JACS 2008 / PMC2680199 Table 1 cited experimental second-order rate constant in water",
)
