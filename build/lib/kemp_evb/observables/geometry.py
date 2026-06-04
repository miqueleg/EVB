from __future__ import annotations

import numpy as np

from ..config import DistanceCVDefinition, ReactionCoordinateDefinition


def compute_distance(positions_nm: np.ndarray, atom1: int, atom2: int) -> float:
    return float(np.linalg.norm(positions_nm[atom1] - positions_nm[atom2]))


def compute_named_distances(positions_nm: np.ndarray, definitions: list[DistanceCVDefinition]) -> dict[str, float]:
    return {definition.name: compute_distance(positions_nm, definition.atom1, definition.atom2) for definition in definitions}


def compute_reaction_coordinate(positions_nm: np.ndarray, definition: ReactionCoordinateDefinition) -> float:
    if definition.kind == "distance":
        return compute_distance(positions_nm, definition.atom1, definition.atom2)
    if definition.kind == "difference_of_distances":
        if definition.atom3 is None:
            raise ValueError(
                f"Reaction coordinate '{definition.name}' of kind difference_of_distances requires atom3."
            )
        return compute_distance(positions_nm, definition.atom1, definition.atom2) - compute_distance(
            positions_nm, definition.atom2, definition.atom3
        )
    raise ValueError(f"Unsupported reaction coordinate kind: {definition.kind!r}")


def compute_named_reaction_coordinates(
    positions_nm: np.ndarray, definitions: list[ReactionCoordinateDefinition]
) -> dict[str, float]:
    return {definition.name: compute_reaction_coordinate(positions_nm, definition) for definition in definitions}
