from __future__ import annotations

import numpy as np


def proton_transfer_coordinate(
    positions_nm: np.ndarray,
    donor_index: int,
    proton_index: int,
    acceptor_index: int,
) -> float:
    donor = positions_nm[donor_index]
    proton = positions_nm[proton_index]
    acceptor = positions_nm[acceptor_index]
    d_donor = np.linalg.norm(donor - proton)
    d_acceptor = np.linalg.norm(acceptor - proton)
    return float(d_donor - d_acceptor)
