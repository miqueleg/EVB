from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DIR = ROOT / "systems" / "KEMP_implicit"
FILES = {
    "RC": SYSTEM_DIR / "RC_final.xyz",
    "TS": SYSTEM_DIR / "TS_final.xyz",
    "PROD": SYSTEM_DIR / "PROD_final.xyz",
}

# 0-based indices from the EVB code path, derived from the QM structures.
PROTON_TRANSFER = {"donor": 6, "proton": 15, "acceptor": 19}
RING_OPENING = {"atom1": 1, "atom2": 11}


def _load_xyz(path: Path) -> tuple[float | None, list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = int(lines[0].strip())
    energy = None
    if len(lines) > 1:
        tokens = lines[1].split()
        for index, token in enumerate(tokens[:-1]):
            if token == "E":
                try:
                    energy = float(tokens[index + 1])
                except ValueError:
                    pass
                break
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + natoms]:
        symbol, xs, ys, zs = line.split()[:4]
        symbols.append(symbol)
        coords.append([float(xs), float(ys), float(zs)])
    return energy, symbols, np.asarray(coords, dtype=float)


def _distance(coords: np.ndarray, atom1: int, atom2: int) -> float:
    return float(np.linalg.norm(coords[atom1] - coords[atom2]))


def main() -> None:
    payload: dict[str, object] = {
        "recommended_indices_0based": {
            "proton_transfer": PROTON_TRANSFER,
            "ring_opening": RING_OPENING,
        },
        "structures": {},
    }
    for label, path in FILES.items():
        energy, symbols, coords = _load_xyz(path)
        proton_transfer_rc = _distance(coords, PROTON_TRANSFER["donor"], PROTON_TRANSFER["proton"]) - _distance(
            coords, PROTON_TRANSFER["proton"], PROTON_TRANSFER["acceptor"]
        )
        ring_opening_rc = _distance(coords, RING_OPENING["atom1"], RING_OPENING["atom2"])
        payload["structures"][label] = {
            "path": str(path),
            "energy_hartree": energy,
            "symbols": symbols,
            "proton_transfer_rc_angstrom": proton_transfer_rc,
            "ring_opening_rc_angstrom": ring_opening_rc,
            "distances_angstrom": {
                "donor_h": _distance(coords, PROTON_TRANSFER["donor"], PROTON_TRANSFER["proton"]),
                "h_acceptor": _distance(coords, PROTON_TRANSFER["proton"], PROTON_TRANSFER["acceptor"]),
                "n_o_breaking": ring_opening_rc,
            },
        }
    output_path = SYSTEM_DIR / "qm_geometry_guide.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
