#!/usr/bin/env python
"""Extract a gap-ordered pseudo trajectory from EVB metadynamics output.

The script bins saved trajectory frames along the shifted EVB energy gap and
selects the lowest EVB lower-surface energy frame in each bin.  The selected
frames are written as a multi-model PDB ordered from negative to positive gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openmm import unit
from openmm.app import PDBFile


KJ_PER_KCAL = 4.184


@dataclass(frozen=True)
class CandidateFrame:
    replica: str
    step: int
    time_ps: float
    dcd_frame: int
    gap_kj_mol: float
    eevb_kj_mol: float
    total_potential_kj_mol: float
    w1: float
    w2: float
    donor_h_nm: float
    h_acceptor_nm: float
    proton_transfer_rc_nm: float


class OpenMMDCDReader:
    """Minimal reader for CHARMM-style DCD files written by OpenMM."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("rb")
        self.n_frames, self.n_atoms, self.has_box = self._read_header()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "OpenMMDCDReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _read_header(self) -> tuple[int, int, bool]:
        handle = self.handle
        handle.seek(0)
        marker = struct.unpack("<i", handle.read(4))[0]
        if marker != 84:
            raise ValueError(f"{self.path} is not an OpenMM little-endian DCD file")
        magic = handle.read(4)
        if magic != b"CORD":
            raise ValueError(f"{self.path} is not a coordinate DCD file")
        n_frames = struct.unpack("<i", handle.read(4))[0]
        handle.seek(48)
        has_box = bool(struct.unpack("<i", handle.read(4))[0])
        handle.seek(264)
        atom_record = struct.unpack("<i", handle.read(4))[0]
        n_atoms = struct.unpack("<i", handle.read(4))[0]
        end_record = struct.unpack("<i", handle.read(4))[0]
        if atom_record != 4 or end_record != 4:
            raise ValueError(f"{self.path} has a corrupt atom-count record")
        return n_frames, n_atoms, has_box

    def _skip_frame(self) -> None:
        handle = self.handle
        if self.has_box:
            self._skip_record(expected_bytes=48)
        coord_bytes = 4 * self.n_atoms
        self._skip_record(expected_bytes=coord_bytes)
        self._skip_record(expected_bytes=coord_bytes)
        self._skip_record(expected_bytes=coord_bytes)

    def _skip_record(self, expected_bytes: int | None = None) -> None:
        handle = self.handle
        n_bytes = struct.unpack("<i", handle.read(4))[0]
        if expected_bytes is not None and n_bytes != expected_bytes:
            raise ValueError(
                f"{self.path} has an unexpected DCD record length "
                f"{n_bytes}; expected {expected_bytes}"
            )
        handle.seek(n_bytes, 1)
        end = struct.unpack("<i", handle.read(4))[0]
        if end != n_bytes:
            raise ValueError(f"{self.path} has a corrupt DCD record")

    def read_selected_frames(self, frame_indices: list[int]) -> dict[int, np.ndarray]:
        """Return selected frame coordinates in nanometers."""

        wanted = sorted(set(frame_indices))
        if not wanted:
            return {}
        if wanted[-1] >= self.n_frames:
            raise ValueError(
                f"Requested frame {wanted[-1]} from {self.path}, "
                f"but DCD contains {self.n_frames} frames"
            )
        selected: dict[int, np.ndarray] = {}
        next_wanted = 0
        for frame_index in range(self.n_frames):
            if next_wanted >= len(wanted):
                break
            if frame_index == wanted[next_wanted]:
                selected[frame_index] = self._read_frame_nm()
                next_wanted += 1
            else:
                self._skip_frame()
        return selected

    def _read_frame_nm(self) -> np.ndarray:
        handle = self.handle
        if self.has_box:
            self._skip_record(expected_bytes=48)
        coords_angstrom = np.empty((self.n_atoms, 3), dtype=np.float32)
        coord_bytes = 4 * self.n_atoms
        for axis in range(3):
            n_bytes = struct.unpack("<i", handle.read(4))[0]
            if n_bytes != coord_bytes:
                raise ValueError(
                    f"{self.path} has an unexpected coordinate record length "
                    f"{n_bytes}; expected {coord_bytes}"
                )
            data = np.fromfile(handle, dtype="<f4", count=self.n_atoms)
            end = struct.unpack("<i", handle.read(4))[0]
            if end != n_bytes:
                raise ValueError(f"{self.path} has a corrupt coordinate record")
            coords_angstrom[:, axis] = data
        return coords_angstrom.astype(np.float64) * 0.1


def _read_candidates(
    base_dir: Path,
    replicas: list[str],
    save_stride: int,
) -> list[CandidateFrame]:
    candidates: list[CandidateFrame] = []
    for replica in replicas:
        path = base_dir / replica / "gap_metad_observables.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                step = int(float(row["step"]))
                if step % save_stride != 0:
                    continue
                dcd_frame = step // save_stride - 1
                if dcd_frame < 0:
                    continue
                candidates.append(
                    CandidateFrame(
                        replica=replica,
                        step=step,
                        time_ps=float(row["time_ps"]),
                        dcd_frame=dcd_frame,
                        gap_kj_mol=float(row["gap_shifted_kJmol"]),
                        eevb_kj_mol=float(row["Eevb_kJmol"]),
                        total_potential_kj_mol=float(row["total_potential_kJmol"]),
                        w1=float(row["w1"]),
                        w2=float(row["w2"]),
                        donor_h_nm=float(row["distance_donor_h_nm"]),
                        h_acceptor_nm=float(row["distance_h_acceptor_nm"]),
                        proton_transfer_rc_nm=float(row["rc_proton_transfer_rc_nm"]),
                    )
                )
    return [c for c in candidates if math.isfinite(c.gap_kj_mol) and math.isfinite(c.eevb_kj_mol)]


def _select_lowest_energy_by_gap_bin(
    candidates: list[CandidateFrame],
    n_bins: int,
    min_gap_kcal: float | None,
    max_gap_kcal: float | None,
) -> tuple[list[CandidateFrame], np.ndarray, list[int]]:
    gaps_kcal = np.array([c.gap_kj_mol / KJ_PER_KCAL for c in candidates], dtype=float)
    lo = gaps_kcal.min() if min_gap_kcal is None else min_gap_kcal
    hi = gaps_kcal.max() if max_gap_kcal is None else max_gap_kcal
    if not lo < hi:
        raise ValueError("The requested gap range is empty")
    edges = np.linspace(lo, hi, n_bins + 1)
    selected: list[CandidateFrame] = []
    empty_bins: list[int] = []
    for bin_index in range(n_bins):
        left = edges[bin_index]
        right = edges[bin_index + 1]
        if bin_index == n_bins - 1:
            in_bin = [
                c for c in candidates if left <= c.gap_kj_mol / KJ_PER_KCAL <= right
            ]
        else:
            in_bin = [
                c for c in candidates if left <= c.gap_kj_mol / KJ_PER_KCAL < right
            ]
        if not in_bin:
            empty_bins.append(bin_index)
            continue
        selected.append(min(in_bin, key=lambda c: c.eevb_kj_mol))
    selected.sort(key=lambda c: c.gap_kj_mol)
    return selected, edges, empty_bins


def _load_selected_coordinates(
    base_dir: Path,
    selected: list[CandidateFrame],
) -> dict[tuple[str, int], np.ndarray]:
    coords: dict[tuple[str, int], np.ndarray] = {}
    by_replica: dict[str, list[int]] = {}
    for frame in selected:
        by_replica.setdefault(frame.replica, []).append(frame.dcd_frame)
    for replica, frames in by_replica.items():
        dcd_path = base_dir / replica / "gap_metad.dcd"
        with OpenMMDCDReader(dcd_path) as reader:
            for dcd_frame, positions_nm in reader.read_selected_frames(frames).items():
                coords[(replica, dcd_frame)] = positions_nm
    return coords


def _write_outputs(
    base_dir: Path,
    selected: list[CandidateFrame],
    coords: dict[tuple[str, int], np.ndarray],
    out_dir: Path,
    topology_replica: str,
    edges: np.ndarray,
    empty_bins: list[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    topology_pdb = PDBFile(str(base_dir / topology_replica / "gap_metad_final.pdb"))
    pdb_path = out_dir / "hg317_gap_metad_pseudo_transition.pdb"
    csv_path = out_dir / "hg317_gap_metad_pseudo_transition_index.csv"
    json_path = out_dir / "hg317_gap_metad_pseudo_transition_summary.json"
    pml_path = out_dir / "load_hg317_gap_metad_pseudo_transition.pml"

    with pdb_path.open("w", encoding="utf-8") as handle:
        PDBFile.writeHeader(topology_pdb.topology, handle)
        for model_index, frame in enumerate(selected, start=1):
            positions_nm = coords[(frame.replica, frame.dcd_frame)] * unit.nanometer
            PDBFile.writeModel(topology_pdb.topology, positions_nm, handle, modelIndex=model_index)
        PDBFile.writeFooter(topology_pdb.topology, handle)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pseudo_model",
                "replica",
                "step",
                "time_ps",
                "dcd_frame",
                "gap_kj_mol",
                "gap_kcal_mol",
                "Eevb_kj_mol",
                "Eevb_kcal_mol",
                "Eevb_relative_kcal_mol",
                "total_potential_kj_mol",
                "w1",
                "w2",
                "distance_donor_h_nm",
                "distance_h_acceptor_nm",
                "proton_transfer_rc_nm",
            ]
        )
        e0 = min(frame.eevb_kj_mol for frame in selected)
        for model_index, frame in enumerate(selected, start=1):
            writer.writerow(
                [
                    model_index,
                    frame.replica,
                    frame.step,
                    f"{frame.time_ps:.6f}",
                    frame.dcd_frame,
                    f"{frame.gap_kj_mol:.8f}",
                    f"{frame.gap_kj_mol / KJ_PER_KCAL:.8f}",
                    f"{frame.eevb_kj_mol:.8f}",
                    f"{frame.eevb_kj_mol / KJ_PER_KCAL:.8f}",
                    f"{(frame.eevb_kj_mol - e0) / KJ_PER_KCAL:.8f}",
                    f"{frame.total_potential_kj_mol:.8f}",
                    f"{frame.w1:.8f}",
                    f"{frame.w2:.8f}",
                    f"{frame.donor_h_nm:.8f}",
                    f"{frame.h_acceptor_nm:.8f}",
                    f"{frame.proton_transfer_rc_nm:.8f}",
                ]
            )

    summary = {
        "description": (
            "Pseudo trajectory built by binning saved EVB metadynamics frames "
            "along shifted EVB gap and selecting the lowest E_EVB frame per bin. "
            "This is for visualization, not a dynamical trajectory."
        ),
        "output_pdb": str(pdb_path),
        "index_csv": str(csv_path),
        "n_models": len(selected),
        "gap_range_kcal_mol": [
            selected[0].gap_kj_mol / KJ_PER_KCAL,
            selected[-1].gap_kj_mol / KJ_PER_KCAL,
        ],
        "bin_edges_kcal_mol": edges.tolist(),
        "empty_gap_bins": empty_bins,
        "selection_energy": "Eevb_kJmol",
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pml_path.write_text(
        "\n".join(
            [
                f"load {pdb_path.resolve()}, hg317_gap_transition",
                "hide everything",
                "show cartoon, polymer.protein",
                "show sticks, not polymer.protein",
                "spectrum count, blue_white_red, hg317_gap_transition",
                "set all_states, on",
                "mplay",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("outputs/hg317_evb_gap_metad"))
    parser.add_argument("--replica", action="append", dest="replicas")
    parser.add_argument("--bins", type=int, default=61)
    parser.add_argument("--save-stride", type=int, default=10000)
    parser.add_argument("--min-gap-kcal", type=float)
    parser.add_argument("--max-gap-kcal", type=float)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/hg317_evb_gap_metad/pseudo_trajectory"),
    )
    args = parser.parse_args()

    replicas = args.replicas or sorted(
        p.name
        for p in args.base.iterdir()
        if p.is_dir() and (p / "gap_metad_observables.csv").exists() and (p / "gap_metad.dcd").exists()
    )
    if not replicas:
        raise SystemExit(f"No replica directories found under {args.base}")

    candidates = _read_candidates(args.base, replicas, args.save_stride)
    if not candidates:
        raise SystemExit("No saved-frame candidates found")
    selected, edges, empty_bins = _select_lowest_energy_by_gap_bin(
        candidates,
        n_bins=args.bins,
        min_gap_kcal=args.min_gap_kcal,
        max_gap_kcal=args.max_gap_kcal,
    )
    coords = _load_selected_coordinates(args.base, selected)
    missing = [(f.replica, f.dcd_frame) for f in selected if (f.replica, f.dcd_frame) not in coords]
    if missing:
        raise SystemExit(f"Missing coordinates for selected frames: {missing[:5]}")
    _write_outputs(args.base, selected, coords, args.out_dir, replicas[0], edges, empty_bins)
    print(f"Wrote {len(selected)} pseudo-trajectory models to {args.out_dir}")
    print(
        "Gap range: "
        f"{selected[0].gap_kj_mol / KJ_PER_KCAL:.1f} to "
        f"{selected[-1].gap_kj_mol / KJ_PER_KCAL:.1f} kcal/mol"
    )
    if empty_bins:
        print(f"Skipped {len(empty_bins)} empty gap bins")


if __name__ == "__main__":
    main()
