from __future__ import annotations

import re
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np


HARTREE_TO_KCAL_MOL = 627.5094740631
HARTREE_TO_KJ_MOL = 2625.4996394799
_ENERGY_PATTERN = re.compile(r"\benergy:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)")
_ROLE_PATTERN = re.compile(r"\b(RC|TS|PROD|PRODUCT|REACTANT)\b", re.IGNORECASE)


@dataclass(slots=True)
class IRCFrame:
    index: int
    symbols: list[str]
    coordinates_angstrom: np.ndarray
    comment: str
    energy_hartree: float | None = None
    role: str | None = None
    original_index: int | None = None

    @property
    def coordinates_nm(self) -> np.ndarray:
        return self.coordinates_angstrom * 0.1


@dataclass(slots=True)
class CanonicalIRCPath:
    frames: list[IRCFrame]
    original_order: str
    canonical_order: str
    original_to_canonical: dict[int, int]
    canonical_to_original: dict[int, int]
    rc_frame: int
    ts_frame: int
    product_frame: int
    warnings: list[str]


@dataclass(slots=True)
class ReferenceProfile:
    units: str
    rc_kj_mol: float
    ts_kj_mol: float
    product_kj_mol: float
    source_label: str | None = None

    @property
    def target_barrier_kj_mol(self) -> float:
        return self.ts_kj_mol - self.rc_kj_mol

    @property
    def target_reaction_free_energy_kj_mol(self) -> float:
        return self.product_kj_mol - self.rc_kj_mol


@dataclass(slots=True)
class IRCSummary:
    n_frames: int
    n_atoms: int
    first_frame: int
    last_frame: int
    minimum_energy_frame: int
    maximum_energy_frame: int
    first_energy_hartree: float
    last_energy_hartree: float
    minimum_energy_hartree: float
    maximum_energy_hartree: float
    energy_span_kcal_mol: float
    forward_endpoint_barrier_kcal_mol: float
    reverse_endpoint_barrier_kcal_mol: float
    endpoint_reaction_energy_kcal_mol: float
    missing_energy_frames: list[int]
    ts_comment_frames: list[int]


@dataclass(slots=True)
class FixedAtomCandidate:
    frame_atom_index: int
    element: str
    x_angstrom: float
    y_angstrom: float
    z_angstrom: float
    rms_displacement_angstrom: float
    max_displacement_angstrom: float


def read_irc_xyz(path: str | Path) -> list[IRCFrame]:
    """Read a multi-frame XYZ IRC.

    Energies in comments are optional metadata only. They are never treated as
    thermodynamic reference energies by this parser.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    frames: list[IRCFrame] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            n_atoms = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(f"Expected atom count at line {i + 1}, got {lines[i]!r}.") from exc
        if i + 1 >= len(lines):
            raise ValueError(f"Missing comment line after frame atom count at line {i + 1}.")
        comment = lines[i + 1].strip()
        match = _ENERGY_PATTERN.search(comment)
        role = infer_role_from_comment(comment)
        symbols: list[str] = []
        coords: list[list[float]] = []
        atom_start = i + 2
        atom_end = atom_start + n_atoms
        if atom_end > len(lines):
            raise ValueError(f"Frame {len(frames)} is truncated: expected {n_atoms} atom lines.")
        for line_number, line in enumerate(lines[atom_start:atom_end], start=atom_start + 1):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid XYZ atom line {line_number}: {line!r}")
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        frames.append(
            IRCFrame(
                index=len(frames),
                symbols=symbols,
                coordinates_angstrom=np.asarray(coords, dtype=float),
                comment=comment,
                energy_hartree=None if match is None else float(match.group(1)),
                role=role,
                original_index=len(frames),
            )
        )
        i = atom_end
    if not frames:
        raise ValueError(f"No XYZ frames found in {path}.")
    n_atoms = len(frames[0].symbols)
    symbols = frames[0].symbols
    for frame in frames[1:]:
        if len(frame.symbols) != n_atoms:
            raise ValueError(f"Frame {frame.index} has {len(frame.symbols)} atoms; expected {n_atoms}.")
        if frame.symbols != symbols:
            raise ValueError(f"Frame {frame.index} has a different atom ordering or element sequence.")
    return frames


def infer_role_from_comment(comment: str) -> str | None:
    match = _ROLE_PATTERN.search(comment)
    if match is None:
        return None
    role = match.group(1).upper()
    if role == "REACTANT":
        return "RC"
    if role == "PRODUCT":
        return "PROD"
    return role


def canonicalize_irc_path(
    frames: list[IRCFrame],
    order: str = "auto",
    rc_frame: int | str | None = "auto",
    ts_frame: int | str | None = "auto",
    product_frame: int | str | None = "auto",
) -> CanonicalIRCPath:
    if not frames:
        raise ValueError("Cannot canonicalize an empty IRC path.")
    order = order.lower()
    if order not in {"rc_ts_prod", "prod_ts_rc", "auto"}:
        raise ValueError("irc.order must be one of: rc_ts_prod, prod_ts_rc, auto.")
    warnings: list[str] = []
    if order == "auto":
        order = _infer_order(frames)
    if order == "prod_ts_rc":
        canonical_frames = list(reversed(frames))
    else:
        canonical_frames = list(frames)
    canonicalized = [
        replace(frame, index=canonical_index, original_index=frame.original_index if frame.original_index is not None else frame.index)
        for canonical_index, frame in enumerate(canonical_frames)
    ]
    original_to_canonical = {frame.original_index if frame.original_index is not None else frame.index: frame.index for frame in canonicalized}
    canonical_to_original = {frame.index: frame.original_index if frame.original_index is not None else frame.index for frame in canonicalized}

    rc_idx = _select_frame("RC", canonicalized, original_to_canonical, rc_frame, default=0, warnings=warnings)
    product_idx = _select_frame("PROD", canonicalized, original_to_canonical, product_frame, default=len(canonicalized) - 1, warnings=warnings)
    ts_default = len(canonicalized) // 2
    ts_idx = _select_frame("TS", canonicalized, original_to_canonical, ts_frame, default=ts_default, warnings=warnings)
    if ts_frame in {None, "auto"} and not any(frame.role == "TS" for frame in canonicalized):
        warnings.append(
            f"No TS label found in IRC comments; selected middle canonical frame {ts_idx}. "
            "Check this manually or set irc.ts_frame explicitly."
        )
    return CanonicalIRCPath(
        frames=canonicalized,
        original_order=order,
        canonical_order="rc_ts_prod",
        original_to_canonical=original_to_canonical,
        canonical_to_original=canonical_to_original,
        rc_frame=rc_idx,
        ts_frame=ts_idx,
        product_frame=product_idx,
        warnings=warnings,
    )


def parse_reference_profile(units: str, rc: float, ts: float, product: float, source_label: str | None = None) -> ReferenceProfile:
    factor = _unit_factor_to_kj(units)
    return ReferenceProfile(
        units=units,
        rc_kj_mol=float(rc) * factor,
        ts_kj_mol=float(ts) * factor,
        product_kj_mol=float(product) * factor,
        source_label=source_label,
    )


def _unit_factor_to_kj(units: str) -> float:
    normalized = units.lower().replace(" ", "")
    if normalized in {"kj/mol", "kjmol", "kj_mol", "kj"}:
        return 1.0
    if normalized in {"kcal/mol", "kcalmol", "kcal_mol", "kcal"}:
        return 4.184
    raise ValueError(f"Unsupported reference_profile.units {units!r}; use kcal/mol or kJ/mol.")


def _infer_order(frames: list[IRCFrame]) -> str:
    labeled = {frame.role: frame.index for frame in frames if frame.role in {"RC", "TS", "PROD"}}
    if {"RC", "TS", "PROD"} <= set(labeled):
        if labeled["RC"] < labeled["TS"] < labeled["PROD"]:
            return "rc_ts_prod"
        if labeled["PROD"] < labeled["TS"] < labeled["RC"]:
            return "prod_ts_rc"
    raise ValueError(
        "Could not infer IRC order confidently. Set irc.order to 'rc_ts_prod' or 'prod_ts_rc', "
        "or provide comments/indices that identify RC, TS, and PROD."
    )


def _select_frame(
    role: str,
    frames: list[IRCFrame],
    original_to_canonical: dict[int, int],
    requested: int | str | None,
    default: int,
    warnings: list[str],
) -> int:
    if requested not in {None, "auto"}:
        original_index = int(requested)
        if original_index not in original_to_canonical:
            raise ValueError(f"Requested original {role} frame {original_index} is outside the IRC frame range.")
        return original_to_canonical[original_index]
    labeled = [frame.index for frame in frames if frame.role == role]
    if labeled:
        return labeled[0]
    warnings.append(f"No {role} label or explicit frame supplied; using canonical frame {default}.")
    return default


def summarize_irc(frames: list[IRCFrame]) -> IRCSummary:
    finite_frames = [frame for frame in frames if frame.energy_hartree is not None]
    if not finite_frames:
        raise ValueError("No IRC frames contain an 'energy: <Hartree>' comment.")
    energies = np.asarray([frame.energy_hartree for frame in finite_frames], dtype=float)
    min_pos = int(np.argmin(energies))
    max_pos = int(np.argmax(energies))
    min_idx = finite_frames[min_pos].index
    max_idx = finite_frames[max_pos].index
    first = _nearest_endpoint_energy(frames, from_start=True)
    last = _nearest_endpoint_energy(frames, from_start=False)
    maximum = float(energies[max_pos])
    minimum = float(energies[min_pos])
    return IRCSummary(
        n_frames=len(frames),
        n_atoms=len(frames[0].symbols),
        first_frame=0,
        last_frame=len(frames) - 1,
        minimum_energy_frame=min_idx,
        maximum_energy_frame=max_idx,
        first_energy_hartree=first,
        last_energy_hartree=last,
        minimum_energy_hartree=minimum,
        maximum_energy_hartree=maximum,
        energy_span_kcal_mol=(maximum - minimum) * HARTREE_TO_KCAL_MOL,
        forward_endpoint_barrier_kcal_mol=(maximum - first) * HARTREE_TO_KCAL_MOL,
        reverse_endpoint_barrier_kcal_mol=(maximum - last) * HARTREE_TO_KCAL_MOL,
        endpoint_reaction_energy_kcal_mol=(last - first) * HARTREE_TO_KCAL_MOL,
        missing_energy_frames=[frame.index for frame in frames if frame.energy_hartree is None],
        ts_comment_frames=[frame.index for frame in frames if "TS" in frame.comment.upper()],
    )


def write_irc_outputs(frames: list[IRCFrame], output_dir: str | Path, title: str = "IRC") -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = summarize_irc(frames)
    table = _irc_table(frames)
    _write_csv(output_path / "irc_profile.csv", table)
    _write_summary(output_path / "irc_summary.json", summary)
    fixed_carbons = identify_fixed_atoms(frames, element="C")
    _write_fixed_atom_candidates(output_path / "fixed_carbon_candidates.json", fixed_carbons)
    _write_fixed_atom_candidates_yaml(output_path / "fixed_carbon_candidates.yaml", fixed_carbons)
    _write_plot(output_path / "irc_energy_profile_kcal.png", table, summary, title=title, zero_mode="first")
    _write_plot(output_path / "irc_energy_profile_min_zero_kcal.png", table, summary, title=title, zero_mode="minimum")
    payload = asdict(summary)
    payload["output_dir"] = str(output_path)
    return payload


def identify_fixed_atoms(
    frames: list[IRCFrame],
    element: str | None = None,
    max_displacement_tolerance_angstrom: float = 1.0e-3,
) -> list[FixedAtomCandidate]:
    coords = np.asarray([frame.coordinates_angstrom for frame in frames], dtype=float)
    symbols = frames[0].symbols
    mean_coords = coords.mean(axis=0)
    rms_displacement = np.sqrt(np.mean(np.sum((coords - mean_coords[None, :, :]) ** 2, axis=2), axis=0))
    max_displacement = np.max(np.sqrt(np.sum((coords - coords[0][None, :, :]) ** 2, axis=2)), axis=0)
    candidates = []
    for atom_index, symbol in enumerate(symbols):
        if element is not None and symbol.upper() != element.upper():
            continue
        if max_displacement[atom_index] > max_displacement_tolerance_angstrom:
            continue
        x, y, z = coords[0, atom_index]
        candidates.append(
            FixedAtomCandidate(
                frame_atom_index=atom_index,
                element=symbol,
                x_angstrom=float(x),
                y_angstrom=float(y),
                z_angstrom=float(z),
                rms_displacement_angstrom=float(rms_displacement[atom_index]),
                max_displacement_angstrom=float(max_displacement[atom_index]),
            )
        )
    return candidates


def _irc_table(frames: list[IRCFrame]) -> list[dict[str, float]]:
    finite = np.asarray([frame.energy_hartree for frame in frames if frame.energy_hartree is not None], dtype=float)
    first = _nearest_endpoint_energy(frames, from_start=True)
    minimum = float(np.min(finite))
    rows = []
    for frame in frames:
        energy = frame.energy_hartree
        rows.append(
            {
                "frame": frame.index,
                "energy_hartree": "" if energy is None else float(energy),
                "relative_to_first_kcal_mol": "" if energy is None else float((energy - first) * HARTREE_TO_KCAL_MOL),
                "relative_to_first_kj_mol": "" if energy is None else float((energy - first) * HARTREE_TO_KJ_MOL),
                "relative_to_minimum_kcal_mol": "" if energy is None else float((energy - minimum) * HARTREE_TO_KCAL_MOL),
                "relative_to_minimum_kj_mol": "" if energy is None else float((energy - minimum) * HARTREE_TO_KJ_MOL),
                "comment": frame.comment,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, summary: IRCSummary) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")


def _write_fixed_atom_candidates(path: Path, candidates: list[FixedAtomCandidate]) -> None:
    payload = [asdict(candidate) for candidate in candidates]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_fixed_atom_candidates_yaml(path: Path, candidates: list[FixedAtomCandidate]) -> None:
    lines = [
        "# Fixed carbon atoms in the IRC cluster model.",
        "# These are alpha-carbon anchor candidates, but they are not mapped to OpenMM atoms yet.",
        "# Complete the openmm_atom_index field after matching against a protein/crystal PDB with CA atoms.",
        "fixed_carbon_candidates:",
    ]
    for candidate in candidates:
        lines.extend(
            [
                f"  - irc_atom_index: {candidate.frame_atom_index}",
                f"    element: {candidate.element}",
                "    openmm_atom_index: null",
                "    coordinate_angstrom: "
                f"[{candidate.x_angstrom:.6f}, {candidate.y_angstrom:.6f}, {candidate.z_angstrom:.6f}]",
                f"    max_displacement_angstrom: {candidate.max_displacement_angstrom:.8g}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, rows: list[dict[str, float]], summary: IRCSummary, title: str, zero_mode: str) -> None:
    import matplotlib.pyplot as plt

    column = "relative_to_first_kcal_mol" if zero_mode == "first" else "relative_to_minimum_kcal_mol"
    finite_rows = [row for row in rows if row[column] != ""]
    x = np.asarray([row["frame"] for row in finite_rows], dtype=float)
    y = np.asarray([row[column] for row in finite_rows], dtype=float)
    plt.figure(figsize=(7.0, 4.4), dpi=220)
    plt.plot(x, y, color="black", linewidth=1.9)
    y_by_frame = {int(row["frame"]): float(row[column]) for row in finite_rows}
    plt.scatter([summary.maximum_energy_frame], [y_by_frame[summary.maximum_energy_frame]], color="#d62728", s=28, label="highest-energy frame")
    plt.scatter([summary.minimum_energy_frame], [y_by_frame[summary.minimum_energy_frame]], color="#1f77b4", s=28, label="minimum-energy frame")
    for ts_frame in summary.ts_comment_frames:
        plt.axvline(ts_frame, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.55)
    plt.xlabel("IRC frame")
    plt.ylabel("Energy / kcal mol$^{-1}$")
    plt.title(title)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linewidth=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _nearest_endpoint_energy(frames: list[IRCFrame], from_start: bool) -> float:
    iterable = frames if from_start else reversed(frames)
    for frame in iterable:
        if frame.energy_hartree is not None:
            return frame.energy_hartree
    raise ValueError("No finite endpoint energy found.")
