from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import openmm
    from openmm import unit
except ImportError:  # pragma: no cover
    openmm = None
    unit = None

from .config import EVBConfig
from .irc import canonicalize_irc_path, identify_fixed_atoms, read_irc_xyz
from .openmm_backend import AmberSystemLoader, EVBSystemBuilder, write_openmm_bundle


@dataclass(slots=True)
class ExceptionRecord:
    atom1: int
    atom2: int
    source_state: str
    reason: str

    @property
    def pair(self) -> tuple[int, int]:
        return tuple(sorted((self.atom1, self.atom2)))


def prepare_evb_ready_inputs(
    config: EVBConfig,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    extra_reactive_atoms: list[int] | None = None,
    extra_reactive_pairs: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Write EVB-ready OpenMM bundles from ordinary AMBER endpoint states.

    AMBER endpoint prmtops often have different bonded/exclusion graphs around
    reactive bonds. A product-state geometry can therefore be a nonbonded
    singularity in the reactant topology, or vice versa. This preprocessor keeps
    the original bonded terms intact, but mirrors zero nonbonded exclusions for
    reactive atoms across both diabatic systems.
    """

    _require_openmm()
    if config.state1.format != "amber" or config.state2.format != "amber":
        raise ValueError("prepare-evb-inputs expects ordinary AMBER inputs as state1/state2. Use format: amber.")
    out_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir) / "evb_ready_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = EVBSystemBuilder(
        AmberSystemLoader(
            nonbonded_method=config.simulation.nonbonded_method,
            constraints=config.simulation.constraints,
        )
    )
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    reactive_atoms = _infer_reactive_atoms(config)
    reactive_atoms.update(int(atom) for atom in (extra_reactive_atoms or []))
    explicit_pairs = _infer_reactive_pairs(config)
    explicit_pairs.update(tuple(sorted((int(a), int(b)))) for a, b in (extra_reactive_pairs or []))
    if not reactive_atoms and not explicit_pairs:
        raise ValueError(
            "No reactive atoms could be inferred. Define reaction.atoms donor/proton/acceptor or pass explicit reactive atoms/pairs."
        )

    records = _collect_reactive_exclusion_union(state1.system, state2.system, reactive_atoms, explicit_pairs)
    added_state1 = _apply_zero_exclusion_union(state1.system, records)
    added_state2 = _apply_zero_exclusion_union(state2.system, records)
    irc_distances = _irc_reactive_pair_distances(config, state1, explicit_pairs)

    state1_dir = out_dir / "state1_evb_ready"
    state2_dir = out_dir / "state2_evb_ready"
    write_openmm_bundle(state1_dir, state1.system, state1.topology, state1.positions_nm, state1.box_vectors_nm)
    write_openmm_bundle(state2_dir, state2.system, state2.topology, state2.positions_nm, state2.box_vectors_nm)

    derived_config = _write_derived_config(config_path, out_dir, state1_dir, state2_dir)
    report = {
        "output_dir": str(out_dir),
        "state1_bundle": str(state1_dir),
        "state2_bundle": str(state2_dir),
        "derived_config": str(derived_config),
        "reactive_atoms": sorted(reactive_atoms),
        "explicit_reactive_pairs": [list(pair) for pair in sorted(explicit_pairs)],
        "union_exclusion_pairs": [
            {"atom1": record.atom1, "atom2": record.atom2, "source_state": record.source_state, "reason": record.reason}
            for record in records
        ],
        "irc_reactive_pair_distances": irc_distances,
        "added_to_state1": [list(pair) for pair in sorted(added_state1)],
        "added_to_state2": [list(pair) for pair in sorted(added_state2)],
        "warnings": [
            "EVB-ready bundles mirror reactive zero nonbonded exclusions across states; original AMBER prmtops are not modified.",
            "This removes endpoint nonbonded singularities but changes diabatic reference energies. Re-run setup-from-irc and refit EVB parameters.",
        ],
    }
    (out_dir / "evb_ready_input_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def prepare_adiabatic_system_from_irc(
    config: EVBConfig,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    write_window_config: bool = False,
    extra_reactive_atoms: list[int] | None = None,
    extra_reactive_pairs: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Create EVB-ready OpenMM bundles and calibrated adiabatic setup from an IRC config."""
    from .config import load_config
    from .irc_setup import setup_from_irc

    prep_report = prepare_evb_ready_inputs(
        config,
        config_path=config_path,
        output_dir=output_dir,
        extra_reactive_atoms=extra_reactive_atoms,
        extra_reactive_pairs=extra_reactive_pairs,
    )
    derived_config_path = prep_report["derived_config"]
    derived_config = load_config(derived_config_path)
    mapping_report = _ensure_missing_alpha_carbon_mapping(derived_config)
    setup_report = setup_from_irc(derived_config, write_window_config=write_window_config)
    report = {
        "status": "adiabatic_system_prepared",
        "source_config": str(config_path),
        "output_dir": prep_report["output_dir"],
        "derived_config": derived_config_path,
        "state1_bundle": prep_report["state1_bundle"],
        "state2_bundle": prep_report["state2_bundle"],
        "evb_ready_input_report": prep_report,
        "mapping_report": mapping_report,
        "setup_from_irc_report": setup_report,
        "warnings": [
            "Adiabatic OpenMM EVB-ready bundles were generated from AMBER endpoint states.",
            "IRC calibration was rerun against the derived EVB-ready config; use the derived config for HG3.17 benchmarks/sampling.",
        ],
    }
    report_path = Path(prep_report["output_dir"]) / "adiabatic_system_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report



def _ensure_missing_alpha_carbon_mapping(config: EVBConfig) -> dict[str, Any]:
    embedding = config.irc.embedding or {}
    mapping_path = embedding.get("alpha_carbon_mapping") or embedding.get("mapping_file")
    if not mapping_path:
        return {"created": False, "reason": "No alpha-carbon mapping path configured."}
    path = Path(mapping_path)
    if path.exists():
        return {"created": False, "path": str(path), "reason": "Mapping file already exists."}
    if not config.irc.path:
        return {"created": False, "path": str(path), "reason": "No IRC path configured."}
    pdb_path = Path(embedding.get("reference_pdb", "inputs/5RGE.pdb"))
    if not pdb_path.exists():
        return {"created": False, "path": str(path), "reason": f"Reference PDB is missing: {pdb_path}"}
    frames = read_irc_xyz(config.irc.path)
    fixed_carbons = identify_fixed_atoms(frames, element="C")
    ca_atoms = _read_ca_atoms_for_mapping(pdb_path)
    mappings = []
    for candidate in fixed_carbons:
        coord = np.asarray([candidate.x_angstrom, candidate.y_angstrom, candidate.z_angstrom], dtype=float)
        distance, atom = min((float(np.linalg.norm(coord - item["coord_angstrom"])), item) for item in ca_atoms)
        mappings.append(
            {
                "irc_atom_index": candidate.frame_atom_index,
                "irc_element": candidate.element,
                "irc_coordinate_angstrom": [candidate.x_angstrom, candidate.y_angstrom, candidate.z_angstrom],
                "pdb": {
                    "id": pdb_path.stem,
                    "atom_serial": atom["serial"],
                    "atom_name": "CA",
                    "chain_id": atom["chain_id"],
                    "residue_name": atom["residue_name"],
                    "residue_number": atom["residue_number"],
                    "coordinate_angstrom": atom["coord_angstrom"].tolist(),
                },
                "match_distance_angstrom": distance,
                "confidence": "exact" if distance < 0.05 else "review" if distance < 0.5 else "low",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"reference_pdb": str(pdb_path), "irc_xyz": str(config.irc.path), "alpha_carbon_mapping": mappings}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return {
        "created": True,
        "path": str(path),
        "n_mappings": len(mappings),
        "max_match_distance_angstrom": max((item["match_distance_angstrom"] for item in mappings), default=None),
    }


def _read_ca_atoms_for_mapping(path: Path) -> list[dict[str, Any]]:
    atoms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        altloc = line[16].strip()
        if atom_name != "CA" or altloc not in ("", "A"):
            continue
        atoms.append(
            {
                "serial": int(line[6:11]),
                "residue_name": line[17:20].strip(),
                "chain_id": line[21].strip(),
                "residue_number": int(line[22:26]),
                "coord_angstrom": np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float),
            }
        )
    if not atoms:
        raise ValueError(f"No alpha carbon atoms found in {path}.")
    return atoms


def _infer_reactive_atoms(config: EVBConfig) -> set[int]:
    atoms = set()
    if config.reaction.atoms is not None:
        atoms.update(
            int(atom)
            for atom in (config.reaction.atoms.donor, config.reaction.atoms.proton, config.reaction.atoms.acceptor)
            if atom is not None
        )
    return atoms


def _infer_reactive_pairs(config: EVBConfig) -> set[tuple[int, int]]:
    pairs = set()
    if config.reaction.atoms is not None:
        donor = int(config.reaction.atoms.donor)
        proton = int(config.reaction.atoms.proton)
        acceptor = int(config.reaction.atoms.acceptor)
        pairs.add(tuple(sorted((donor, proton))))
        pairs.add(tuple(sorted((proton, acceptor))))
    return pairs


def _collect_reactive_exclusion_union(system1, system2, reactive_atoms: set[int], explicit_pairs: set[tuple[int, int]]) -> list[ExceptionRecord]:
    records_by_pair: dict[tuple[int, int], ExceptionRecord] = {}
    for label, system in (("state1", system1), ("state2", system2)):
        nonbonded = _nonbonded_force(system)
        for index in range(nonbonded.getNumExceptions()):
            atom1, atom2, chargeprod, _sigma, epsilon = nonbonded.getExceptionParameters(index)
            pair = tuple(sorted((int(atom1), int(atom2))))
            if pair not in explicit_pairs and not (pair[0] in reactive_atoms or pair[1] in reactive_atoms):
                continue
            charge_value = chargeprod.value_in_unit(unit.elementary_charge**2)
            epsilon_value = epsilon.value_in_unit(unit.kilojoule_per_mole)
            if abs(charge_value) > 1.0e-12 or abs(epsilon_value) > 1.0e-12:
                continue
            records_by_pair.setdefault(
                pair,
                ExceptionRecord(pair[0], pair[1], label, "zero exception involving a reactive atom"),
            )
    for atom1, atom2 in explicit_pairs:
        pair = tuple(sorted((atom1, atom2)))
        records_by_pair.setdefault(pair, ExceptionRecord(pair[0], pair[1], "explicit", "explicit reactive pair"))
    return [records_by_pair[pair] for pair in sorted(records_by_pair)]


def _apply_zero_exclusion_union(system, records: list[ExceptionRecord]) -> set[tuple[int, int]]:
    nonbonded = _nonbonded_force(system)
    existing = {frozenset(nonbonded.getExceptionParameters(index)[:2]) for index in range(nonbonded.getNumExceptions())}
    added = set()
    for record in records:
        key = frozenset(record.pair)
        if key in existing:
            continue
        nonbonded.addException(record.atom1, record.atom2, 0.0, 0.1, 0.0, replace=False)
        existing.add(key)
        added.add(record.pair)
    return added


def _write_derived_config(config_path: str | Path, out_dir: Path, state1_dir: Path, state2_dir: Path) -> Path:
    source = Path(config_path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload.setdefault("states", {})
    payload["states"]["state1"] = {
        "format": "openmm",
        "topology": str(state1_dir / "system.xml"),
        "coordinates": str(state1_dir / "coordinates.pdb"),
    }
    payload["states"]["state2"] = {
        "format": "openmm",
        "topology": str(state2_dir / "system.xml"),
        "coordinates": str(state2_dir / "coordinates.pdb"),
    }
    payload.setdefault("project", {})
    payload["project"]["output_dir"] = str(out_dir / "setup_from_irc")
    output = out_dir / f"{source.stem}_evb_ready.yaml"
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output


def _irc_reactive_pair_distances(config: EVBConfig, state, pairs: set[tuple[int, int]]) -> dict[str, Any]:
    if not config.irc.path or not pairs:
        return {"available": False, "reason": "No IRC path or reactive pairs were provided."}
    try:
        frames = read_irc_xyz(config.irc.path)
        canonical = canonicalize_irc_path(
            frames,
            order=config.irc.order,
            rc_frame=config.irc.rc_frame,
            ts_frame=config.irc.ts_frame,
            product_frame=config.irc.product_frame,
        )
        positions_by_role = {}
        if len(canonical.frames[0].symbols) == state.system.getNumParticles():
            for role, frame_index in (("RC", canonical.rc_frame), ("TS", canonical.ts_frame), ("PROD", canonical.product_frame)):
                positions_by_role[role] = canonical.frames[frame_index].coordinates_nm
        else:
            from .irc_setup import _make_cluster_embedder

            embedder = _make_cluster_embedder(config, canonical, state)
            if embedder is None:
                return {"available": False, "reason": "IRC is not full-system and no embedder was created."}
            for role, frame_index in (("RC", canonical.rc_frame), ("TS", canonical.ts_frame), ("PROD", canonical.product_frame)):
                positions_by_role[role] = embedder.embed(canonical.frames[frame_index])
        distances = {}
        for role, positions in positions_by_role.items():
            distances[role] = []
            for atom1, atom2 in sorted(pairs):
                distance_angstrom = float(np.linalg.norm(positions[atom1] - positions[atom2]) * 10.0)
                distances[role].append({"atom1": atom1, "atom2": atom2, "distance_angstrom": distance_angstrom})
        return {"available": True, "distances": distances}
    except Exception as exc:
        return {"available": False, "reason": f"Could not compute IRC reactive-pair distances: {exc}"}


def _nonbonded_force(system):
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            return force
    raise ValueError("System does not contain a NonbondedForce.")


def _require_openmm() -> None:
    if openmm is None or unit is None:
        raise ImportError("OpenMM is required to prepare EVB-ready inputs.")
