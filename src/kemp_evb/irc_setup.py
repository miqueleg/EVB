from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import EVBConfig
from .evb import EVBHamiltonian, EVBParameters, fit_evb_reference_profile
from .irc import CanonicalIRCPath, ReferenceProfile, canonicalize_irc_path, parse_reference_profile, read_irc_xyz
from .openmm_backend import (
    AmberSystemLoader,
    EVBSystemBuilder,
    OpenMMStateEvaluator,
    build_absolute_positional_restraint_force,
    write_pdb,
    _to_box_vectors,
    _to_openmm_positions,
)

try:
    import openmm
    from openmm import unit
except ImportError:  # pragma: no cover
    openmm = None
    unit = None


KJ_TO_KCAL = 1.0 / 4.184


class ClusterIRCEmbedder:
    def __init__(
        self,
        reference_positions_nm: np.ndarray,
        irc_to_openmm: dict[int, int],
        rotation: np.ndarray,
        translation_angstrom: np.ndarray,
        warnings: list[str],
    ):
        self.reference_positions_nm = np.asarray(reference_positions_nm, dtype=float)
        self.irc_to_openmm = dict(irc_to_openmm)
        self.rotation = np.asarray(rotation, dtype=float)
        self.translation_angstrom = np.asarray(translation_angstrom, dtype=float)
        self.warnings = list(warnings)

    def embed(self, frame, base_positions_nm: np.ndarray | None = None) -> np.ndarray:
        positions = self.reference_positions_nm.copy() if base_positions_nm is None else np.asarray(base_positions_nm, dtype=float).copy()
        transformed = _apply_transform(frame.coordinates_angstrom, self.rotation, self.translation_angstrom)
        for irc_index, openmm_index in self.irc_to_openmm.items():
            positions[openmm_index] = transformed[irc_index] * 0.1
        return positions


def setup_from_irc(config: EVBConfig, write_window_config: bool = False) -> dict[str, Any]:
    if config.irc.path is None:
        raise ValueError("setup-from-irc requires irc.path in the config or --irc on the CLI.")
    if config.reference_profile is None:
        raise ValueError("setup-from-irc requires a top-level reference_profile section.")
    if config.reference_profile.rc is None or config.reference_profile.ts is None or config.reference_profile.product is None:
        raise ValueError("reference_profile requires rc, ts, and product values.")

    output_dir = Path(config.output_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = read_irc_xyz(config.irc.path)
    canonical = canonicalize_irc_path(
        frames,
        order=config.irc.order,
        rc_frame=config.irc.rc_frame,
        ts_frame=config.irc.ts_frame,
        product_frame=config.irc.product_frame,
    )
    reference = parse_reference_profile(
        config.reference_profile.units,
        config.reference_profile.rc,
        config.reference_profile.ts,
        config.reference_profile.product,
        source_label=config.reference_profile.source_label,
    )

    builder = EVBSystemBuilder(
        AmberSystemLoader(
            nonbonded_method=config.simulation.nonbonded_method,
            constraints=config.simulation.constraints,
        )
    )
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    embedder = _make_cluster_embedder(config, canonical, state1)
    if embedder is None:
        _validate_irc_against_topology(canonical.frames, state1)
    relaxed_positions, relaxation_report = _relax_irc_seeds_if_requested(config, canonical, state1, state2, builder, embedder, output_dir)

    evaluator1 = OpenMMStateEvaluator(state1, platform_name=config.simulation.platform)
    evaluator2 = OpenMMStateEvaluator(state2, platform_name=config.simulation.platform)
    rows = _evaluate_diabatic_scan(
        canonical,
        evaluator1,
        evaluator2,
        delta_alpha=None,
        h12=None,
        embedder=embedder,
        positions_by_frame=relaxed_positions,
    )
    _write_scan_csv(output_dir / "irc_diabatic_scan_prefit.csv", rows)

    fit = fit_evb_from_irc_roles(canonical, rows, reference)
    postfit_rows = _apply_fit_to_scan(canonical, rows, fit.parameters)
    _write_scan_csv(output_dir / "irc_diabatic_scan.csv", postfit_rows)

    warnings = list(canonical.warnings)
    if embedder is not None:
        warnings.extend(embedder.warnings)
    if relaxation_report is not None:
        warnings.extend(relaxation_report.get("warnings", []))
    warnings.extend(_diagnose_scan(postfit_rows, canonical, fit))
    report = _build_report(config, canonical, reference, fit, postfit_rows, warnings)
    if relaxation_report is not None:
        report["irc_seed_relaxation"] = relaxation_report
    (output_dir / "evb_reference_fit_from_irc.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if config.sampling.windows.gap_umbrella.from_irc_scan or write_window_config:
        window_report = _write_gap_window_proposal(config, canonical, postfit_rows, output_dir)
        report["generated_gap_windows"] = window_report
        (output_dir / "evb_reference_fit_from_irc.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def fit_evb_from_irc_roles(scan: CanonicalIRCPath, rows: list[dict[str, Any]], reference: ReferenceProfile):
    by_frame = {int(row["canonical_frame"]): row for row in rows}
    rc = by_frame[scan.rc_frame]
    prod = by_frame[scan.product_frame]
    ts = by_frame[scan.ts_frame]
    return fit_evb_reference_profile(
        e_mm_min1_state1=float(rc["E1_kj_mol"]),
        e_mm_min1_state2=float(rc["E2_kj_mol"]),
        e_mm_min2_state1=float(prod["E1_kj_mol"]),
        e_mm_min2_state2=float(prod["E2_kj_mol"]),
        e_mm_ts_state1=float(ts["E1_kj_mol"]),
        e_mm_ts_state2=float(ts["E2_kj_mol"]),
        e_qmmm_min1=reference.rc_kj_mol,
        e_qmmm_min2=reference.product_kj_mol,
        e_qmmm_ts=reference.ts_kj_mol,
    )


def _evaluate_diabatic_scan(
    canonical: CanonicalIRCPath,
    evaluator1: OpenMMStateEvaluator,
    evaluator2: OpenMMStateEvaluator,
    delta_alpha: float | None,
    h12: float | None,
    embedder: "ClusterIRCEmbedder | None" = None,
    positions_by_frame: dict[int, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for frame in canonical.frames:
        if positions_by_frame is not None and frame.index in positions_by_frame:
            positions_nm = positions_by_frame[frame.index]
        else:
            positions_nm = embedder.embed(frame) if embedder is not None else frame.coordinates_nm
        e1, _ = evaluator1.evaluate(positions_nm)
        e2, _ = evaluator2.evaluate(positions_nm)
        row = {
            "frame": frame.original_index if frame.original_index is not None else frame.index,
            "canonical_frame": frame.index,
            "role": _role_for_frame(canonical, frame.index, frame.role),
            "path_order": canonical.original_order,
            "E1_kj_mol": e1,
            "E2_kj_mol": e2,
            "gap_raw_kj_mol": e1 - e2,
            "delta_alpha_kj_mol": delta_alpha,
            "gap_shifted_kj_mol": None,
            "Eevb_kj_mol": None,
            "w1": None,
            "w2": None,
            "comment": frame.comment,
        }
        if delta_alpha is not None and h12 is not None:
            ham = EVBHamiltonian(EVBParameters(delta_alpha=delta_alpha, h12=h12))
            evb, w1, w2 = ham.lower_eigenvalue(e1, e2)
            row["gap_shifted_kj_mol"] = ham.gap(e1, e2)
            row["Eevb_kj_mol"] = evb
            row["w1"] = w1
            row["w2"] = w2
        rows.append(row)
    return rows


def _apply_fit_to_scan(canonical: CanonicalIRCPath, rows: list[dict[str, Any]], parameters: EVBParameters) -> list[dict[str, Any]]:
    ham = EVBHamiltonian(parameters)
    postfit = []
    for row in rows:
        updated = dict(row)
        e1 = float(row["E1_kj_mol"])
        e2 = float(row["E2_kj_mol"])
        evb, w1, w2 = ham.lower_eigenvalue(e1, e2)
        updated["delta_alpha_kj_mol"] = parameters.delta_alpha
        updated["gap_shifted_kj_mol"] = ham.gap(e1, e2)
        updated["Eevb_kj_mol"] = evb
        updated["w1"] = w1
        updated["w2"] = w2
        postfit.append(updated)
    return postfit


def _validate_irc_against_topology(frames, state) -> None:
    topology_symbols = []
    for atom in state.topology.atoms():
        if atom.element is None:
            topology_symbols.append(atom.name[0].upper())
        else:
            topology_symbols.append(atom.element.symbol)
    if len(frames[0].symbols) != len(topology_symbols):
        raise ValueError(
            f"IRC atom count ({len(frames[0].symbols)}) does not match OpenMM topology atom count ({len(topology_symbols)}). "
            "Provide a full-system IRC/coordinate path or a tested atom mapping before running setup-from-irc."
        )
    if [symbol.upper() for symbol in frames[0].symbols] != [symbol.upper() for symbol in topology_symbols]:
        raise ValueError("IRC element sequence does not match the OpenMM topology atom order.")


def _relax_irc_seeds_if_requested(
    config: EVBConfig,
    canonical: CanonicalIRCPath,
    state1,
    state2,
    builder: EVBSystemBuilder,
    embedder: ClusterIRCEmbedder | None,
    output_dir: Path,
) -> tuple[dict[int, np.ndarray] | None, dict[str, Any] | None]:
    spec = config.irc.relaxation
    if not spec.enabled:
        return None, None
    if embedder is None:
        raise ValueError("IRC seed relaxation currently requires an embedded cluster/full-system IRC path.")
    if openmm is None or unit is None:
        raise ImportError("OpenMM is required for IRC seed relaxation.")
    if spec.mode != "mapped":
        raise ValueError("IRC seed relaxation currently supports mode: mapped.")
    frame_indices = _selected_relaxation_frames(canonical, spec.frame_stride, spec.frame_indices)
    relaxed_dir = output_dir / spec.output_subdir
    relaxed_dir.mkdir(parents=True, exist_ok=True)
    platform_name = spec.platform or config.simulation.platform or config.sampling.md.platform
    base_positions, pre_relaxation_report = _pre_relax_reference_structure(
        config,
        state1,
        state2,
        builder,
        embedder,
        relaxed_dir,
        platform_name,
    )
    relaxed_positions: dict[int, np.ndarray] = {}
    rows = []
    warnings = []
    for frame_index in frame_indices:
        frame = canonical.frames[frame_index]
        initial_positions = embedder.embed(frame, base_positions_nm=base_positions)
        mobile_atoms = _resolve_relaxation_mobile_atoms(config, initial_positions, embedder)
        restrained_atoms = [index for index in range(state1.system.getNumParticles()) if index not in mobile_atoms]
        lambda_value = 0.0 if len(canonical.frames) == 1 else frame_index / (len(canonical.frames) - 1)
        mapped_system = builder.build_openmm_mapped_system(
            state1,
            state2,
            lambda_value=lambda_value,
            delta_alpha=config.evb_parameters.delta_alpha or 0.0,
            add_cmmotion_remover=False,
        )
        if spec.restrain_nonmobile and restrained_atoms:
            mapped_system.system.addForce(
                build_absolute_positional_restraint_force(
                    base_positions,
                    atom_indices=restrained_atoms,
                    force_constant_kj_mol_nm2=spec.nonmobile_restraint_kj_mol_nm2,
                    parameter_name="k_irc_nonmobile",
                )
            )
        if spec.irc_atom_restraint_kj_mol_nm2 > 0.0 and embedder.irc_to_openmm:
            mapped_system.system.addForce(
                build_absolute_positional_restraint_force(
                    initial_positions,
                    atom_indices=sorted(set(embedder.irc_to_openmm.values())),
                    force_constant_kj_mol_nm2=spec.irc_atom_restraint_kj_mol_nm2,
                    parameter_name="k_irc_seed",
                )
            )
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        context = _create_openmm_context(
            mapped_system.system,
            integrator,
            platform_name,
            require_platform=spec.require_platform,
        )
        if state1.box_vectors_nm is not None:
            context.setPeriodicBoxVectors(*_to_box_vectors(state1.box_vectors_nm))
        context.setPositions(_to_openmm_positions(initial_positions))
        actual_platform = context.getPlatform().getName()
        initial_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        openmm.LocalEnergyMinimizer.minimize(
            context,
            spec.minimization_tolerance_kj_mol_nm * unit.kilojoule_per_mole / unit.nanometer,
            spec.minimization_steps,
        )
        final_state = context.getState(getEnergy=True, getPositions=True)
        final_energy = final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        final_positions = np.asarray(final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        relaxed_positions[frame_index] = final_positions
        pdb_path = relaxed_dir / f"irc_relaxed_c{frame_index:04d}_o{canonical.canonical_to_original[frame_index]:04d}.pdb"
        write_pdb(str(pdb_path), state1.topology, final_positions)
        rows.append(
            {
                "canonical_frame": frame_index,
                "original_frame": canonical.canonical_to_original[frame_index],
                "lambda_value": lambda_value,
                "n_mobile_atoms": len(mobile_atoms),
                "n_restrained_atoms": len(restrained_atoms),
                "platform": actual_platform,
                "initial_energy_kj_mol": float(initial_energy),
                "final_energy_kj_mol": float(final_energy),
                "pdb": str(pdb_path),
            }
        )
    _write_relaxation_csv(relaxed_dir / "irc_seed_relaxation.csv", rows)
    if len(frame_indices) != len(canonical.frames):
        warnings.append(
            f"Relaxed {len(frame_indices)} of {len(canonical.frames)} IRC frames; unrelaxed frames will use directly embedded coordinates for scans."
        )
    report = {
        "enabled": True,
        "mode": spec.mode,
        "n_relaxed_frames": len(frame_indices),
        "frame_indices": frame_indices,
        "requested_platform": platform_name,
        "output_dir": str(relaxed_dir),
        "pre_relaxation": pre_relaxation_report,
        "warnings": warnings,
        "frames": rows,
    }
    (relaxed_dir / "irc_seed_relaxation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return relaxed_positions if spec.use_relaxed_for_scan else None, report


def _pre_relax_reference_structure(
    config: EVBConfig,
    state1,
    state2,
    builder: EVBSystemBuilder,
    embedder: ClusterIRCEmbedder,
    relaxed_dir: Path,
    platform_name: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = config.irc.relaxation
    if not spec.pre_relaxation_enabled:
        return state1.positions_nm.copy(), {"enabled": False}

    pre_dir = relaxed_dir / "pre_relaxation"
    pre_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "enabled": True,
        "requested_platform": platform_name,
        "stages": [],
    }
    positions = state1.positions_nm.copy()
    solvent_atoms = _topology_solvent_atoms(state1.topology, spec.solvent_residue_names)
    ca_atoms = _topology_alpha_carbon_atoms(state1.topology) if spec.fix_alpha_carbons else set()

    solvent_mobile = set(solvent_atoms)
    positions, solvent_row = _run_minimization_stage(
        config,
        state1,
        state2,
        builder,
        positions,
        mobile_atoms=solvent_mobile,
        restrained_atoms=[index for index in range(state1.system.getNumParticles()) if index not in solvent_mobile],
        ca_atoms=[],
        restraint_reference_nm=state1.positions_nm,
        stage_name="solvent",
        output_pdb=pre_dir / "solvent_minimized.pdb",
        minimization_steps=spec.solvent_minimization_steps,
        nonmobile_restraint_kj_mol_nm2=spec.pre_relax_nonmobile_restraint_kj_mol_nm2,
        ca_restraint_kj_mol_nm2=spec.alpha_carbon_restraint_kj_mol_nm2,
        platform_name=platform_name,
    )
    report["stages"].append(solvent_row)

    local_mobile = _local_mobile_atoms(
        positions,
        anchor_atoms=set(embedder.irc_to_openmm.values()),
        radius_nm=spec.pre_relax_mobile_radius_nm,
    )
    local_mobile.update(solvent_atoms)
    local_mobile.update(int(index) for index in spec.mobile_atoms)
    local_mobile.update(int(index) for index in embedder.irc_to_openmm.values())
    positions, protein_row = _run_minimization_stage(
        config,
        state1,
        state2,
        builder,
        positions,
        mobile_atoms=local_mobile,
        restrained_atoms=[index for index in range(state1.system.getNumParticles()) if index not in local_mobile],
        ca_atoms=sorted(ca_atoms),
        restraint_reference_nm=positions,
        stage_name="protein_local",
        output_pdb=pre_dir / "protein_local_minimized.pdb",
        minimization_steps=spec.protein_minimization_steps,
        nonmobile_restraint_kj_mol_nm2=spec.pre_relax_nonmobile_restraint_kj_mol_nm2,
        ca_restraint_kj_mol_nm2=spec.alpha_carbon_restraint_kj_mol_nm2,
        platform_name=platform_name,
    )
    report["stages"].append(protein_row)
    report["output_positions_pdb"] = protein_row["pdb"]
    report["n_solvent_atoms"] = len(solvent_atoms)
    report["n_alpha_carbon_restraints"] = len(ca_atoms)
    report["n_local_mobile_atoms"] = len(local_mobile)
    (pre_dir / "pre_relaxation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return positions, report


def _run_minimization_stage(
    config: EVBConfig,
    state1,
    state2,
    builder: EVBSystemBuilder,
    positions_nm: np.ndarray,
    mobile_atoms: set[int],
    restrained_atoms: list[int],
    ca_atoms: list[int],
    restraint_reference_nm: np.ndarray,
    stage_name: str,
    output_pdb: Path,
    minimization_steps: int,
    nonmobile_restraint_kj_mol_nm2: float,
    ca_restraint_kj_mol_nm2: float,
    platform_name: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = config.irc.relaxation
    mapped_system = builder.build_openmm_mapped_system(
        state1,
        state2,
        lambda_value=0.0,
        delta_alpha=config.evb_parameters.delta_alpha or 0.0,
        add_cmmotion_remover=False,
    )
    if restrained_atoms:
        mapped_system.system.addForce(
            build_absolute_positional_restraint_force(
                restraint_reference_nm,
                atom_indices=restrained_atoms,
                force_constant_kj_mol_nm2=nonmobile_restraint_kj_mol_nm2,
                parameter_name=f"k_{stage_name}_nonmobile",
            )
        )
    if ca_atoms:
        mapped_system.system.addForce(
            build_absolute_positional_restraint_force(
                restraint_reference_nm,
                atom_indices=ca_atoms,
                force_constant_kj_mol_nm2=ca_restraint_kj_mol_nm2,
                parameter_name=f"k_{stage_name}_ca",
            )
        )
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = _create_openmm_context(
        mapped_system.system,
        integrator,
        platform_name,
        require_platform=spec.require_platform,
    )
    if state1.box_vectors_nm is not None:
        context.setPeriodicBoxVectors(*_to_box_vectors(state1.box_vectors_nm))
    context.setPositions(_to_openmm_positions(positions_nm))
    actual_platform = context.getPlatform().getName()
    initial_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    openmm.LocalEnergyMinimizer.minimize(
        context,
        spec.minimization_tolerance_kj_mol_nm * unit.kilojoule_per_mole / unit.nanometer,
        int(minimization_steps),
    )
    final_state = context.getState(getEnergy=True, getPositions=True)
    final_energy = final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    final_positions = np.asarray(final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    write_pdb(str(output_pdb), state1.topology, final_positions)
    return final_positions, {
        "stage": stage_name,
        "platform": actual_platform,
        "n_mobile_atoms": len(mobile_atoms),
        "n_restrained_atoms": len(restrained_atoms),
        "n_alpha_carbon_restraints": len(ca_atoms),
        "minimization_steps": int(minimization_steps),
        "initial_energy_kj_mol": float(initial_energy),
        "final_energy_kj_mol": float(final_energy),
        "pdb": str(output_pdb),
    }


def _create_openmm_context(system, integrator, platform_name: str | None, require_platform: bool = False):
    if platform_name:
        try:
            platform = openmm.Platform.getPlatformByName(platform_name)
            return openmm.Context(system, integrator, platform)
        except Exception as exc:
            if require_platform:
                raise RuntimeError(f"Requested OpenMM platform '{platform_name}' is not usable for IRC relaxation.") from exc
            return openmm.Context(system, integrator)
    return openmm.Context(system, integrator)


def _selected_relaxation_frames(canonical: CanonicalIRCPath, stride: int, explicit: list[int]) -> list[int]:
    if explicit:
        result = sorted({int(index) for index in explicit})
    else:
        stride = max(1, int(stride))
        result = list(range(0, len(canonical.frames), stride))
        for required in (canonical.rc_frame, canonical.ts_frame, canonical.product_frame):
            if required not in result:
                result.append(required)
        result = sorted(set(result))
    for index in result:
        if index < 0 or index >= len(canonical.frames):
            raise ValueError(f"IRC relaxation frame index {index} is outside the canonical IRC frame range.")
    return result


def _resolve_relaxation_mobile_atoms(config: EVBConfig, positions_nm: np.ndarray, embedder: ClusterIRCEmbedder) -> set[int]:
    spec = config.irc.relaxation
    mobile = set(int(index) for index in spec.mobile_atoms)
    anchor_atoms = set(embedder.irc_to_openmm.values())
    mobile.update(anchor_atoms)
    mobile.update(_local_mobile_atoms(positions_nm, anchor_atoms, spec.mobile_radius_nm))
    return mobile


def _local_mobile_atoms(positions_nm: np.ndarray, anchor_atoms: set[int], radius_nm: float) -> set[int]:
    if radius_nm <= 0.0 or not anchor_atoms:
        return set()
    anchor_coords = positions_nm[sorted(anchor_atoms)]
    distances = np.min(np.linalg.norm(positions_nm[:, None, :] - anchor_coords[None, :, :], axis=2), axis=1)
    return set(np.where(distances <= radius_nm)[0].astype(int).tolist())


def _topology_solvent_atoms(topology, solvent_residue_names: list[str]) -> set[int]:
    solvent_names = {name.upper() for name in solvent_residue_names}
    atoms = set()
    for atom in topology.atoms():
        if atom.residue.name.upper() in solvent_names:
            atoms.add(int(atom.index))
    return atoms


def _topology_alpha_carbon_atoms(topology) -> set[int]:
    return {int(atom.index) for atom in topology.atoms() if atom.name.upper() == "CA"}


def _write_relaxation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "canonical_frame",
        "original_frame",
        "lambda_value",
        "n_mobile_atoms",
        "n_restrained_atoms",
        "platform",
        "initial_energy_kj_mol",
        "final_energy_kj_mol",
        "pdb",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_cluster_embedder(config: EVBConfig, canonical: CanonicalIRCPath, state) -> ClusterIRCEmbedder | None:
    embedding = getattr(config.irc, "embedding", {}) or {}
    enabled = bool(embedding.get("enabled", False))
    if not enabled and len(canonical.frames[0].symbols) == state.system.getNumParticles():
        return None
    if not enabled:
        raise ValueError(
            f"IRC atom count ({len(canonical.frames[0].symbols)}) does not match OpenMM topology atom count "
            f"({state.system.getNumParticles()}). For cluster-model IRCs, set irc.embedding.enabled: true "
            "and provide an alpha-carbon mapping file."
        )
    mapping_file = embedding.get("alpha_carbon_mapping") or embedding.get("mapping_file")
    if mapping_file is None:
        raise ValueError("Cluster IRC embedding requires irc.embedding.alpha_carbon_mapping or mapping_file.")
    mapping_payload = _load_yaml_mapping(mapping_file)
    anchor_records = mapping_payload.get("alpha_carbon_mapping", mapping_payload)
    if not isinstance(anchor_records, list) or not anchor_records:
        raise ValueError(f"No alpha_carbon_mapping entries found in {mapping_file}.")
    max_match_angstrom = float(embedding.get("max_match_angstrom", 0.45))
    include_hydrogens = bool(embedding.get("include_hydrogens", True))
    max_anchor_error = float(embedding.get("max_anchor_error_angstrom", 0.35))
    topology_atoms = list(state.topology.atoms())
    positions_angstrom = np.asarray(state.positions_nm, dtype=float) * 10.0
    system_shift = _infer_reference_to_openmm_translation(anchor_records, topology_atoms, positions_angstrom)
    anchor_pairs, anchor_warnings = _build_anchor_pairs(
        anchor_records,
        topology_atoms,
        positions_angstrom,
        system_shift,
        max_anchor_error,
    )
    if len(anchor_pairs) < 3:
        raise ValueError(f"Only {len(anchor_pairs)} usable alpha-carbon anchors found; at least 3 are required for cluster embedding.")
    irc_anchor = np.asarray([pair[0] for pair in anchor_pairs], dtype=float)
    openmm_anchor = np.asarray([pair[1] for pair in anchor_pairs], dtype=float)
    rotation, translation = _fit_rigid_transform(irc_anchor, openmm_anchor)
    transformed_first = _apply_transform(canonical.frames[0].coordinates_angstrom, rotation, translation)
    auto_match = bool(embedding.get("auto_match", True))
    if auto_match:
        irc_to_openmm, map_warnings = _auto_match_cluster_atoms(
            canonical.frames[0].symbols,
            transformed_first,
            topology_atoms,
            positions_angstrom,
            max_match_angstrom=max_match_angstrom,
            include_hydrogens=include_hydrogens,
        )
    else:
        irc_to_openmm = {}
        map_warnings = ["Automatic same-element cluster-to-OpenMM matching is disabled; using explicit IRC-to-OpenMM mappings only."]
    explicit_mapping = _coerce_explicit_irc_mapping(embedding.get("irc_to_openmm"))
    if explicit_mapping:
        irc_to_openmm = _apply_explicit_mapping(irc_to_openmm, explicit_mapping, canonical.frames[0].symbols, topology_atoms)
        map_warnings.append(f"Applied {len(explicit_mapping)} explicit IRC-to-OpenMM atom mapping overrides.")
    if not irc_to_openmm:
        raise ValueError("Cluster IRC embedding found no same-element atom matches in the OpenMM system.")
    warnings = [
        "Cluster IRC coordinates were embedded into the full OpenMM system; unmatched cluster atoms remain at the reference OpenMM positions.",
        f"Embedded {len(irc_to_openmm)} of {len(canonical.frames[0].symbols)} IRC atoms using alpha-carbon anchored alignment.",
    ]
    warnings.extend(anchor_warnings)
    warnings.extend(map_warnings)
    return ClusterIRCEmbedder(state.positions_nm, irc_to_openmm, rotation, translation, warnings)


def _coerce_explicit_irc_mapping(payload) -> dict[int, int]:
    if not payload:
        return {}
    if isinstance(payload, dict):
        return {int(key): int(value) for key, value in payload.items()}
    if isinstance(payload, list):
        mapping = {}
        for item in payload:
            if isinstance(item, dict):
                mapping[int(item["irc_atom_index"])] = int(item["openmm_atom_index"])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                mapping[int(item[0])] = int(item[1])
            else:
                raise ValueError("irc.embedding.irc_to_openmm list entries must be mappings or 2-item pairs.")
        return mapping
    raise ValueError("irc.embedding.irc_to_openmm must be a mapping or list.")


def _apply_explicit_mapping(auto_mapping: dict[int, int], explicit: dict[int, int], symbols: list[str], topology_atoms) -> dict[int, int]:
    mapping = {irc: openmm for irc, openmm in auto_mapping.items() if irc not in explicit and openmm not in set(explicit.values())}
    atoms = list(topology_atoms)
    for irc_index, openmm_index in explicit.items():
        if irc_index < 0 or irc_index >= len(symbols):
            raise ValueError(f"Explicit IRC atom index {irc_index} is outside the IRC atom range.")
        if openmm_index < 0 or openmm_index >= len(atoms):
            raise ValueError(f"Explicit OpenMM atom index {openmm_index} is outside the topology atom range.")
        expected = symbols[irc_index].upper()
        actual = atoms[openmm_index].element.symbol.upper() if atoms[openmm_index].element is not None else atoms[openmm_index].name[0].upper()
        if expected != actual:
            raise ValueError(f"Explicit IRC/OpenMM mapping {irc_index}->{openmm_index} has element mismatch: {expected} vs {actual}.")
        mapping[irc_index] = openmm_index
    return mapping


def _load_yaml_mapping(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _infer_reference_to_openmm_translation(anchor_records, topology_atoms, positions_angstrom: np.ndarray) -> np.ndarray:
    translations = []
    for record in anchor_records:
        pdb = record.get("pdb", {})
        residue_name = pdb.get("residue_name")
        pdb_coord = np.asarray(pdb.get("coordinate_angstrom", record.get("irc_coordinate_angstrom")), dtype=float)
        if residue_name is None or pdb_coord.shape != (3,):
            continue
        for atom in topology_atoms:
            if atom.name != "CA" or atom.residue.name != residue_name:
                continue
            translations.append(positions_angstrom[atom.index] - pdb_coord)
    if not translations:
        return np.zeros(3, dtype=float)
    translations = np.asarray(translations, dtype=float)
    best_index = 0
    best_count = -1
    for index, candidate in enumerate(translations):
        distances = np.linalg.norm(translations - candidate[None, :], axis=1)
        count = int(np.sum(distances < 0.25))
        if count > best_count:
            best_index = index
            best_count = count
    close = np.linalg.norm(translations - translations[best_index][None, :], axis=1) < 0.25
    return np.median(translations[close], axis=0)


def _build_anchor_pairs(anchor_records, topology_atoms, positions_angstrom: np.ndarray, system_shift: np.ndarray, max_error: float):
    pairs = []
    warnings = []
    for record in anchor_records:
        pdb = record.get("pdb", {})
        residue_name = pdb.get("residue_name")
        pdb_coord = np.asarray(pdb.get("coordinate_angstrom", record.get("irc_coordinate_angstrom")), dtype=float)
        irc_coord = np.asarray(record.get("irc_coordinate_angstrom"), dtype=float)
        if residue_name is None or pdb_coord.shape != (3,) or irc_coord.shape != (3,):
            warnings.append(f"Skipped malformed alpha-carbon mapping entry for IRC atom {record.get('irc_atom_index')}.")
            continue
        target = pdb_coord + system_shift
        candidates = [
            atom for atom in topology_atoms
            if atom.name == "CA" and atom.residue.name == residue_name
        ]
        if not candidates:
            warnings.append(f"No OpenMM CA candidate found for {residue_name} anchor IRC atom {record.get('irc_atom_index')}.")
            continue
        best = min(candidates, key=lambda atom: float(np.linalg.norm(positions_angstrom[atom.index] - target)))
        error = float(np.linalg.norm(positions_angstrom[best.index] - target))
        if error > max_error:
            warnings.append(
                f"Skipped {residue_name} anchor IRC atom {record.get('irc_atom_index')}: nearest OpenMM CA is {error:.3f} A from shifted 5RGE coordinate."
            )
            continue
        pairs.append((irc_coord, positions_angstrom[best.index]))
    return pairs, warnings


def _fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source0 = source - source_center
    target0 = target - target_center
    u, _, vt = np.linalg.svd(source0.T @ target0)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _apply_transform(coords_angstrom: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return coords_angstrom @ rotation.T + translation


def _auto_match_cluster_atoms(
    symbols: list[str],
    transformed_angstrom: np.ndarray,
    topology_atoms,
    positions_angstrom: np.ndarray,
    max_match_angstrom: float,
    include_hydrogens: bool,
) -> tuple[dict[int, int], list[str]]:
    warnings = []
    used_openmm: set[int] = set()
    mapping: dict[int, int] = {}
    openmm_by_element: dict[str, list[int]] = {}
    for atom in topology_atoms:
        if atom.element is None:
            symbol = atom.name[0].upper()
        else:
            symbol = atom.element.symbol.upper()
        openmm_by_element.setdefault(symbol, []).append(atom.index)
    pairs: list[tuple[float, int, int]] = []
    for irc_index, symbol in enumerate(symbols):
        symbol = symbol.upper()
        if symbol == "H" and not include_hydrogens:
            continue
        candidates = openmm_by_element.get(symbol, [])
        if not candidates:
            continue
        distances = np.linalg.norm(positions_angstrom[candidates] - transformed_angstrom[irc_index][None, :], axis=1)
        for candidate, distance in zip(candidates, distances):
            distance = float(distance)
            if distance <= max_match_angstrom:
                pairs.append((distance, irc_index, int(candidate)))
    for _, irc_index, openmm_index in sorted(pairs, key=lambda item: item[0]):
        if irc_index in mapping or openmm_index in used_openmm:
            continue
        mapping[irc_index] = openmm_index
        used_openmm.add(openmm_index)
    skipped_symbols = {
        index for index, symbol in enumerate(symbols)
        if not (symbol.upper() == "H" and not include_hydrogens)
    }
    unmatched = len(skipped_symbols - set(mapping))
    if unmatched:
        warnings.append(
            f"{unmatched} IRC atoms were not mapped within {max_match_angstrom:.3f} A after alpha-carbon alignment; "
            "they will not be inserted into the full-system coordinates."
        )
    return mapping, warnings


def _write_scan_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "frame",
        "canonical_frame",
        "role",
        "path_order",
        "E1_kj_mol",
        "E2_kj_mol",
        "gap_raw_kj_mol",
        "delta_alpha_kj_mol",
        "gap_shifted_kj_mol",
        "Eevb_kj_mol",
        "w1",
        "w2",
        "comment",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def _build_report(config, canonical, reference, fit, rows, warnings):
    by_role = {row["role"]: row for row in rows if row["role"] in {"RC", "TS", "PROD"}}
    return {
        "input_irc_path": config.irc.path,
        "original_irc_order": canonical.original_order,
        "canonical_order_used": canonical.canonical_order,
        "selected_frames": {
            "rc": _frame_selection_payload(canonical, canonical.rc_frame),
            "ts": _frame_selection_payload(canonical, canonical.ts_frame),
            "product": _frame_selection_payload(canonical, canonical.product_frame),
        },
        "reference_profile": {
            "input_units": reference.units,
            "rc_kj_mol": reference.rc_kj_mol,
            "ts_kj_mol": reference.ts_kj_mol,
            "product_kj_mol": reference.product_kj_mol,
            "target_barrier_kj_mol": reference.target_barrier_kj_mol,
            "target_reaction_free_energy_kj_mol": reference.target_reaction_free_energy_kj_mol,
            "source_label": reference.source_label,
        },
        "fit": {
            "delta_alpha_kj_mol": fit.parameters.delta_alpha,
            "h12_kj_mol": fit.parameters.h12,
            "fitted_barrier_kj_mol": fit.fitted_barrier,
            "fitted_reaction_free_energy_kj_mol": fit.fitted_reaction_free_energy,
            "ts_weight2": fit.ts_weight2,
            "objective_value": fit.objective_value,
        },
        "role_energies": {
            role: {
                "E1_kj_mol": row["E1_kj_mol"],
                "E2_kj_mol": row["E2_kj_mol"],
                "gap_shifted_kj_mol": row["gap_shifted_kj_mol"],
                "Eevb_kj_mol": row["Eevb_kj_mol"],
                "w1": row["w1"],
                "w2": row["w2"],
            }
            for role, row in by_role.items()
        },
        "diagnostics": _diagnostic_values(rows, canonical, fit),
        "warnings": warnings,
    }


def _diagnostic_values(rows, canonical, fit) -> dict[str, Any]:
    gaps = np.asarray([float(row["gap_shifted_kj_mol"]) for row in rows], dtype=float)
    jumps = np.abs(np.diff(gaps)) if len(gaps) > 1 else np.asarray([0.0])
    ts_row = rows[canonical.ts_frame]
    return {
        "largest_adjacent_gap_jump_kj_mol": float(np.max(jumps)),
        "largest_abs_gap_kj_mol": float(np.max(np.abs(gaps))),
        "ts_gap_shifted_kj_mol": float(ts_row["gap_shifted_kj_mol"]),
        "ts_weight1": float(ts_row["w1"]),
        "ts_weight2": float(ts_row["w2"]),
        "barrier_equals_reaction_free_energy": abs(fit.fitted_barrier - fit.fitted_reaction_free_energy) < 1.0e-3,
    }


def _diagnose_scan(rows, canonical, fit) -> list[str]:
    warnings = []
    values = _diagnostic_values(rows, canonical, fit)
    if values["ts_weight2"] < 0.2 or values["ts_weight2"] > 0.8:
        warnings.append(f"TS EVB state-2 weight is {values['ts_weight2']:.3f}; expected a more mixed TS for a well-calibrated two-state EVB model.")
    if values["largest_adjacent_gap_jump_kj_mol"] > 1000.0:
        warnings.append(f"Largest adjacent shifted-gap jump is {values['largest_adjacent_gap_jump_kj_mol']:.3f} kJ/mol; check atom mapping/path smoothness.")
    if values["largest_abs_gap_kj_mol"] > 4184.0:
        warnings.append(f"Shifted gaps reach {values['largest_abs_gap_kj_mol'] * KJ_TO_KCAL:.3f} kcal/mol; this often indicates mapping/topology incompatibility.")
    if values["barrier_equals_reaction_free_energy"]:
        warnings.append("Fitted barrier equals reaction free energy within tolerance; this may indicate a monotonic or broken profile.")
    if canonical.ts_frame in {canonical.rc_frame, canonical.product_frame}:
        warnings.append("Selected TS frame is the same as an endpoint frame.")
    return warnings


def _write_gap_window_proposal(config: EVBConfig, canonical: CanonicalIRCPath, rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    n_windows = config.sampling.windows.gap_umbrella.n_windows or len(config.sampling.windows.gap_umbrella.centers_kj_mol) or 41
    gaps = np.asarray([float(row["gap_shifted_kj_mol"]) for row in rows], dtype=float)
    jumps = np.abs(np.diff(gaps)) if len(gaps) > 1 else np.asarray([0.0])
    warnings = []
    largest_jump = float(np.max(jumps))
    largest_abs_gap = float(np.max(np.abs(gaps)))
    pathological = largest_jump > 10000.0 or largest_abs_gap > 41840.0
    if pathological and not config.sampling.windows.gap_umbrella.allow_pathological_irc_windows:
        payload = {
            "n_windows": 0,
            "status": "blocked_pathological_irc_scan",
            "warnings": [
                f"Refused to generate IRC-derived gap umbrella windows because largest adjacent gap jump is {largest_jump:.3f} kJ/mol "
                f"and largest absolute shifted gap is {largest_abs_gap:.3f} kJ/mol.",
                "This usually indicates cluster-to-OpenMM atom mapping, topology, or diabatic-state incompatibility. Fix the IRC embedding before production sampling.",
                "Set sampling.windows.gap_umbrella.allow_pathological_irc_windows: true only for debugging, not for production.",
            ],
            "windows": [],
        }
        (output_dir / "irc_gap_umbrella_windows.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
    strategy = config.sampling.windows.gap_umbrella.center_strategy
    if largest_jump > 1000.0:
        lo, hi = np.percentile(gaps, [5, 95])
        warnings.append("Large adjacent gap jump detected; using 5th/95th percentile gap range for proposed windows.")
        centers = np.linspace(float(lo), float(hi), n_windows)
        center_strategy = "linear_percentile_range_due_to_large_gap_jump"
    else:
        endpoint1 = float(gaps[canonical.rc_frame])
        endpoint2 = float(gaps[canonical.product_frame])
        ts_gap = float(gaps[canonical.ts_frame])
        if strategy == "irc_ts":
            centers = _ts_focused_gap_centers(endpoint1, ts_gap, endpoint2, n_windows)
            center_strategy = "ts_focused_from_irc_role_gaps"
        elif strategy == "linear":
            lo = min(endpoint1, endpoint2) - config.sampling.windows.gap_umbrella.basin_extension_kj_mol
            hi = max(endpoint1, endpoint2) + config.sampling.windows.gap_umbrella.basin_extension_kj_mol
            centers = np.linspace(lo, hi, n_windows)
            center_strategy = "linear_endpoint_range"
        elif strategy == "evb_mixing":
            centers = _mixing_focused_gap_centers(
                endpoint1,
                endpoint2,
                n_windows,
                mixing_gap=config.sampling.windows.gap_umbrella.mixing_gap_kj_mol,
                basin_extension=config.sampling.windows.gap_umbrella.basin_extension_kj_mol,
            )
            center_strategy = "evb_mixing_focused_from_irc_endpoint_gaps"
        else:
            raise ValueError("sampling.windows.gap_umbrella.center_strategy must be one of: evb_mixing, irc_ts, linear")
    seeds_dir = output_dir / "irc_window_seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for index, center in enumerate(centers):
        frame_index = int(np.argmin(np.abs(gaps - center)))
        frame = canonical.frames[frame_index]
        seed_path = seeds_dir / f"w{index:03d}_seed.pdb"
        try:
            # PDB output is only possible when OpenMM is installed and topology is available through sampling later.
            _write_xyz_seed(seeds_dir / f"w{index:03d}_seed.xyz", frame.symbols, frame.coordinates_angstrom, frame.comment)
            seed_file = seeds_dir / f"w{index:03d}_seed.xyz"
        except Exception:
            seed_file = ""
        seed_rows.append(
            {
                "window_id": f"w{index:03d}",
                "gap_center_kj_mol": float(center),
                "seed_canonical_frame": frame_index,
                "seed_original_frame": frame.original_index,
                "seed_coordinates": str(seed_file),
            }
        )
    with (output_dir / "irc_gap_umbrella_windows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    payload = {
        "n_windows": n_windows,
        "center_strategy": center_strategy,
        "labels": {
            "endpoint1": "E1",
            "transition_region": "EVB mixing" if center_strategy == "evb_mixing_focused_from_irc_endpoint_gaps" else "TS",
            "endpoint2": "E2",
        },
        "warnings": warnings,
        "windows": seed_rows,
    }
    (output_dir / "irc_gap_umbrella_windows.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _ts_focused_gap_centers(endpoint1: float, ts_gap: float, endpoint2: float, n_windows: int) -> np.ndarray:
    if n_windows <= 1:
        return np.asarray([ts_gap], dtype=float)
    # Keep TS at the center index for odd n, and dense central windows for even n.
    left_n = n_windows // 2 + 1
    right_n = n_windows - left_n + 1
    left = _dense_toward_target(endpoint1, ts_gap, left_n)
    right = _dense_toward_target(endpoint2, ts_gap, right_n)[::-1]
    centers = np.concatenate([left[:-1], right])
    if endpoint1 > endpoint2:
        centers = centers[::-1]
    return centers.astype(float)


def _mixing_focused_gap_centers(
    endpoint1: float,
    endpoint2: float,
    n_windows: int,
    mixing_gap: float = 0.0,
    basin_extension: float = 0.0,
) -> np.ndarray:
    if n_windows <= 1:
        return np.asarray([mixing_gap], dtype=float)
    low_endpoint = min(endpoint1, endpoint2)
    high_endpoint = max(endpoint1, endpoint2)
    low = min(low_endpoint, mixing_gap) - max(0.0, basin_extension)
    high = max(high_endpoint, mixing_gap) + max(0.0, basin_extension)
    if not (low < mixing_gap < high):
        return np.linspace(low, high, n_windows)
    left_n = n_windows // 2 + 1
    right_n = n_windows - left_n + 1
    left = np.linspace(low, mixing_gap, left_n)
    right = np.linspace(high, mixing_gap, right_n)[::-1]
    return np.concatenate([left[:-1], right]).astype(float)


def _dense_toward_target(start: float, target: float, count: int) -> np.ndarray:
    if count <= 1:
        return np.asarray([target], dtype=float)
    t = np.linspace(0.0, 1.0, count)
    # 1-(1-t)^2 approaches the target with decreasing step size, so windows are denser near TS.
    scaled = 1.0 - (1.0 - t) ** 2
    return start + (target - start) * scaled


def _write_xyz_seed(path: Path, symbols: list[str], coordinates_angstrom: np.ndarray, comment: str) -> None:
    lines = [str(len(symbols)), comment]
    for symbol, xyz in zip(symbols, coordinates_angstrom):
        lines.append(f"{symbol} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _role_for_frame(canonical: CanonicalIRCPath, frame_index: int, comment_role: str | None) -> str | None:
    if frame_index == canonical.rc_frame:
        return "RC"
    if frame_index == canonical.ts_frame:
        return "TS"
    if frame_index == canonical.product_frame:
        return "PROD"
    return comment_role


def _frame_selection_payload(canonical: CanonicalIRCPath, canonical_index: int) -> dict[str, int]:
    return {
        "canonical_frame": canonical_index,
        "original_frame": canonical.canonical_to_original[canonical_index],
    }
