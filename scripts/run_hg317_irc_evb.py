from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from kemp_evb.evb import EVBHamiltonian, EVBParameters
from kemp_evb.fitting.irc_fit import fit_evb_to_irc_profile
from kemp_evb.irc import read_irc_xyz
from kemp_evb.openmm_backend import AmberSystemLoader, EVBSystemBuilder, _to_box_vectors, _to_openmm_positions


ROOT = Path(__file__).resolve().parents[1]
IRC_XYZ = ROOT / "examples" / "HG3.17_CM_IRC.xyz"
PDB_5RGE = ROOT / "inputs" / "5RGE.pdb"
PREP = ROOT / "prep" / "hg317_full_irc"
OUT = ROOT / "outputs" / "hg317_irc_evb"

STATE1_PRMTOP = PREP / "state1_reactant_matched16.prmtop"
STATE1_INPCRD = PREP / "state1_reactant_matched16.inpcrd"
STATE2_PRMTOP = PREP / "state2_product_matched16.prmtop"
STATE2_INPCRD = PREP / "state2_product_matched16.inpcrd"

IRC_TO_LIGAND_ATOMS = [6, 7, 62, 63, 64, 65, 66, 67, 68, 81, 82, 83, 194, 195, 196, 197]
SUBSTRATE_PROTON_ATOM = 197
REACTANT_DONOR_ATOM = 68
PRODUCT_ACCEPTOR_ATOM = 76
HARTREE_TO_KJ_MOL = 2625.499638
KJ_TO_KCAL_MOL = 1.0 / 4.184


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = read_irc_xyz(IRC_XYZ)
    proton_profile = _proton_transfer_profile(frames)
    _write_proton_profile(proton_profile)

    loader = AmberSystemLoader(nonbonded_method="PME", constraints="None")
    state1 = loader.load(str(STATE1_PRMTOP), str(STATE1_INPCRD))
    state2 = loader.load(str(STATE2_PRMTOP), str(STATE2_INPCRD))
    EVBSystemBuilder.validate_compatibility(state1, state2)
    reference_state = loader.load(str(PREP / "state1_reactant.prmtop"), str(PREP / "state1_reactant.inpcrd"))
    if reference_state.atom_names != state1.atom_names or not np.allclose(reference_state.masses_amu, state1.masses_amu):
        raise ValueError("Original state-1 coordinates are not compatible with the matched draft topology.")

    ligand_indices = _ligand_atom_indices(state1)
    alignment_shift_angstrom = _irc_to_amber_translation(frames, reference_state.positions_nm, ligand_indices)
    platform_name = _choose_platform()
    rows, actual_platform_name = _evaluate_irc_frames(
        frames, state1, state2, reference_state.positions_nm, ligand_indices, alignment_shift_angstrom, platform_name
    )
    _write_singlepoint_csv(rows)

    finite = [row for row in rows if row["qm_relative_kj_mol"] is not None]
    e1 = np.asarray([row["e1_kj_mol"] for row in finite], dtype=float)
    e2 = np.asarray([row["e2_kj_mol"] for row in finite], dtype=float)
    qm = np.asarray([row["qm_relative_kj_mol"] for row in finite], dtype=float)
    fit, evb_relative = fit_evb_to_irc_profile(e1, e2, qm, levels=7, samples_per_axis=121)
    ham = EVBHamiltonian(EVBParameters(fit.delta_alpha_kj_mol, fit.h12_kj_mol))
    for row in rows:
        evb, w1, w2 = ham.lower_eigenvalue(row["e1_kj_mol"], row["e2_kj_mol"])
        row["gap_kj_mol"] = ham.gap(row["e1_kj_mol"], row["e2_kj_mol"])
        row["evb_kj_mol"] = evb
        row["weight1"] = w1
        row["weight2"] = w2
    evb0 = next(row["evb_kj_mol"] for row in rows if row["qm_relative_kj_mol"] is not None)
    for row in rows:
        row["evb_relative_kj_mol"] = row["evb_kj_mol"] - evb0
    _add_kcal_columns(rows)
    _write_fitted_csv(rows)
    _write_plots(rows)

    acceptor_mapping = _map_irc_atom_to_5rge(frames[0].coordinates_angstrom[PRODUCT_ACCEPTOR_ATOM])
    fit_kcal = {
        "delta_alpha_kcal_mol": fit.delta_alpha_kj_mol * KJ_TO_KCAL_MOL,
        "h12_kcal_mol": fit.h12_kj_mol * KJ_TO_KCAL_MOL,
        "objective_rmse_kcal_mol": fit.objective_rmse_kj_mol * KJ_TO_KCAL_MOL,
        "barrier_kcal_mol": fit.barrier_kj_mol * KJ_TO_KCAL_MOL,
        "reaction_energy_kcal_mol": fit.reaction_energy_kj_mol * KJ_TO_KCAL_MOL,
    }
    payload = {
        "status": "completed_draft_matched_topology",
        "warning": (
            "This EVB run uses synchronized draft matched topologies. It is useful for testing the IRC-driven EVB workflow, "
            "but the product-state ASP127-OD2--H197 parameters are a minimal matched-topology repair. Publication-level EVB "
            "calibration should replace these placeholder reactive-region terms with parameters validated against the QM model."
        ),
        "requested_platform": platform_name or "OpenMM default",
        "actual_platform": actual_platform_name,
        "irc_xyz": str(IRC_XYZ.relative_to(ROOT)),
        "state1": str(STATE1_PRMTOP.relative_to(ROOT)),
        "state2": str(STATE2_PRMTOP.relative_to(ROOT)),
        "n_frames": len(rows),
        "n_finite_qm_frames": len(finite),
        "irc_to_amber_translation_angstrom": alignment_shift_angstrom.tolist(),
        "fit": asdict(fit),
        "fit_kcal_mol": fit_kcal,
        "proton_transfer": {
            "moving_proton_irc_atom": SUBSTRATE_PROTON_ATOM,
            "reactant_substrate_donor_irc_atom": REACTANT_DONOR_ATOM,
            "product_acceptor_irc_atom": PRODUCT_ACCEPTOR_ATOM,
            "product_acceptor_5rge_mapping": acceptor_mapping,
        },
        "outputs": {
            "singlepoints_csv": str((OUT / "irc_singlepoints.csv").relative_to(ROOT)),
            "fitted_profile_csv": str((OUT / "irc_evb_fitted_profile.csv").relative_to(ROOT)),
            "qm_vs_evb_plot": str((OUT / "irc_qm_vs_evb_kcal_mol.png").relative_to(ROOT)),
            "gap_plot": str((OUT / "irc_gap_kcal_mol.png").relative_to(ROOT)),
            "weights_plot": str((OUT / "irc_weights.png").relative_to(ROOT)),
            "proton_transfer_plot": str((OUT / "h197_transfer_distances.png").relative_to(ROOT)),
        },
    }
    (OUT / "irc_evb_run_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown_report(payload)
    print(json.dumps(payload, indent=2))


def _choose_platform() -> str | None:
    try:
        import openmm

        names = [openmm.Platform.getPlatform(index).getName() for index in range(openmm.Platform.getNumPlatforms())]
        return "CUDA" if "CUDA" in names else None
    except Exception:
        return None


def _ligand_atom_indices(state) -> list[int]:
    residues = list(state.topology.residues())
    ligands = [residue for residue in residues if residue.name in {"SBR", "SDM"}]
    if len(ligands) != 1:
        raise ValueError(f"Expected exactly one matched ligand residue, found {[residue.name for residue in ligands]}.")
    indices = [atom.index for atom in ligands[0].atoms()]
    if len(indices) != len(IRC_TO_LIGAND_ATOMS):
        raise ValueError(f"Expected {len(IRC_TO_LIGAND_ATOMS)} ligand atoms, found {len(indices)}.")
    return indices


def _irc_to_amber_translation(frames, reference_positions_nm: np.ndarray, ligand_indices: list[int]) -> np.ndarray:
    reactant = frames[-1].coordinates_angstrom[IRC_TO_LIGAND_ATOMS]
    amber = np.asarray(reference_positions_nm[ligand_indices], dtype=float) * 10.0
    return amber.mean(axis=0) - reactant.mean(axis=0)


def _evaluate_irc_frames(
    frames,
    state1,
    state2,
    reference_positions_nm: np.ndarray,
    ligand_indices: list[int],
    alignment_shift_angstrom: np.ndarray,
    platform_name: str | None,
) -> tuple[list[dict], str]:
    import openmm

    integrator1 = openmm.VerletIntegrator(0.001)
    integrator2 = openmm.VerletIntegrator(0.001)
    if platform_name:
        try:
            platform = openmm.Platform.getPlatformByName(platform_name)
            context1 = openmm.Context(state1.system, integrator1, platform)
            context2 = openmm.Context(state2.system, integrator2, platform)
        except Exception:
            integrator1 = openmm.VerletIntegrator(0.001)
            integrator2 = openmm.VerletIntegrator(0.001)
            context1 = openmm.Context(state1.system, integrator1)
            context2 = openmm.Context(state2.system, integrator2)
    else:
        context1 = openmm.Context(state1.system, integrator1)
        context2 = openmm.Context(state2.system, integrator2)
    if state1.box_vectors_nm is not None:
        context1.setPeriodicBoxVectors(*_to_box_vectors(state1.box_vectors_nm))
    if state2.box_vectors_nm is not None:
        context2.setPeriodicBoxVectors(*_to_box_vectors(state2.box_vectors_nm))
    actual_platform_name = context1.getPlatform().getName()

    rows = []
    reference_positions = np.asarray(reference_positions_nm, dtype=float)
    qm_reference = next(frame.energy_hartree for frame in frames if frame.energy_hartree is not None)
    for frame in frames:
        positions = reference_positions.copy()
        for ligand_index, irc_index in zip(ligand_indices, IRC_TO_LIGAND_ATOMS):
            positions[ligand_index] = (frame.coordinates_angstrom[irc_index] + alignment_shift_angstrom) * 0.1
        context1.setPositions(_to_openmm_positions(positions))
        context2.setPositions(_to_openmm_positions(positions))
        e1 = context1.getState(getEnergy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        e2 = context2.getState(getEnergy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        qm_relative = None if frame.energy_hartree is None else (frame.energy_hartree - qm_reference) * HARTREE_TO_KJ_MOL
        rows.append(
            {
                "frame": frame.index,
                "qm_energy_hartree": frame.energy_hartree,
                "qm_relative_kj_mol": qm_relative,
                "e1_kj_mol": float(e1),
                "e2_kj_mol": float(e2),
                "raw_gap_kj_mol": float(e1 - e2),
            }
        )
    return rows, actual_platform_name


def _add_kcal_columns(rows: list[dict]) -> None:
    fields = [
        "qm_relative",
        "e1",
        "e2",
        "raw_gap",
        "gap",
        "evb",
        "evb_relative",
    ]
    for row in rows:
        for field in fields:
            kj_key = f"{field}_kj_mol"
            if kj_key in row:
                row[f"{field}_kcal_mol"] = None if row[kj_key] is None else row[kj_key] * KJ_TO_KCAL_MOL


def _proton_transfer_profile(frames) -> list[dict]:
    rows = []
    for frame in frames:
        h = frame.coordinates_angstrom[SUBSTRATE_PROTON_ATOM]
        donor = frame.coordinates_angstrom[REACTANT_DONOR_ATOM]
        acceptor = frame.coordinates_angstrom[PRODUCT_ACCEPTOR_ATOM]
        rows.append(
            {
                "frame": frame.index,
                "energy_hartree": frame.energy_hartree,
                "h197_to_c68_angstrom": float(np.linalg.norm(h - donor)),
                "h197_to_o76_angstrom": float(np.linalg.norm(h - acceptor)),
            }
        )
    return rows


def _map_irc_atom_to_5rge(coord_angstrom: np.ndarray) -> dict:
    best = None
    for line in PDB_5RGE.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        altloc = line[16].strip()
        if altloc not in ("", "A"):
            continue
        xyz = np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
        distance = float(np.linalg.norm(coord_angstrom - xyz))
        if best is None or distance < best["distance_angstrom"]:
            best = {
                "distance_angstrom": distance,
                "atom_name": line[12:16].strip(),
                "residue_name": line[17:20].strip(),
                "chain_id": line[21].strip(),
                "residue_number": int(line[22:26]),
                "serial": int(line[6:11]),
                "element": (line[76:78].strip() or line[12:16].strip()[0]).upper(),
            }
    if best is None:
        raise ValueError("No atoms found in 5RGE PDB.")
    return best


def _write_proton_profile(rows: list[dict]) -> None:
    with (OUT / "h197_transfer_distances.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "energy_hartree", "h197_to_c68_angstrom", "h197_to_o76_angstrom"])
        writer.writeheader()
        writer.writerows(rows)


def _write_singlepoint_csv(rows: list[dict]) -> None:
    _add_kcal_columns(rows)
    fields = [
        "frame",
        "qm_energy_hartree",
        "qm_relative_kj_mol",
        "qm_relative_kcal_mol",
        "e1_kj_mol",
        "e1_kcal_mol",
        "e2_kj_mol",
        "e2_kcal_mol",
        "raw_gap_kj_mol",
        "raw_gap_kcal_mol",
    ]
    with (OUT / "irc_singlepoints.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_fitted_csv(rows: list[dict]) -> None:
    fields = [
        "frame",
        "qm_energy_hartree",
        "qm_relative_kj_mol",
        "qm_relative_kcal_mol",
        "e1_kj_mol",
        "e1_kcal_mol",
        "e2_kj_mol",
        "e2_kcal_mol",
        "raw_gap_kj_mol",
        "raw_gap_kcal_mol",
        "gap_kj_mol",
        "gap_kcal_mol",
        "evb_kj_mol",
        "evb_kcal_mol",
        "evb_relative_kj_mol",
        "evb_relative_kcal_mol",
        "weight1",
        "weight2",
    ]
    with (OUT / "irc_evb_fitted_profile.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(rows: list[dict]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    x = np.asarray([row["frame"] for row in rows], dtype=float)
    qm = np.asarray([np.nan if row["qm_relative_kcal_mol"] is None else row["qm_relative_kcal_mol"] for row in rows], dtype=float)
    evb = np.asarray([row["evb_relative_kcal_mol"] for row in rows], dtype=float)
    gap = np.asarray([row["gap_kcal_mol"] for row in rows], dtype=float)
    w1 = np.asarray([row["weight1"] for row in rows], dtype=float)
    w2 = np.asarray([row["weight2"] for row in rows], dtype=float)

    plt.figure(figsize=(7, 4))
    plt.plot(x, qm, label="QM IRC", linewidth=2)
    plt.plot(x, evb, label="EVB fit", linewidth=2)
    plt.xlabel("IRC frame")
    plt.ylabel("Relative energy (kcal/mol)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "irc_qm_vs_evb_kcal_mol.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x, gap, linewidth=2)
    plt.xlabel("IRC frame")
    plt.ylabel("EVB gap E1 - E2 - delta_alpha (kcal/mol)")
    plt.tight_layout()
    plt.savefig(OUT / "irc_gap_kcal_mol.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x, w1, label="state 1 weight", linewidth=2)
    plt.plot(x, w2, label="state 2 weight", linewidth=2)
    plt.xlabel("IRC frame")
    plt.ylabel("EVB weight")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "irc_weights.png", dpi=200)
    plt.close()

    proton_rows = _proton_transfer_profile(read_irc_xyz(IRC_XYZ))
    frames = [row["frame"] for row in proton_rows]
    donor = [row["h197_to_c68_angstrom"] for row in proton_rows]
    acceptor = [row["h197_to_o76_angstrom"] for row in proton_rows]
    plt.figure(figsize=(7, 4))
    plt.plot(frames, donor, label="H197-C68 substrate donor")
    plt.plot(frames, acceptor, label="H197-O76 / ASP127 OD2 acceptor")
    plt.xlabel("IRC frame")
    plt.ylabel("Distance (A)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "h197_transfer_distances.png", dpi=200)
    plt.close()


def _write_markdown_report(payload: dict) -> None:
    fit = payload["fit_kcal_mol"]
    mapping = payload["proton_transfer"]["product_acceptor_5rge_mapping"]
    lines = [
        "# HG3.17 IRC EVB Run Report",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Fit",
        "",
        f"- delta_alpha: `{fit['delta_alpha_kcal_mol']:.6f}` kcal/mol",
        f"- H12: `{fit['h12_kcal_mol']:.6f}` kcal/mol",
        f"- RMSE: `{fit['objective_rmse_kcal_mol']:.6f}` kcal/mol",
        f"- fitted barrier: `{fit['barrier_kcal_mol']:.6f}` kcal/mol",
        f"- reaction energy: `{fit['reaction_energy_kcal_mol']:.6f}` kcal/mol",
        "",
        "## Proton Transfer Mapping",
        "",
        f"- Moving proton: IRC atom `{SUBSTRATE_PROTON_ATOM}`",
        f"- Reactant donor: IRC atom `{REACTANT_DONOR_ATOM}`",
        f"- Product acceptor: IRC atom `{PRODUCT_ACCEPTOR_ATOM}`",
        (
            "- Product acceptor maps to "
            f"{mapping['residue_name']} {mapping['chain_id']}{mapping['residue_number']} "
            f"{mapping['atom_name']} serial {mapping['serial']} "
            f"({mapping['distance_angstrom']:.3f} A)"
        ),
        "",
        "## Warning",
        "",
        payload["warning"],
    ]
    (OUT / "irc_evb_run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
