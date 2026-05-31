from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QM_DIR = ROOT / "systems" / "KEMP_implicit"
OLD_DIR = ROOT / "systems" / "KEMP-solvent"
OUT_DIR = ROOT / "prep" / "kemp_qm_openmm"

RC_XYZ = QM_DIR / "RC_final.xyz"
PROD_XYZ = QM_DIR / "PROD_final.xyz"
OLD_RC_PDB = OLD_DIR / "RC.pdb"
OLD_PROD_PDB = OLD_DIR / "PROD.pdb"

RC_SUBSTRATE_NAMES = [
    "N1", "N2", "C1", "C2", "C3", "C4", "C5", "C6",
    "C7", "O1", "O2", "O3", "H1", "H2", "H3", "H4",
]
RC_PARTNER_NAMES = ["C8", "C9", "O4", "O5", "H5", "H6", "H7"]
PROD_SUBSTRATE_NAMES = RC_SUBSTRATE_NAMES[:15]
PROD_PARTNER_NAMES = ["H4", "C8", "C9", "O4", "O5", "H5", "H6", "H7"]

RC_SUBSTRATE_RESNAME = "SBR"
RC_PARTNER_RESNAME = "BAR"
PROD_SUBSTRATE_RESNAME = "SDP"
PROD_PARTNER_RESNAME = "ACP"

ALIGNMENT_ATOM_INDICES = list(range(15))
REACTIVE_ATOM_COUNT = 23
OLD_RC_BOX = (31.466, 31.835, 30.366)


def _load_xyz(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = int(lines[0].strip())
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + natoms]:
        symbol, xs, ys, zs = line.split()[:4]
        symbols.append(symbol)
        coords.append([float(xs), float(ys), float(zs)])
    return symbols, coords


def _load_first_n_pdb_coords(path: Path, n: int) -> list[list[float]]:
    coords: list[list[float]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                coords.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
                if len(coords) == n:
                    break
    if len(coords) != n:
        raise ValueError(f"{path} did not contain {n} ATOM/HETATM records")
    return coords


def _centroid(coords: list[list[float]]) -> list[float]:
    n = float(len(coords))
    return [sum(c[i] for c in coords) / n for i in range(3)]


def _apply_rot(coords: list[list[float]], rot: list[list[float]]) -> list[list[float]]:
    return [[sum(rot[i][j] * c[j] for j in range(3)) for i in range(3)] for c in coords]


def _power_iteration_symmetric4(mat: list[list[float]], n_iter: int = 64) -> list[float]:
    vec = [1.0, 0.0, 0.0, 0.0]
    for _ in range(n_iter):
        nxt = [sum(mat[i][j] * vec[j] for j in range(4)) for i in range(4)]
        norm = math.sqrt(sum(x * x for x in nxt))
        if norm == 0.0:
            break
        vec = [x / norm for x in nxt]
    return vec


def _rotation_from_quaternion(q: list[float]) -> list[list[float]]:
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _kabsch_align(mobile: list[list[float]], target: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    mob_c = _centroid(mobile)
    tar_c = _centroid(target)
    x = [[c[i] - mob_c[i] for i in range(3)] for c in mobile]
    y = [[c[i] - tar_c[i] for i in range(3)] for c in target]
    s = [[0.0, 0.0, 0.0] for _ in range(3)]
    for xm, yt in zip(x, y):
        for i in range(3):
            for j in range(3):
                s[i][j] += xm[i] * yt[j]
    sxx, sxy, sxz = s[0]
    syx, syy, syz = s[1]
    szx, szy, szz = s[2]
    k = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]
    q = _power_iteration_symmetric4(k)
    rot = _rotation_from_quaternion(q)
    return rot, mob_c, tar_c


def _transform_all(coords: list[list[float]], rot: list[list[float]], mob_c: list[float], tar_c: list[float]) -> list[list[float]]:
    centered = [[c[i] - mob_c[i] for i in range(3)] for c in coords]
    rotated = _apply_rot(centered, rot)
    return [[c[i] + tar_c[i] for i in range(3)] for c in rotated]


def _write_pdb_with_names(
    path: Path,
    symbols: list[str],
    coords: list[list[float]],
    first_names: list[str],
    first_resname: str,
    second_names: list[str],
    second_resname: str,
) -> None:
    first_count = len(first_names)
    with path.open("w", encoding="utf-8") as handle:
        for index, (symbol, (x, y, z), atom_name) in enumerate(zip(symbols[:first_count], coords[:first_count], first_names), start=1):
            handle.write(
                f"HETATM{index:5d} {atom_name:>4s} {first_resname:>3s} A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol:>2s}\n"
            )
        for offset, (symbol, (x, y, z), atom_name) in enumerate(zip(symbols[first_count:], coords[first_count:], second_names), start=1):
            handle.write(
                f"HETATM{first_count + offset:5d} {atom_name:>4s} {second_resname:>3s} A   2    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol:>2s}\n"
            )
        handle.write("END\n")


def _write_simple_pdb(path: Path, symbols: list[str], coords: list[list[float]], residue_name: str) -> None:
    counts: dict[str, int] = defaultdict(int)
    with path.open("w", encoding="utf-8") as handle:
        for index, (symbol, (x, y, z)) in enumerate(zip(symbols, coords), start=1):
            counts[symbol] += 1
            atom_name = f"{symbol}{counts[symbol]}"
            handle.write(
                f"HETATM{index:5d} {atom_name:>4s} {residue_name:>3s} A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol:>2s}\n"
            )
        handle.write("END\n")


def _write_names_file(path: Path, names: list[str]) -> None:
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def _write_runner() -> None:
    (OUT_DIR / "run_from_previous_solvent.sh").write_text(
        """#!/usr/bin/env bash
source "$HOME/.bashrc"
conda activate ambertraj
set -euo pipefail
export PATH="$CONDA_PREFIX/bin:$PATH"

cd "$(dirname "$0")"
mkdir -p 02_ambertools 03_solvated_rc 03_solvated_prod 05_templates 06_minimized_states

antechamber -i 01_qm_inputs/rc_substrate_simple.pdb -fi pdb -o 02_ambertools/rc_substrate_raw.mol2 -fo mol2 -c bcc -s 2 -nc 0
python ../../scripts/rename_kemp_qm_mol2.py --input 02_ambertools/rc_substrate_raw.mol2 --output 02_ambertools/rc_substrate_named.mol2 --resname SBR --names-file 01_qm_inputs/rc_substrate.names
parmchk2 -i 02_ambertools/rc_substrate_named.mol2 -f mol2 -o 02_ambertools/rc_substrate.frcmod

antechamber -i 01_qm_inputs/rc_partner_simple.pdb -fi pdb -o 02_ambertools/rc_partner_raw.mol2 -fo mol2 -c bcc -s 2 -nc -1
python ../../scripts/rename_kemp_qm_mol2.py --input 02_ambertools/rc_partner_raw.mol2 --output 02_ambertools/rc_partner_named.mol2 --resname BAR --names-file 01_qm_inputs/rc_partner.names
parmchk2 -i 02_ambertools/rc_partner_named.mol2 -f mol2 -o 02_ambertools/rc_partner.frcmod

antechamber -i 01_qm_inputs/prod_substrate_simple.pdb -fi pdb -o 02_ambertools/prod_substrate_raw.mol2 -fo mol2 -c bcc -s 2 -nc -1
python ../../scripts/rename_kemp_qm_mol2.py --input 02_ambertools/prod_substrate_raw.mol2 --output 02_ambertools/prod_substrate_named.mol2 --resname SDP --names-file 01_qm_inputs/prod_substrate.names
parmchk2 -i 02_ambertools/prod_substrate_named.mol2 -f mol2 -o 02_ambertools/prod_substrate.frcmod

antechamber -i 01_qm_inputs/prod_partner_simple.pdb -fi pdb -o 02_ambertools/prod_partner_raw.mol2 -fo mol2 -c bcc -s 2 -nc 0
python ../../scripts/rename_kemp_qm_mol2.py --input 02_ambertools/prod_partner_raw.mol2 --output 02_ambertools/prod_partner_named.mol2 --resname ACP --names-file 01_qm_inputs/prod_partner.names
parmchk2 -i 02_ambertools/prod_partner_named.mol2 -f mol2 -o 02_ambertools/prod_partner.frcmod

python ../../scripts/extract_solvent_template_from_pdb.py \
  --input-pdb ../../systems/KEMP-solvent/RC.pdb \
  --output-pdb 05_templates/solvent_only_from_old_rc.pdb \
  --reactive-atom-count 23

python ../../scripts/build_kemp_prod_from_rc_template.py \
  --rc-master-pdb ../../systems/KEMP-solvent/RC.pdb \
  --rc-reactive-pdb 01_qm_inputs/rc_complex_fragmented.pdb \
  --prod-reactive-pdb 01_qm_inputs/prod_complex_fragmented.pdb \
  --rc-output-pdb 05_templates/rc_in_old_rc_frame_fragmented.pdb \
  --prod-output-pdb 05_templates/prod_in_old_rc_frame_fragmented.pdb \
  --reactive-atom-count 23

tleap -f tleap_rc_from_previous_solvent.in
tleap -f tleap_prod_from_previous_solvent.in

echo 'Initial solvated QM-guided states are under 03_solvated_rc/ and 03_solvated_prod/.'
""",
        encoding="utf-8",
    )


def _write_tleap_templates() -> None:
    box_line = f"set RC box {{ {OLD_RC_BOX[0]:.3f} {OLD_RC_BOX[1]:.3f} {OLD_RC_BOX[2]:.3f} }}"
    (OUT_DIR / "tleap_rc_from_previous_solvent.in").write_text(
        f"""source leaprc.gaff2
source leaprc.water.tip3p

loadamberparams 02_ambertools/rc_substrate.frcmod
loadamberparams 02_ambertools/rc_partner.frcmod

SUBR = loadmol2 02_ambertools/rc_substrate_named.mol2
BAR = loadmol2 02_ambertools/rc_partner_named.mol2
SOLV = loadpdb 05_templates/solvent_only_from_old_rc.pdb

RC = combine {{ SUBR BAR SOLV }}
{box_line}
check RC

savepdb RC 03_solvated_rc/RC_solvated_initial.pdb
saveamberparm RC 03_solvated_rc/RC_solvated_initial.prmtop 03_solvated_rc/RC_solvated_initial.inpcrd
quit
""",
        encoding="utf-8",
    )
    (OUT_DIR / "tleap_prod_from_previous_solvent.in").write_text(
        f"""source leaprc.gaff2
source leaprc.water.tip3p

loadamberparams 02_ambertools/prod_substrate.frcmod
loadamberparams 02_ambertools/prod_partner.frcmod

SUBP = loadmol2 02_ambertools/prod_substrate_named.mol2
ACP = loadmol2 02_ambertools/prod_partner_named.mol2
SOLV = loadpdb 05_templates/solvent_only_from_old_rc.pdb

PROD = combine {{ SUBP ACP SOLV }}
set PROD box {{ {OLD_RC_BOX[0]:.3f} {OLD_RC_BOX[1]:.3f} {OLD_RC_BOX[2]:.3f} }}
check PROD

savepdb PROD 03_solvated_prod/PROD_solvated_initial.pdb
saveamberparm PROD 03_solvated_prod/PROD_solvated_initial.prmtop 03_solvated_prod/PROD_solvated_initial.inpcrd
quit
""",
        encoding="utf-8",
    )


def _write_minimization_runner() -> None:
    (OUT_DIR / "run_openmm_minimizations.sh").write_text(
        """#!/usr/bin/env bash
source "$HOME/.bashrc"
conda activate OpenMM
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p 06_minimized_states/RC 06_minimized_states/PROD
rm -f 06_minimized_states/RC/* 06_minimized_states/PROD/*
python ../../scripts/equilibrate_solvated_state_openmm.py \
  --prmtop 03_solvated_rc/RC_solvated_initial.prmtop \
  --inpcrd 03_solvated_rc/RC_solvated_initial.inpcrd \
  --coordinates-pdb 03_solvated_rc/RC_solvated_initial.pdb \
  --output-dir 06_minimized_states/RC \
  --prefix RC \
  --minimize-only \
  --restraint-k-kcal-a2 250.0
python ../../scripts/equilibrate_solvated_state_openmm.py \
  --prmtop 03_solvated_prod/PROD_solvated_initial.prmtop \
  --inpcrd 03_solvated_prod/PROD_solvated_initial.inpcrd \
  --coordinates-pdb 03_solvated_prod/PROD_solvated_initial.pdb \
  --output-dir 06_minimized_states/PROD \
  --prefix PROD \
  --minimize-only \
  --restraint-k-kcal-a2 250.0

PYTHONPATH=../../src python ../../scripts/build_openmm_bundle_from_amber.py \
  --prmtop 03_solvated_rc/RC_solvated_initial.prmtop \
  --coordinates 06_minimized_states/RC/RC_minimized.pdb \
  --output-dir 07_openmm_bundles/RC \
  --nonbonded-method PME \
  --constraints None

PYTHONPATH=../../src python ../../scripts/build_openmm_bundle_from_amber.py \
  --prmtop 03_solvated_prod/PROD_solvated_initial.prmtop \
  --coordinates 06_minimized_states/PROD/PROD_minimized.pdb \
  --output-dir 07_openmm_bundles/PROD \
  --nonbonded-method PME \
  --constraints None
""",
        encoding="utf-8",
    )


def _write_manifest() -> None:
    payload = {
        "source_qm_xyz": {
            "RC": str(RC_XYZ),
            "PROD": str(PROD_XYZ),
        },
        "previous_explicit_solvent_template": str(OLD_RC_PDB),
        "reactive_atom_count": REACTIVE_ATOM_COUNT,
        "box_angstrom": OLD_RC_BOX,
        "reaction_coordinates_0based": {
            "proton_transfer": {"donor": 6, "proton": 15, "acceptor": 19},
            "ring_opening": {"atom1": 1, "atom2": 11},
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    for subdir in ["01_qm_inputs", "02_ambertools", "03_solvated_rc", "03_solvated_prod", "05_templates", "06_minimized_states", "07_openmm_bundles"]:
        (OUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    rc_symbols, rc_coords = _load_xyz(RC_XYZ)
    prod_symbols, prod_coords = _load_xyz(PROD_XYZ)
    target_old_rc = _load_first_n_pdb_coords(OLD_RC_PDB, REACTIVE_ATOM_COUNT)

    rc_rot, rc_mob_c, rc_tar_c = _kabsch_align([rc_coords[i] for i in ALIGNMENT_ATOM_INDICES], [target_old_rc[i] for i in ALIGNMENT_ATOM_INDICES])
    rc_coords_aligned = _transform_all(rc_coords, rc_rot, rc_mob_c, rc_tar_c)

    prod_rot, prod_mob_c, prod_tar_c = _kabsch_align([prod_coords[i] for i in ALIGNMENT_ATOM_INDICES], [target_old_rc[i] for i in ALIGNMENT_ATOM_INDICES])
    prod_coords_aligned = _transform_all(prod_coords, prod_rot, prod_mob_c, prod_tar_c)

    input_dir = OUT_DIR / "01_qm_inputs"
    _write_pdb_with_names(
        input_dir / "rc_complex_fragmented.pdb",
        rc_symbols,
        rc_coords_aligned,
        RC_SUBSTRATE_NAMES,
        RC_SUBSTRATE_RESNAME,
        RC_PARTNER_NAMES,
        RC_PARTNER_RESNAME,
    )
    _write_pdb_with_names(
        input_dir / "prod_complex_fragmented.pdb",
        prod_symbols,
        prod_coords_aligned,
        PROD_SUBSTRATE_NAMES,
        PROD_SUBSTRATE_RESNAME,
        PROD_PARTNER_NAMES,
        PROD_PARTNER_RESNAME,
    )
    _write_simple_pdb(input_dir / "rc_substrate_simple.pdb", rc_symbols[:16], rc_coords_aligned[:16], RC_SUBSTRATE_RESNAME)
    _write_simple_pdb(input_dir / "rc_partner_simple.pdb", rc_symbols[16:23], rc_coords_aligned[16:23], RC_PARTNER_RESNAME)
    _write_simple_pdb(input_dir / "prod_substrate_simple.pdb", prod_symbols[:15], prod_coords_aligned[:15], PROD_SUBSTRATE_RESNAME)
    _write_simple_pdb(input_dir / "prod_partner_simple.pdb", prod_symbols[15:23], prod_coords_aligned[15:23], PROD_PARTNER_RESNAME)
    _write_names_file(input_dir / "rc_substrate.names", RC_SUBSTRATE_NAMES)
    _write_names_file(input_dir / "rc_partner.names", RC_PARTNER_NAMES)
    _write_names_file(input_dir / "prod_substrate.names", PROD_SUBSTRATE_NAMES)
    _write_names_file(input_dir / "prod_partner.names", PROD_PARTNER_NAMES)

    _write_tleap_templates()
    _write_runner()
    _write_minimization_runner()
    _write_manifest()


if __name__ == "__main__":
    main()
