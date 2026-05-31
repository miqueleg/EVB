from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
import math


ROOT = Path(__file__).resolve().parents[1]
QM_DIR = ROOT / "systems" / "KEMP_implicit"
OUT_DIR = ROOT / "prep" / "kemp_qm_evb"

RC_XYZ = QM_DIR / "RC_final.xyz"
TS_XYZ = QM_DIR / "TS_final.xyz"
PROD_XYZ = QM_DIR / "PROD_final.xyz"

REACTIVE_RESIDUE_NAME = "EVB"
COMMON_ATOM_NAMES = [
    "N1",
    "N2",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "O1",
    "O2",
    "O3",
    "H1",
    "H2",
    "H3",
    "H4",
    "C8",
    "C9",
    "O4",
    "O5",
    "H5",
    "H6",
    "H7",
]
RC_SUBSTRATE_NAMES = COMMON_ATOM_NAMES[:16]
RC_PARTNER_NAMES = COMMON_ATOM_NAMES[16:23]
PROD_SUBSTRATE_NAMES = COMMON_ATOM_NAMES[:15]
PROD_PARTNER_NAMES = COMMON_ATOM_NAMES[15:23]
RC_SUBSTRATE_RESNAME = "SBR"
RC_PARTNER_RESNAME = "BAR"
PROD_SUBSTRATE_RESNAME = "SDP"
PROD_PARTNER_RESNAME = "ACP"
ALIGNMENT_ATOM_INDICES = list(range(15))


def _load_xyz(path: Path) -> tuple[float | None, list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = int(lines[0].strip())
    energy = None
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
    return energy, symbols, coords


def _centroid(coords: list[list[float]]) -> list[float]:
    n = float(len(coords))
    return [sum(c[i] for c in coords) / n for i in range(3)]


def _matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _transpose3(a: list[list[float]]) -> list[list[float]]:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def _apply_rot(coords: list[list[float]], rot: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(rot[i][j] * c[j] for j in range(3))
            for i in range(3)
        ]
        for c in coords
    ]


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
    # Cross-covariance matrix
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


def _write_xyz(path: Path, symbols: list[str], coords: list[list[float]], comment: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(f"{comment}\n")
        for symbol, (x, y, z) in zip(symbols, coords):
            handle.write(f"{symbol:<2s} {x:14.6f} {y:14.6f} {z:14.6f}\n")


def _write_pdb(path: Path, symbols: list[str], coords: list[list[float]]) -> None:
    _write_pdb_with_names(path, symbols, coords, COMMON_ATOM_NAMES, REACTIVE_RESIDUE_NAME)


def _write_pdb_with_names(path: Path, symbols: list[str], coords: list[list[float]], atom_names: list[str], residue_name: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, (symbol, (x, y, z), atom_name) in enumerate(zip(symbols, coords, atom_names), start=1):
            handle.write(
                f"HETATM{index:5d} {atom_name:>4s} {residue_name:>3s} A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol:>2s}\n"
            )
        handle.write("END\n")


def _write_two_residue_pdb(
    path: Path,
    symbols: list[str],
    coords: list[list[float]],
    first_names: list[str],
    first_resname: str,
    second_names: list[str],
    second_resname: str,
) -> None:
    first_count = len(first_names)
    second_count = len(second_names)
    assert len(symbols) == first_count + second_count
    assert len(coords) == len(symbols)
    with path.open("w", encoding="utf-8") as handle:
        for index, (symbol, (x, y, z), atom_name) in enumerate(
            zip(symbols[:first_count], coords[:first_count], first_names),
            start=1,
        ):
            handle.write(
                f"HETATM{index:5d} {atom_name:>4s} {first_resname:>3s} A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol:>2s}\n"
            )
        for offset, (symbol, (x, y, z), atom_name) in enumerate(
            zip(symbols[first_count:], coords[first_count:], second_names),
            start=1,
        ):
            handle.write(
                f"HETATM{first_count + offset:5d} {atom_name:>4s} {second_resname:>3s} A   2    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol:>2s}\n"
            )
        handle.write("END\n")


def _write_names_file(path: Path, names: list[str]) -> None:
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def _simple_atom_names(symbols: list[str]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    names: list[str] = []
    for symbol in symbols:
        counts[symbol] += 1
        names.append(f"{symbol}{counts[symbol]}")
    return names


def _write_manifest(rc_energy: float | None, ts_energy: float | None, prod_energy: float | None) -> None:
    payload = {
        "source_xyz": {
            "RC": str(RC_XYZ),
            "TS": str(TS_XYZ),
            "PROD": str(PROD_XYZ),
        },
        "residue_name": REACTIVE_RESIDUE_NAME,
        "atom_names": COMMON_ATOM_NAMES,
        "state_charges": {
            "RC": 0,
            "PROD": -1,
        },
        "qm_geometry_implied_fragment_charges": {
            "rc_substrate": 0,
            "rc_partner": -1,
            "prod_substrate": -1,
            "prod_partner": 0,
        },
        "qm_energies_hartree": {
            "RC": rc_energy,
            "TS": ts_energy,
            "PROD": prod_energy,
        },
        "recommended_reaction_coordinates_0based": {
            "proton_transfer": {"donor": 6, "proton": 15, "acceptor": 19},
            "ring_opening": {"atom1": 1, "atom2": 11},
        },
        "warning": "The current QM RC geometry is a disconnected substrate + acetate complex. That implies RC total charge -1, not 0. If you truly want RC = 0, you need a different QM RC geometry with a protonated partner or different reactant chemistry.",
    }
    (OUT_DIR / "reactive_region_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_tleap_templates() -> None:
    (OUT_DIR / "tleap_rc_solvate.in").write_text(
        """source leaprc.gaff2
source leaprc.water.tip3p

loadamberparams 02_ambertools/rc_substrate.frcmod
loadamberparams 02_ambertools/rc_partner.frcmod

SUBR = loadmol2 02_ambertools/rc_substrate_named.mol2
BASR = loadmol2 02_ambertools/rc_partner_named.mol2
RC = combine { SUBR BASR }
check RC
solvatebox RC TIP3PBOX 12.0
addions RC Na+ 0

savepdb RC 03_solvated_rc/RC_solvated_initial.pdb
saveamberparm RC 03_solvated_rc/RC_solvated_initial.prmtop 03_solvated_rc/RC_solvated_initial.inpcrd
quit
""",
        encoding="utf-8",
    )
    (OUT_DIR / "tleap_prod_solvate_from_template.in").write_text(
        """source leaprc.gaff2
source leaprc.water.tip3p

loadamberparams 02_ambertools/prod_substrate.frcmod
loadamberparams 02_ambertools/prod_partner.frcmod

SUBP = loadmol2 02_ambertools/prod_substrate_named.mol2
ACP = loadmol2 02_ambertools/prod_partner_named.mol2
SOLV = loadpdb 05_templates/solvent_only_template.pdb

PROD = combine { SUBP ACP SOLV }
set PROD box { 37.435 39.930 31.855 }
check PROD

savepdb PROD 03_solvated_prod/PROD_solvated_initial.pdb
saveamberparm PROD 03_solvated_prod/PROD_solvated_initial.prmtop 03_solvated_prod/PROD_solvated_initial.inpcrd
quit
""",
        encoding="utf-8",
    )
    (OUT_DIR / "tleap_prod_from_template.in").write_text(
        """source leaprc.gaff2
source leaprc.water.tip3p

loadamberparams 02_ambertools/prod_substrate.frcmod
loadamberparams 02_ambertools/prod_partner.frcmod

SUBP = loadmol2 02_ambertools/prod_substrate_named.mol2
ACDP = loadmol2 02_ambertools/prod_partner_named.mol2
saveoff SUBP 02_ambertools/prod_substrate.lib
saveoff ACDP 02_ambertools/prod_partner.lib
loadoff 02_ambertools/prod_substrate.lib
loadoff 02_ambertools/prod_partner.lib
addPdbResMap {
  { 0 "SDP" "SDP" } { 1 "SDP" "SDP" }
  { 0 "ACP" "ACP" } { 1 "ACP" "ACP" }
}

PROD = loadpdb 05_templates/prod_from_rc_template_fragmented.pdb
check PROD

savepdb PROD 07_final_evb_states/PROD_final.pdb
saveamberparm PROD 07_final_evb_states/PROD_final.prmtop 07_final_evb_states/PROD_final.inpcrd
quit
""",
        encoding="utf-8",
    )
    (OUT_DIR / "tleap_rc_from_template.in").write_text(
        """source leaprc.gaff2
source leaprc.water.tip3p

loadamberparams 02_ambertools/rc_substrate.frcmod
loadamberparams 02_ambertools/rc_partner.frcmod

SUBR = loadmol2 02_ambertools/rc_substrate_named.mol2
BASR = loadmol2 02_ambertools/rc_partner_named.mol2
saveoff SUBR 02_ambertools/rc_substrate.lib
saveoff BASR 02_ambertools/rc_partner.lib
loadoff 02_ambertools/rc_substrate.lib
loadoff 02_ambertools/rc_partner.lib
addPdbResMap {
  { 0 "SBR" "SBR" } { 1 "SBR" "SBR" }
  { 0 "BAR" "BAR" } { 1 "BAR" "BAR" }
}

RC = loadpdb 05_templates/rc_from_master_template_fragmented.pdb
check RC

savepdb RC 07_final_evb_states/RC_final.pdb
saveamberparm RC 07_final_evb_states/RC_final.prmtop 07_final_evb_states/RC_final.inpcrd
quit
""",
        encoding="utf-8",
    )


def _write_runner_script() -> None:
    (OUT_DIR / "run_from_qm_xyz.sh").write_text(
        """#!/usr/bin/env bash
source "$HOME/.bashrc"
conda activate ambertraj
set -euo pipefail
export PATH="$CONDA_PREFIX/bin:$PATH"

cd "$(dirname "$0")"

mkdir -p 02_ambertools 03_solvated_rc 05_templates 07_final_evb_states

# 1. Generate AM1-BCC mol2/frcmod for the full reactive complex in each EVB state.
antechamber -i 01_qm_inputs/rc_complex_ordered.pdb -fi pdb -o 02_ambertools/rc_complex_raw.mol2 -fo mol2 -c bcc -s 2 -nc 0
python ../../scripts/rename_kemp_qm_mol2.py --input 02_ambertools/rc_complex_raw.mol2 --output 02_ambertools/rc_complex_named.mol2 --resname EVB
parmchk2 -i 02_ambertools/rc_complex_named.mol2 -f mol2 -o 02_ambertools/rc_complex.frcmod

antechamber -i 01_qm_inputs/prod_complex_ordered.pdb -fi pdb -o 02_ambertools/prod_complex_raw.mol2 -fo mol2 -c bcc -s 2 -nc -1
python ../../scripts/rename_kemp_qm_mol2.py --input 02_ambertools/prod_complex_raw.mol2 --output 02_ambertools/prod_complex_named.mol2 --resname EVB
parmchk2 -i 02_ambertools/prod_complex_named.mol2 -f mol2 -o 02_ambertools/prod_complex.frcmod

# 2. Build one master solvated RC box only once.
tleap -f tleap_rc_solvate.in

# 3. You now equilibrate 03_solvated_rc/RC_solvated_initial.* externally.
#    Use the equilibrated snapshot as rc_master_snapshot.pdb and place it under 04_equilibrated_rc/.
if [[ ! -f 04_equilibrated_rc/rc_master_snapshot.pdb ]]; then
  echo 'RC master snapshot not found yet. Stop here, equilibrate the RC box, then rerun this script.'
  exit 0
fi

# 4. Rebuild PROD from the RC master solvent template.
python ../../scripts/build_kemp_prod_from_rc_template.py \\
  --rc-master-pdb 04_equilibrated_rc/rc_master_snapshot.pdb \\
  --rc-reactive-pdb 01_qm_inputs/rc_complex_ordered.pdb \\
  --prod-reactive-pdb 01_qm_inputs/prod_complex_ordered.pdb \\
  --rc-output-pdb 05_templates/rc_from_master_template.pdb \\
  --prod-output-pdb 05_templates/prod_from_rc_template.pdb \\
  --reactive-atom-count 23

# 5. Generate final EVB topologies from the shared template box and shared solvent ordering.
tleap -f tleap_rc_from_template.in
tleap -f tleap_prod_from_template.in

echo "Final files are under 07_final_evb_states/"
""",
        encoding="utf-8",
    )
    (OUT_DIR / "run_fragment_route.sh").write_text(
        """#!/usr/bin/env bash
source "$HOME/.bashrc"
conda activate ambertraj
set -euo pipefail
export PATH="$CONDA_PREFIX/bin:$PATH"

cd "$(dirname "$0")"

mkdir -p 02_ambertools 03_solvated_rc 03_solvated_prod 05_templates 06_equilibrated_states

# Charges implied by the current QM geometries:
#   RC substrate   =  0
#   RC partner     = -1
#   PROD substrate = -1
#   PROD partner   =  0
# This means the current QM RC/PROD pair is effectively -1 / -1 overall.
# If you require RC = 0, you need a different RC QM input geometry.

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

tleap -f tleap_rc_solvate.in
python ../../scripts/extract_solvent_template_from_pdb.py \\
  --input-pdb 03_solvated_rc/RC_solvated_initial.pdb \\
  --output-pdb 05_templates/solvent_only_template.pdb \\
  --reactive-atom-count 23
tleap -f tleap_prod_solvate_from_template.in

echo 'Initial solvated EVB files are under 03_solvated_rc/ and 03_solvated_prod/.'
""",
        encoding="utf-8",
    )


def _write_checklist() -> None:
    (OUT_DIR / "README.md").write_text(
        """# KEMP EVB Prep From QM XYZ

This prep set uses the reactive coordinates from:

- `systems/KEMP_implicit/RC_final.xyz`
- `systems/KEMP_implicit/PROD_final.xyz`

The runnable route is the fragment route:

- RC substrate: `SBR`
- RC partner: `BAR`
- PROD substrate: `SDP`
- PROD partner: `ACP`

## Charge warning

The current QM geometries imply these fragment charges:

- RC substrate: `0`
- RC partner: `-1`
- PROD substrate: `-1`
- PROD partner: `0`

So the QM inputs you provided correspond to:

- RC total charge: `-1`
- PROD total charge: `-1`

If you truly need `RC = 0`, the RC QM geometry itself must change. The current
`RC_final.xyz` is a neutral substrate plus an acetate-like anion.

## Files

- `01_qm_inputs/rc_complex_fragmented.pdb`
- `01_qm_inputs/prod_complex_fragmented.pdb`
- `01_qm_inputs/rc_substrate.pdb`
- `01_qm_inputs/rc_partner.pdb`
- `01_qm_inputs/prod_substrate.pdb`
- `01_qm_inputs/prod_partner.pdb`
- `reactive_region_manifest.json`
- `tleap_rc_solvate.in`
- `tleap_rc_from_template.in`
- `tleap_prod_from_template.in`
- `run_fragment_route.sh`

## Workflow

1. Run `run_fragment_route.sh` after confirming AmberTools is available in `ambertraj`.
2. This produces initial solvated inputs for both EVB states:
   - `03_solvated_rc/RC_solvated_initial.prmtop`
   - `03_solvated_rc/RC_solvated_initial.inpcrd`
   - `03_solvated_prod/PROD_solvated_initial.prmtop`
   - `03_solvated_prod/PROD_solvated_initial.inpcrd`
3. Equilibrate RC and PROD separately in OpenMM using those fixed prmtops.
4. Keep the initial prmtops and use the new equilibrated coordinates as the EVB coordinates.

## Important

Do not solvate RC and PROD independently.
Build the solvent once from RC, derive PROD from that exact solvent template, then equilibrate RC and PROD separately while keeping the initial prmtops fixed.
""",
        encoding="utf-8",
    )


def main() -> None:
    input_dir = OUT_DIR / "01_qm_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["02_ambertools", "03_solvated_rc", "03_solvated_prod", "04_equilibrated_rc", "05_templates", "06_equilibrated_states", "07_final_evb_states"]:
        (OUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    rc_energy, rc_symbols, rc_coords = _load_xyz(RC_XYZ)
    ts_energy, _, _ = _load_xyz(TS_XYZ)
    prod_energy, prod_symbols, prod_coords_raw = _load_xyz(PROD_XYZ)
    mobile = [prod_coords_raw[i] for i in ALIGNMENT_ATOM_INDICES]
    target = [rc_coords[i] for i in ALIGNMENT_ATOM_INDICES]
    rot, mob_c, tar_c = _kabsch_align(mobile, target)
    prod_coords = _transform_all(prod_coords_raw, rot, mob_c, tar_c)

    _write_xyz(input_dir / "rc_complex_ordered.xyz", rc_symbols, rc_coords, "RC reactive complex from QM with EVB atom order")
    _write_xyz(input_dir / "prod_complex_ordered.xyz", prod_symbols, prod_coords, "PROD reactive complex from QM with EVB atom order")
    _write_pdb(input_dir / "rc_complex_ordered.pdb", rc_symbols, rc_coords)
    _write_pdb(input_dir / "prod_complex_ordered.pdb", prod_symbols, prod_coords)
    _write_two_residue_pdb(
        input_dir / "rc_complex_fragmented.pdb",
        rc_symbols,
        rc_coords,
        RC_SUBSTRATE_NAMES,
        RC_SUBSTRATE_RESNAME,
        RC_PARTNER_NAMES,
        RC_PARTNER_RESNAME,
    )
    _write_two_residue_pdb(
        input_dir / "prod_complex_fragmented.pdb",
        prod_symbols,
        prod_coords,
        PROD_SUBSTRATE_NAMES,
        PROD_SUBSTRATE_RESNAME,
        PROD_PARTNER_NAMES,
        PROD_PARTNER_RESNAME,
    )
    _write_xyz(input_dir / "rc_substrate.xyz", rc_symbols[:16], rc_coords[:16], "RC substrate fragment from QM")
    _write_xyz(input_dir / "rc_partner.xyz", rc_symbols[16:23], rc_coords[16:23], "RC acetate-like partner from QM")
    _write_xyz(input_dir / "prod_substrate.xyz", prod_symbols[:15], prod_coords[:15], "PROD substrate fragment from QM")
    _write_xyz(input_dir / "prod_partner.xyz", prod_symbols[15:23], prod_coords[15:23], "PROD protonated partner from QM")
    _write_pdb_with_names(input_dir / "rc_substrate.pdb", rc_symbols[:16], rc_coords[:16], RC_SUBSTRATE_NAMES, RC_SUBSTRATE_RESNAME)
    _write_pdb_with_names(input_dir / "rc_partner.pdb", rc_symbols[16:23], rc_coords[16:23], RC_PARTNER_NAMES, RC_PARTNER_RESNAME)
    _write_pdb_with_names(input_dir / "prod_substrate.pdb", prod_symbols[:15], prod_coords[:15], PROD_SUBSTRATE_NAMES, PROD_SUBSTRATE_RESNAME)
    _write_pdb_with_names(input_dir / "prod_partner.pdb", prod_symbols[15:23], prod_coords[15:23], PROD_PARTNER_NAMES, PROD_PARTNER_RESNAME)
    _write_pdb_with_names(input_dir / "rc_substrate_simple.pdb", rc_symbols[:16], rc_coords[:16], _simple_atom_names(rc_symbols[:16]), RC_SUBSTRATE_RESNAME)
    _write_pdb_with_names(input_dir / "rc_partner_simple.pdb", rc_symbols[16:23], rc_coords[16:23], _simple_atom_names(rc_symbols[16:23]), RC_PARTNER_RESNAME)
    _write_pdb_with_names(input_dir / "prod_substrate_simple.pdb", prod_symbols[:15], prod_coords[:15], _simple_atom_names(prod_symbols[:15]), PROD_SUBSTRATE_RESNAME)
    _write_pdb_with_names(input_dir / "prod_partner_simple.pdb", prod_symbols[15:23], prod_coords[15:23], _simple_atom_names(prod_symbols[15:23]), PROD_PARTNER_RESNAME)
    _write_names_file(input_dir / "rc_substrate.names", RC_SUBSTRATE_NAMES)
    _write_names_file(input_dir / "rc_partner.names", RC_PARTNER_NAMES)
    _write_names_file(input_dir / "prod_substrate.names", PROD_SUBSTRATE_NAMES)
    _write_names_file(input_dir / "prod_partner.names", PROD_PARTNER_NAMES)

    _write_manifest(rc_energy, ts_energy, prod_energy)
    _write_tleap_templates()
    _write_runner_script()
    _write_checklist()
    print(OUT_DIR)


if __name__ == "__main__":
    main()
