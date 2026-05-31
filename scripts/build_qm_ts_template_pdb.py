from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QM_DIR = ROOT / "systems" / "KEMP_implicit"

TS_XYZ = QM_DIR / "TS_final.xyz"
REACTIVE_ATOM_COUNT = 23
ALIGNMENT_ATOM_INDICES = list(range(15))
TS_SUBSTRATE_NAMES = [
    "N1", "N2", "C1", "C2", "C3", "C4", "C5", "C6",
    "C7", "O1", "O2", "O3", "H1", "H2", "H3", "H4",
]
TS_PARTNER_NAMES = ["C8", "C9", "O4", "O5", "H5", "H6", "H7"]


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
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                if len(coords) == n:
                    break
    if len(coords) != n:
        raise ValueError(f"{path} did not contain {n} atoms")
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
        norm = sum(x * x for x in nxt) ** 0.5
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


def _replace_reactive_region(template_pdb: Path, output_pdb: Path, coords: list[list[float]], symbols: list[str]) -> None:
    lines = template_pdb.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    atom_index = 0
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and atom_index < REACTIVE_ATOM_COUNT:
            x, y, z = coords[atom_index]
            if atom_index < len(TS_SUBSTRATE_NAMES):
                atom_name = TS_SUBSTRATE_NAMES[atom_index]
                resname = "SBR"
                resid = 1
            else:
                atom_name = TS_PARTNER_NAMES[atom_index - len(TS_SUBSTRATE_NAMES)]
                resname = "BAR"
                resid = 2
            symbol = symbols[atom_index]
            out_lines.append(
                f"HETATM{atom_index + 1:5d} {atom_name:>4s} {resname:>3s} A{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{line[54:76]}{symbol:>2s}"
            )
            atom_index += 1
        else:
            out_lines.append(line)
    output_pdb.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def build_ts_template(template_pdb: Path, output_pdb: Path) -> None:
    symbols, ts_coords = _load_xyz(TS_XYZ)
    target_coords = _load_first_n_pdb_coords(template_pdb, REACTIVE_ATOM_COUNT)
    rot, mob_c, tar_c = _kabsch_align(
        [ts_coords[i] for i in ALIGNMENT_ATOM_INDICES],
        [target_coords[i] for i in ALIGNMENT_ATOM_INDICES],
    )
    ts_coords_aligned = _transform_all(ts_coords, rot, mob_c, tar_c)
    _replace_reactive_region(template_pdb, output_pdb, ts_coords_aligned, symbols)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a solvated TS template PDB by aligning the QM TS geometry into a reference explicit-solvent PDB.")
    parser.add_argument("--template-pdb", required=True)
    parser.add_argument("--output-pdb", required=True)
    args = parser.parse_args()
    build_ts_template(Path(args.template_pdb), Path(args.output_pdb))


if __name__ == "__main__":
    main()
