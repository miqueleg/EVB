from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .simulation import ensure_output_dir


def _read_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise ImportError("PyYAML is required for candidate generation.")
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    if yaml is None:  # pragma: no cover
        raise ImportError("PyYAML is required for candidate generation.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def make_qregion_candidates(config_path: str | Path, output: str | Path, *, cutoffs: list[float] | None = None) -> dict[str, Any]:
    root = ensure_output_dir(output)
    candidates = ensure_output_dir(root / "candidates")
    base = _read_yaml(config_path)
    cutoffs = cutoffs or [0.8, 1.0, 1.2, 1.5]
    written = []
    for baseline in ["state1", "state2"]:
        name = f"shared_nonbonded_baseline_{baseline}"
        payload = _candidate_payload(base, baseline, "shared_nonbonded_model", None, None)
        path = candidates / f"{name}.yaml"
        _write_yaml(path, payload)
        written.append({"name": name, "path": str(path), "mode": "shared_nonbonded_model"})
    for cutoff in cutoffs:
        for correction_atoms in ["q_atoms", "q_plus_shell"]:
            name = f"local_pme_{correction_atoms}_cutoff_{cutoff:g}"
            payload = _candidate_payload(base, "state1", "local_pme_approx", correction_atoms, cutoff)
            path = candidates / f"{name}.yaml"
            _write_yaml(path, payload)
            written.append({"name": name, "path": str(path), "mode": "local_pme_approx"})
    name = "local_pme_all_atoms_cutoff_1.2"
    payload = _candidate_payload(base, "state1", "local_pme_approx", "all_atoms", 1.2)
    path = candidates / f"{name}.yaml"
    _write_yaml(path, payload)
    written.append({"name": name, "path": str(path), "mode": "local_pme_approx"})
    summary = {"candidate_dir": str(candidates), "candidates": written}
    (root / "candidate_generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _candidate_payload(base: dict[str, Any], baseline: str, mode: str, correction_atoms: str | None, cutoff: float | None) -> dict[str, Any]:
    payload = json.loads(json.dumps(base))
    evb = payload.setdefault("evb", {})
    evb["representation"] = "q_region"
    qreg = evb.setdefault("q_region", {})
    qreg["enabled"] = True
    qreg["baseline_state"] = baseline
    nb = qreg.setdefault("nonbonded", {})
    nb["mode"] = mode
    if mode == "shared_nonbonded_model":
        nb["pme_policy"] = "shared_baseline"
        nb["local_approx_enabled"] = False
    else:
        nb["pme_policy"] = "local_direct_space_correction"
        nb["local_approx_enabled"] = True
        nb["correction_atoms"] = correction_atoms
        nb["correction_cutoff_nm"] = cutoff
        nb.setdefault("include_q_q", True)
        nb.setdefault("include_q_environment", True)
        nb.setdefault("include_q_water", True)
        nb.setdefault("include_exceptions", True)
    return payload


def copy_example_dataset(source: str | Path, destination: str | Path) -> str:
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    return str(dest)
