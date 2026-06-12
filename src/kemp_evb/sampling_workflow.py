from __future__ import annotations

import csv
import json
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .reference_profile import KJ_PER_KCAL
from .simulation import ensure_output_dir


@dataclass(slots=True)
class WorkflowSettings:
    mode: str
    umbrella_steps: int
    metad_steps: int
    replicas: int


def settings_for_mode(mode: str, umbrella_steps: int | None = None, metad_steps: int | None = None, replicas: int | None = None) -> WorkflowSettings:
    defaults = {"quick": (1000, 5000, 1), "pilot": (20000, 100000, 2), "production": (200000, 1000000, 3)}
    if mode not in defaults:
        raise ValueError("mode must be quick, pilot, or production")
    u, m, r = defaults[mode]
    return WorkflowSettings(mode, int(umbrella_steps or u), int(metad_steps or m), int(replicas or r))


def read_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise ImportError("PyYAML is required for workflow generation.")
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    if yaml is None:  # pragma: no cover
        raise ImportError("PyYAML is required for workflow generation.")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def derive_gap_windows(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    cfg = read_yaml(config_path)
    sampling = cfg.get("sampling") or {}
    umbrella = sampling.get("umbrella") or sampling.get("windows", {}).get("gap_umbrella", {})
    n = int(umbrella.get("windows") or umbrella.get("n_windows") or 21)
    if umbrella.get("centers_kj_mol"):
        centers_kj = [float(x) for x in umbrella["centers_kj_mol"]]
    else:
        low = float(umbrella.get("gap_min_kcal_mol", -100.0)) * KJ_PER_KCAL
        high = float(umbrella.get("gap_max_kcal_mol", 100.0)) * KJ_PER_KCAL
        import numpy as np
        centers_kj = [float(x) for x in np.linspace(low, high, n)]
    k = float(umbrella.get("force_constant_kj_mol2") or float(umbrella.get("force_constant_kcal_mol2", 0.005)) / KJ_PER_KCAL)
    out = ensure_output_dir(output)
    path = out / "umbrella_windows.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["window_id", "center_kj_mol", "center_kcal_mol", "force_constant_kj_mol2"])
        for i, center in enumerate(centers_kj):
            writer.writerow([f"w{i:03d}", center, center / KJ_PER_KCAL, k])
    return {"windows_csv": str(path), "n_windows": len(centers_kj), "centers_kj_mol": centers_kj, "force_constant_kj_mol2": k}


def generate_workflow_configs(config_path: str | Path, output: str | Path, *, mode: str = "quick", workflow: str = "all", umbrella_steps: int | None = None, metad_steps: int | None = None, replicas: int | None = None) -> dict[str, Any]:
    out = ensure_output_dir(output)
    cfg_dir = ensure_output_dir(out / "configs")
    cfg = read_yaml(config_path)
    settings = settings_for_mode(mode, umbrella_steps, metad_steps, replicas)
    written: dict[str, Any] = {"settings": asdict(settings)}
    if workflow in {"umbrella", "all"}:
        umbrella_cfg = json.loads(json.dumps(cfg))
        umbrella_cfg.setdefault("project", {})["output_dir"] = str(out / "umbrella")
        umbrella_cfg.setdefault("sampling", {})["mode"] = "gap_umbrella" if umbrella_cfg.get("evb", {}).get("representation", "full_state") == "full_state" else "q_region_gap_umbrella"
        umbrella_cfg["sampling"].setdefault("umbrella", {})["production_steps"] = settings.umbrella_steps
        path = cfg_dir / "umbrella.yaml"
        write_yaml(path, umbrella_cfg)
        written["umbrella_config"] = str(path)
        written["umbrella_windows"] = derive_gap_windows(path, cfg_dir)
    if workflow in {"metad", "all"}:
        metad_cfg = json.loads(json.dumps(cfg))
        metad_cfg.setdefault("project", {})["output_dir"] = str(out / "metad")
        metad_cfg.setdefault("sampling", {})["mode"] = "gap_table_metadynamics"
        native = metad_cfg["sampling"].setdefault("native_gap_bias", {})
        m = metad_cfg["sampling"].get("metad", {})
        native.setdefault("method", "well_tempered_metadynamics")
        native.setdefault("cv", "gap")
        native.setdefault("bias_factor", m.get("bias_factor", 15.0))
        native.setdefault("height_kj_mol", m.get("height_kj_mol", 0.2))
        width = m.get("bias_width_kcal_mol")
        native.setdefault("bias_width", float(width) * KJ_PER_KCAL if width is not None else m.get("bias_width", 750.0))
        native.setdefault("frequency", m.get("frequency", 1000))
        native.setdefault("grid_width", m.get("grid_width", 1001))
        native.setdefault("min_value", float(m.get("grid_min_kcal_mol", -5000.0)) * KJ_PER_KCAL)
        native.setdefault("max_value", float(m.get("grid_max_kcal_mol", 5000.0)) * KJ_PER_KCAL)
        metad_cfg.setdefault("simulation", {})["steps"] = settings.metad_steps
        path = cfg_dir / "metad.yaml"
        write_yaml(path, metad_cfg)
        written["metad_config"] = str(path)
    written["run_scripts"] = write_run_scripts(out)
    (out / "workflow_generation_summary.json").write_text(json.dumps(written, indent=2), encoding="utf-8")
    return written


def write_run_scripts(output: str | Path, script_dir: str | Path | None = None) -> dict[str, str]:
    out = ensure_output_dir(Path(output) / "run_commands")
    template = ensure_output_dir(script_dir) if script_dir is not None else None
    scripts = {"run_umbrella.sh": _runner("umbrella"), "run_metad.sh": _runner("metad"), "run_workflow.sh": _runner("all"), "monitor.sh": "#!/usr/bin/env bash\nset -euo pipefail\nOUT=${1:-outputs/evb_workflow}\nfind \"$OUT\" -maxdepth 4 -name '*.json' -o -name '*.log'\n", "resume.sh": _runner("all", resume=True)}
    written = {}
    for name, text in scripts.items():
        dirs = [out] + ([template] if template is not None else [])
        for directory in dirs:
            path = directory / name
            path.write_text(text, encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        written[name] = str(out / name)
    return written


def _runner(workflow: str, resume: bool = False) -> str:
    extra = " --resume --skip-existing" if resume else ""
    return """#!/usr/bin/env bash
set -euo pipefail
CONFIG=${CONFIG:-config.yaml}
OUTPUT=${OUTPUT:-outputs/evb_workflow}
PLATFORM=${PLATFORM:-CUDA}
MODE=${MODE:-production}
mkdir -p "$OUTPUT/logs"
.venv/bin/evb run-workflow \\
  --config "$CONFIG" \\
  --output "$OUTPUT" \\
  --platform "$PLATFORM" \\
  --mode "$MODE" \\
  --workflow {workflow}{extra} 2>&1 | tee -a "$OUTPUT/logs/run.log"
""".replace("{workflow}", workflow).replace("{extra}", extra)


def validate_workflow_config(config_path: str | Path) -> dict[str, Any]:
    cfg = read_yaml(config_path)
    errors = []
    for key in ["project", "states", "evb", "sampling"]:
        if key not in cfg:
            errors.append(f"missing required section: {key}")
    if "reference_profile" not in cfg and not cfg.get("reference"):
        errors.append("missing reference_profile section or reference path")
    return {"valid": not errors, "errors": errors}
