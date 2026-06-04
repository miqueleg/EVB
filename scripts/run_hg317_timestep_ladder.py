from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_TIMESTEPS_FS = [0.1, 0.2, 0.5, 1.0]
DEFAULT_WINDOWS = ["u000", "u019", "u038", "u048"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short CUDA timestep-stability tests for HG3.17 EVB gap umbrellas.")
    parser.add_argument("--base-config", default="runs/hg317_evb_gap_umbrella/configs/rep01.yaml")
    parser.add_argument("--output-root", default="outputs/hg317_timestep_ladder")
    parser.add_argument("--timesteps-fs", default=",".join(str(value) for value in DEFAULT_TIMESTEPS_FS))
    parser.add_argument("--windows", default=",".join(DEFAULT_WINDOWS))
    parser.add_argument("--production-ps", type=float, default=5.0)
    parser.add_argument("--equilibration-ps", type=float, default=1.0)
    parser.add_argument("--report-stride-ps", type=float, default=0.1)
    args = parser.parse_args()

    base_path = Path(args.base_config)
    payload = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    timesteps = [float(value) for value in args.timesteps_fs.split(",") if value.strip()]
    windows = [value.strip() for value in args.windows.split(",") if value.strip()]

    results = []
    for timestep_fs in timesteps:
        config = copy.deepcopy(payload)
        config["project"]["name"] = f"hg317-timestep-{timestep_fs:g}fs"
        config["project"]["output_dir"] = str(output_root / f"dt_{_label(timestep_fs)}")
        config["sampling"]["integrator"]["timestep_fs"] = timestep_fs
        config["sampling"]["md"]["equilibration_steps"] = _steps(args.equilibration_ps, timestep_fs)
        config["sampling"]["md"]["production_steps"] = _steps(args.production_ps, timestep_fs)
        config["sampling"]["md"]["report_stride"] = max(_steps(args.report_stride_ps, timestep_fs), 1)
        config["sampling"]["md"]["minimize_steps"] = min(int(config["sampling"]["md"].get("minimize_steps", 0)), 100)
        config_path = config_dir / f"dt_{_label(timestep_fs)}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        for window in windows:
            command = [sys.executable, "-m", "evb.cli", "sample-window", "--config", str(config_path), "--window", window]
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
            result = {
                "timestep_fs": timestep_fs,
                "window": window,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr_tail": completed.stderr[-2000:],
            }
            results.append(result)
            print(f"dt={timestep_fs:g} fs window={window} returncode={completed.returncode}")
            if completed.returncode != 0:
                print(result["stderr_tail"])
                break

    (output_root / "timestep_ladder_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {output_root / 'timestep_ladder_results.json'}")


def _steps(ps: float, timestep_fs: float) -> int:
    return int(round(ps * 1000.0 / timestep_fs))


def _label(timestep_fs: float) -> str:
    return str(timestep_fs).replace(".", "p")


if __name__ == "__main__":
    main()
