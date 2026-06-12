#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kemp_evb.config import load_config
from kemp_evb.hg317_reproduction import run_reproduction_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or stage HG3.17 Q-region PMF/metadynamics reproduction workflows.")
    parser.add_argument("--config", default="examples/hg317_evb_gap_metad.yaml")
    parser.add_argument("--reference", default="examples/hg317_gxtb_reference_profile.yaml")
    parser.add_argument("--output", default="outputs/hg317_qregion_reproduction")
    parser.add_argument("--platform", default=None)
    parser.add_argument("--mode", choices=["quick", "pilot", "production"], default="quick")
    parser.add_argument("--workflow", choices=["umbrella", "metad", "all", "analysis-only"], default="all")
    parser.add_argument("--replicas", type=int)
    parser.add_argument("--umbrella-steps", type=int)
    parser.add_argument("--metad-steps", type=int)
    parser.add_argument("--write-run-scripts-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--barrier-warning-kcal", type=float, default=2.0)
    parser.add_argument("--barrier-fail-kcal", type=float, default=3.0)
    args = parser.parse_args()

    config = load_config(args.config)
    summary = run_reproduction_workflow(
        config,
        Path(args.config),
        Path(args.reference),
        Path(args.output),
        platform=args.platform,
        mode=args.mode,
        workflow=args.workflow,
        replicas=args.replicas,
        umbrella_steps=args.umbrella_steps,
        metad_steps=args.metad_steps,
        write_run_scripts_only=args.write_run_scripts_only,
        resume=args.resume,
        skip_existing=args.skip_existing,
        barrier_warning_kcal=args.barrier_warning_kcal,
        barrier_fail_kcal=args.barrier_fail_kcal,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
