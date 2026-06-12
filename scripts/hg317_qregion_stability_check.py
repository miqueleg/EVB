#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kemp_evb.hg317_stability import run_hg317_qregion_stability_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HG3.17 calibrated Q-region stability checks.")
    parser.add_argument("--config", default="outputs/hg317_qregion_gxtb_calibrated/fitted/local_pme_q_atoms_cutoff_0.8/fitted_config.yaml")
    parser.add_argument("--output", default="outputs/hg317_qregion_stability")
    parser.add_argument("--platform", default=None)
    parser.add_argument("--plain-steps", type=int, default=100000)
    parser.add_argument("--table-metad-steps", type=int, default=100000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--report-interval", type=int, default=None)
    args = parser.parse_args()

    summary = run_hg317_qregion_stability_check(
        Path(args.config),
        Path(args.output),
        args.platform,
        plain_steps=args.plain_steps,
        table_metad_steps=args.table_metad_steps,
        quick=args.quick,
        report_interval=args.report_interval,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
