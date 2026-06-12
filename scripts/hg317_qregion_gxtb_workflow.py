#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kemp_evb.config import load_config
from kemp_evb.hg317_gxtb_calibration import hg317_qregion_gxtb_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate HG3.17 Q-region candidates to the g-xTB RC/TS/PROD profile.")
    parser.add_argument("--config", default="examples/hg317_evb_gap_metad.yaml")
    parser.add_argument("--reference", default="examples/hg317_gxtb_reference_profile.yaml")
    parser.add_argument("--output", default="outputs/hg317_qregion_gxtb_workflow")
    parser.add_argument("--profile", default="relative_to_RC")
    parser.add_argument("--platform", default=None)
    parser.add_argument("--smoke-steps", type=int, default=2000)
    parser.add_argument("--smoke-policy", choices=["exploratory", "force"], default="exploratory")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = hg317_qregion_gxtb_workflow(
        config,
        Path(args.config),
        Path(args.reference),
        Path(args.output),
        args.profile,
        args.platform,
        args.smoke_steps,
        args.smoke_policy,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
