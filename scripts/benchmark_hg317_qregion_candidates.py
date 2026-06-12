#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kemp_evb.config import load_config
from kemp_evb.hg317_qregion import make_hg317_qregion_candidates, q_region_fit_irc, validate_hg317_qregion_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--include-reaction-atoms", action="store_true", default=True)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--force-smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2000)
    args = parser.parse_args()

    config = load_config(args.config)
    output = Path(args.output)
    generation = make_hg317_qregion_candidates(config, args.config, output, include_reaction_atoms=args.include_reaction_atoms)
    fits = []
    if args.fit:
        for candidate in generation["candidates"]:
            if candidate["mode"] == "shared_nonbonded_model":
                candidate_config = load_config(candidate["path"])
                fits.append(q_region_fit_irc(candidate_config, candidate["path"], output / "fits" / candidate["name"]))
    validation = validate_hg317_qregion_candidates(
        output,
        output / "validation",
        args.platform,
        run_smoke=args.run_smoke,
        smoke_steps=args.smoke_steps,
        force_smoke=args.force_smoke,
    )
    summary = {"generation": generation, "fits": fits, "validation": validation}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
