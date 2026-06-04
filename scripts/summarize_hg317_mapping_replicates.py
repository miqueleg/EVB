from __future__ import annotations

from pathlib import Path

from kemp_evb.analysis.replicates import summarize_replicates
from kemp_evb.io import write_json


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outputs_root = repo_root / "outputs" / "hg317_evb_mapping_replicates"
    replicate_dirs = sorted(path for path in outputs_root.glob("rep*") if (path / "analysis" / "pmf_gap.csv").is_file())
    if not replicate_dirs:
        raise SystemExit(f"No analyzed mapped-replica PMFs found under {outputs_root}")
    destination = outputs_root / "summary"
    payload = summarize_replicates(replicate_dirs, destination)
    write_json(destination / "replicate_inputs.json", {"replicates": [str(path) for path in replicate_dirs]})
    print(f"Summarized {len(replicate_dirs)} mapped replicates into {destination}")
    print(payload["summary"])


if __name__ == "__main__":
    main()
