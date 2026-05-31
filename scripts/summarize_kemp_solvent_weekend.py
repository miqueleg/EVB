from __future__ import annotations

from pathlib import Path

from kemp_evb.analysis import summarize_replicates


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / "outputs" / "kemp_solvent_weekend"
    replicate_dirs = [output_root / f"rep{index:02d}" for index in range(1, 31)]
    existing = [path for path in replicate_dirs if (path / "analysis" / "pmf_gap.csv").exists()]
    if not existing:
        raise SystemExit("No finished replicate analyses found under outputs/kemp_solvent_weekend.")
    summary_dir = output_root / "summary"
    summarize_replicates(existing, summary_dir)
    print(summary_dir)


if __name__ == "__main__":
    main()
