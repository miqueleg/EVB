from __future__ import annotations

import csv
from pathlib import Path

from kemp_evb.analysis import summarize_replicates


def _write_barrier(path: Path, barrier: float, reaction: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'{{"barrier_forward_kj_mol": {barrier}, "reaction_free_energy_kj_mol": {reaction}}}',
        encoding="utf-8",
    )


def _write_pmf(path: Path, rows: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gap_kj_mol", "free_energy_kj_mol", "probability", "counts"])
        for gap, fe in rows:
            writer.writerow([gap, fe, 0.1, 1])


def test_summarize_replicates(tmp_path: Path):
    rep1 = tmp_path / "rep1"
    rep2 = tmp_path / "rep2"
    _write_barrier(rep1 / "analysis" / "barrier_estimate.json", 10.0, 2.0)
    _write_barrier(rep2 / "analysis" / "barrier_estimate.json", 14.0, 4.0)
    rows1 = [(-10.0, 0.0), (0.0, 1.0), (10.0, 2.0)]
    rows2 = [(-10.0, 0.0), (0.0, 2.0), (10.0, 4.0)]
    _write_pmf(rep1 / "analysis" / "pmf_gap.csv", rows1)
    _write_pmf(rep2 / "analysis" / "pmf_gap.csv", rows2)

    result = summarize_replicates([rep1, rep2], tmp_path / "summary")
    summary = result["summary"]
    assert summary.n_replicates == 2
    assert summary.barrier_mean_kj_mol == 12.0
    assert summary.reaction_free_energy_mean_kj_mol == 3.0
    assert (tmp_path / "summary" / "replicate_summary.json").exists()
    assert (tmp_path / "summary" / "pmf_gap_replicates.csv").exists()
