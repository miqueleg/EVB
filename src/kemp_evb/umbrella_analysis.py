from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

K_B_KCAL = 0.00198720425864083


def read_gap_samples(paths: list[str | Path]) -> np.ndarray:
    values = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                gap = row.get("shifted_gap_kcal_mol") or row.get("gap_kcal_mol")
                if gap is None:
                    kj = row.get("shifted_gap") or row.get("shifted_gap_kj_mol") or row.get("gap_kj_mol")
                    gap = None if kj is None else float(kj) / 4.184
                if gap is not None and gap != "":
                    values.append(float(gap))
    return np.asarray(values, dtype=float)


def histogram_pmf(samples: np.ndarray, bins: int = 100, temperature_k: float = 300.0) -> tuple[np.ndarray, np.ndarray]:
    if len(samples) == 0:
        grid = np.linspace(-1.0, 1.0, bins)
        return grid, np.full_like(grid, np.nan)
    hist, edges = np.histogram(samples, bins=min(bins, max(5, len(samples) // 5)))
    grid = 0.5 * (edges[:-1] + edges[1:])
    prob = np.maximum(hist.astype(float), 1.0e-12)
    pmf = -K_B_KCAL * temperature_k * np.log(prob)
    pmf -= np.nanmin(pmf)
    return grid, pmf


def calculate_barrier(gap_kcal_mol, pmf_kcal_mol) -> dict[str, float | None]:
    gap = np.asarray(gap_kcal_mol, dtype=float)
    pmf = np.asarray(pmf_kcal_mol, dtype=float)
    mask = np.isfinite(gap) & np.isfinite(pmf)
    gap = gap[mask]; pmf = pmf[mask]
    if len(gap) == 0:
        return {"barrier_from_left_kcal": None, "barrier_from_right_kcal": None, "deltaG_reaction_kcal": None, "pmf_at_gap0_kcal": None}
    left = gap <= 0; right = gap >= 0
    left_min = float(np.min(pmf[left])) if np.any(left) else float(np.min(pmf))
    right_min = float(np.min(pmf[right])) if np.any(right) else float(np.min(pmf))
    order = np.argsort(gap)
    pmf0 = float(np.interp(0.0, gap[order], pmf[order])) if len(gap) > 1 else float(pmf[0])
    return {"barrier_from_left_kcal": pmf0 - left_min, "barrier_from_right_kcal": pmf0 - right_min, "deltaG_reaction_kcal": right_min - left_min, "pmf_at_gap0_kcal": pmf0}


def analyze_umbrella_outputs(input_paths: list[str | Path], output: str | Path, *, method: str = "wham_fallback", temperature_k: float = 300.0) -> dict[str, Any]:
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    samples = read_gap_samples(input_paths)
    grid, pmf = histogram_pmf(samples, temperature_k=temperature_k)
    with (out / "pmf.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["gap_kcal_mol", "pmf_kcal_mol"])
        for x, y in zip(grid, pmf): writer.writerow([float(x), float(y)])
    barrier = calculate_barrier(grid, pmf)
    barrier["analysis_method"] = method
    barrier["n_samples"] = int(len(samples))
    (out / "barrier_summary.json").write_text(json.dumps(barrier, indent=2), encoding="utf-8")
    return barrier
