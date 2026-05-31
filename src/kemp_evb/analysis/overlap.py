from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EVBConfig
from .histograms import WindowObservableData, histogram_edges


@dataclass(slots=True)
class WindowOverlap:
    window_id: str
    overlaps: dict[str, float]


def compute_window_overlap_matrix(config: EVBConfig, window_data: list[WindowObservableData]) -> list[WindowOverlap]:
    edges = histogram_edges(config)
    histograms: dict[str, np.ndarray] = {}
    for window in window_data:
        counts, _ = np.histogram(window.values_kj_mol, bins=edges)
        total = counts.sum()
        histograms[window.window_id] = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
    ordered = [window.window_id for window in window_data]
    results: list[WindowOverlap] = []
    for row_id in ordered:
        overlaps = {}
        for col_id in ordered:
            overlaps[col_id] = float(np.minimum(histograms[row_id], histograms[col_id]).sum())
        results.append(WindowOverlap(window_id=row_id, overlaps=overlaps))
    return results
