from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EVBConfig
from .histograms import WindowObservableData


@dataclass(slots=True)
class WindowStatistics:
    window_id: str
    n_frames: int
    mean_gap_kj_mol: float
    std_gap_kj_mol: float
    statistical_inefficiency: float
    effective_sample_size: float
    block_sem_kj_mol: float | None


def compute_window_statistics(config: EVBConfig, window_data: list[WindowObservableData]) -> list[WindowStatistics]:
    return [_statistics_for_window(config, window) for window in window_data]


def _statistics_for_window(config: EVBConfig, window: WindowObservableData) -> WindowStatistics:
    values = np.asarray(window.values_kj_mol, dtype=float)
    n_frames = len(values)
    inefficiency = _statistical_inefficiency(values)
    block_sem = _block_sem(values, max(config.analysis.uncertainty.blocks, 1))
    return WindowStatistics(
        window_id=window.window_id,
        n_frames=n_frames,
        mean_gap_kj_mol=float(np.mean(values)) if n_frames else float("nan"),
        std_gap_kj_mol=float(np.std(values, ddof=1)) if n_frames > 1 else 0.0,
        statistical_inefficiency=float(inefficiency),
        effective_sample_size=float(n_frames / inefficiency) if inefficiency > 0 else 0.0,
        block_sem_kj_mol=block_sem,
    )


def _statistical_inefficiency(values: np.ndarray) -> float:
    n = len(values)
    if n < 3:
        return 1.0
    centered = values - np.mean(values)
    variance = float(np.dot(centered, centered) / n)
    if variance <= 0.0:
        return 1.0
    g = 1.0
    max_lag = min(n - 1, 1000)
    for lag in range(1, max_lag + 1):
        corr = float(np.dot(centered[:-lag], centered[lag:]) / ((n - lag) * variance))
        if corr <= 0.0:
            break
        g += 2.0 * corr * (1.0 - lag / n)
    return max(g, 1.0)


def _block_sem(values: np.ndarray, blocks: int) -> float | None:
    n = len(values)
    if blocks < 2 or n < blocks:
        return None
    block_size = n // blocks
    if block_size < 1:
        return None
    trimmed = values[: block_size * blocks]
    means = trimmed.reshape(blocks, block_size).mean(axis=1)
    return float(np.std(means, ddof=1) / np.sqrt(blocks))
