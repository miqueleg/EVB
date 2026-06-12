from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def reconstruct_wt_fel_from_bias_table(table_csv: str | Path, output: str | Path, *, bias_factor: float) -> dict[str, Any]:
    gaps = []
    bias = []
    with Path(table_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gap = row.get("gap_kj_mol") or row.get("x") or row.get("grid")
            val = row.get("bias_kj_mol") or row.get("bias") or row.get("value")
            if gap is not None and val is not None:
                gaps.append(float(gap)); bias.append(float(val))
    gamma = float(bias_factor)
    if gamma <= 1.0:
        raise ValueError("WT-MetaD bias_factor must be > 1 for FEL reconstruction.")
    gap = np.asarray(gaps, dtype=float)
    fel = -gamma / (gamma - 1.0) * np.asarray(bias, dtype=float)
    if len(fel): fel -= np.nanmin(fel)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    with (out / "metad_fel.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["gap_kj_mol", "fel_kj_mol", "gap_kcal_mol", "fel_kcal_mol"])
        for x, y in zip(gap, fel): writer.writerow([float(x), float(y), float(x)/4.184, float(y)/4.184])
    summary = {"analysis_method": "well_tempered_bias_table", "bias_factor": gamma, "n_grid_points": int(len(gap)), "fel_csv": str(out / "metad_fel.csv")}
    (out / "metad_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
