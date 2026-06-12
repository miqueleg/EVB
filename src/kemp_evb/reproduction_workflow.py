from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .reference_profile import load_reference_profile
from .simulation import ensure_output_dir


def compare_profile(reference_path: str | Path, output: str | Path, *, umbrella_barrier_json: str | Path | None = None, metad_barrier_json: str | Path | None = None, legacy_barrier_json: str | Path | None = None, warning_kcal: float = 2.0, fail_kcal: float = 3.0) -> dict[str, Any]:
    reference = load_reference_profile(reference_path)
    target = reference.barrier_kj_mol()
    target_kcal = None if target is None else target / 4.184
    rows = [{"method": reference.method_label, "barrier_kcal_mol": target_kcal, "delta_vs_reference": 0.0, "status": "reference"}]
    for label, path in [("umbrella", umbrella_barrier_json), ("metad", metad_barrier_json), ("legacy", legacy_barrier_json)]:
        if path and Path(path).exists():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            barrier = data.get("barrier_from_left_kcal") or data.get("barrier_kcal_mol")
            delta = None if barrier is None or target_kcal is None else float(barrier) - target_kcal
            status = "missing" if delta is None else ("fail" if abs(delta) > fail_kcal else "warning" if abs(delta) > warning_kcal else "pass")
            rows.append({"method": label, "barrier_kcal_mol": barrier, "delta_vs_reference": delta, "status": status})
    out = ensure_output_dir(output)
    with (out / "barrier_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "barrier_kcal_mol", "delta_vs_reference", "status"])
        writer.writeheader(); writer.writerows(rows)
    with (out / "barrier_comparison.md").open("w", encoding="utf-8") as handle:
        handle.write("| method | barrier kcal/mol | delta vs reference | status |\n| --- | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(f"| {row['method']} | {row.get('barrier_kcal_mol')} | {row.get('delta_vs_reference')} | {row.get('status')} |\n")
    report = {"barriers": rows, "warning_kcal": warning_kcal, "fail_kcal": fail_kcal}
    (out / "reproduction_report.md").write_text((out / "barrier_comparison.md").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "comparison_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
