from __future__ import annotations

import os
from pathlib import Path

from kemp_evb.sampling_workflow import write_run_scripts


def test_generic_run_scripts_are_executable_and_generic(tmp_path: Path):
    scripts = write_run_scripts(tmp_path / "run")

    for path in scripts.values():
        p = Path(path)
        assert p.exists()
        assert os.access(p, os.X_OK)
        assert "run-workflow" in p.read_text(encoding="utf-8") or p.name == "monitor.sh"
