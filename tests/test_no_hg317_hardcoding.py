from __future__ import annotations

from pathlib import Path


def test_no_system_specific_hardcoding_in_source():
    forbidden = ["HG3", "hg317", "g-xtb", "gxtb", "local_pme_q_atoms_cutoff_0.8"]
    matches = []
    for path in Path("src/kemp_evb").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                matches.append(f"{path}:{token}")
    assert matches == []
