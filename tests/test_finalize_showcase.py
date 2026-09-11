from __future__ import annotations

import sys
from pathlib import Path

from scripts.finalize_showcase import read_jsonl


def test_finalize_showcase_requires_exactly_64_selections(tmp_path: Path) -> None:
    scored = tmp_path / "scored.jsonl"
    selection = tmp_path / "selection.jsonl"
    scored.write_text("", encoding="utf-8")
    selection.write_text("", encoding="utf-8")
    assert read_jsonl(scored) == []

    root = Path(__file__).resolve().parents[1]
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "scripts/finalize_showcase.py",
            "--scored",
            str(scored),
            "--selection",
            str(selection),
            "--output",
            str(tmp_path / "final"),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "exactly 64" in result.stderr
