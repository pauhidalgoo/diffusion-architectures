from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_showcase_candidates import jobs, load_existing, write_manifest


def test_showcase_jobs_are_deterministic_and_unique(tmp_path: Path) -> None:
    prompts = [
        {"id": "one", "category": "a", "prompt": "first"},
        {"id": "two", "category": "b", "prompt": "second"},
    ]
    planned = jobs(prompts, 4, 1000, 15, 5.0, "bad", tmp_path)
    assert len(planned) == 8
    assert len({row["candidate_id"] for row in planned}) == 8
    assert [row["seed"] for row in planned[:4]] == [1000, 1001, 1002, 1003]
    assert [row["seed"] for row in planned[4:]] == [1100, 1101, 1102, 1103]


def test_showcase_manifest_round_trip_is_sorted(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    rows = {
        "b": {"candidate_id": "b", "image": "b.png"},
        "a": {"candidate_id": "a", "image": "a.png"},
    }
    write_manifest(path, rows)
    assert [json.loads(line)["candidate_id"] for line in path.read_text().splitlines()] == [
        "a",
        "b",
    ]
    assert load_existing(path) == rows
