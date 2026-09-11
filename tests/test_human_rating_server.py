from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.human_rating_server import StudyStore


FIELDS = (
    "item_id",
    "assignment_id",
    "prompt",
    "category",
    "left_image",
    "right_image",
    "alignment",
    "quality",
    "overall",
    "rater_id",
)


def _study(path: Path, left: Path, right: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for slot in range(3):
            writer.writerow(
                {
                    "item_id": "one",
                    "assignment_id": f"0000-{slot}",
                    "prompt": "a prompt",
                    "category": "test",
                    "left_image": left,
                    "right_image": right,
                    "alignment": "",
                    "quality": "",
                    "overall": "",
                    "rater_id": "",
                }
            )


def test_rating_store_saves_only_assigned_slot(tmp_path: Path) -> None:
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    study = tmp_path / "study.csv"
    _study(study, left, right)

    store = StudyStore(study, "rater-a", 1)
    assert store.state()["total"] == 1
    assert store.image("0000-1", "left") == left.resolve()
    final = store.rate(
        {
            "assignment_id": "0000-1",
            "alignment": "left",
            "quality": "tie",
            "overall": "right",
        }
    )
    assert final["done"] is True

    rows = list(csv.DictReader(study.open(encoding="utf-8")))
    assert rows[1]["rater_id"] == "rater-a"
    assert rows[1]["quality"] == "tie"
    assert rows[0]["rater_id"] == rows[2]["rater_id"] == ""
    with pytest.raises(ValueError, match="another rater"):
        StudyStore(study, "different-rater", 1)


def test_rating_store_rejects_incomplete_or_duplicate_votes(tmp_path: Path) -> None:
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    study = tmp_path / "study.csv"
    _study(study, left, right)
    store = StudyStore(study, "rater-a", 0)

    with pytest.raises(ValueError, match="Every rating"):
        store.rate(
            {
                "assignment_id": "0000-0",
                "alignment": "left",
                "quality": "tie",
            }
        )
    store.rate(
        {
            "assignment_id": "0000-0",
            "alignment": "left",
            "quality": "tie",
            "overall": "right",
        }
    )
    with pytest.raises(ValueError, match="already rated"):
        store.rate(
            {
                "assignment_id": "0000-0",
                "alignment": "right",
                "quality": "right",
                "overall": "right",
            }
        )
