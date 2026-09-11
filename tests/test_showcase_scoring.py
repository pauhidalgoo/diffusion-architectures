from __future__ import annotations

from PIL import Image

from scripts.score_showcase_candidates import (
    add_composite_scores,
    rank01,
    technical_metrics,
)


def test_showcase_technical_metrics_are_finite() -> None:
    image = Image.new("RGB", (32, 32), (30, 100, 200))
    metrics = technical_metrics(image)
    assert set(metrics) == {
        "sharpness",
        "entropy",
        "clipped_fraction",
        "channel_spread",
    }
    assert all(value == value for value in metrics.values())


def test_showcase_ranking_and_composite_are_per_prompt() -> None:
    assert rank01([1.0, 3.0, 2.0]) == [0.0, 1.0, 0.5]
    rows = []
    for prompt_id in ("a", "b"):
        for index in range(4):
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "clip_alignment": float(index),
                    "sharpness": float(index),
                    "entropy": float(index),
                    "clipped_fraction": float(3 - index),
                }
            )
    add_composite_scores(rows)
    for prompt_id in ("a", "b"):
        scores = [
            row["automatic_score"]
            for row in rows
            if row["prompt_id"] == prompt_id
        ]
        assert scores == sorted(scores)
