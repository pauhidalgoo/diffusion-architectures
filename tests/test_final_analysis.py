from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_analysis_module():
    path = Path(__file__).parents[1] / "scripts" / "analyze_final_delivery.py"
    spec = importlib.util.spec_from_file_location("analyze_final_delivery", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compact_metrics_omits_large_feature_payloads() -> None:
    analysis = _load_analysis_module()
    compact = analysis._compact_test_metrics(
        {
            "samples": 2,
            "cmmd": 0.5,
            "fid": 12.0,
            "paired_features": {"row_ids": ["a", "b"]},
            "per_source": {
                "source-a": {
                    "samples": 2,
                    "cmmd": 0.6,
                    "clip_score_mean": 0.2,
                    "unused": [1, 2, 3],
                }
            },
        }
    )

    assert compact["samples"] == 2
    assert "paired_features" not in compact
    assert compact["per_source"]["source-a"] == {
        "samples": 2,
        "cmmd": 0.6,
        "precision": None,
        "recall": None,
        "clip_score_mean": 0.2,
        "siglip_alignment_mean": None,
        "hpsv2_mean": None,
    }


def test_binning_deduplicates_resumed_steps() -> None:
    analysis = _load_analysis_module()
    rows = [
        {"step": 1, "loss": 3.0},
        {"step": 2, "loss": 2.0},
        {"step": 2, "loss": 1.0},
        {"step": 3, "loss": 0.5},
    ]

    binned = analysis._bin_rows(rows, value_keys=("loss",), bins=10)

    assert [row["step"] for row in binned] == [1.0, 2.0, 3.0]
    assert [row["loss"] for row in binned] == [3.0, 1.0, 0.5]


def test_histogram_is_normalized_to_peak_bin() -> None:
    analysis = _load_analysis_module()
    histogram = analysis._histogram([0.1, 0.1, 0.2, 0.9], bins=4)

    assert len(histogram) == 4
    assert max(value for _, value in histogram) == 1.0
