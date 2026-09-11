from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


def _module():
    path = Path("scripts/cloud/freeze_stage.py")
    spec = importlib.util.spec_from_file_location("freeze_stage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_config(path: Path, backbone: str, batch: int = 128) -> None:
    path.mkdir(parents=True)
    (path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"backbone": backbone},
                "objective": {"name": "flow"},
                "data": {"source_temperature": 1.0},
                "train": {
                    "batch_size": batch,
                    "effective_batch_size": 768,
                },
            }
        ),
        encoding="utf-8",
    )


def test_freeze_selects_best_efficiency_compatible_transformer(tmp_path: Path) -> None:
    module = _module()
    unet = tmp_path / "unet"
    mmdit = tmp_path / "mmdit"
    pixart = tmp_path / "pixart"
    _run_config(unet, "unet")
    _run_config(mmdit, "mmdit")
    _run_config(pixart, "pixart", batch=384)
    ranking = tmp_path / "ranking.json"
    ranking.write_text(
        json.dumps(
            [
                {"name": "unet", "path": str(unet)},
                {"name": "mmdit", "path": str(mmdit)},
                {"name": "pixart", "path": str(pixart)},
            ]
        ),
        encoding="utf-8",
    )
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "model": {"backbone": "dit"},
                "train": {"batch_size": 64, "effective_batch_size": 768},
            }
        ),
        encoding="utf-8",
    )
    module.freeze_winner(
        ranking,
        base,
        "backbone-transformer",
        tmp_path / "record.json",
    )
    frozen = yaml.safe_load(base.read_text(encoding="utf-8"))
    assert frozen["model"]["backbone"] == "pixart"
    assert frozen["train"]["batch_size"] == 384


def test_profile_batches_and_confirmation_are_materialized(tmp_path: Path) -> None:
    module = _module()
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        yaml.safe_dump(
            {
                "runs": [
                    {"name": "a", "overrides": {}},
                    {"name": "b", "overrides": {}},
                ]
            }
        ),
        encoding="utf-8",
    )
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "profiles": {
                    name: {
                        "results": {
                            "256": {"status": "ok", "images_per_second": speed},
                            "384": {
                                "status": "ok",
                                "images_per_second": speed + 1,
                            },
                        }
                    }
                    for name, speed in (("a", 10), ("b", 20))
                }
            }
        ),
        encoding="utf-8",
    )
    module.apply_profiled_batches(sweep, profiles)
    updated = yaml.safe_load(sweep.read_text(encoding="utf-8"))
    assert all(
        run["overrides"]["train"]
        == {"batch_size": 384, "effective_batch_size": 768}
        for run in updated["runs"]
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    _run_config(first, "pixart")
    _run_config(second, "dit")
    ranking = tmp_path / "ranking.json"
    ranking.write_text(
        json.dumps(
            [
                {"name": "first", "path": str(first)},
                {"name": "second", "path": str(second)},
            ]
        ),
        encoding="utf-8",
    )
    confirmation = tmp_path / "confirmation.yaml"
    module.prepare_confirmation(ranking, confirmation, max_eur=0.36)
    payload = yaml.safe_load(confirmation.read_text(encoding="utf-8"))
    assert payload["max_eur"] == 0.36
    assert len(payload["runs"]) == 6
    assert [run["overrides"]["train"]["seed"] for run in payload["runs"]] == [
        101,
        202,
        303,
        101,
        202,
        303,
    ]
    assert [
        run["overrides"]["model"]["backbone"] for run in payload["runs"]
    ] == ["pixart", "pixart", "pixart", "dit", "dit", "dit"]
    assert all(
        run["overrides"]["objective"] == {"name": "flow"}
        for run in payload["runs"]
    )
