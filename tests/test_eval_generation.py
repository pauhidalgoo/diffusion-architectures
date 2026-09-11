from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

from hemera.config import config_from_dict
from hemera.data import _raw_cache_root, _raw_image_path
from hemera.eval_generation import materialize_audit_images


def test_materialize_audit_images_reuses_complete_raw_cache(
    tmp_path: Path, monkeypatch
) -> None:
    config = config_from_dict(
        {
            "data": {
                "dataset_revision": "fixture-revision",
                "raw_cache_dir": str(tmp_path / "raw-cache"),
                "image_size": 16,
            }
        }
    )
    cached = _raw_image_path(_raw_cache_root(config), 7)
    cached.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), (12, 34, 56)).save(cached)
    manifest = tmp_path / "audit.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "cached/row",
                "row_index": 7,
                "prompt": "a cached image",
                "source": "fixture",
                "license": "CC0",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def unexpected_stream(_):
        raise AssertionError("complete raw cache must bypass dataset streaming")

    monkeypatch.setattr("hemera.eval_generation._load_photonyx", unexpected_stream)
    output = tmp_path / "materialized"
    result = materialize_audit_images(
        config, manifest, output, "train", samples=1, seed=1234
    )
    row = json.loads(result.read_text(encoding="utf-8"))
    materialized = Path(row["image"])

    assert materialized.exists()
    assert materialized.read_bytes() == cached.read_bytes()
    assert row["id"] == "cached/row"
    assert row["materialization_seed"] == 1234
    if os.name != "nt":
        assert materialized.stat().st_ino == cached.stat().st_ino
