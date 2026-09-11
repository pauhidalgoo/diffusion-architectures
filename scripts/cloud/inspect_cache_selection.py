#!/usr/bin/env python3
"""Report which audited rows remain absent from the reusable raw-image cache."""

from __future__ import annotations

import json
from pathlib import Path

from hemera.config import load_config
from hemera.data import _raw_cache_root, _raw_image_path, select_manifest_subset


def main() -> None:
    config = load_config("configs/experiment-base.yaml")
    manifest = Path("data/manifests/photonyx.jsonl")
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = select_manifest_subset(records, config.data.max_items)
    raw_root = _raw_cache_root(config)
    missing = [
        record
        for record in selected
        if not _raw_image_path(raw_root, int(record["row_index"])).exists()
    ]
    layout = json.loads(
        manifest.with_suffix(".shards.json").read_text(encoding="utf-8")
    )
    counts: list[dict[str, int]] = []
    offset = 0
    for shard, length in enumerate(layout["shard_lengths"]):
        end = offset + int(length)
        count = sum(
            offset <= int(record["row_index"]) < end for record in missing
        )
        if count:
            counts.append({"shard": shard, "missing": count})
        offset = end
    print(
        json.dumps(
            {
                "selected": len(selected),
                "raw_present": len(selected) - len(missing),
                "missing": len(missing),
                "missing_shards": len(counts),
                "largest_missing_shards": sorted(
                    counts, key=lambda item: -item["missing"]
                )[:20],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
