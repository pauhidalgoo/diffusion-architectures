from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hemera.config import load_config
from hemera.data import (
    _materialize_parquet_shard,
    _raw_cache_root,
    _records_by_parquet_shard,
    select_manifest_subset,
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a nonoverlapping range of pinned Photonyx shards"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--first-shard", type=int, required=True)
    parser.add_argument("--last-shard", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress", required=True)
    args = parser.parse_args()
    if args.first_shard < 0 or args.last_shard <= args.first_shard:
        raise ValueError("Require 0 <= first-shard < last-shard")
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    config = load_config(args.config)
    manifest = Path(args.manifest)
    layout = json.loads(
        manifest.with_suffix(".shards.json").read_text(encoding="utf-8")
    )
    files = list(layout["files"])
    lengths = [int(value) for value in layout["shard_lengths"]]
    if args.last_shard > len(files):
        raise ValueError("Requested shard range exceeds the pinned layout")
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = select_manifest_subset(records, config.data.max_items)
    assignments = _records_by_parquet_shard(selected, lengths)
    raw_root = _raw_cache_root(config)
    started = time.monotonic()
    completed = 0
    observed = 0
    newly_materialized = 0
    jobs = [
        (shard_index, files[shard_index], offset, shard_records)
        for shard_index, offset, shard_records in assignments[
            args.first_shard : args.last_shard
        ]
        if shard_records
    ]

    def write(stage: str) -> None:
        atomic_json(
            Path(args.progress),
            {
                "stage": stage,
                "first_shard": args.first_shard,
                "last_shard_exclusive": args.last_shard,
                "jobs": len(jobs),
                "completed_jobs": completed,
                "selected_records_in_completed_jobs": observed,
                "newly_materialized": newly_materialized,
                "elapsed_seconds": time.monotonic() - started,
            },
        )

    write("starting")
    if not jobs:
        write("complete")
        return
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = {
            pool.submit(
                _materialize_parquet_shard,
                url,
                offset,
                shard_records,
                raw_root,
                config.data.image_size,
            ): shard_index
            for shard_index, url, offset, shard_records in jobs
        }
        for future in as_completed(futures):
            shard_observed, shard_new = future.result()
            completed += 1
            observed += shard_observed
            newly_materialized += shard_new
            write(f"materializing:{completed}/{len(jobs)}")
    write("complete")


if __name__ == "__main__":
    main()
