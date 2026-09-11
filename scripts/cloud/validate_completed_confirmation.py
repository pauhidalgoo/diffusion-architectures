#!/usr/bin/env python3
"""Validate immutable confirmation outputs without rebuilding mutable sweep configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


SEEDS = (101, 202, 303)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/confirmatory"))
    parser.add_argument(
        "--efficiency-ranking",
        type=Path,
        default=Path("reports/efficiency-ranking.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ranking = json.loads(args.efficiency_ranking.read_text(encoding="utf-8"))
    if len(ranking) < 2:
        raise SystemExit("Efficiency ranking does not contain two finalists")
    expected_recipes = {
        "a": load_yaml(Path(ranking[0]["path"]) / "config.yaml"),
        "b": load_yaml(Path(ranking[1]["path"]) / "config.yaml"),
    }
    validated: list[dict] = []
    group_configs: dict[str, list[dict]] = {"a": [], "b": []}

    for label in ("a", "b"):
        for seed in SEEDS:
            name = f"finalist-{label}-seed-{seed}"
            run_dir = args.root / name
            manifest_path = run_dir / "manifest.json"
            checkpoint = run_dir / "checkpoint-last.pt"
            evaluation = run_dir / "evaluation.json"
            for required in (manifest_path, checkpoint, evaluation):
                if not required.exists():
                    raise SystemExit(f"Missing completed artifact: {required}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = manifest["config"]
            if manifest.get("name") != name or config.get("name") != name:
                raise SystemExit(f"Confirmation name mismatch: {run_dir}")
            if int(config["train"]["seed"]) != seed:
                raise SystemExit(f"Confirmation seed mismatch: {run_dir}")
            if int(config["train"]["effective_batch_size"]) != 768:
                raise SystemExit(f"Effective batch mismatch: {run_dir}")
            expected = expected_recipes[label]
            if config["model"] != expected["model"]:
                raise SystemExit(f"Finalist model recipe mismatch: {run_dir}")
            if config["objective"] != expected["objective"]:
                raise SystemExit(f"Finalist objective mismatch: {run_dir}")
            expected_hash = manifest.get("checkpoint_hashes", {}).get(
                checkpoint.name
            )
            observed_hash = sha256(checkpoint)
            if not expected_hash or observed_hash != expected_hash:
                raise SystemExit(f"Checkpoint checksum mismatch: {checkpoint}")
            if int(manifest.get("samples_seen", 0)) <= 0:
                raise SystemExit(f"Confirmation run contains no samples: {run_dir}")
            group_configs[label].append(
                {
                    "model": config["model"],
                    "objective": config["objective"],
                    "data": config["data"],
                    "effective_batch_size": config["train"][
                        "effective_batch_size"
                    ],
                }
            )
            validated.append(
                {
                    "name": name,
                    "seed": seed,
                    "physical_batch_size": config["train"]["batch_size"],
                    "effective_batch_size": config["train"][
                        "effective_batch_size"
                    ],
                    "samples_seen": manifest["samples_seen"],
                    "checkpoint_sha256": observed_hash,
                }
            )

    for label, configs in group_configs.items():
        if any(config != configs[0] for config in configs[1:]):
            raise SystemExit(f"Finalist {label} seeds do not share one recipe")

    payload = {
        "status": "complete",
        "runs": validated,
        "finalist_a_source": ranking[0],
        "finalist_b_source": ranking[1],
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
