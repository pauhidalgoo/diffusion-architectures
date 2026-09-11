from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .config import config_from_dict
from .evaluation import evaluate_checkpoint
from .runtime import sha256_file
from .trainer import train_from_config


def deep_update(target: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _validated_completed_checkpoint(
    run_dir: Path, expected_config: dict[str, Any]
) -> Path | None:
    """Return a complete exact-config checkpoint, or fail closed on corruption."""
    manifest_path = run_dir / "manifest.json"
    checkpoint = run_dir / "checkpoint-last.pt"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config") != expected_config:
        raise RuntimeError(
            f"Completed sweep directory has a different config: {run_dir}"
        )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Completed sweep manifest is missing checkpoint-last.pt: {run_dir}"
        )
    expected_hash = manifest.get("checkpoint_hashes", {}).get(
        checkpoint.name
    )
    if not expected_hash or sha256_file(checkpoint) != expected_hash:
        raise RuntimeError(f"Sweep checkpoint checksum mismatch: {checkpoint}")
    return checkpoint


def _preserve_partial_attempt(run_dir: Path) -> Path | None:
    """Move an incomplete run aside rather than silently overwriting evidence."""
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return None
    failures = run_dir.parent / "_failed_attempts"
    failures.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        destination = failures / f"{run_dir.name}-attempt-{attempt:03d}"
        if not destination.exists():
            break
        attempt += 1
    os.replace(run_dir, destination)
    return destination


def run_sweep(path: str | Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    base_path = Path(path).parent / payload["base"]
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError("Sweep must define at least one run")
    total_budget = float(payload.get("max_eur", base.get("budget", {}).get("max_eur", 0.0)))
    per_run_budget = total_budget / len(runs) if total_budget else 0.0
    results: list[dict[str, Any]] = []
    for run in runs:
        raw = deep_update(copy.deepcopy(base), run.get("overrides", {}))
        raw["name"] = run["name"]
        raw.setdefault("train", {})["output_dir"] = str(
            Path(payload.get("output_dir", "runs/sweep")) / run["name"]
        )
        if per_run_budget:
            raw.setdefault("budget", {})["max_eur"] = per_run_budget
        config = config_from_dict(raw)
        run_dir = Path(config.train.output_dir)
        checkpoint = _validated_completed_checkpoint(
            run_dir, config.to_dict()
        )
        evaluation_path = run_dir / "evaluation.json"
        if checkpoint is None:
            _preserve_partial_attempt(run_dir)
            checkpoint = train_from_config(config)
            metrics = evaluate_checkpoint(config, checkpoint)
        elif evaluation_path.exists():
            metrics = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
        else:
            metrics = evaluate_checkpoint(config, checkpoint)
        results.append({"name": run["name"], "checkpoint": str(checkpoint), **metrics})
    output = Path(payload.get("output_dir", "runs/sweep")) / "sweep-results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
