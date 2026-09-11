#!/usr/bin/env python3
"""Freeze predeclared sweep winners and profiled batches into later stages."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _ranked_config(row: dict) -> dict:
    path = Path(row["path"]) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Ranked run config is missing: {path}")
    return _load_yaml(path)


def freeze_winner(
    ranking_path: Path,
    base_path: Path,
    stage: str,
    record_path: Path,
) -> None:
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    if not ranking:
        raise ValueError(f"Empty ranking: {ranking_path}")
    selected_row = ranking[0]
    if stage == "backbone-transformer":
        selected_row = next(
            row
            for row in ranking
            if row["name"]
            not in {"unet", "mmdit", "representation-unet", "representation-mmdit"}
        )
    selected = _ranked_config(selected_row)
    base = _load_yaml(base_path)
    if stage == "objective":
        base["objective"] = copy.deepcopy(selected["objective"])
    elif stage in {"conditioning", "backbone-transformer", "efficiency"}:
        base["model"] = copy.deepcopy(selected["model"])
        base["train"]["batch_size"] = selected["train"]["batch_size"]
        base["train"]["effective_batch_size"] = selected["train"][
            "effective_batch_size"
        ]
    elif stage == "data":
        base["data"]["source_temperature"] = selected["data"][
            "source_temperature"
        ]
    else:
        raise ValueError(f"Unsupported freeze stage: {stage}")
    _write_yaml(base_path, base)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "stage": stage,
                "ranking": str(ranking_path),
                "selected": selected_row,
                "selected_config": str(Path(selected_row["path"]) / "config.yaml"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def apply_profiled_batches(
    sweep_path: Path,
    profile_path: Path,
) -> None:
    sweep = _load_yaml(sweep_path)
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))["profiles"]
    effective_batch = 768
    for run in sweep["runs"]:
        profile = profiles[run["name"]]
        successful = [
            (int(size), metrics)
            for size, metrics in profile["results"].items()
            if metrics["status"] == "ok" and effective_batch % int(size) == 0
        ]
        if not successful:
            raise RuntimeError(f"No safe common-batch profile for {run['name']}")
        physical, _ = max(
            successful,
            key=lambda item: item[1]["images_per_second"],
        )
        train = run.setdefault("overrides", {}).setdefault("train", {})
        train["batch_size"] = physical
        train["effective_batch_size"] = effective_batch
    _write_yaml(sweep_path, sweep)


def prepare_confirmation(
    ranking_path: Path,
    output_path: Path,
    max_eur: float = 0.9,
) -> None:
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    if len(ranking) < 2:
        raise ValueError("Confirmation requires two ranked complete recipes")
    candidates = [_ranked_config(row) for row in ranking[:2]]
    runs = []
    for label, candidate in zip(("a", "b"), candidates, strict=True):
        for seed in (101, 202, 303):
            runs.append(
                {
                    "name": f"finalist-{label}-seed-{seed}",
                    "overrides": {
                        "model": copy.deepcopy(candidate["model"]),
                        "objective": copy.deepcopy(candidate["objective"]),
                        "train": {"seed": seed},
                    },
                }
            )
    payload = {
        "base": "../experiment-base.yaml",
        "output_dir": "runs/confirmatory",
        "max_eur": max_eur,
        "runs": runs,
    }
    _write_yaml(output_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--ranking", type=Path, required=True)
    freeze.add_argument("--base", type=Path, required=True)
    freeze.add_argument(
        "--stage",
        choices=[
            "objective",
            "conditioning",
            "backbone-transformer",
            "efficiency",
            "data",
        ],
        required=True,
    )
    freeze.add_argument("--record", type=Path, required=True)

    batches = subparsers.add_parser("apply-batches")
    batches.add_argument("--sweep", type=Path, required=True)
    batches.add_argument("--profiles", type=Path, required=True)

    confirmation = subparsers.add_parser("prepare-confirmation")
    confirmation.add_argument("--ranking", type=Path, required=True)
    confirmation.add_argument("--output", type=Path, required=True)
    confirmation.add_argument("--max-eur", type=float, default=0.9)

    args = parser.parse_args()
    if args.command == "freeze":
        freeze_winner(args.ranking, args.base, args.stage, args.record)
    elif args.command == "apply-batches":
        apply_profiled_batches(args.sweep, args.profiles)
    else:
        prepare_confirmation(args.ranking, args.output, args.max_eur)


if __name__ == "__main__":
    main()
