"""Validate a Hemera checkpoint without taking memory from the training GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from hemera.runtime import sha256_file


def tensor_stats(value: Any) -> tuple[int, int]:
    tensors = 0
    elements = 0
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Checkpoint contains a non-finite tensor")
        return 1, value.numel()
    if isinstance(value, dict):
        for child in value.values():
            child_tensors, child_elements = tensor_stats(child)
            tensors += child_tensors
            elements += child_elements
    elif isinstance(value, (list, tuple)):
        for child in value:
            child_tensors, child_elements = tensor_stats(child)
            tensors += child_tensors
            elements += child_elements
    return tensors, elements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "step", "samples_seen", "ema", "config"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")

    sections = {}
    for name in ("model", "optimizer", "ema"):
        tensors, elements = tensor_stats(payload[name])
        sections[name] = {"tensors": tensors, "elements": elements}

    result = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "step": int(payload["step"]),
        "samples_seen": int(payload["samples_seen"]),
        "best_validation_loss": float(payload["best_validation_loss"]),
        "sections": sections,
        "all_floating_tensors_finite": True,
    }
    encoded = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, output)
    print(encoded)


if __name__ == "__main__":
    main()
