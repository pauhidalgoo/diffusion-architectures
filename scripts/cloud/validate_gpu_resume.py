from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hemera.checkpoint import load_checkpoint
from hemera.config import load_config
from hemera.models import build_model
from hemera.trainer import train_from_config


def state(path: Path, config) -> dict[str, torch.Tensor]:
    model = build_model(config.model)
    load_checkpoint(path, model, restore_rng=False)
    return model.state_dict()


def configured(output: Path, steps: int):
    config = load_config("configs/smoke.yaml")
    config.name = "hemera-cuda-resume-preflight"
    config.train.output_dir = str(output)
    config.train.device = "cuda"
    config.train.precision = "bfloat16"
    config.train.steps = steps
    config.train.save_every = 2
    config.train.log_every = 1
    return config


def main() -> None:
    root = Path("preflight/cuda-resume")
    uninterrupted_config = configured(root / "uninterrupted", 4)
    uninterrupted = train_from_config(uninterrupted_config)

    first_config = configured(root / "resumed", 2)
    first = train_from_config(first_config)
    resumed_config = configured(root / "resumed", 4)
    resumed_config.train.resume = str(first)
    resumed = train_from_config(resumed_config)

    expected = state(uninterrupted, uninterrupted_config)
    actual = state(resumed, resumed_config)
    mismatches: list[str] = []
    maximum_difference = 0.0
    for name in expected:
        difference = (expected[name] - actual[name]).abs().max().item()
        maximum_difference = max(maximum_difference, difference)
        if difference != 0.0:
            mismatches.append(name)
    print(
        {
            "exact": not mismatches,
            "mismatched_tensors": mismatches,
            "maximum_absolute_difference": maximum_difference,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        }
    )
    if mismatches:
        raise SystemExit("CUDA resume did not reproduce uninterrupted weights exactly")


if __name__ == "__main__":
    main()
