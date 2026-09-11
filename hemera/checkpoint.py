from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    samples_seen: int,
    ema_state: dict[str, torch.Tensor] | None,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "samples_seen": samples_seen,
            "ema": ema_state,
            "rng": rng_state(),
            "config": config,
        }
    if extra:
        payload.update(extra)
    atomic_torch_save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng and "rng" in payload:
        restore_rng_state(payload["rng"])
    return payload
