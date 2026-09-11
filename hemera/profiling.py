from __future__ import annotations

import contextlib
import gc
import json
import statistics
import time
import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

import yaml

from .config import ExperimentConfig, config_from_dict, load_config
from .data import (
    ShardedLatentDataset,
    build_dataset,
    collate_examples,
    deterministic_dataset_indices,
)
from .models import build_model, count_parameters
from .objectives import build_objective
from .runtime import resolve_device, set_seed


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device, precision: str):
    if precision == "float32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _training_dtype(config: ExperimentConfig, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.train.precision]


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _profile_auxiliary_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    efficiency: str,
) -> Tensor | None:
    auxiliary: Tensor | None = None
    if efficiency == "repa":
        prediction = getattr(model, "take_repa_prediction")()
        target = batch.get("repa_target")
        if prediction is None or target is None:
            raise RuntimeError("REPA profiling requires precomputed targets")
        if target.shape[1] != prediction.shape[1]:
            side_in = int(target.shape[1] ** 0.5)
            side_out = int(prediction.shape[1] ** 0.5)
            target = F.interpolate(
                target.transpose(1, 2).reshape(
                    target.shape[0], target.shape[2], side_in, side_in
                ),
                size=(side_out, side_out),
                mode="bilinear",
                align_corners=False,
            ).flatten(2).transpose(1, 2)
        auxiliary = 0.5 * (
            F.normalize(prediction.float(), dim=-1)
            - F.normalize(target.float(), dim=-1)
        ).square().mean()
    moe_loss = getattr(model, "moe_balance_loss", lambda: None)()
    if moe_loss is not None:
        weighted = 0.01 * moe_loss
        auxiliary = weighted if auxiliary is None else auxiliary + weighted
    return auxiliary


def profile_batch_sizes(
    config: ExperimentConfig,
    output: str | Path,
    batch_sizes: list[int],
    warmup: int = 5,
    iterations: int = 20,
) -> dict[str, Any]:
    """Measure complete optimizer steps and recommend the fastest safe batch."""
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    device = resolve_device(config.train.device)
    dataset = build_dataset(config, "train")
    dtype = _training_dtype(config, device)
    results: dict[str, Any] = {}
    for batch_size in batch_sizes:
        model = None
        optimizer = None
        try:
            set_seed(config.train.seed)
            model = build_model(config.model).to(device).train()
            optimizer_options: dict[str, Any] = {"lr": config.train.learning_rate}
            if device.type == "cuda":
                optimizer_options["fused"] = True
                torch.cuda.reset_peak_memory_stats(device)
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)
            objective = build_objective(config.objective)
            timings: list[float] = []
            for step in range(warmup + iterations):
                indices = deterministic_dataset_indices(
                    dataset, batch_size, step, config.train.seed
                )
                if isinstance(dataset, ShardedLatentDataset):
                    batch = dataset.collate_indices(indices)
                else:
                    batch = collate_examples([dataset[index] for index in indices])
                latents = batch["latents"].to(device=device, dtype=dtype)
                condition = batch["condition"].to(device, dtype)
                if "repa_target" in batch:
                    batch["repa_target"] = batch["repa_target"].to(
                        device=device, dtype=dtype
                    )
                target = objective.make_training_batch(latents)
                optimizer.zero_grad(set_to_none=True)
                _synchronize(device)
                started = time.perf_counter()
                with _autocast(device, config.train.precision):
                    prediction = model(target.noisy, target.time, condition)
                    loss = objective.loss(prediction, target)
                    auxiliary = _profile_auxiliary_loss(
                        model, batch, config.model.efficiency
                    )
                    if auxiliary is not None:
                        loss = loss + auxiliary
                loss.backward()
                optimizer.step()
                _synchronize(device)
                if step >= warmup:
                    timings.append(time.perf_counter() - started)
            mean_seconds = statistics.mean(timings)
            results[str(batch_size)] = {
                "status": "ok",
                "mean_train_step_seconds": mean_seconds,
                "images_per_second": batch_size / mean_seconds,
                "peak_vram_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0
                ),
                "iterations": iterations,
            }
        except (torch.OutOfMemoryError, RuntimeError) as error:
            if not isinstance(error, torch.OutOfMemoryError) and "out of memory" not in str(error).lower():
                raise
            results[str(batch_size)] = {
                "status": "out_of_memory",
                "error": str(error).splitlines()[0],
            }
        finally:
            del optimizer, model
            _release_cuda()
    successful = [
        (int(size), values)
        for size, values in results.items()
        if values["status"] == "ok"
    ]
    recommended = (
        max(successful, key=lambda item: item[1]["images_per_second"])[0]
        if successful
        else None
    )
    payload = {
        "results": results,
        "recommended_batch_size": recommended,
        "device": str(device),
        "precision": config.train.precision,
        "compiled": False,
        "backbone": config.model.backbone,
    }
    target_path = Path(output)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def profile_sweep_batch_sizes(
    sweep_path: str | Path,
    output: str | Path,
    cache_dir: str | Path,
    batch_sizes: list[int],
    warmup: int = 2,
    iterations: int = 5,
) -> dict[str, Any]:
    """Profile every recipe in a sweep against the same real latent cache."""
    source = Path(sweep_path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    base = load_config(source.parent / payload["base"]).to_dict()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for run in payload.get("runs", []):
        raw = _deep_update(copy.deepcopy(base), run.get("overrides", {}))
        raw["name"] = f"profile-{run['name']}"
        raw["data"]["cache_dir"] = str(cache_dir)
        raw["train"]["dataset_mode"] = "shards"
        raw["train"]["compile"] = False
        config = config_from_dict(raw)
        results[run["name"]] = profile_batch_sizes(
            config,
            root / f"{run['name']}.json",
            batch_sizes,
            warmup,
            iterations,
        )
    summary = {
        "sweep": str(source),
        "cache_dir": str(cache_dir),
        "profiles": results,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def profile_attention(
    config: ExperimentConfig,
    output: str | Path,
    warmup: int = 5,
    iterations: int = 20,
) -> dict[str, Any]:
    """Compare complete train steps; window attention advances only at >=15%."""
    device = resolve_device(config.train.device)
    dataset = build_dataset(config, "train")
    objective = build_objective(config.objective)
    results: dict[str, Any] = {}
    for attention in ("full", "window"):
        set_seed(config.train.seed)
        model_config = replace(config.model, attention=attention)
        model = build_model(model_config).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        timings: list[float] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(warmup + iterations):
            indices = deterministic_dataset_indices(
                dataset, config.train.batch_size, step, config.train.seed
            )
            if isinstance(dataset, ShardedLatentDataset):
                batch = dataset.collate_indices(indices)
            else:
                batch = collate_examples([dataset[index] for index in indices])
            dtype = _training_dtype(config, device)
            latents = batch["latents"].to(device=device, dtype=dtype)
            condition = batch["condition"].to(device, dtype)
            if "repa_target" in batch:
                batch["repa_target"] = batch["repa_target"].to(
                    device=device, dtype=dtype
                )
            target = objective.make_training_batch(latents)
            _synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, config.train.precision):
                prediction = model(target.noisy, target.time, condition)
                loss = objective.loss(prediction, target)
                auxiliary = _profile_auxiliary_loss(
                    model, batch, config.model.efficiency
                )
                if auxiliary is not None:
                    loss = loss + auxiliary
            loss.backward()
            optimizer.step()
            _synchronize(device)
            if step >= warmup:
                timings.append(time.perf_counter() - started)
        mean_seconds = statistics.mean(timings)
        results[attention] = {
            "mean_train_step_seconds": mean_seconds,
            "images_per_second": config.train.batch_size / mean_seconds,
            "parameter_count": count_parameters(model),
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "iterations": iterations,
        }
    speedup = results["full"]["mean_train_step_seconds"] / results["window"][
        "mean_train_step_seconds"
    ] - 1.0
    payload = {
        "full": results["full"],
        "window": results["window"],
        "window_speedup_fraction": speedup,
        "threshold_fraction": 0.15,
        "train_window_candidate": speedup >= 0.15,
        "device": str(device),
        "latent_size": config.data.latent_size,
        "batch_size": config.train.batch_size,
    }
    target_path = Path(output)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
