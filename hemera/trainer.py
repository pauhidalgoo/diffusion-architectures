from __future__ import annotations

import contextlib
import json
import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig, save_config
from .data import (
    build_dataset,
    collate_examples,
    deterministic_dataset_indices,
    ShardedLatentDataset,
    source_groups,
)
from .interfaces import TextCondition
from .models import build_model, count_active_parameters, count_parameters
from .objectives import GenerativeObjective, build_objective
from .runtime import (
    BudgetWatchdog,
    JsonlLogger,
    RunManifest,
    hardware_details,
    resolve_device,
    set_seed,
    sha256_file,
)


class EMA:
    def __init__(self, model: nn.Module, half_life_images: int, batch_size: int):
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.decay = math.exp(math.log(0.5) * batch_size / max(half_life_images, 1))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach().float(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        restored: dict[str, Tensor] = {}
        for key, value in state.items():
            device = self.shadow[key].device if key in self.shadow else value.device
            restored[key] = value.to(device=device, dtype=torch.float32).clone()
        self.shadow = restored

    @contextlib.contextmanager
    def apply(self, model: nn.Module):
        original: dict[str, Tensor] = {}
        try:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in self.shadow:
                        original[name] = parameter.detach().clone()
                        parameter.copy_(self.shadow[name].to(parameter))
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in original:
                        parameter.copy_(original[name])


def _move_condition(condition: TextCondition, device: torch.device, dtype: torch.dtype) -> TextCondition:
    return condition.to(device, dtype)


def _learning_rate(
    config: ExperimentConfig,
    step: int,
    budget_progress: float | None = None,
) -> float:
    if budget_progress is not None:
        warmup_fraction = config.train.warmup_ratio
        if budget_progress < warmup_fraction:
            multiplier = budget_progress / max(warmup_fraction, 1e-8)
        else:
            progress = (budget_progress - warmup_fraction) / max(
                1.0 - warmup_fraction, 1e-8
            )
            cosine = 0.5 * (
                1.0 + math.cos(math.pi * min(progress, 1.0))
            )
            multiplier = config.train.min_learning_rate_ratio + (
                1.0 - config.train.min_learning_rate_ratio
            ) * cosine
    else:
        total = max(config.train.steps, 1)
        warmup = max(1, int(total * config.train.warmup_ratio))
        if step < warmup:
            multiplier = (step + 1) / warmup
        else:
            progress = (step - warmup) / max(total - warmup, 1)
            cosine = 0.5 * (
                1.0 + math.cos(math.pi * min(progress, 1.0))
            )
            multiplier = config.train.min_learning_rate_ratio + (
                1.0 - config.train.min_learning_rate_ratio
            ) * cosine
    return config.train.learning_rate * multiplier


def _microdit_mask_ratio(
    config: ExperimentConfig,
    step: int,
    budget_progress: float | None,
) -> float:
    if config.model.efficiency != "microdit":
        return 0.0
    progress = (
        budget_progress
        if budget_progress is not None
        else step / max(config.train.steps, 1)
    )
    return (
        config.model.mask_ratio
        if progress < 1.0 - config.model.mask_finish_ratio
        else 0.0
    )


def _autocast(device: torch.device, precision: str):
    if precision == "float32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _model_dtype(config: ExperimentConfig, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.train.precision]


class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.started = time.monotonic()
        self.config = config
        # Cloud scripts export the provider's live all-in hourly price. Record
        # and enforce that value instead of a stale config estimate.
        if os.environ.get("HEMERA_HOURLY_EUR"):
            config.budget.hourly_eur = float(os.environ["HEMERA_HOURLY_EUR"])
        self.watchdog = BudgetWatchdog(
            config.budget.max_eur,
            config.budget.hourly_eur,
            config.budget.reserve_minutes,
            config.budget.already_spent_eur,
        )
        self.device = resolve_device(config.train.device)
        set_seed(config.train.seed)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.dataset = build_dataset(config, "train")
        self.groups = (
            source_groups(self.dataset)
            if config.data.source_temperature < 0.999
            and not isinstance(self.dataset, ShardedLatentDataset)
            else None
        )
        self.raw_model = build_model(config.model).to(self.device)
        self.model = self.raw_model
        if config.train.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.objective = build_objective(config.objective)
        optimizer_options: dict[str, Any] = {
            "lr": config.train.learning_rate,
            "betas": (config.train.beta1, config.train.beta2),
            "weight_decay": config.train.weight_decay,
        }
        if self.device.type == "cuda":
            optimizer_options["fused"] = True
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_options)
        self.ema = EMA(
            self.raw_model,
            config.train.ema_half_life_images,
            config.train.effective_batch_size,
        )
        self.use_device_data_cache = (
            self.device.type == "cuda"
            and isinstance(self.dataset, ShardedLatentDataset)
            and os.environ.get("HEMERA_GPU_DATA_CACHE", "1") != "0"
        )
        if self.use_device_data_cache:
            self.dataset.cache_tensors_on_device(
                self.device,
                _model_dtype(config, self.device),
                include_repa=config.model.efficiency == "repa",
            )
        self.output = Path(config.train.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        save_config(config, self.output / "config.yaml")
        self.logger = JsonlLogger(self.output / "metrics.jsonl")
        self.step = 0
        self.samples_seen = 0
        self.best_validation_loss = float("inf")
        self.saved_budget_milestones: set[int] = set()
        self.manifest = RunManifest(
            name=config.name,
            config=config.to_dict(),
            seed=config.train.seed,
            device=str(self.device),
            parameter_count=count_parameters(self.raw_model),
            active_parameter_count=count_active_parameters(self.raw_model),
            hardware=hardware_details(self.device),
            data_artifacts=self._data_artifacts(),
        )
        if config.train.resume:
            payload = load_checkpoint(config.train.resume, self.raw_model, self.optimizer)
            self.step = int(payload["step"])
            self.samples_seen = int(payload.get("samples_seen", 0))
            if payload.get("ema"):
                self.ema.load_state_dict(payload["ema"])
            self.best_validation_loss = float(payload.get("best_validation_loss", float("inf")))
            self.manifest.estimated_flops = payload.get("estimated_flops")

    def _data_artifacts(self) -> dict[str, Any]:
        artifacts: dict[str, Any] = {
            "dataset_id": self.config.data.dataset_id,
            "dataset_revision": self.config.data.dataset_revision,
            "dataset_mode": self.config.train.dataset_mode,
        }
        if self.config.train.dataset_mode != "shards":
            return artifacts
        index_path = Path(self.config.data.cache_dir) / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing dataset cache index: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        artifacts.update(
            {
                "cache_index": str(index_path.resolve()),
                "cache_index_sha256": sha256_file(index_path),
                "selection_sha256": index.get("representation", {}).get(
                    "selection_sha256"
                ),
                "representation": index.get("representation"),
                "shard_count": len(index.get("shards", [])),
                "examples": sum(
                    int(shard["count"]) for shard in index.get("shards", [])
                ),
            }
        )
        return artifacts

    def _cpu_batch(self, step: int) -> dict[str, Any]:
        indices = deterministic_dataset_indices(
            self.dataset,
            self.config.train.batch_size,
            step,
            self.config.train.seed,
            self.config.data.source_temperature,
            self.groups,
        )
        posterior_generator = torch.Generator(device="cpu").manual_seed(
            self.config.train.seed * 1_000_003 + step + 97_409
        )
        if isinstance(self.dataset, ShardedLatentDataset):
            return self.dataset.collate_indices(
                indices, generator=posterior_generator
            )
        return collate_examples(
            [self.dataset[index] for index in indices],
            generator=posterior_generator,
        )

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = dict(batch)
        dtype = _model_dtype(self.config, self.device)
        batch["latents"] = batch["latents"].to(self.device, dtype=dtype)
        batch["condition"] = _move_condition(batch["condition"], self.device, dtype)
        if "repa_target" in batch:
            batch["repa_target"] = batch["repa_target"].to(self.device, dtype=dtype)
        return batch

    def _batch(self, step: int) -> dict[str, Any]:
        if self.use_device_data_cache:
            assert isinstance(self.dataset, ShardedLatentDataset)
            indices = deterministic_dataset_indices(
                self.dataset,
                self.config.train.batch_size,
                step,
                self.config.train.seed,
                self.config.data.source_temperature,
                self.groups,
            )
            generator = torch.Generator(device=self.device).manual_seed(
                self.config.train.seed * 1_000_003 + step + 97_409
            )
            return self.dataset.collate_device_indices(indices, generator)
        return self._move_batch(self._cpu_batch(step))

    def _auxiliary_loss(self, batch: dict[str, Any]) -> tuple[Tensor | None, dict[str, float]]:
        auxiliary: Tensor | None = None
        metrics: dict[str, float] = {}
        unwrapped = self.raw_model
        if self.config.model.efficiency == "repa":
            prediction = unwrapped.take_repa_prediction()
            target = batch.get("repa_target")
            if prediction is None or target is None:
                raise RuntimeError("REPA training requires precomputed `repa_target` tensors")
            if target.shape[1] != prediction.shape[1]:
                side_in = int(math.sqrt(target.shape[1]))
                side_out = int(math.sqrt(prediction.shape[1]))
                if side_in * side_in != target.shape[1] or side_out * side_out != prediction.shape[1]:
                    raise ValueError("REPA token grids must be square")
                target = F.interpolate(
                    target.transpose(1, 2).reshape(target.shape[0], target.shape[2], side_in, side_in),
                    size=(side_out, side_out),
                    mode="bilinear",
                    align_corners=False,
                ).flatten(2).transpose(1, 2)
            auxiliary = 0.5 * (
                F.normalize(prediction.float(), dim=-1) - F.normalize(target.float(), dim=-1)
            ).square().mean()
            metrics["repa_loss"] = float(auxiliary.detach())
        moe_loss = getattr(unwrapped, "moe_balance_loss", lambda: None)()
        if moe_loss is not None:
            weighted = 0.01 * moe_loss
            auxiliary = weighted if auxiliary is None else auxiliary + weighted
            metrics["moe_balance_loss"] = float(moe_loss.detach())
        return auxiliary, metrics

    def train(self) -> Path:
        config = self.config
        accumulation = config.train.effective_batch_size // config.train.batch_size
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and config.train.precision == "float16",
        )
        self.model.train()
        last_checkpoint = self.output / "checkpoint-last.pt"
        started = self.started
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        pending_batch: Future[dict[str, Any]] | None = None
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hemera-batch"
        ) as batch_executor:
            while self.step < config.train.steps:
                if self.watchdog.should_stop():
                    self.logger.log(
                        {
                            "event": "budget_stop",
                            "step": self.step,
                            "spent_eur": self.watchdog.spent_eur,
                        }
                    )
                    break
                budget_progress = self.watchdog.progress
                active_mask_ratio = _microdit_mask_ratio(
                    config, self.step, budget_progress
                )
                if config.model.efficiency == "microdit":
                    setter = getattr(self.raw_model, "set_patch_mask_ratio", None)
                    if setter is not None:
                        setter(active_mask_ratio)
                step_started = time.monotonic()
                running_loss = 0.0
                auxiliary_metrics: dict[str, float] = {}
                for micro_step in range(accumulation):
                    logical_micro_step = self.step * accumulation + micro_step
                    if self.use_device_data_cache:
                        batch = self._batch(logical_micro_step)
                    else:
                        if pending_batch is None:
                            pending_batch = batch_executor.submit(
                                self._cpu_batch, logical_micro_step
                            )
                        cpu_batch = pending_batch.result()
                        pending_batch = batch_executor.submit(
                            self._cpu_batch, logical_micro_step + 1
                        )
                        batch = self._move_batch(cpu_batch)
                    drop = (
                        torch.rand(config.train.batch_size, device=self.device)
                        < config.train.cfg_dropout
                    )
                    condition = batch["condition"].dropped(drop)
                    target = self.objective.make_training_batch(batch["latents"])
                    with _autocast(self.device, config.train.precision):
                        if self.manifest.estimated_flops is None:
                            try:
                                from torch.utils.flop_counter import FlopCounterMode

                                with FlopCounterMode(display=False) as flop_counter:
                                    prediction = self.model(
                                        target.noisy, target.time, condition
                                    )
                                self.manifest.estimated_flops = (
                                    float(flop_counter.get_total_flops())
                                    / config.train.batch_size
                                )
                            except (ImportError, RuntimeError, AttributeError):
                                prediction = self.model(target.noisy, target.time, condition)
                        else:
                            prediction = self.model(target.noisy, target.time, condition)
                        loss = self.objective.loss(prediction, target)
                        auxiliary, values = self._auxiliary_loss(batch)
                        if auxiliary is not None:
                            loss = loss + auxiliary
                    scaler.scale(loss / accumulation).backward()
                    running_loss += float(loss.detach()) / accumulation
                    auxiliary_metrics.update(values)
                scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.train.grad_clip
                )
                lr = _learning_rate(config, self.step, budget_progress)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.raw_model)
                self.step += 1
                self.samples_seen += config.train.effective_batch_size
                elapsed = time.monotonic() - step_started
                if self.step == 1 or self.step % config.train.log_every == 0:
                    self.logger.log(
                        {
                            "step": self.step,
                            "loss": running_loss,
                            "lr": lr,
                            "grad_norm": float(grad_norm),
                            "samples_seen": self.samples_seen,
                            "step_seconds": elapsed,
                            "images_per_second": config.train.effective_batch_size
                            / max(elapsed, 1e-8),
                            "spent_eur": self.watchdog.spent_eur,
                            "budget_progress": budget_progress,
                            "patch_mask_ratio": active_mask_ratio,
                            **auxiliary_metrics,
                        }
                    )
                if (
                    self.step % config.train.save_every == 0
                    or self.step >= config.train.steps
                ):
                    self._save(last_checkpoint)
                self._save_budget_milestone()
                if (
                    config.train.sample_every > 0
                    and self.step % config.train.sample_every == 0
                ):
                    validation_loss = self._quick_validation()
                    self.logger.log(
                        {"step": self.step, "validation_loss": validation_loss}
                    )
                    if validation_loss < self.best_validation_loss:
                        self.best_validation_loss = validation_loss
                        self._save(self.output / "checkpoint-best.pt")

        self._save(last_checkpoint)
        elapsed_total = time.monotonic() - started
        self.manifest.finished_at = time.time()
        self.manifest.wall_seconds = elapsed_total
        self.manifest.gpu_seconds = elapsed_total if self.device.type == "cuda" else 0.0
        self.manifest.estimated_eur = (
            self.watchdog.spent_eur if self.device.type == "cuda" else 0.0
        )
        self.manifest.samples_seen = self.samples_seen
        self.manifest.mean_throughput = (
            self.samples_seen / elapsed_total if elapsed_total > 0 else 0.0
        )
        if self.device.type == "cuda":
            self.manifest.peak_vram_bytes = torch.cuda.max_memory_allocated(self.device)
        self.manifest.checkpoint_hashes = {
            checkpoint.name: sha256_file(checkpoint)
            for checkpoint in sorted(self.output.glob("checkpoint-*.pt"))
        }
        self.manifest.save(self.output / "manifest.json")
        return last_checkpoint

    def _save_budget_milestone(self) -> None:
        if (
            self.config.budget.max_eur <= 0
            or not self.config.budget.save_milestones
        ):
            return
        fraction = self.watchdog.spent_eur / self.config.budget.max_eur
        for percentage in (25, 50, 75, 95):
            if percentage not in self.saved_budget_milestones and fraction >= percentage / 100:
                self._save(self.output / f"checkpoint-{percentage}pct.pt")
                self.saved_budget_milestones.add(percentage)

    @torch.no_grad()
    def _quick_validation(self) -> float:
        split = "validation" if self.config.train.dataset_mode == "shards" else "train"
        dataset = build_dataset(self.config, split)
        count = min(self.config.train.batch_size, len(dataset))
        indices = deterministic_dataset_indices(
            dataset, count, 0, self.config.train.seed + 91
        )
        if isinstance(dataset, ShardedLatentDataset):
            batch = dataset.collate_indices(indices, sample_posterior=False)
        else:
            batch = collate_examples(
                [dataset[index] for index in indices], sample_posterior=False
            )
        dtype = _model_dtype(self.config, self.device)
        clean = batch["latents"].to(self.device, dtype=dtype)
        condition = batch["condition"].to(self.device, dtype)
        generator = torch.Generator(device=self.device).manual_seed(self.config.train.seed + 177)
        target = self.objective.make_training_batch(clean, generator)
        was_training = self.model.training
        self.model.eval()
        with self.ema.apply(self.raw_model), _autocast(self.device, self.config.train.precision):
            prediction = self.model(target.noisy, target.time, condition)
            loss = self.objective.loss(prediction, target)
        self.model.train(was_training)
        return float(loss)

    def _save(self, path: Path) -> None:
        save_checkpoint(
            path,
            self.raw_model,
            self.optimizer,
            self.step,
            self.samples_seen,
            self.ema.state_dict(),
            self.config.to_dict(),
            {
                "best_validation_loss": self.best_validation_loss,
                "estimated_flops": self.manifest.estimated_flops,
            },
        )


def train_from_config(config: ExperimentConfig) -> Path:
    return Trainer(config).train()
