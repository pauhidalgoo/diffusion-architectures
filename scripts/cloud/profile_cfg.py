from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hemera.checkpoint import load_checkpoint
from hemera.config import load_config
from hemera.data import ShardedLatentDataset, collate_examples
from hemera.interfaces import TextCondition
from hemera.models import build_model
from hemera.objectives import GenerativeObjective, build_objective
from hemera.sampling import sample_latents


@torch.no_grad()
def sequential_cfg_sample(
    model,
    objective: GenerativeObjective,
    shape: tuple[int, ...],
    condition: TextCondition,
    steps: int,
    guidance: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Reference implementation with two denoiser calls per Euler step."""
    device = condition.hidden_states.device
    x = torch.randn(
        shape,
        device=device,
        dtype=condition.hidden_states.dtype,
        generator=generator,
    )
    unconditional = TextCondition.unconditional_like(condition)
    times = objective.sampling_times(steps, device)
    for index in range(steps):
        current, following = times[index], times[index + 1]
        batch_time = current.expand(shape[0])
        conditional = model(x, batch_time, condition)
        unconditioned = model(x, batch_time, unconditional)
        prediction = unconditioned + guidance * (conditional - unconditioned)
        x = x + (following - current) * objective.velocity(
            prediction, x, batch_time
        )
    return x


def _timed(
    function,
    *,
    repeats: int,
    seed_start: int,
    device: torch.device,
) -> tuple[list[float], torch.Tensor, int]:
    timings: list[float] = []
    final: torch.Tensor | None = None
    torch.cuda.reset_peak_memory_stats(device)
    for repeat in range(repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        final = function(seed_start + repeat)
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
    assert final is not None
    return timings, final, torch.cuda.max_memory_allocated(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile sequential versus concatenated classifier-free guidance."
    )
    parser.add_argument("--config", default="configs/gpu-preflight.yaml")
    parser.add_argument("--cache-dir", default="preflight/cache")
    parser.add_argument(
        "--checkpoint", default="preflight/gpu-overfit/checkpoint-last.pt"
    )
    parser.add_argument("--output", default="preflight/cfg-profile.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CFG profiling requires CUDA")

    device = torch.device("cuda")
    config = load_config(args.config)
    config.data.cache_dir = args.cache_dir
    model = build_model(config.model).to(device=device, dtype=torch.bfloat16).eval()
    payload = load_checkpoint(args.checkpoint, model, restore_rng=False)
    if payload.get("ema"):
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in payload["ema"].items():
                if name in parameters:
                    parameters[name].copy_(value.to(parameters[name]))
    objective = build_objective(config.objective)
    dataset = ShardedLatentDataset(args.cache_dir, "train")
    batch = collate_examples(
        [dataset[index % len(dataset)] for index in range(args.batch_size)],
        sample_posterior=False,
    )
    condition = batch["condition"].to(device, torch.bfloat16)
    shape = (
        args.batch_size,
        config.data.latent_channels,
        config.data.latent_size,
        config.data.latent_size,
    )

    def batched(seed: int) -> torch.Tensor:
        return sample_latents(
            model,
            objective,
            shape,
            condition,
            args.steps,
            args.guidance,
            torch.Generator(device=device).manual_seed(seed),
        )

    def sequential(seed: int) -> torch.Tensor:
        return sequential_cfg_sample(
            model,
            objective,
            shape,
            condition,
            args.steps,
            args.guidance,
            torch.Generator(device=device).manual_seed(seed),
        )

    for offset in range(args.warmup):
        batched(20260726 + offset)
        sequential(20260726 + offset)
    sequential_times, sequential_output, sequential_vram = _timed(
        sequential,
        repeats=args.repeats,
        seed_start=20260800,
        device=device,
    )
    batched_times, batched_output, batched_vram = _timed(
        batched,
        repeats=args.repeats,
        seed_start=20260800,
        device=device,
    )
    difference = sequential_output.float() - batched_output.float()
    max_difference = float(difference.abs().max())
    difference_rmse = float(difference.square().mean().sqrt())
    reference_rms = float(sequential_output.float().square().mean().sqrt())
    relative_rmse = difference_rmse / max(reference_rms, 1e-8)
    sequential_mean = sum(sequential_times) / len(sequential_times)
    batched_mean = sum(batched_times) / len(batched_times)
    report = {
        "device": torch.cuda.get_device_name(device),
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "guidance": args.guidance,
        "repeats": args.repeats,
        "sequential_seconds": sequential_times,
        "batched_seconds": batched_times,
        "sequential_mean_seconds": sequential_mean,
        "batched_mean_seconds": batched_mean,
        "speedup": sequential_mean / batched_mean,
        "sequential_peak_vram_bytes": sequential_vram,
        "batched_peak_vram_bytes": batched_vram,
        "max_absolute_output_difference": max_difference,
        "output_difference_rmse": difference_rmse,
        "output_difference_relative_rmse": relative_rmse,
        "bf16_numerical_equivalence_gate": bool(
            max_difference <= 0.05 and relative_rmse <= 0.01
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
