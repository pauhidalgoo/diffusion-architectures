from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hemera.checkpoint import load_checkpoint
from hemera.config import load_config
from hemera.data import ShardedLatentDataset, collate_examples
from hemera.models import build_model
from hemera.objectives import build_objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.cache_dir:
        config.data.cache_dir = args.cache_dir
    dataset = ShardedLatentDataset(config.data.cache_dir, "train")
    first = dataset[0]
    report = {
        "count": len(dataset),
        "keys": sorted(first),
        "latent_mean": list(first["latent_mean"].shape),
        "latent_dtype": str(first["latent_mean"].dtype),
        "latent_mean_abs": float(first["latent_mean"].abs().mean()),
        "latent_std": float(first["latent_mean"].std()),
        "text_hidden": list(first["text_hidden"].shape),
        "text_dtype": str(first["text_hidden"].dtype),
        "valid_tokens": int(first["text_mask"].sum()),
        "pooled": list(first["pooled"].shape),
        "repa_target": (
            list(first["repa_target"].shape) if "repa_target" in first else None
        ),
        "id": first["id"],
        "source": first["source"],
        "license": first["license"],
        "prompt": first["prompt"][:240],
    }
    if args.checkpoint:
        device = torch.device("cuda")
        model = build_model(config.model).to(device=device).eval()
        payload = load_checkpoint(args.checkpoint, model, restore_rng=False)
        if payload.get("ema"):
            parameters = dict(model.named_parameters())
            with torch.no_grad():
                for name, value in payload["ema"].items():
                    if name in parameters:
                        parameters[name].copy_(value.to(parameters[name]))
        count = min(8, len(dataset))
        batch = collate_examples(
            [dataset[index] for index in range(count)], sample_posterior=False
        )
        clean = batch["latents"].to(device=device, dtype=torch.bfloat16)
        condition = batch["condition"].to(device, clean.dtype)
        swapped = type(condition)(
            hidden_states=condition.hidden_states.roll(1, 0),
            attention_mask=condition.attention_mask.roll(1, 0),
            pooled=(
                None if condition.pooled is None else condition.pooled.roll(1, 0)
            ),
        )
        objective = build_objective(config.objective)
        correct_losses: list[float] = []
        wrong_losses: list[float] = []
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            for seed in range(20260726, 20260734):
                target = objective.make_training_batch(
                    clean, torch.Generator(device=device).manual_seed(seed)
                )
                correct_losses.append(
                    float(
                        objective.loss(
                            model(target.noisy, target.time, condition), target
                        )
                    )
                )
                wrong_losses.append(
                    float(
                        objective.loss(
                            model(target.noisy, target.time, swapped), target
                        )
                    )
                )
        correct = sum(correct_losses) / len(correct_losses)
        wrong = sum(wrong_losses) / len(wrong_losses)
        report["conditioning"] = {
            "noise_seeds": len(correct_losses),
            "correct_prompt_loss": correct,
            "shuffled_prompt_loss": wrong,
            "correct_is_better": correct < wrong,
            "relative_gap": (wrong - correct) / max(correct, 1e-8),
            "per_seed_loss_gap": [
                wrong_value - correct_value
                for correct_value, wrong_value in zip(correct_losses, wrong_losses)
            ],
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
