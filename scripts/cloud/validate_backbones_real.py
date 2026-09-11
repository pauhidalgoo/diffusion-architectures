from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hemera.config import config_from_dict
from hemera.data import ShardedLatentDataset, collate_examples
from hemera.interfaces import TextCondition
from hemera.models import build_model, count_parameters
from hemera.objectives import build_objective
from hemera.sweep import deep_update


def _condition_loss(
    model: torch.nn.Module,
    objective: Any,
    clean: torch.Tensor,
    condition: TextCondition,
    *,
    seed_start: int,
    seeds: int,
) -> dict[str, Any]:
    shuffled = TextCondition(
        hidden_states=condition.hidden_states.roll(1, 0),
        attention_mask=condition.attention_mask.roll(1, 0),
        pooled=None if condition.pooled is None else condition.pooled.roll(1, 0),
    )
    correct: list[float] = []
    wrong: list[float] = []
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for seed in range(seed_start, seed_start + seeds):
            generator = torch.Generator(device=clean.device).manual_seed(seed)
            target = objective.make_training_batch(clean, generator)
            correct.append(
                float(objective.loss(model(target.noisy, target.time, condition), target))
            )
            wrong.append(
                float(objective.loss(model(target.noisy, target.time, shuffled), target))
            )
    correct_mean = sum(correct) / len(correct)
    shuffled_mean = sum(wrong) / len(wrong)
    gaps = [bad - good for good, bad in zip(correct, wrong)]
    return {
        "correct_prompt_loss": correct_mean,
        "shuffled_prompt_loss": shuffled_mean,
        "correct_is_better": correct_mean < shuffled_mean,
        "all_seeds_correct_are_better": all(gap > 0 for gap in gaps),
        "relative_gap": (shuffled_mean - correct_mean) / max(correct_mean, 1e-8),
        "per_seed_loss_gap": gaps,
    }


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overfit every backbone on real cached Photonyx image-caption pairs."
    )
    parser.add_argument("--base", default="configs/experiment-base.yaml")
    parser.add_argument("--sweep", default="configs/sweeps/backbones.yaml")
    parser.add_argument("--cache-dir", default="preflight/cache")
    parser.add_argument("--output", default="preflight/backbone-real-validation.json")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument(
        "--backbone",
        action="append",
        help="Run only these sweep names; may be specified more than once.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This validation gate requires CUDA")

    torch.manual_seed(20260726)
    torch.cuda.manual_seed_all(20260726)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    base = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    sweep = yaml.safe_load(Path(args.sweep).read_text(encoding="utf-8"))
    selected = set(args.backbone or [])
    recipes = [
        run for run in sweep["runs"] if not selected or run["name"] in selected
    ]
    missing = selected - {run["name"] for run in recipes}
    if missing:
        raise ValueError(f"Unknown backbone sweep names: {sorted(missing)}")

    dataset = ShardedLatentDataset(args.cache_dir, "train")
    if len(dataset) < 2:
        raise ValueError("At least two real cached examples are required")
    evaluation_count = min(max(args.batch_size, 2), len(dataset))
    evaluation_batch = collate_examples(
        [dataset[index] for index in range(evaluation_count)],
        sample_posterior=False,
    )
    evaluation_clean = evaluation_batch["latents"].to(
        device=device, dtype=torch.bfloat16
    )
    evaluation_condition = evaluation_batch["condition"].to(
        device, torch.bfloat16
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "real_examples": len(dataset),
        "evaluation_examples": evaluation_count,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "precision": "bfloat16",
        "device": torch.cuda.get_device_name(device),
        "results": [],
    }
    output = Path(args.output)

    for recipe_index, recipe in enumerate(recipes):
        raw = copy.deepcopy(base)
        deep_update(raw, recipe.get("overrides", {}))
        raw["name"] = f"real-gate-{recipe['name']}"
        raw["data"]["cache_dir"] = args.cache_dir
        raw["train"]["compile"] = False
        raw["objective"]["name"] = "flow"
        config = config_from_dict(raw)

        torch.manual_seed(20260726 + recipe_index)
        torch.cuda.manual_seed_all(20260726 + recipe_index)
        model = build_model(config.model).to(device)
        objective = build_objective(config.objective)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            fused=True,
        )
        before = _condition_loss(
            model,
            objective,
            evaluation_clean,
            evaluation_condition,
            seed_start=20260726,
            seeds=args.seeds,
        )

        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        losses: list[float] = []
        model.train()
        for step in range(args.steps):
            indices = [
                (step * args.batch_size + offset) % len(dataset)
                for offset in range(args.batch_size)
            ]
            batch = collate_examples(
                [dataset[index] for index in indices], sample_posterior=False
            )
            clean = batch["latents"].to(device=device, dtype=torch.bfloat16)
            condition = batch["condition"].to(device, torch.bfloat16)
            generator = torch.Generator(device=device).manual_seed(
                20260726 + recipe_index * 100_000 + step
            )
            target = objective.make_training_batch(clean, generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(target.noisy, target.time, condition)
                loss = objective.loss(prediction, target)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                raise RuntimeError(
                    f"{recipe['name']} produced non-finite values at step {step}"
                )
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        after = _condition_loss(
            model,
            objective,
            evaluation_clean,
            evaluation_condition,
            seed_start=20260726,
            seeds=args.seeds,
        )
        result = {
            "name": recipe["name"],
            "backbone": config.model.backbone,
            "conditioning": config.model.conditioning,
            "attention": config.model.attention,
            "ffn": config.model.ffn,
            "parameters": count_parameters(model),
            "first_training_loss": losses[0],
            "final_training_loss": losses[-1],
            "minimum_training_loss": min(losses),
            "loss_reduction": losses[0] - losses[-1],
            "elapsed_seconds": elapsed,
            "images_per_second": args.steps * args.batch_size / elapsed,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
            "conditioning_before": before,
            "conditioning_after": after,
        }
        report["results"].append(result)
        _atomic_json(report, output)
        del optimizer, model
        torch.cuda.empty_cache()

    report["all_finite_and_trained"] = all(
        result["loss_reduction"] > 0 for result in report["results"]
    )
    report["all_conditioning_gates_passed"] = all(
        result["conditioning_after"]["correct_is_better"]
        and result["conditioning_after"]["all_seeds_correct_are_better"]
        for result in report["results"]
    )
    _atomic_json(report, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
