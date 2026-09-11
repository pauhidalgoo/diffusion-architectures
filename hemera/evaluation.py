from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .checkpoint import load_checkpoint
from .config import ExperimentConfig
from .data import (
    ShardedLatentDataset,
    build_dataset,
    collate_examples,
    deterministic_dataset_indices,
)
from .interfaces import TextCondition
from .models import build_model
from .objectives import build_objective
from .runtime import resolve_device, set_seed
from .runtime import sha256_file, source_tree_sha256
from .sampling import sample_latents
from .encoders import torch_dtype


def squared_distances(x: Tensor, y: Tensor) -> Tensor:
    return x.square().sum(1, keepdim=True) + y.square().sum(1)[None] - 2 * x @ y.T


def cmmd(real_features: Tensor, generated_features: Tensor, bandwidth: float = 10.0) -> float:
    """Official CMMD minimum-variance estimator, including its ×1000 scale."""
    x = real_features.float()
    y = generated_features.float()
    if not x.numel() or not y.numel():
        raise ValueError("CMMD requires non-empty real and generated features")
    kxx = torch.exp(-squared_distances(x, x) / (2 * bandwidth**2))
    kyy = torch.exp(-squared_distances(y, y) / (2 * bandwidth**2))
    kxy = torch.exp(-squared_distances(x, y) / (2 * bandwidth**2))
    return float((1000.0 * (kxx.mean() + kyy.mean() - 2 * kxy.mean())).cpu())


def frechet_distance(real_features: Tensor, generated_features: Tensor) -> float:
    """Small-sample diagnostic FID-like distance on supplied features."""
    x = real_features.double()
    y = generated_features.double()
    mean_x, mean_y = x.mean(0), y.mean(0)
    cov_x = torch.cov(x.T)
    cov_y = torch.cov(y.T)
    eigenvalues = torch.linalg.eigvals(cov_x @ cov_y).real.clamp_min(0)
    trace_sqrt = eigenvalues.sqrt().sum()
    value = (mean_x - mean_y).square().sum() + torch.trace(cov_x + cov_y) - 2 * trace_sqrt
    return float(value.clamp_min(0).cpu())


def bootstrap_mean_interval(
    values: Tensor, confidence: float = 0.95, samples: int = 2_000, seed: int = 0
) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    count = values.numel()
    indices = torch.randint(count, (samples, count), generator=generator)
    means = values.flatten()[indices].float().mean(1).sort().values
    tail = (1 - confidence) / 2
    return float(means[int(tail * samples)]), float(means[min(samples - 1, int((1 - tail) * samples))])


def latent_features(latents: Tensor, dimensions: int = 128) -> Tensor:
    flat = latents.float().flatten(1)
    if flat.shape[1] <= dimensions:
        return flat
    generator = torch.Generator(device=flat.device).manual_seed(17)
    projection = torch.randn(flat.shape[1], dimensions, device=flat.device, generator=generator)
    projection /= math.sqrt(flat.shape[1])
    return flat @ projection


@torch.no_grad()
def evaluate_checkpoint(config: ExperimentConfig, checkpoint: str | Path) -> dict[str, Any]:
    """Dependency-light latent evaluation used for local and cloud screening."""
    started = time.monotonic()
    device = resolve_device(config.train.device)
    set_seed(config.eval.seed)
    dtype = (
        torch.float32
        if device.type == "cpu"
        else torch_dtype(config.train.precision)
    )
    model = build_model(config.model).to(device=device, dtype=dtype).eval()
    payload = load_checkpoint(checkpoint, model, restore_rng=False)
    if payload.get("ema"):
        named_parameters = dict(model.named_parameters())
        for name, value in payload["ema"].items():
            if name in named_parameters:
                named_parameters[name].copy_(value.to(named_parameters[name]))
    objective = build_objective(config.objective)
    dataset = build_dataset(config, "validation" if config.train.dataset_mode == "shards" else "train")
    sample_count = min(config.eval.samples, len(dataset))
    real_batches: list[Tensor] = []
    generated_batches: list[Tensor] = []
    losses: list[float] = []
    shuffled_condition_losses: list[float] = []
    conditioning_prediction_deltas: list[float] = []
    for start in range(0, sample_count, config.eval.batch_size):
        count = min(config.eval.batch_size, sample_count - start)
        indices = deterministic_dataset_indices(
            dataset, count, start, config.eval.seed
        )
        if isinstance(dataset, ShardedLatentDataset):
            batch = dataset.collate_indices(indices, sample_posterior=False)
        else:
            batch = collate_examples(
                [dataset[index] for index in indices], sample_posterior=False
            )
        real = batch["latents"].to(device=device, dtype=dtype)
        condition: TextCondition = batch["condition"].to(device, dtype)
        target = objective.make_training_batch(real)
        prediction = model(target.noisy, target.time, condition)
        losses.append(float(objective.loss(prediction, target)))
        if count > 1:
            # Hold image latent, noise and timestep fixed while changing only
            # the paired caption. A conditioned model should normally incur a
            # higher denoising loss and change its prediction under this test.
            permutation = torch.roll(
                torch.arange(count, device=device), shifts=1
            )
            shuffled_prediction = model(
                target.noisy,
                target.time,
                condition.index_select(permutation),
            )
            shuffled_condition_losses.append(
                float(objective.loss(shuffled_prediction, target))
            )
            conditioning_prediction_deltas.append(
                float(
                    (prediction.float() - shuffled_prediction.float())
                    .square()
                    .mean()
                )
            )
        generator = torch.Generator(device=device).manual_seed(config.eval.seed + start)
        generated = sample_latents(
            model,
            objective,
            tuple(real.shape),
            condition,
            config.eval.num_inference_steps,
            config.eval.guidance_scale,
            generator,
        )
        real_batches.append(real.cpu())
        generated_batches.append(generated.cpu())
    real_features = latent_features(torch.cat(real_batches))
    generated_features = latent_features(torch.cat(generated_batches))
    result = {
        "validation_loss": sum(losses) / len(losses),
        "latent_cmmd": cmmd(real_features, generated_features, config.eval.cmmd_bandwidth),
        "latent_frechet": frechet_distance(real_features, generated_features),
        "generated_mean": float(torch.cat(generated_batches).mean()),
        "generated_std": float(torch.cat(generated_batches).std()),
        "samples": float(sample_count),
    }
    if shuffled_condition_losses:
        shuffled_loss = sum(shuffled_condition_losses) / len(
            shuffled_condition_losses
        )
        result["shuffled_condition_validation_loss"] = shuffled_loss
        result["conditioning_loss_delta"] = (
            shuffled_loss - result["validation_loss"]
        )
        result["conditioning_prediction_mse"] = sum(
            conditioning_prediction_deltas
        ) / len(conditioning_prediction_deltas)
    result["evaluation_wall_seconds"] = time.monotonic() - started
    result["evaluation_estimated_eur"] = (
        result["evaluation_wall_seconds"] * config.budget.hourly_eur / 3600.0
        if device.type == "cuda"
        else 0.0
    )
    result["evaluation_source_tree_sha256"] = source_tree_sha256()
    result["evaluated_checkpoint_sha256"] = sha256_file(checkpoint)
    output = Path(config.train.output_dir) / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def percentile_selection_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank CMMD down, HPS and CLIP up; use throughput as a deterministic tiebreaker."""
    if not rows:
        return []
    metrics = [("cmmd", False), ("hpsv2", True), ("clip_score", True)]
    for metric, higher in metrics:
        ordered = sorted(
            range(len(rows)),
            key=lambda index: rows[index][metric],
            reverse=higher,
        )
        denominator = max(len(rows) - 1, 1)
        position = 0
        while position < len(ordered):
            end = position + 1
            value = rows[ordered[position]][metric]
            while (
                end < len(ordered)
                and rows[ordered[end]][metric] == value
            ):
                end += 1
            average_rank = 0.5 * (position + end - 1)
            percentile = 1.0 - average_rank / denominator
            for index in ordered[position:end]:
                rows[index][f"{metric}_percentile"] = percentile
            position = end
    for row in rows:
        row["selection_score"] = sum(row[f"{name}_percentile"] for name, _ in metrics) / len(metrics)
    return sorted(
        rows,
        key=lambda row: (
            -row["selection_score"],
            -row.get("images_per_second", 0),
            row.get("peak_vram_bytes", float("inf")),
        ),
    )


def _load_paired_features(
    metrics_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    path = Path(metrics_path)
    metrics = json.loads(path.read_text(encoding="utf-8"))
    artifact = metrics.get("paired_features")
    if not artifact:
        raise ValueError(f"Missing paired feature artifact in {path}")
    feature_path = Path(artifact["path"])
    if not feature_path.exists():
        feature_path = path.parent / feature_path.name
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    if sha256_file(feature_path) != artifact["sha256"]:
        raise RuntimeError(f"Paired feature checksum mismatch: {feature_path}")
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError(
            "Paired benchmark comparison requires safetensors"
        ) from error
    return metrics, load_file(str(feature_path), device="cpu")


def compare_paired_benchmarks(
    candidate_metrics: str | Path,
    baseline_metrics: str | Path,
    output: str | Path,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired prompt bootstrap for the pre-registered automated composite."""
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 paired bootstrap samples")
    candidate, candidate_features = _load_paired_features(candidate_metrics)
    baseline, baseline_features = _load_paired_features(baseline_metrics)
    candidate_ids = candidate["paired_features"]["row_ids"]
    baseline_ids = baseline["paired_features"]["row_ids"]
    if candidate_ids != baseline_ids:
        raise ValueError("Candidate and baseline prompt/image row IDs differ")
    required = {
        "real_clip",
        "generated_clip",
        "clip_alignment",
        "hpsv2",
    }
    for label, features in (
        ("candidate", candidate_features),
        ("baseline", baseline_features),
    ):
        missing = required - features.keys()
        if missing:
            raise ValueError(f"{label} paired features are missing {sorted(missing)}")
    torch.testing.assert_close(
        candidate_features["real_clip"],
        baseline_features["real_clip"],
        rtol=0,
        atol=1e-6,
    )

    real = candidate_features["real_clip"].float()
    candidate_generated = candidate_features["generated_clip"].float()
    baseline_generated = baseline_features["generated_clip"].float()
    count = real.shape[0]
    if count < 2:
        raise ValueError("Paired comparison requires at least two prompts")
    bandwidth = float(
        candidate.get("cmmd_estimator", {}).get("bandwidth", 10.0)
    )
    if bandwidth != float(
        baseline.get("cmmd_estimator", {}).get("bandwidth", 10.0)
    ):
        raise ValueError("Candidate and baseline CMMD bandwidths differ")

    def kernel(left: Tensor, right: Tensor) -> Tensor:
        return torch.exp(
            -squared_distances(left, right).clamp_min(0)
            / (2 * bandwidth**2)
        )

    k_real = kernel(real, real)
    k_candidate = kernel(candidate_generated, candidate_generated)
    k_baseline = kernel(baseline_generated, baseline_generated)
    k_real_candidate = kernel(real, candidate_generated)
    k_real_baseline = kernel(real, baseline_generated)
    candidate_clip = candidate_features["clip_alignment"].float()
    baseline_clip = baseline_features["clip_alignment"].float()
    candidate_hps = candidate_features["hpsv2"].float()
    baseline_hps = baseline_features["hpsv2"].float()

    def values(indices: Tensor) -> dict[str, float]:
        grid = indices[:, None], indices[None, :]
        real_term = k_real[grid].mean()
        candidate_cmmd = 1000.0 * (
            real_term
            + k_candidate[grid].mean()
            - 2 * k_real_candidate[grid].mean()
        )
        baseline_cmmd = 1000.0 * (
            real_term
            + k_baseline[grid].mean()
            - 2 * k_real_baseline[grid].mean()
        )
        cmmd_improvement = baseline_cmmd - candidate_cmmd
        hps_improvement = (
            candidate_hps[indices] - baseline_hps[indices]
        ).mean()
        clip_improvement = (
            candidate_clip[indices] - baseline_clip[indices]
        ).mean()
        relative = torch.stack(
            [
                cmmd_improvement / baseline_cmmd.abs().clamp_min(1e-8),
                hps_improvement
                / baseline_hps[indices].mean().abs().clamp_min(1e-8),
                clip_improvement
                / baseline_clip[indices].mean().abs().clamp_min(1e-8),
            ]
        )
        return {
            "cmmd_improvement": float(cmmd_improvement),
            "hpsv2_improvement": float(hps_improvement),
            "clip_improvement": float(clip_improvement),
            "relative_composite_improvement": float(relative.mean()),
        }

    observed = values(torch.arange(count))
    # Represent every resample by its histogram over prompt indices. For any
    # kernel K, indexing an n-by-n grid with a bootstrap sample is equivalent
    # to counts @ K @ counts / n**2. Computing all resamples as matrix
    # products avoids thousands of large advanced-indexing operations.
    generator = torch.Generator().manual_seed(seed)
    sampled = torch.randint(
        count,
        (bootstrap_samples, count),
        generator=generator,
    )
    counts = torch.zeros(
        (bootstrap_samples, count),
        dtype=k_real.dtype,
    )
    counts.scatter_add_(
        1,
        sampled,
        torch.ones_like(sampled, dtype=counts.dtype),
    )
    normalization = float(count**2)

    def quadratic_means(matrix: Tensor) -> Tensor:
        return ((counts @ matrix) * counts).sum(dim=1) / normalization

    real_terms = quadratic_means(k_real)
    candidate_cmmds = 1000.0 * (
        real_terms
        + quadratic_means(k_candidate)
        - 2 * quadratic_means(k_real_candidate)
    )
    baseline_cmmds = 1000.0 * (
        real_terms
        + quadratic_means(k_baseline)
        - 2 * quadratic_means(k_real_baseline)
    )
    cmmd_improvements = baseline_cmmds - candidate_cmmds
    candidate_hps_means = counts @ candidate_hps / count
    baseline_hps_means = counts @ baseline_hps / count
    hps_improvements = candidate_hps_means - baseline_hps_means
    candidate_clip_means = counts @ candidate_clip / count
    baseline_clip_means = counts @ baseline_clip / count
    clip_improvements = candidate_clip_means - baseline_clip_means
    relative_composite_improvements = torch.stack(
        [
            cmmd_improvements / baseline_cmmds.abs().clamp_min(1e-8),
            hps_improvements / baseline_hps_means.abs().clamp_min(1e-8),
            clip_improvements / baseline_clip_means.abs().clamp_min(1e-8),
        ],
        dim=1,
    ).mean(dim=1)
    bootstrap = {
        "cmmd_improvement": cmmd_improvements,
        "hpsv2_improvement": hps_improvements,
        "clip_improvement": clip_improvements,
        "relative_composite_improvement": relative_composite_improvements,
    }

    intervals: dict[str, list[float]] = {}
    for metric, metric_values in bootstrap.items():
        distribution = metric_values.sort().values
        lower = int(0.025 * bootstrap_samples)
        upper = min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))
        intervals[metric] = [
            float(distribution[lower]),
            float(distribution[upper]),
        ]
    composite_interval = intervals["relative_composite_improvement"]
    result = {
        "candidate_metrics": str(candidate_metrics),
        "baseline_metrics": str(baseline_metrics),
        "paired_prompts": count,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "composite_definition": (
            "mean relative improvement of CMMD (lower), HPSv2 (higher), "
            "and CLIP alignment (higher), recomputed on paired prompt resamples"
        ),
        "observed": observed,
        "confidence_intervals_95": intervals,
        "candidate_beats_baseline": (
            observed["relative_composite_improvement"] > 0
        ),
        "positive_at_95_confidence": composite_interval[0] > 0,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return result


def compare_confirmation_benchmarks(
    candidate_metrics: list[str | Path],
    baseline_metrics: list[str | Path],
    output: str | Path,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Three-seed comparison with a prompt-cluster paired bootstrap."""
    if len(candidate_metrics) != len(baseline_metrics) or len(candidate_metrics) < 2:
        raise ValueError("Confirmation requires equal candidate/baseline seed lists")
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 paired bootstrap samples")
    candidate_runs = [
        _load_paired_features(path) for path in candidate_metrics
    ]
    baseline_runs = [
        _load_paired_features(path) for path in baseline_metrics
    ]
    reference_ids = candidate_runs[0][0]["paired_features"]["row_ids"]
    required = {"real_clip", "generated_clip", "clip_alignment", "hpsv2"}
    for index, ((candidate, candidate_features), (baseline, baseline_features)) in enumerate(
        zip(candidate_runs, baseline_runs, strict=True)
    ):
        if candidate["paired_features"]["row_ids"] != reference_ids:
            raise ValueError(f"Candidate seed {index} has different prompt IDs")
        if baseline["paired_features"]["row_ids"] != reference_ids:
            raise ValueError(f"Baseline seed {index} has different prompt IDs")
        for label, features in (
            ("candidate", candidate_features),
            ("baseline", baseline_features),
        ):
            missing = required - features.keys()
            if missing:
                raise ValueError(
                    f"{label} seed {index} is missing {sorted(missing)}"
                )
        torch.testing.assert_close(
            candidate_features["real_clip"],
            baseline_features["real_clip"],
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            candidate_runs[0][1]["real_clip"],
            candidate_features["real_clip"],
            rtol=0,
            atol=1e-6,
        )

    real = torch.cat([features["real_clip"] for _, features in candidate_runs]).float()
    candidate_generated = torch.cat(
        [features["generated_clip"] for _, features in candidate_runs]
    ).float()
    baseline_generated = torch.cat(
        [features["generated_clip"] for _, features in baseline_runs]
    ).float()
    candidate_clip = torch.cat(
        [features["clip_alignment"] for _, features in candidate_runs]
    ).float()
    baseline_clip = torch.cat(
        [features["clip_alignment"] for _, features in baseline_runs]
    ).float()
    candidate_hps = torch.cat(
        [features["hpsv2"] for _, features in candidate_runs]
    ).float()
    baseline_hps = torch.cat(
        [features["hpsv2"] for _, features in baseline_runs]
    ).float()
    prompts = len(reference_ids)
    seeds = len(candidate_runs)
    bandwidth = float(
        candidate_runs[0][0].get("cmmd_estimator", {}).get(
            "bandwidth", 10.0
        )
    )
    for metrics, _ in (*candidate_runs, *baseline_runs):
        if float(
            metrics.get("cmmd_estimator", {}).get("bandwidth", 10.0)
        ) != bandwidth:
            raise ValueError("Confirmation CMMD bandwidths differ")

    def kernel(left: Tensor, right: Tensor) -> Tensor:
        return torch.exp(
            -squared_distances(left, right).clamp_min(0)
            / (2 * bandwidth**2)
        )

    k_real = kernel(real, real)
    k_candidate = kernel(candidate_generated, candidate_generated)
    k_baseline = kernel(baseline_generated, baseline_generated)
    k_real_candidate = kernel(real, candidate_generated)
    k_real_baseline = kernel(real, baseline_generated)

    def values(indices: Tensor) -> dict[str, float]:
        grid = indices[:, None], indices[None, :]
        real_term = k_real[grid].mean()
        candidate_cmmd = 1000.0 * (
            real_term
            + k_candidate[grid].mean()
            - 2 * k_real_candidate[grid].mean()
        )
        baseline_cmmd = 1000.0 * (
            real_term
            + k_baseline[grid].mean()
            - 2 * k_real_baseline[grid].mean()
        )
        improvements = torch.stack(
            [
                (baseline_cmmd - candidate_cmmd)
                / baseline_cmmd.abs().clamp_min(1e-8),
                (candidate_hps[indices] - baseline_hps[indices]).mean()
                / baseline_hps[indices].mean().abs().clamp_min(1e-8),
                (candidate_clip[indices] - baseline_clip[indices]).mean()
                / baseline_clip[indices].mean().abs().clamp_min(1e-8),
            ]
        )
        return {
            "cmmd_improvement": float(baseline_cmmd - candidate_cmmd),
            "hpsv2_improvement": float(
                (candidate_hps[indices] - baseline_hps[indices]).mean()
            ),
            "clip_improvement": float(
                (candidate_clip[indices] - baseline_clip[indices]).mean()
            ),
            "relative_composite_improvement": float(improvements.mean()),
        }

    all_indices = torch.arange(seeds * prompts)
    observed = values(all_indices)
    per_seed = [
        values(torch.arange(seed_index * prompts, (seed_index + 1) * prompts))
        for seed_index in range(seeds)
    ]
    generator = torch.Generator().manual_seed(seed)
    distributions = {metric: [] for metric in observed}
    seed_offsets = torch.arange(seeds)[:, None] * prompts
    for _ in range(bootstrap_samples):
        sampled_prompts = torch.randint(
            prompts, (prompts,), generator=generator
        )
        # Every sampled prompt brings all seeds with it; this is the unit of
        # independence promised by the confirmation protocol.
        indices = (seed_offsets + sampled_prompts[None, :]).flatten()
        sample_values = values(indices)
        for metric, value in sample_values.items():
            distributions[metric].append(value)
    intervals: dict[str, list[float]] = {}
    for metric, values_distribution in distributions.items():
        ordered = torch.tensor(values_distribution).sort().values
        intervals[metric] = [
            float(ordered[int(0.025 * bootstrap_samples)]),
            float(
                ordered[
                    min(
                        bootstrap_samples - 1,
                        int(0.975 * bootstrap_samples),
                    )
                ]
            ),
        ]
    result = {
        "candidate_metrics": [str(path) for path in candidate_metrics],
        "baseline_metrics": [str(path) for path in baseline_metrics],
        "seeds": seeds,
        "paired_prompts": prompts,
        "bootstrap_unit": "prompt cluster containing all seeds",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "observed": observed,
        "per_seed_observed": per_seed,
        "confidence_intervals_95": intervals,
        "candidate_beats_baseline_in_every_seed": all(
            item["relative_composite_improvement"] > 0 for item in per_seed
        ),
        "aggregate_positive_at_95_confidence": intervals[
            "relative_composite_improvement"
        ][0]
        > 0,
    }
    result["confirmation_gate_passed"] = (
        result["candidate_beats_baseline_in_every_seed"]
        and result["aggregate_positive_at_95_confidence"]
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return result


def create_blinded_human_study(
    prompts: list[str],
    model_a_images: list[str],
    model_b_images: list[str],
    output: str | Path,
    seed: int = 0,
    ratings_per_prompt: int = 3,
    item_ids: list[str] | None = None,
    categories: list[str] | None = None,
) -> None:
    if not (len(prompts) == len(model_a_images) == len(model_b_images)):
        raise ValueError("Prompts and image lists must have equal lengths")
    item_ids = item_ids or [str(index) for index in range(len(prompts))]
    categories = categories or ["unknown"] * len(prompts)
    if not (len(prompts) == len(item_ids) == len(categories)):
        raise ValueError("Human-study metadata lists must have equal lengths")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Human-study item IDs must be unique")
    randomizer = random.Random(seed)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    key_rows: list[dict[str, str]] = []
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "assignment_id",
                "prompt",
                "category",
                "left_image",
                "right_image",
                "alignment",
                "quality",
                "overall",
                "rater_id",
            ],
        )
        writer.writeheader()
        for index, (item_id, category, prompt, a_image, b_image) in enumerate(
            zip(
                item_ids,
                categories,
                prompts,
                model_a_images,
                model_b_images,
                strict=True,
            )
        ):
            for rating_index in range(ratings_per_prompt):
                a_left = bool(randomizer.getrandbits(1))
                writer.writerow(
                    {
                        "item_id": item_id,
                        "assignment_id": f"{index:04d}-{rating_index}",
                        "prompt": prompt,
                        "category": category,
                        "left_image": a_image if a_left else b_image,
                        "right_image": b_image if a_left else a_image,
                        "alignment": "",
                        "quality": "",
                        "overall": "",
                        "rater_id": "",
                    }
                )
                key_rows.append(
                    {
                        "assignment_id": f"{index:04d}-{rating_index}",
                        "hidden_left_model": "a" if a_left else "b",
                    }
                )
    key_path = target.with_suffix(".key.csv")
    with key_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["assignment_id", "hidden_left_model"]
        )
        writer.writeheader()
        writer.writerows(key_rows)


def create_blinded_human_study_from_manifests(
    model_a_manifest: str | Path,
    model_b_manifest: str | Path,
    output: str | Path,
    seed: int = 0,
    ratings_per_prompt: int = 3,
    model_a_image_root: str | Path | None = None,
    model_b_image_root: str | Path | None = None,
) -> None:
    """Create a study only after exact prompt-ID and prompt-text alignment."""

    def image_index(root: str | Path | None) -> dict[str, Path] | None:
        if root is None:
            return None
        directory = Path(root).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        candidates: dict[str, list[Path]] = {}
        for image in directory.rglob("*"):
            if image.is_file() and image.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }:
                candidates.setdefault(image.name, []).append(image.resolve())
        duplicates = {
            name: paths for name, paths in candidates.items() if len(paths) > 1
        }
        if duplicates:
            example = next(iter(duplicates))
            raise ValueError(
                f"Image root contains duplicate basename {example!r}; "
                "use a narrower model-specific root"
            )
        return {name: paths[0] for name, paths in candidates.items()}

    def load(
        path: str | Path, root: str | Path | None
    ) -> dict[str, dict[str, Any]]:
        indexed_images = image_index(root)
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected: dict[str, dict[str, Any]] = {}
        for row in rows:
            identifier = str(row["id"])
            generation_index = int(row.get("generation_index", 0))
            if generation_index != 0:
                continue
            if identifier in selected:
                raise ValueError(f"Duplicate first-generation ID {identifier!r}")
            if indexed_images is not None:
                basename = Path(str(row["image"])).name
                try:
                    row["image"] = str(indexed_images[basename])
                except KeyError as error:
                    raise FileNotFoundError(
                        f"No image named {basename!r} under {root}"
                    ) from error
            selected[identifier] = row
        if not selected:
            raise ValueError(f"No first-generation images found in {path}")
        return selected

    model_a = load(model_a_manifest, model_a_image_root)
    model_b = load(model_b_manifest, model_b_image_root)
    if model_a.keys() != model_b.keys():
        raise ValueError("Human-study manifests contain different prompt IDs")
    ids = list(model_a)
    for identifier in ids:
        if model_a[identifier]["prompt"] != model_b[identifier]["prompt"]:
            raise ValueError(f"Prompt mismatch for human-study ID {identifier!r}")
    create_blinded_human_study(
        [str(model_a[identifier]["prompt"]) for identifier in ids],
        [str(model_a[identifier]["image"]) for identifier in ids],
        [str(model_b[identifier]["image"]) for identifier in ids],
        output,
        seed,
        ratings_per_prompt,
        ids,
        [str(model_a[identifier].get("category", "unknown")) for identifier in ids],
    )


def analyze_human_study(path: str | Path) -> dict[str, Any]:
    study_path = Path(path)
    rows = list(csv.DictReader(study_path.open("r", encoding="utf-8")))
    key_path = study_path.with_suffix(".key.csv")
    if not key_path.exists():
        raise FileNotFoundError(f"Missing blinded assignment key: {key_path}")
    keys = {
        row["assignment_id"]: row["hidden_left_model"]
        for row in csv.DictReader(key_path.open("r", encoding="utf-8"))
    }
    for row in rows:
        try:
            row["hidden_left_model"] = keys[row["assignment_id"]]
        except KeyError as error:
            raise ValueError(
                f"No blinding key for assignment {row.get('assignment_id')!r}"
            ) from error

    def normalize_vote(row: dict[str, str], field: str) -> str | None:
        vote = row[field].strip().lower()
        if vote not in {"left", "right", "tie"}:
            return None
        if vote == "tie":
            return "tie"
        left = row["hidden_left_model"]
        return left if vote == "left" else ("b" if left == "a" else "a")

    def cluster_interval(groups: dict[str, list[str]], seed: int = 0) -> tuple[float, float]:
        item_ids = sorted(groups)
        if not item_ids:
            return float("nan"), float("nan")
        generator = torch.Generator().manual_seed(seed)
        bootstrap: list[float] = []
        for _ in range(2_000):
            sampled = torch.randint(
                len(item_ids), (len(item_ids),), generator=generator
            ).tolist()
            votes = [
                vote
                for item_index in sampled
                for vote in groups[item_ids[item_index]]
                if vote != "tie"
            ]
            if votes:
                bootstrap.append(sum(vote == "a" for vote in votes) / len(votes))
        values = torch.tensor(bootstrap).sort().values
        if not values.numel():
            return float("nan"), float("nan")
        return float(values[int(0.025 * len(values))]), float(
            values[min(len(values) - 1, int(0.975 * len(values)))]
        )

    def fleiss_kappa(groups: dict[str, list[str]]) -> float:
        complete = [votes for votes in groups.values() if len(votes) >= 2]
        if not complete:
            return float("nan")
        categories = ("a", "tie", "b")
        agreements = []
        totals = Counter()
        for votes in complete:
            counts = Counter(votes)
            count = len(votes)
            agreements.append(
                sum(counts[category] * (counts[category] - 1) for category in categories)
                / (count * (count - 1))
            )
            totals.update(votes)
        total_votes = sum(totals.values())
        expected = sum((totals[category] / total_votes) ** 2 for category in categories)
        observed = sum(agreements) / len(agreements)
        return (observed - expected) / max(1 - expected, 1e-12)

    dimensions: dict[str, Any] = {}
    for field in ("alignment", "quality", "overall"):
        groups: dict[str, list[str]] = {}
        for row in rows:
            vote = normalize_vote(row, field)
            if vote is not None:
                groups.setdefault(row["item_id"], []).append(vote)
        votes = [vote for group in groups.values() for vote in group]
        decisive = [vote for vote in votes if vote != "tie"]
        wins = sum(vote == "a" for vote in decisive)
        losses = sum(vote == "b" for vote in decisive)
        dimensions[field] = {
            "model_a_wins": wins,
            "ties": sum(vote == "tie" for vote in votes),
            "model_a_losses": losses,
            "model_a_win_rate_excluding_ties": (
                wins / len(decisive) if decisive else float("nan")
            ),
            "cluster_bootstrap_confidence_interval_95": cluster_interval(groups),
            "fleiss_kappa": fleiss_kappa(groups),
            "rated_prompts": len(groups),
            "ratings": len(votes),
        }
    overall = dimensions["overall"]
    return {
        "dimensions": dimensions,
        # Backwards-compatible headline fields.
        "model_a_win_rate_excluding_ties": overall[
            "model_a_win_rate_excluding_ties"
        ],
        "confidence_interval_95": overall[
            "cluster_bootstrap_confidence_interval_95"
        ],
        "decisive_votes": overall["model_a_wins"] + overall["model_a_losses"],
        "ties": overall["ties"],
        "inter_rater_agreement_fleiss_kappa": overall["fleiss_kappa"],
    }
