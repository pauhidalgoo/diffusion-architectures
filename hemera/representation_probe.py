from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import ExperimentConfig
from .data import (
    ShardedLatentDataset,
    _load_raw_image,
    _raw_cache_root,
    _raw_image_path,
    image_to_tensor,
)
from .encoders import VAEAdapter, torch_dtype
from .runtime import resolve_device, sha256_file, source_tree_sha256


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stratified_probe_indices(
    source_groups: dict[str, list[int]], count: int
) -> list[int]:
    """Select a stable proportional sample while retaining every source."""
    sources = sorted(source_groups)
    total = sum(len(source_groups[source]) for source in sources)
    if count <= 0 or count > total:
        raise ValueError("Probe sample count must be within the available rows")
    allocations = {source: 0 for source in sources}
    remaining = count
    if count >= len(sources):
        for source in sources:
            allocations[source] = 1
        remaining -= len(sources)
    capacity = {
        source: len(source_groups[source]) - allocations[source]
        for source in sources
    }
    total_capacity = sum(capacity.values())
    exact = {
        source: remaining * capacity[source] / max(total_capacity, 1)
        for source in sources
    }
    for source in sources:
        addition = min(capacity[source], int(exact[source]))
        allocations[source] += addition
        remaining -= addition
    order = sorted(
        sources,
        key=lambda source: (-(exact[source] - int(exact[source])), source),
    )
    while remaining:
        progressed = False
        for source in order:
            if allocations[source] < len(source_groups[source]):
                allocations[source] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("Could not allocate the requested probe rows")
    selected: list[int] = []
    for source in sources:
        candidates = sorted(
            source_groups[source],
            key=lambda index: hashlib.sha256(
                f"hemera-representation-probe-v1:{source}:{index}".encode("utf-8")
            ).digest(),
        )
        selected.extend(candidates[: allocations[source]])
    # Shard-local order avoids repeatedly reloading large tensor shards.
    return sorted(selected)


def _retrieval_metrics(text: Tensor, image: Tensor, ids: list[str]) -> dict[str, float]:
    order = sorted(
        range(len(ids)),
        key=lambda index: hashlib.sha256(
            f"hemera-retrieval-v1:{ids[index]}".encode("utf-8")
        ).digest(),
    )
    train_count = max(2, int(0.7 * len(order)))
    train_indices = order[:train_count]
    test_indices = order[train_count:]
    if len(test_indices) < 2:
        raise ValueError("Text-image retrieval probe requires at least seven examples")
    x_train = text[train_indices].double()
    x_test = text[test_indices].double()
    y_train = F.normalize(image[train_indices].double(), dim=-1)
    y_test = F.normalize(image[test_indices].double(), dim=-1)
    mean = x_train.mean(0, keepdim=True)
    scale = x_train.std(0, keepdim=True).clamp_min(1e-5)
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    x_train = torch.cat([x_train, torch.ones(len(x_train), 1)], dim=1)
    x_test = torch.cat([x_test, torch.ones(len(x_test), 1)], dim=1)
    ridge = 1e-2
    kernel = x_train @ x_train.T
    alpha = torch.linalg.solve(
        kernel + ridge * torch.eye(len(kernel), dtype=kernel.dtype),
        y_train,
    )
    predicted = F.normalize(x_test @ x_train.T @ alpha, dim=-1)
    similarity = predicted @ y_test.T
    ranking = similarity.argsort(dim=1, descending=True)
    targets = torch.arange(len(test_indices))[:, None]
    ranks = (ranking == targets).nonzero()[:, 1] + 1
    return {
        "train_examples": float(len(train_indices)),
        "test_examples": float(len(test_indices)),
        "recall_at_1": float((ranks <= 1).float().mean()),
        "recall_at_5": float((ranks <= 5).float().mean()),
        "mean_reciprocal_rank": float((1.0 / ranks.float()).mean()),
    }


@torch.inference_mode()
def probe_representation(
    config: ExperimentConfig,
    output: str | Path,
    samples: int = 256,
    batch_size: int = 16,
    device: str = "auto",
    evaluator_id: str = "openai/clip-vit-large-patch14-336",
) -> dict[str, Any]:
    resolved = resolve_device(device)
    dtype = (
        torch.float32
        if resolved.type == "cpu"
        else torch_dtype(config.representation.precompute_dtype)
    )
    dataset = ShardedLatentDataset(config.data.cache_dir, "validation")
    count = min(samples, len(dataset))
    if count < 7:
        raise ValueError("Representation probe requires at least seven validation rows")
    selected_indices = _stratified_probe_indices(dataset.grouped_indices(), count)
    selected = [dataset[index] for index in selected_indices]
    raw_root = _raw_cache_root(config)
    missing = [
        int(item["row_index"])
        for item in selected
        if not _raw_image_path(raw_root, int(item["row_index"])).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} raw validation images are missing; precompute the selected manifest first"
        )

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from transformers import AutoModel, AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "Representation probes require torchmetrics[image] and transformers"
        ) from error

    vae = VAEAdapter(config.representation, resolved, dtype)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=False, reduction="none"
    ).to(resolved)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(resolved)
    processor = AutoProcessor.from_pretrained(evaluator_id)
    evaluator = AutoModel.from_pretrained(evaluator_id).to(resolved).eval()
    evaluator.requires_grad_(False)

    squared_errors: list[Tensor] = []
    lpips_values: list[Tensor] = []
    image_features: list[Tensor] = []
    text_features: list[Tensor] = []
    ids: list[str] = []
    decode_seconds = 0.0
    for start in range(0, count, batch_size):
        items = selected[start : start + batch_size]
        raw_pil = [
            _load_raw_image(_raw_image_path(raw_root, int(item["row_index"])))
            for item in items
        ]
        raw = torch.stack(
            [image_to_tensor(image, config.data.image_size) for image in raw_pil]
        ).to(resolved)
        latents = torch.stack([item["latent_mean"] for item in items]).to(
            device=resolved, dtype=dtype
        )
        _sync(resolved)
        started = time.perf_counter()
        reconstructed = vae.decode(latents).clamp(-1, 1)
        _sync(resolved)
        decode_seconds += time.perf_counter() - started
        squared_errors.append((reconstructed.float() - raw.float()).square().flatten(1).mean(1).cpu())
        lpips_values.append(lpips(reconstructed.float(), raw.float()).reshape(-1).cpu())
        real_uint8 = raw.add(1).mul(127.5).round().clamp(0, 255).byte()
        recon_uint8 = reconstructed.add(1).mul(127.5).round().clamp(0, 255).byte()
        fid.update(real_uint8, real=True)
        fid.update(recon_uint8, real=False)
        inputs = processor(images=raw_pil, return_tensors="pt").to(resolved)
        image_features.append(evaluator.get_image_features(**inputs).float().cpu())
        text_features.append(torch.stack([item["pooled"] for item in items]).float())
        ids.extend(str(item["id"]) for item in items)

    mse = torch.cat(squared_errors)
    # Input tensors use [-1, 1], so the peak-to-peak range is two.
    psnr = 10.0 * torch.log10(4.0 / mse.clamp_min(1e-12))
    cache_index = json.loads(
        (Path(config.data.cache_dir) / "index.json").read_text(encoding="utf-8")
    )
    token_counts = torch.tensor(
        [int(item["text_mask"].sum()) for item in selected],
        dtype=torch.float32,
    )
    prompt_lengths = torch.tensor(
        [len(str(item.get("prompt", ""))) for item in selected],
        dtype=torch.float32,
    )
    latent_means = torch.stack(
        [item["latent_mean"].float() for item in selected]
    )
    latent_logvars = torch.stack(
        [item["latent_logvar"].float() for item in selected]
    )
    per_channel_std = latent_means.permute(1, 0, 2, 3).flatten(1).std(1)
    index_path = Path(config.data.cache_dir) / "index.json"
    result: dict[str, Any] = {
        "samples": count,
        "source_counts": {
            source: sum(
                str(item.get("source", "unknown")) == source for item in selected
            )
            for source in sorted(
                {str(item.get("source", "unknown")) for item in selected}
            )
        },
        "vae": config.representation.vae_id,
        "text_encoder": config.representation.text_encoder_id,
        "reconstruction": {
            "psnr_mean_db": float(psnr.mean()),
            "psnr_std_db": float(psnr.std()),
            "lpips_mean": float(torch.cat(lpips_values).mean()),
            "fid": float(fid.compute()),
        },
        "retrieval": _retrieval_metrics(
            torch.cat(text_features), torch.cat(image_features), ids
        ),
        "text_diagnostics": {
            "mean_non_padding_tokens": float(token_counts.mean()),
            "fraction_at_token_capacity": float(
                (token_counts >= config.data.text_length).float().mean()
            ),
            "mean_prompt_characters": float(prompt_lengths.mean()),
            "token_capacity": config.data.text_length,
            "note": (
                "Capacity saturation is an upper-bound diagnostic for "
                "truncation; exactly full prompts may not be truncated."
            ),
        },
        "latent_diagnostics": {
            "mean": float(latent_means.mean()),
            "std": float(latent_means.std()),
            "absolute_mean": float(latent_means.abs().mean()),
            "channel_std_min": float(per_channel_std.min()),
            "channel_std_max": float(per_channel_std.max()),
            "posterior_std_mean": float(
                torch.exp(0.5 * latent_logvars).mean()
            ),
        },
        "performance": {
            "decode_images_per_second": count / max(decode_seconds, 1e-8),
            **cache_index.get("statistics", {}),
        },
        "evaluator": evaluator_id,
        "cache_index_sha256": sha256_file(index_path),
        "probe_source_tree_sha256": source_tree_sha256(),
        "device": str(resolved),
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(resolved)
            if resolved.type == "cuda"
            else 0
        ),
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
