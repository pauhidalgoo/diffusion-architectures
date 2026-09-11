from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor

from .evaluation import cmmd, frechet_distance, squared_distances
from .runtime import sha256_file


def clip_cmmd(
    real_features: Tensor,
    generated_features: Tensor,
    bandwidth: float = 10.0,
) -> float:
    """CMMD on L2-normalized CLIP embeddings, matching Scenic CLIP defaults."""
    real = torch.nn.functional.normalize(real_features.float(), dim=-1)
    generated = torch.nn.functional.normalize(
        generated_features.float(), dim=-1
    )
    return cmmd(real, generated, bandwidth)


def manifold_precision_recall(real: Tensor, generated: Tensor, neighbors: int = 3) -> tuple[float, float]:
    real = torch.nn.functional.normalize(real.float(), dim=-1)
    generated = torch.nn.functional.normalize(generated.float(), dim=-1)
    real_distances = squared_distances(real, real).clamp_min(0)
    generated_distances = squared_distances(generated, generated).clamp_min(0)
    real_radius = real_distances.topk(neighbors + 1, largest=False).values[:, -1]
    generated_radius = generated_distances.topk(neighbors + 1, largest=False).values[:, -1]
    cross = squared_distances(real, generated).clamp_min(0)
    precision = (cross <= real_radius[:, None]).any(dim=0).float().mean()
    recall = (cross <= generated_radius[None, :]).any(dim=1).float().mean()
    return float(precision), float(recall)


class CLIPBenchmarkEncoder:
    def __init__(self, model_id: str, device: torch.device):
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError("Image-space benchmarking requires `transformers`") from error
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode_images(self, paths: list[str], batch_size: int) -> Tensor:
        features: list[Tensor] = []
        for start in range(0, len(paths), batch_size):
            images = [Image.open(path).convert("RGB") for path in paths[start : start + batch_size]]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            features.append(self.model.get_image_features(**inputs).float().cpu())
        return torch.cat(features)

    @torch.no_grad()
    def encode_text(self, prompts: list[str], batch_size: int) -> Tensor:
        features: list[Tensor] = []
        for start in range(0, len(prompts), batch_size):
            inputs = self.processor(
                text=prompts[start : start + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            features.append(self.model.get_text_features(**inputs).float().cpu())
        return torch.cat(features)


class DINOv2BenchmarkEncoder:
    """Frozen DINOv2 image encoder for appearance-based memorization checks."""

    def __init__(self, model_id: str, device: torch.device):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise RuntimeError("DINOv2 benchmarking requires `transformers`") from error
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode_images(self, paths: list[str], batch_size: int) -> Tensor:
        features: list[Tensor] = []
        for start in range(0, len(paths), batch_size):
            with_images = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    with_images.append(image.convert("RGB").copy())
            inputs = self.processor(images=with_images, return_tensors="pt").to(self.device)
            output = self.model(**inputs)
            # The CLS token is the documented global representation and is
            # available across transformers' DINOv2 model variants.
            features.append(output.last_hidden_state[:, 0].float().cpu())
        return torch.cat(features)


def load_pair_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {"prompt", "real_image", "generated_image"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Pair manifest row {index} is missing {sorted(missing)}")
    return rows


def hpsv2_scores(
    image_paths: list[str],
    prompts: list[str],
    batch_size: int,
    device: torch.device,
    version: str = "v2.1",
) -> Tensor:
    """Score matched image/prompt pairs with one official HPSv2 model load."""
    if len(image_paths) != len(prompts):
        raise ValueError("HPSv2 image and prompt batches must have equal lengths")
    import huggingface_hub
    import hpsv2
    from hpsv2 import img_score
    from hpsv2.src.open_clip import get_tokenizer
    from hpsv2.utils import hps_version_map

    img_score.initialize_model()
    model = img_score.model_dict["model"]
    preprocess = img_score.model_dict["preprocess_val"]
    checkpoint_path = huggingface_hub.hf_hub_download(
        "xswu/HPSv2", hps_version_map[version]
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    tokenizer = get_tokenizer("ViT-H-14")
    values: list[Tensor] = []

    def preprocess_path(path: str) -> Tensor:
        with Image.open(path) as image:
            return preprocess(image.convert("RGB"))

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        images = torch.stack(
            [preprocess_path(path) for path in batch_paths]
        ).to(device=device, non_blocking=True)
        tokens = tokenizer(batch_prompts).to(device=device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model(images, tokens)
            scores = torch.diagonal(
                outputs["image_features"] @ outputs["text_features"].T
            )
        values.append(scores.float().cpu())
    # Avoid keeping the large HPS model alive through subsequent evaluations.
    img_score.model_dict.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(values)


def evaluate_image_pairs(
    manifest: str | Path,
    output: str | Path,
    model_id: str = "openai/clip-vit-large-patch14-336",
    batch_size: int = 32,
    device: str = "auto",
    siglip_model_id: str | None = None,
) -> dict[str, Any]:
    resolved = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    )
    rows = load_pair_manifest(manifest)
    prompts = [row["prompt"] for row in rows]
    real_paths = [row["real_image"] for row in rows]
    generated_paths = [row["generated_image"] for row in rows]
    encoder = CLIPBenchmarkEncoder(model_id, resolved)
    real = encoder.encode_images(real_paths, batch_size)
    generated = encoder.encode_images(generated_paths, batch_size)
    text = encoder.encode_text(prompts, batch_size)
    normalized_real = torch.nn.functional.normalize(real, dim=-1)
    normalized_generated = torch.nn.functional.normalize(generated, dim=-1)
    normalized_text = torch.nn.functional.normalize(text, dim=-1)
    alignment = (
        normalized_generated * normalized_text
    ).sum(-1)
    precision, recall = manifold_precision_recall(real, generated)
    result: dict[str, Any] = {
        "samples": len(rows),
        "cmmd": cmmd(normalized_real, normalized_generated),
        "semantic_frechet": frechet_distance(
            normalized_real, normalized_generated
        ),
        "clip_score_mean": float(alignment.mean()),
        "clip_score_std": float(alignment.std()),
        "precision": precision,
        "recall": recall,
        "feature_model": model_id,
        "feature_normalization": "l2",
        "cmmd_estimator": {
            "kernel": "rbf",
            "bandwidth": 10.0,
            "biased": True,
            "scale": 1000.0,
        },
    }
    per_source: dict[str, Any] = {}
    sources = sorted({str(row.get("source", "unknown")) for row in rows})
    for source in sources:
        indices = [
            index
            for index, row in enumerate(rows)
            if str(row.get("source", "unknown")) == source
        ]
        source_alignment = alignment[indices]
        source_result: dict[str, Any] = {
            "samples": len(indices),
            "clip_score_mean": float(source_alignment.mean()),
        }
        if len(indices) >= 2:
            source_result["cmmd"] = cmmd(
                normalized_real[indices], normalized_generated[indices]
            )
            source_result["semantic_frechet"] = frechet_distance(
                normalized_real[indices], normalized_generated[indices]
            )
        if len(indices) > 3:
            source_result["precision"], source_result["recall"] = (
                manifold_precision_recall(real[indices], generated[indices])
            )
        per_source[source] = source_result
    result["per_source"] = per_source
    siglip_alignment: Tensor | None = None
    if siglip_model_id:
        # Keep CMMD tied to the preregistered CLIP-L/14@336 representation,
        # while adding an independent SigLIP prompt-alignment diagnostic.
        secondary = CLIPBenchmarkEncoder(siglip_model_id, resolved)
        secondary_generated = torch.nn.functional.normalize(
            secondary.encode_images(generated_paths, batch_size), dim=-1
        )
        secondary_text = torch.nn.functional.normalize(
            secondary.encode_text(prompts, batch_size), dim=-1
        )
        siglip_alignment = (
            secondary_generated * secondary_text
        ).sum(-1)
        result.update(
            {
                "siglip_alignment_mean": float(siglip_alignment.mean()),
                "siglip_alignment_std": float(siglip_alignment.std()),
                "siglip_model": siglip_model_id,
            }
        )
        for source, source_result in per_source.items():
            indices = [
                index
                for index, row in enumerate(rows)
                if str(row.get("source", "unknown")) == source
            ]
            source_result["siglip_alignment_mean"] = float(
                siglip_alignment[indices].mean()
            )
        del secondary
        if resolved.type == "cuda":
            torch.cuda.empty_cache()
    try:
        import numpy as np
        from torchmetrics.image.fid import FrechetInceptionDistance

        fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(resolved)
        for start in range(0, len(rows), batch_size):
            real_images = torch.stack(
                [
                    torch.from_numpy(
                        np.asarray(Image.open(path).convert("RGB")).copy()
                    )
                    .permute(2, 0, 1)
                    for path in real_paths[start : start + batch_size]
                ]
            ).to(resolved)
            generated_images = torch.stack(
                [
                    torch.from_numpy(
                        np.asarray(Image.open(path).convert("RGB")).copy()
                    )
                    .permute(2, 0, 1)
                    for path in generated_paths[start : start + batch_size]
                ]
            ).to(resolved)
            fid_metric.update(real_images, real=True)
            fid_metric.update(generated_images, real=False)
        result["fid"] = float(fid_metric.compute())
    except (ImportError, ModuleNotFoundError):
        result["fid"] = None
        result["fid_note"] = "Install torchmetrics[image] and torch-fidelity to enable canonical FID."
    hps_scores: Tensor | None = None
    try:
        hps_scores = hpsv2_scores(
            generated_paths, prompts, batch_size, resolved, version="v2.1"
        )
        result["hpsv2_mean"] = float(hps_scores.mean())
        result["hpsv2_std"] = float(hps_scores.std())
        result["hpsv2_version"] = "v2.1"
        for source, source_result in per_source.items():
            indices = [
                index
                for index, row in enumerate(rows)
                if str(row.get("source", "unknown")) == source
            ]
            source_result["hpsv2_mean"] = float(hps_scores[indices].mean())
    except ImportError:
        result["hpsv2_mean"] = None
        result["hpsv2_note"] = "Install hpsv2 on the evaluation instance to enable this metric."
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError(
            "Image benchmarking requires safetensors for paired feature artifacts"
        ) from error
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    feature_path = target.with_suffix(".features.safetensors")
    feature_tensors = {
        "real_clip": normalized_real.contiguous(),
        "generated_clip": normalized_generated.contiguous(),
        "text_clip": normalized_text.contiguous(),
        "clip_alignment": alignment.float().contiguous(),
    }
    if hps_scores is not None:
        feature_tensors["hpsv2"] = hps_scores.contiguous()
    if siglip_alignment is not None:
        feature_tensors["siglip_alignment"] = (
            siglip_alignment.float().contiguous()
        )
    temporary_features = feature_path.with_suffix(
        feature_path.suffix + ".tmp"
    )
    save_file(feature_tensors, str(temporary_features))
    os.replace(temporary_features, feature_path)
    result["paired_features"] = {
        "path": str(feature_path),
        "sha256": sha256_file(feature_path),
        "row_ids": [
            str(row.get("id", index)) for index, row in enumerate(rows)
        ],
    }
    temporary_metrics = target.with_suffix(target.suffix + ".tmp")
    temporary_metrics.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    os.replace(temporary_metrics, target)
    return result


def nearest_neighbors(
    generated_features: Tensor,
    training_features: Tensor,
    generated_ids: list[str],
    training_ids: list[str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    generated = torch.nn.functional.normalize(generated_features.float(), dim=-1)
    training = torch.nn.functional.normalize(training_features.float(), dim=-1)
    similarities = generated @ training.T
    values, indices = similarities.topk(min(top_k, training.shape[0]), dim=1)
    return [
        {
            "generated_id": generated_ids[row],
            "neighbors": [
                {"training_id": training_ids[index], "cosine_similarity": float(score)}
                for score, index in zip(values[row], indices[row], strict=True)
            ],
        }
        for row in range(generated.shape[0])
    ]


def chunked_nearest_neighbors(
    generated_features: Tensor,
    training_features: Tensor,
    generated_ids: list[str],
    training_ids: list[str],
    top_k: int = 5,
    chunk_size: int = 16_384,
) -> list[dict[str, Any]]:
    """Exact cosine top-k without allocating the full generated×training matrix."""
    if len(generated_ids) != generated_features.shape[0]:
        raise ValueError("Generated feature count and IDs differ")
    if len(training_ids) != training_features.shape[0]:
        raise ValueError("Training feature count and IDs differ")
    if top_k < 1 or chunk_size < 1 or not training_ids:
        raise ValueError("top_k, chunk_size, and the training index must be non-empty")
    generated = torch.nn.functional.normalize(generated_features.float(), dim=-1)
    training = torch.nn.functional.normalize(training_features.float(), dim=-1)
    keep = min(top_k, training.shape[0])
    best_values = torch.full((generated.shape[0], keep), -torch.inf)
    best_indices = torch.full((generated.shape[0], keep), -1, dtype=torch.long)
    for start in range(0, training.shape[0], chunk_size):
        similarities = generated @ training[start : start + chunk_size].T
        local_keep = min(keep, similarities.shape[1])
        values, indices = similarities.topk(local_keep, dim=1)
        indices += start
        merged_values = torch.cat((best_values, values), dim=1)
        merged_indices = torch.cat((best_indices, indices), dim=1)
        best_values, positions = merged_values.topk(keep, dim=1)
        best_indices = merged_indices.gather(1, positions)
    return [
        {
            "generated_id": generated_ids[row],
            "neighbors": [
                {
                    "training_id": training_ids[index],
                    "cosine_similarity": float(score),
                }
                for score, index in zip(
                    best_values[row], best_indices[row], strict=True
                )
            ],
        }
        for row in range(generated.shape[0])
    ]


def build_image_feature_index(
    manifest: str | Path,
    output: str | Path,
    image_field: str = "image",
    encoder: str = "clip",
    model_id: str | None = None,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    """Encode a JSONL image manifest into a checksum-verified feature index."""
    rows = [
        json.loads(line)
        for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Image feature indexing requires a non-empty manifest")
    for index, row in enumerate(rows):
        if image_field not in row:
            raise ValueError(
                f"Manifest row {index} requires {image_field!r}"
            )
    used_fallback_ids = any("id" not in row for row in rows)
    ids = [
        str(row.get("id", f"row-{index:09d}"))
        for index, row in enumerate(rows)
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("Image feature index IDs must be unique")
    paths = [str(row[image_field]) for row in rows]
    resolved = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else ("cpu" if device == "auto" else device)
    )
    if encoder == "clip":
        model_id = model_id or "openai/clip-vit-large-patch14-336"
        feature_encoder = CLIPBenchmarkEncoder(model_id, resolved)
    elif encoder == "dinov2":
        model_id = model_id or "facebook/dinov2-small"
        feature_encoder = DINOv2BenchmarkEncoder(model_id, resolved)
    else:
        raise ValueError("encoder must be 'clip' or 'dinov2'")
    features = torch.nn.functional.normalize(
        feature_encoder.encode_images(paths, batch_size).float(), dim=-1
    ).contiguous()
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("Image indexing requires safetensors") from error
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    feature_path = target.with_suffix(".safetensors")
    temporary_features = feature_path.with_suffix(".safetensors.tmp")
    save_file({"features": features}, str(temporary_features))
    os.replace(temporary_features, feature_path)
    result = {
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": sha256_file(manifest),
        "image_field": image_field,
        "encoder": encoder,
        "model_id": model_id,
        "feature_normalization": "l2",
        "id_policy": (
            "manifest-id-with-row-index-fallback"
            if used_fallback_ids
            else "manifest-id"
        ),
        "samples": len(rows),
        "feature_file": str(feature_path.resolve()),
        "feature_sha256": sha256_file(feature_path),
        "ids": ids,
        "image_paths": paths,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return result


def _load_image_feature_index(path: str | Path) -> tuple[dict[str, Any], Tensor]:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    feature_path = Path(metadata["feature_file"])
    if not feature_path.exists():
        feature_path = Path(path).parent / feature_path.name
    if sha256_file(feature_path) != metadata["feature_sha256"]:
        raise RuntimeError(f"Image feature checksum mismatch: {feature_path}")
    from safetensors.torch import load_file

    return metadata, load_file(str(feature_path), device="cpu")["features"]


def audit_nearest_neighbors(
    generated_index: str | Path,
    training_index: str | Path,
    output: str | Path,
    top_k: int = 5,
    chunk_size: int = 16_384,
) -> dict[str, Any]:
    generated_meta, generated = _load_image_feature_index(generated_index)
    training_meta, training = _load_image_feature_index(training_index)
    if (
        generated_meta["encoder"] != training_meta["encoder"]
        or generated_meta["model_id"] != training_meta["model_id"]
    ):
        raise ValueError("Generated and training feature encoders differ")
    rows = chunked_nearest_neighbors(
        generated,
        training,
        generated_meta["ids"],
        training_meta["ids"],
        top_k,
        chunk_size,
    )
    maxima = torch.tensor(
        [row["neighbors"][0]["cosine_similarity"] for row in rows]
    )
    result = {
        "encoder": generated_meta["encoder"],
        "model_id": generated_meta["model_id"],
        "generated_samples": len(rows),
        "training_samples": len(training_meta["ids"]),
        "top_k": min(top_k, len(training_meta["ids"])),
        "maximum_similarity": float(maxima.max()),
        "mean_nearest_similarity": float(maxima.mean()),
        "quantiles": {
            str(quantile): float(torch.quantile(maxima, quantile))
            for quantile in (0.5, 0.9, 0.95, 0.99)
        },
        "neighbors": rows,
        "interpretation": (
            "Diagnostic only: high embedding similarity is not proof of memorization; "
            "the highest matches require blinded visual inspection."
        ),
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return result
