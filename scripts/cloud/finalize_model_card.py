"""Render the final model card from authoritative run artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hemera.config import load_config
from hemera.runtime import sha256_file


def read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return (
        json.loads(target.read_text(encoding="utf-8"))
        if target.exists()
        else {}
    )


def value(payload: dict[str, Any], key: str, digits: int = 5) -> str:
    item = payload.get(key)
    if item is None:
        return "Not available"
    if isinstance(item, float):
        return f"{item:.{digits}f}"
    return str(item)


def suite_images(path: str | Path) -> int:
    return int(read_json(path).get("images", 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/final-hemera-nano.yaml"
    )
    parser.add_argument(
        "--run", default="runs/final/hemera-nano"
    )
    parser.add_argument(
        "--test-metrics", default="runs/final/test/metrics.json"
    )
    parser.add_argument(
        "--locked", default="runs/final/locked-inference.json"
    )
    parser.add_argument("--output", default="MODEL_CARD.md")
    args = parser.parse_args()

    config = load_config(args.config)
    run_root = Path(args.run)
    manifest = read_json(run_root / "manifest.json")
    metrics = read_json(args.test_metrics)
    locked = read_json(args.locked).get("winner", {})
    clip_neighbors = read_json(
        "runs/final/memorization/neighbors-clip.json"
    )
    dino_neighbors = read_json(
        "runs/final/memorization/neighbors-dinov2.json"
    )
    checkpoint = Path(
        locked.get("checkpoint", run_root / "checkpoint-best.pt")
    )
    checkpoint_hash = (
        sha256_file(checkpoint)
        if checkpoint.exists()
        else str(locked.get("checkpoint_sha256", "Not available"))
    )

    phase_start = Path(".hemera-final-start")
    phase_hours = (
        max(0.0, time.time() - int(phase_start.read_text().strip())) / 3600
        if phase_start.exists()
        else float(manifest.get("wall_seconds", 0.0)) / 3600
    )
    hourly_eur = float(
        os.environ.get("HEMERA_HOURLY_EUR", config.budget.hourly_eur)
    )
    estimated_phase_cost = phase_hours * hourly_eur

    geneval_summary = Path("runs/final/geneval/summary.txt")
    geneval_score = "Not scored"
    if geneval_summary.exists():
        match = re.search(
            r"Overall score \(avg\. over tasks\):\s*([0-9.]+)",
            geneval_summary.read_text(encoding="utf-8"),
        )
        if match:
            geneval_score = match.group(1)
    geneval_images = suite_images(
        "runs/final/geneval/generations/generation-summary.json"
    )
    human_images = suite_images(
        "runs/final/human-study/generation-summary.json"
    )
    safety_images = suite_images(
        "runs/final/safety-bias/generation-summary.json"
    )

    card = f"""---
library_name: hemera
pipeline_tag: text-to-image
license: apache-2.0
---

# Hemera-Nano

Hemera-Nano is a {manifest.get("parameter_count", 29_756_564):,}-parameter
256×256 text-to-image denoiser trained from scratch under a micro-budget. It
uses frozen pretrained representation models; the denoiser weights are the
only trained component.

## Model details

- Backbone: standard flat DiT with global self-attention and adaLN-Zero
- Conditioning: pooled frozen CLIP text representation
- Feed-forward network: dense SwiGLU
- Objective: rectified-flow velocity prediction
- Trainable parameters: {manifest.get("parameter_count", "Not available")}
- Frozen VAE: `{config.representation.vae_id}`
- Frozen text encoder: `{config.representation.text_encoder_id}`
- Native resolution: {config.data.image_size}×{config.data.image_size}
- Training dataset: Photonyx, pinned revision
  `{config.data.dataset_revision}`
- Audited examples: {manifest.get("data_artifacts", {}).get("examples", "Not available")}
- Final checkpoint SHA-256: `{checkpoint_hash}`
- Locked inference: CFG {locked.get("guidance", "Not available")},
  NFE {locked.get("nfe", "Not available")}
- Estimated complete final-instance cost at card generation:
  €{estimated_phase_cost:.2f} ({phase_hours:.2f} hours at
  €{hourly_eur:.4f}/hour)

## Intended use

Research and education around micro-budget text-to-image training. The model is
not intended for safety-critical, identity-sensitive, medical, legal, or
factual visual tasks.

## Locked test evaluation

Inference settings were selected on the validation split before the test split
was generated.

| Metric | Value |
|---|---:|
| Photonyx test CMMD | {value(metrics, "cmmd")} |
| Canonical FID | {value(metrics, "fid")} |
| Precision | {value(metrics, "precision")} |
| Recall | {value(metrics, "recall")} |
| HPSv2 | {value(metrics, "hpsv2_mean")} |
| CLIP alignment | {value(metrics, "clip_score_mean")} |
| SigLIP alignment | {value(metrics, "siglip_alignment_mean")} |
| GenEval | {geneval_score} |

Memorization diagnostics compare the locked test generations against a
deterministic, source-stratified subset of 50,000 training images. Similarity
is diagnostic only and is not evidence by itself that an image was memorized.

| Neighbor encoder | Training references | Mean nearest cosine | 99th percentile | Maximum |
|---|---:|---:|---:|---:|
| CLIP | {value(clip_neighbors, "training_samples", 0)} | {value(clip_neighbors, "mean_nearest_similarity")} | {value(clip_neighbors.get("quantiles", {}), "0.99")} | {value(clip_neighbors, "maximum_similarity")} |
| DINOv2 | {value(dino_neighbors, "training_samples", 0)} | {value(dino_neighbors, "mean_nearest_similarity")} | {value(dino_neighbors.get("quantiles", {}), "0.99")} | {value(dino_neighbors, "maximum_similarity")} |

Additional generated protocols:

- Human-study candidate images: {human_images}; blinded matched-baseline
  ratings remain pending until human raters complete the protocol.
- Safety/bias audit images: {safety_images}; this suite documents behavior and
  does not establish safety.
- Official GenEval-format images: {geneval_images}; detector scoring is
  reported only when the isolated official evaluator completes successfully.

## Limitations

Photonyx mixes real, artwork, and synthetic sources and may contain caption,
selection, aesthetic, demographic, and cultural biases. A small 256×256 model
is expected to struggle with text rendering, hands, counting, spatial
relations, rare concepts, faces, and photorealistic detail. Automated metrics
are imperfect and the Photonyx test distribution is not an independent
real-world benchmark.

The frozen CLIP encoder has a 77-token limit and pooled conditioning discards
token-level spatial detail. Outputs may reproduce stereotypes or unsafe visual
associations present in the data and pretrained representation models.

## Licensing

Hemera's original code and released denoiser weights are Apache-2.0. Photonyx
is CC-BY as declared by its owner and is attributed in the release. CLIP and
DC-AE weights are not redistributed; they are downloaded separately from
their upstream repositories and remain under their respective terms.

## Reproducibility

The release bundle includes the exact YAML configuration, immutable dataset and
prompt manifests, run manifest, dependency lock, checkpoint hashes, telemetry,
validation sweep, locked inference record, generated test pairs, metric
features, milestone canaries, and cost ledger.
"""
    target = Path(args.output)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(card, encoding="utf-8")
    os.replace(temporary, target)
    print(target)


if __name__ == "__main__":
    main()
