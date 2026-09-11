"""Generate a restartable multi-seed Hemera-Nano showcase candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hemera.eval_generation import load_prompt_suite  # noqa: E402
from hemera.pipeline import HemeraPipeline  # noqa: E402


DEFAULT_PROMPTS = ROOT / "evaluation" / "prompts" / "showcase-v1.jsonl"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "publication-candidate"
DEFAULT_OUTPUT = ROOT / "artifacts" / "showcase"
DEFAULT_NEGATIVE = "blurry, malformed, distorted, text, watermark, logo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["candidate_id"])] = row
    return rows


def write_manifest(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in sorted(rows.values(), key=lambda value: value["candidate_id"])
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def jobs(
    prompts: list[dict[str, Any]],
    variants: int,
    base_seed: int,
    steps: int,
    guidance: float,
    negative_prompt: str,
    output: Path,
) -> list[dict[str, Any]]:
    values = []
    for prompt_index, row in enumerate(prompts):
        for variant in range(variants):
            candidate_id = f"{row['id']}-v{variant:02d}"
            seed = base_seed + prompt_index * 100 + variant
            values.append(
                {
                    "candidate_id": candidate_id,
                    "prompt_id": str(row["id"]),
                    "category": str(row.get("category", "unknown")),
                    "prompt": str(row["prompt"]),
                    "negative_prompt": str(
                        row.get("negative_prompt", negative_prompt)
                    ),
                    "variant": variant,
                    "seed": seed,
                    "nfe": steps,
                    "guidance": guidance,
                    "image": str((output / "candidates" / f"{candidate_id}.png").resolve()),
                }
            )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=640000)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 2 <= args.variants <= 16:
        raise ValueError("Use between 2 and 16 variants per prompt")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    prompts = load_prompt_suite(args.prompts)
    output = args.output.resolve()
    candidate_dir = output / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "candidates.jsonl"
    existing = load_existing(manifest_path)
    planned = jobs(
        prompts,
        args.variants,
        args.base_seed,
        args.steps,
        args.guidance,
        args.negative_prompt,
        output,
    )
    pending = [
        row
        for row in planned
        if row["candidate_id"] not in existing
        or not Path(existing[row["candidate_id"]]["image"]).is_file()
    ]
    print(
        json.dumps(
            {
                "prompts": len(prompts),
                "planned": len(planned),
                "complete": len(planned) - len(pending),
                "pending": len(pending),
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )
    if not pending:
        return

    pipeline = HemeraPipeline.from_pretrained(
        args.checkpoint.resolve(),
        device=args.device,
        dtype=args.dtype,
        load_vae=True,
    )
    print(f"Loaded pipeline on {pipeline.device} ({pipeline.dtype})", flush=True)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        generators = [
            torch.Generator(device=pipeline.device).manual_seed(int(row["seed"]))
            for row in batch
        ]
        images = pipeline(
            [str(row["prompt"]) for row in batch],
            negative_prompt=[str(row["negative_prompt"]) for row in batch],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generators,
        )
        for row, image in zip(batch, images, strict=True):
            image_path = Path(row["image"])
            image.save(image_path)
            row["image_sha256"] = sha256(image_path)
            existing[row["candidate_id"]] = row
        write_manifest(manifest_path, existing)
        done = min(start + len(batch), len(pending))
        print(f"Generated {done}/{len(pending)} pending candidates", flush=True)

    summary = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_model_sha256": sha256(
            args.checkpoint.resolve() / "model.safetensors"
        ),
        "prompt_file": str(args.prompts.resolve()),
        "prompt_file_sha256": sha256(args.prompts.resolve()),
        "prompts": len(prompts),
        "variants_per_prompt": args.variants,
        "candidates": len(planned),
        "base_seed": args.base_seed,
        "nfe": args.steps,
        "guidance": args.guidance,
        "negative_prompt": args.negative_prompt,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
    }
    temporary = (output / "generation-summary.json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output / "generation-summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
