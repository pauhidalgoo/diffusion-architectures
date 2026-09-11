from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .data import (
    _load_photonyx,
    _decode_audit_image,
    _disable_streaming_image_decode,
    _load_raw_image,
    _materialize_selected_parallel,
    _raw_cache_root,
    _raw_image_path,
    _save_raw_image,
    select_manifest_subset,
    validate_streamed_record,
)
from .pipeline import HemeraPipeline
from .runtime import sha256_file


def select_manifest_records(
    manifest_path: str | Path,
    split: str,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    parsed = (
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line
    )
    records = [record for record in parsed if record.get("split") == split]
    if not records:
        raise ValueError(f"No {split!r} records found in {manifest_path}")
    return select_manifest_subset(
        records,
        min(samples, len(records)),
        salt=f"hemera-evaluation-v1:{seed}",
    )


def materialize_audit_images(
    config: ExperimentConfig,
    audit_manifest: str | Path,
    output_dir: str | Path,
    split: str,
    samples: int,
    seed: int,
) -> Path:
    """Materialize a fixed, stratified audit subset for image-space diagnostics."""
    selected = select_manifest_records(audit_manifest, split, samples, seed)
    output = Path(output_dir)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def output_path(record: dict[str, Any]) -> Path:
        row_index = int(record["row_index"])
        identifier = str(record["id"]).replace("/", "_").replace("\\", "_")
        return image_dir / f"{row_index:09d}-{identifier}.png"

    def append_record(record: dict[str, Any], path: Path) -> None:
        rows.append(
            {
                **record,
                "image": str(path.resolve()),
                "materialization_seed": seed,
            }
        )

    # Final-training preprocessing already stores every audited image under a
    # checksum/versioned raw-cache contract. Reuse those bytes through hard
    # links instead of re-streaming Photonyx or consuming a second copy of the
    # image corpus. Cross-filesystem outputs fall back to a regular copy.
    raw_root = _raw_cache_root(config)
    wanted: dict[int, dict[str, Any]] = {}
    for record in selected:
        row_index = int(record["row_index"])
        cached = _raw_image_path(raw_root, row_index)
        path = output_path(record)
        if cached.exists():
            if not path.exists():
                try:
                    os.link(cached, path)
                except OSError:
                    shutil.copy2(cached, path)
            append_record(record, path)
        else:
            wanted[row_index] = record

    if wanted:
        photonyx = _disable_streaming_image_decode(_load_photonyx(config.data))
        for row_index, row in enumerate(photonyx):
            if row_index not in wanted:
                continue
            record = wanted.pop(row_index)
            validate_streamed_record(row_index, row, record)
            path = output_path(record)
            _save_raw_image(
                _decode_audit_image(row["image"]),
                path,
                config.data.image_size,
            )
            append_record(record, path)
            if not wanted:
                break
    if wanted:
        raise RuntimeError(f"Could not materialize {len(wanted)} audited rows")
    rows.sort(key=lambda item: int(item["row_index"]))
    manifest = output / "images.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def load_prompt_suite(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL metadata while preserving fields required by external suites."""
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not str(value.get("prompt", "")).strip():
            raise ValueError(f"Prompt-suite row {index} requires a non-empty prompt")
        row = dict(value)
        row.setdefault("id", f"{index:05d}")
        rows.append(row)
    if not rows:
        raise ValueError("Prompt suite is empty")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Prompt-suite IDs must be unique")
    return rows


@torch.no_grad()
def generate_prompt_suite(
    checkpoint: str | Path,
    prompt_file: str | Path,
    output_dir: str | Path,
    generations_per_prompt: int,
    nfe: int,
    guidance: float,
    seed: int,
    device: str = "auto",
    dtype: str = "bfloat16",
    geneval_layout: bool = False,
    batch_size: int = 16,
) -> Path:
    """Generate fixed-seed prompt suites, including the official GenEval layout."""
    if generations_per_prompt < 1:
        raise ValueError("generations_per_prompt must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = load_prompt_suite(prompt_file)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = HemeraPipeline.from_pretrained(
        checkpoint, device=device, dtype=dtype, load_vae=True
    )
    jobs: list[dict[str, Any]] = []
    for prompt_index, row in enumerate(rows):
        if geneval_layout:
            item_root = output / f"{prompt_index:05d}"
            sample_root = item_root / "samples"
            sample_root.mkdir(parents=True, exist_ok=True)
            (item_root / "metadata.jsonl").write_text(
                json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        else:
            sample_root = output / "images"
            sample_root.mkdir(parents=True, exist_ok=True)
        for generation_index in range(generations_per_prompt):
            sample_seed = (
                seed + prompt_index * generations_per_prompt + generation_index
            )
            if geneval_layout:
                image_path = sample_root / f"{generation_index:04d}.png"
            else:
                safe_id = str(row["id"]).replace("/", "_").replace("\\", "_")
                image_path = (
                    sample_root
                    / f"{prompt_index:05d}-{safe_id}-{generation_index:02d}.png"
                )
            jobs.append(
                {
                    "row": row,
                    "prompt_index": prompt_index,
                    "generation_index": generation_index,
                    "seed": sample_seed,
                    "image_path": image_path,
                }
            )
    generated_rows: list[dict[str, Any]] = []
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        generators = [
            torch.Generator(device=pipeline.device).manual_seed(job["seed"])
            for job in batch
        ]
        images = pipeline(
            [str(job["row"]["prompt"]) for job in batch],
            num_inference_steps=nfe,
            guidance_scale=guidance,
            generator=generators,
            output_type="pil",
        )
        for job, image in zip(batch, images, strict=True):
            image.save(job["image_path"])
            row = job["row"]
            generated_rows.append(
                {
                    **row,
                    "prompt_index": job["prompt_index"],
                    "generation_index": job["generation_index"],
                    "image": str(job["image_path"].resolve()),
                    "seed": job["seed"],
                    "nfe": nfe,
                    "guidance": guidance,
                }
            )
    manifest = output / "generations.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in generated_rows
        ),
        encoding="utf-8",
    )
    summary = {
        "prompt_file": str(Path(prompt_file).resolve()),
        "prompt_file_sha256": sha256_file(prompt_file),
        "prompts": len(rows),
        "generations_per_prompt": generations_per_prompt,
        "images": len(generated_rows),
        "seed": seed,
        "nfe": nfe,
        "guidance": guidance,
        "batch_size": batch_size,
        "geneval_layout": geneval_layout,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
    }
    (output / "generation-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return manifest


@torch.no_grad()
def generate_evaluation_pairs(
    config: ExperimentConfig,
    checkpoint: str | Path,
    audit_manifest: str | Path,
    output_dir: str | Path,
    split: str,
    samples: int,
    nfe: int,
    guidance: float,
    seed: int,
    batch_size: int,
    device: str = "auto",
    dtype: str = "bfloat16",
    materialize_workers: int = 16,
) -> Path:
    """Materialize deterministic real/generated pairs for image-space evaluation."""
    selected = select_manifest_records(audit_manifest, split, samples, seed)
    wanted = {int(record["row_index"]): record for record in selected}
    raw_root = _raw_cache_root(config)
    missing = {
        row_index
        for row_index in wanted
        if not _raw_image_path(raw_root, row_index).exists()
    }
    manifest_path = Path(audit_manifest)
    if (
        missing
        and materialize_workers > 1
        and manifest_path.with_suffix(".shards.json").exists()
    ):
        missing_records = sorted(
            (wanted[row_index] for row_index in missing),
            key=lambda record: int(record["row_index"]),
        )
        _materialize_selected_parallel(
            missing_records,
            manifest_path,
            raw_root,
            config.data.image_size,
            materialize_workers,
            lambda *_: None,
        )
        missing = {
            row_index
            for row_index in wanted
            if not _raw_image_path(raw_root, row_index).exists()
        }
    if missing:
        last_missing = max(missing)
        photonyx = _disable_streaming_image_decode(
            _load_photonyx(config.data)
        )
        for row_index, row in enumerate(photonyx):
            if row_index > last_missing:
                break
            if row_index in missing:
                validate_streamed_record(row_index, row, wanted[row_index])
                _save_raw_image(
                    _decode_audit_image(row["image"]),
                    _raw_image_path(raw_root, row_index),
                    config.data.image_size,
                )
                missing.remove(row_index)
                if not missing:
                    break
    if missing:
        raise RuntimeError(f"Could not reload {len(missing)} audited Photonyx rows")

    output = Path(output_dir)
    real_dir = output / "real"
    generated_dir = output / "generated"
    real_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    pipeline = HemeraPipeline.from_pretrained(
        checkpoint, device=device, dtype=dtype, load_vae=True
    )
    paired_rows: list[dict[str, Any]] = []
    for start in range(0, len(selected), batch_size):
        batch_records = selected[start : start + batch_size]
        prompts = [str(record["prompt"]) for record in batch_records]
        generator = torch.Generator(device=pipeline.device).manual_seed(seed + start)
        generated = pipeline(
            prompts,
            num_inference_steps=nfe,
            guidance_scale=guidance,
            generator=generator,
            output_type="pil",
        )
        for offset, (record, image) in enumerate(
            zip(batch_records, generated, strict=True)
        ):
            identifier = str(record["id"]).replace("/", "_").replace("\\", "_")
            real_path = real_dir / f"{start + offset:06d}-{identifier}.png"
            generated_path = generated_dir / f"{start + offset:06d}-{identifier}.png"
            _load_raw_image(
                _raw_image_path(raw_root, int(record["row_index"]))
            ).save(real_path)
            image.save(generated_path)
            paired_rows.append(
                {
                    "id": record["id"],
                    "prompt": record["prompt"],
                    "source": record["source"],
                    "license": record["license"],
                    "split": split,
                    "real_image": str(real_path.resolve()),
                    "generated_image": str(generated_path.resolve()),
                    "seed": seed + start,
                    "nfe": nfe,
                    "guidance": guidance,
                }
            )
    pair_manifest = output / "pairs.jsonl"
    pair_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in paired_rows),
        encoding="utf-8",
    )
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = (
            checkpoint_path / "model.safetensors"
            if (checkpoint_path / "model.safetensors").exists()
            else checkpoint_path / "checkpoint.pt"
        )
    summary = {
        "audit_manifest": str(Path(audit_manifest).resolve()),
        "audit_manifest_sha256": sha256_file(audit_manifest),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split": split,
        "requested_samples": samples,
        "realized_samples": len(paired_rows),
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(str(record["id"]) for record in selected).encode(
                "utf-8"
            )
        ).hexdigest(),
        "nfe": nfe,
        "guidance": guidance,
        "seed": seed,
        "batch_size": batch_size,
        "dtype": dtype,
        "materialize_workers": materialize_workers,
        "pair_manifest": str(pair_manifest.resolve()),
        "pair_manifest_sha256": sha256_file(pair_manifest),
    }
    (output / "generation-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return pair_manifest
