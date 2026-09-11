"""Score Hemera showcase candidates and render visual-review contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, CLIPModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts" / "showcase" / "candidates.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "showcase" / "review"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def technical_metrics(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    horizontal = np.abs(np.diff(gray, axis=1)).mean()
    vertical = np.abs(np.diff(gray, axis=0)).mean()
    histogram, _ = np.histogram(gray, bins=64, range=(0, 1))
    probabilities = histogram / max(histogram.sum(), 1)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum() / math.log2(64))
    clipped = float(((rgb < 0.01) | (rgb > 0.99)).mean())
    channel_spread = float(np.std(rgb.mean(axis=(0, 1))))
    return {
        "sharpness": float(horizontal + vertical),
        "entropy": entropy,
        "clipped_fraction": clipped,
        "channel_spread": channel_spread,
    }


def rank01(values: list[float], higher_is_better: bool = True) -> list[float]:
    if len(values) == 1:
        return [1.0]
    order = sorted(
        range(len(values)), key=lambda index: values[index], reverse=higher_is_better
    )
    scores = [0.0] * len(values)
    for rank, index in enumerate(order):
        scores[index] = 1.0 - rank / (len(values) - 1)
    return scores


def score_clip(
    rows: list[dict[str, Any]],
    model_id: str,
    batch_size: int,
    device: str,
) -> list[float]:
    resolved = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else ("cpu" if device == "auto" else device)
    )
    model = CLIPModel.from_pretrained(model_id).to(resolved).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    scores: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = [Image.open(row["image"]).convert("RGB") for row in batch]
        inputs = processor(
            text=[str(row["prompt"]) for row in batch],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(resolved) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = torch.nn.functional.normalize(
                outputs.image_embeds.float(), dim=-1
            )
            text_features = torch.nn.functional.normalize(
                outputs.text_embeds.float(), dim=-1
            )
            values = (image_features * text_features).sum(dim=-1)
        scores.extend(float(value) for value in values.cpu())
        print(f"CLIP-scored {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return scores


def add_composite_scores(rows: list[dict[str, Any]]) -> None:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["prompt_id"])].append(row)
    for candidates in by_prompt.values():
        clip_ranks = rank01([float(row["clip_alignment"]) for row in candidates])
        sharp_ranks = rank01([float(row["sharpness"]) for row in candidates])
        entropy_ranks = rank01([float(row["entropy"]) for row in candidates])
        clipped_ranks = rank01(
            [float(row["clipped_fraction"]) for row in candidates],
            higher_is_better=False,
        )
        for row, clip, sharp, entropy, clipped in zip(
            candidates,
            clip_ranks,
            sharp_ranks,
            entropy_ranks,
            clipped_ranks,
            strict=True,
        ):
            row["automatic_score"] = (
                0.70 * clip + 0.15 * sharp + 0.10 * entropy + 0.05 * clipped
            )


def render_sheets(
    rows: list[dict[str, Any]], output: Path, prompts_per_sheet: int = 4
) -> list[Path]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_order: list[str] = []
    for row in rows:
        prompt_id = str(row["prompt_id"])
        if prompt_id not in by_prompt:
            prompt_order.append(prompt_id)
        by_prompt[prompt_id].append(row)
    font = ImageFont.load_default()
    sheets = []
    tile = 256
    gap = 12
    label_height = 54
    prompt_height = 48
    columns = max(len(values) for values in by_prompt.values())
    width = gap + columns * (tile + gap)
    row_height = prompt_height + tile + label_height + gap
    for sheet_index, start in enumerate(
        range(0, len(prompt_order), prompts_per_sheet)
    ):
        identifiers = prompt_order[start : start + prompts_per_sheet]
        canvas = Image.new("RGB", (width, gap + len(identifiers) * row_height), "white")
        draw = ImageDraw.Draw(canvas)
        y = gap
        for identifier in identifiers:
            candidates = sorted(
                by_prompt[identifier],
                key=lambda row: float(row["automatic_score"]),
                reverse=True,
            )
            prompt = str(candidates[0]["prompt"])
            header = f"{identifier}: {prompt}"
            draw.multiline_text(
                (gap, y),
                "\n".join(textwrap.wrap(header, width=135)[:2]),
                fill="black",
                font=font,
                spacing=2,
            )
            image_y = y + prompt_height
            for column, row in enumerate(candidates):
                x = gap + column * (tile + gap)
                image = Image.open(row["image"]).convert("RGB").resize((tile, tile))
                canvas.paste(image, (x, image_y))
                label = (
                    f"{row['candidate_id']}  seed {row['seed']}\n"
                    f"auto {row['automatic_score']:.3f}  "
                    f"CLIP {row['clip_alignment']:.3f}  "
                    f"sharp {row['sharpness']:.3f}"
                )
                draw.multiline_text(
                    (x, image_y + tile + 4),
                    label,
                    fill="black",
                    font=font,
                    spacing=2,
                )
            y += row_height
        path = output / f"review-{sheet_index:02d}.png"
        canvas.save(path)
        sheets.append(path)
    return sheets


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Candidate manifest is empty")
    for row in rows:
        path = Path(row["image"])
        if not path.is_file():
            raise FileNotFoundError(path)
        row.update(technical_metrics(Image.open(path)))
    for row, score in zip(
        rows,
        score_clip(rows, args.model_id, args.batch_size, args.device),
        strict=True,
    ):
        row["clip_alignment"] = score
    add_composite_scores(rows)
    rows.sort(key=lambda row: (str(row["prompt_id"]), -row["automatic_score"]))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scored = output / "scored-candidates.jsonl"
    write_jsonl(scored, rows)
    automatic = []
    seen = set()
    for row in rows:
        if row["prompt_id"] not in seen:
            automatic.append(
                {
                    "prompt_id": row["prompt_id"],
                    "candidate_id": row["candidate_id"],
                    "automatic_score": row["automatic_score"],
                }
            )
            seen.add(row["prompt_id"])
    write_jsonl(output / "automatic-selection.jsonl", automatic)
    sheets = render_sheets(rows, output)
    summary = {
        "schema_version": 1,
        "candidates": len(rows),
        "prompts": len(seen),
        "scored_manifest": str(scored),
        "scored_manifest_sha256": sha256(scored),
        "automatic_selection": str(output / "automatic-selection.jsonl"),
        "review_sheets": [str(path) for path in sheets],
    }
    (output / "scoring-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
