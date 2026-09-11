#!/usr/bin/env python3
"""Build a deterministic, side-by-side representation screening contact sheet."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _read_pairs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=12)
    args = parser.parse_args()

    candidates = ["dc-clip", "dc-t5", "sd-clip", "sd-t5"]
    pairs = {
        name: _read_pairs(args.root / name / "image-eval" / "pairs.jsonl")
        for name in candidates
    }
    selected_ids = [[row["id"] for row in pairs[name]] for name in candidates]
    if any(ids != selected_ids[0] for ids in selected_ids[1:]):
        raise RuntimeError("Representation evaluations do not contain identical row IDs")

    rows = min(args.rows, len(pairs[candidates[0]]))
    tile = 256
    caption_height = 72
    header_height = 36
    columns = ["real", *candidates]
    canvas = Image.new(
        "RGB",
        (tile * len(columns), header_height + (tile + caption_height) * rows),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, label in enumerate(columns):
        draw.text((column * tile + 8, 12), label, fill="black", font=font)

    reference = pairs[candidates[0]]
    for row_index in range(rows):
        y = header_height + row_index * (tile + caption_height)
        paths = [reference[row_index]["real_image"]]
        paths.extend(pairs[name][row_index]["generated_image"] for name in candidates)
        for column, path in enumerate(paths):
            image = Image.open(path).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(image, (column * tile, y))
        prompt = reference[row_index]["prompt"]
        caption = "\n".join(textwrap.wrap(prompt, width=190)[:3])
        draw.multiline_text(
            (8, y + tile + 4),
            f'{row_index:02d} [{reference[row_index]["source"]}] {caption}',
            fill="black",
            font=font,
            spacing=2,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    print(args.output)


if __name__ == "__main__":
    main()
