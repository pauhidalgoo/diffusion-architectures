"""Package 64 manually selected Hemera showcase images and exact metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORED = (
    ROOT / "artifacts" / "showcase" / "review" / "scored-candidates.jsonl"
)
DEFAULT_SELECTION = (
    ROOT / "artifacts" / "showcase" / "review" / "manual-selection.jsonl"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "showcase" / "final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render_contact_sheet(
    rows: list[dict[str, Any]], path: Path, image_root: Path | None = None
) -> None:
    columns = 8
    tile = 256
    label = 28
    rows_count = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile, rows_count * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        x = (index % columns) * tile
        y = (index // columns) * (tile + label)
        image_path = Path(row["selected_image"])
        if image_root is not None and not image_path.is_absolute():
            image_path = image_root / image_path
        image = Image.open(image_path).convert("RGB").resize((tile, tile))
        canvas.paste(image, (x, y))
        draw.text(
            (x + 4, y + tile + 7),
            f"{index + 1:02d}  {row['prompt_id']}",
            fill="black",
            font=font,
        )
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored",
        type=Path,
        action="append",
        help="Scored candidate JSONL; repeat to merge rescue pools",
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored_paths = args.scored or [DEFAULT_SCORED]
    candidate_rows = [
        row for path in scored_paths for row in read_jsonl(path)
    ]
    candidates = {str(row["candidate_id"]): row for row in candidate_rows}
    if len(candidates) != len(candidate_rows):
        raise ValueError("Scored manifests contain duplicate candidate IDs")
    decisions = read_jsonl(args.selection)
    if len(decisions) != 64:
        raise ValueError(f"Expected exactly 64 selections, found {len(decisions)}")
    prompt_counts = Counter(str(row["prompt_id"]) for row in decisions)
    duplicates = [key for key, count in prompt_counts.items() if count != 1]
    if duplicates or len(prompt_counts) != 64:
        raise ValueError("Selection must contain exactly one row per prompt")

    output = args.output.resolve()
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    # A revised selection may change filenames. Remove only files previously
    # emitted by this packager so stale numbered images cannot survive.
    for previous in images.glob("[0-9][0-9]-showcase-*.png"):
        previous.unlink()
    packaged: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions):
        candidate_id = str(decision["candidate_id"])
        try:
            candidate = dict(candidates[candidate_id])
        except KeyError as error:
            raise ValueError(f"Unknown candidate {candidate_id!r}") from error
        if str(candidate["prompt_id"]) != str(decision["prompt_id"]):
            raise ValueError(f"Prompt/candidate mismatch for {candidate_id}")
        source = Path(candidate["image"])
        destination = images / f"{index + 1:02d}-{candidate['prompt_id']}.png"
        shutil.copy2(source, destination)
        candidate.pop("image", None)
        candidate["selected_image"] = destination.relative_to(output).as_posix()
        candidate["selected_image_sha256"] = sha256(destination)
        candidate["selection_notes"] = str(decision.get("notes", ""))
        candidate["selection_rank"] = index + 1
        packaged.append(candidate)

    jsonl = output / "showcase.jsonl"
    temporary = jsonl.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in packaged
        ),
        encoding="utf-8",
    )
    os.replace(temporary, jsonl)

    with (output / "showcase.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "selection_rank",
            "prompt_id",
            "category",
            "prompt",
            "negative_prompt",
            "seed",
            "nfe",
            "guidance",
            "candidate_id",
            "selected_image",
            "selected_image_sha256",
            "selection_notes",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(packaged)

    markdown = [
        "# Hemera-Nano selected showcase",
        "",
        "These 64 images were selected from a deterministic multi-seed candidate",
        "pool. Every image below has its exact prompt and inference parameters",
        "preserved in `showcase.jsonl` and `showcase.csv`.",
        "",
    ]
    for row in packaged:
        relative = str(row["selected_image"])
        markdown.extend(
            [
                f"## {row['selection_rank']:02d}. {row['prompt_id']}",
                "",
                f"![{row['prompt_id']}]({relative})",
                "",
                f"**Prompt:** {row['prompt']}",
                "",
                f"**Negative:** {row['negative_prompt']}",
                "",
                (
                    f"**Parameters:** seed {row['seed']}; NFE {row['nfe']}; "
                    f"CFG {row['guidance']}"
                ),
                "",
            ]
        )
    (output / "README.md").write_text("\n".join(markdown), encoding="utf-8")
    render_contact_sheet(packaged, output / "contact-sheet.png", output)

    checksum_files = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(output).as_posix()}\n"
            for path in sorted(checksum_files)
        ),
        encoding="utf-8",
    )
    summary = {
        "images": len(packaged),
        "categories": dict(Counter(row["category"] for row in packaged)),
        "showcase_manifest_sha256": sha256(jsonl),
        "contact_sheet": str(output / "contact-sheet.png"),
        "output": str(output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
