#!/usr/bin/env python
"""Create compact, publication-ready summaries from a Hemera delivery archive."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import tarfile
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


RANKING_MEMBERS = {
    "representation": "reports/representation-ranking.json",
    "objective": "reports/objective-ranking.json",
    "conditioning": "reports/conditioning-ranking.json",
    "backbone": "reports/backbone-ranking.json",
    "efficiency": "reports/efficiency-ranking.json",
    "data_sampling": "reports/data-sampling-ranking.json",
    "decision_sprint": "reports/decision-sprint-ranking.json",
}


def _read_bytes(archive: tarfile.TarFile, member: str) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(f"Archive member is not a file: {member}")
    return extracted.read()


def _read_json(archive: tarfile.TarFile, member: str) -> Any:
    return json.loads(_read_bytes(archive, member).decode("utf-8"))


def _read_jsonl(archive: tarfile.TarFile, member: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in _read_bytes(archive, member).decode("utf-8").splitlines()
        if line.strip()
    ]


def _compact_test_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "samples",
        "cmmd",
        "fid",
        "precision",
        "recall",
        "semantic_frechet",
        "clip_score_mean",
        "clip_score_std",
        "siglip_alignment_mean",
        "siglip_alignment_std",
        "hpsv2_mean",
        "hpsv2_std",
    )
    compact = {key: metrics.get(key) for key in keys}
    compact["per_source"] = {
        source: {
            key: values.get(key)
            for key in (
                "samples",
                "cmmd",
                "precision",
                "recall",
                "clip_score_mean",
                "siglip_alignment_mean",
                "hpsv2_mean",
            )
        }
        for source, values in metrics.get("per_source", {}).items()
    }
    return compact


def _bin_rows(
    rows: Sequence[dict[str, Any]],
    *,
    value_keys: Sequence[str],
    bins: int = 500,
) -> list[dict[str, float]]:
    if not rows:
        return []
    by_step = {int(row["step"]): row for row in rows if row.get("step") is not None}
    ordered = [by_step[step] for step in sorted(by_step)]
    chunk = max(1, math.ceil(len(ordered) / bins))
    output: list[dict[str, float]] = []
    for start in range(0, len(ordered), chunk):
        group = ordered[start : start + chunk]
        item: dict[str, float] = {
            "step": float(statistics.fmean(float(row["step"]) for row in group))
        }
        for key in value_keys:
            values = [
                float(row[key])
                for row in group
                if isinstance(row.get(key), (int, float))
            ]
            if values:
                item[key] = statistics.fmean(values)
        output.append(item)
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _points(
    values: Sequence[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    ranges: tuple[float, float, float, float] | None = None,
) -> str:
    left, top, width, height = bounds
    if not values:
        return ""
    if ranges is None:
        xs, ys = zip(*values)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
    else:
        x_min, x_max, y_min, y_max = ranges
    x_span = max(1e-12, x_max - x_min)
    y_span = max(1e-12, y_max - y_min)
    return " ".join(
        f"{left + (x - x_min) / x_span * width:.2f},"
        f"{top + height - (y - y_min) / y_span * height:.2f}"
        for x, y in values
    )


def _escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_chart(
    path: Path,
    *,
    title: str,
    panels: Sequence[dict[str, Any]],
    width: int = 1200,
) -> None:
    panel_height = 270
    height = 70 + panel_height * len(panels)
    palette = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#172033}'
        ".title{font-size:24px;font-weight:700}.panel{font-size:16px;"
        "font-weight:650}.axis{font-size:12px;fill:#526078}.legend{font-size:12px}"
        ".grid{stroke:#e4e9f0;stroke-width:1}.axisline{stroke:#8994a6;"
        "stroke-width:1}</style>",
        f'<text x="50" y="38" class="title">{_escape(title)}</text>',
    ]
    for panel_index, panel in enumerate(panels):
        x, y, chart_width, chart_height = 100, 92 + panel_index * panel_height, 1020, 170
        series = panel["series"]
        all_values = [point for item in series for point in item["values"]]
        if not all_values:
            continue
        xs, ys = zip(*all_values)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        padding = max(1e-12, (y_max - y_min) * 0.08)
        y_min, y_max = y_min - padding, y_max + padding
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            grid_y = y + chart_height * fraction
            tick = y_max - (y_max - y_min) * fraction
            svg.append(
                f'<line x1="{x}" y1="{grid_y:.2f}" x2="{x + chart_width}" '
                f'y2="{grid_y:.2f}" class="grid"/>'
            )
            svg.append(
                f'<text x="{x - 10}" y="{grid_y + 4:.2f}" text-anchor="end" '
                f'class="axis">{tick:.4g}</text>'
            )
        svg.extend(
            [
                f'<line x1="{x}" y1="{y + chart_height}" x2="{x + chart_width}" '
                f'y2="{y + chart_height}" class="axisline"/>',
                f'<text x="{x}" y="{y - 14}" class="panel">'
                f'{_escape(panel["title"])}</text>',
                f'<text x="{x}" y="{y + chart_height + 22}" class="axis">'
                f'{x_min:,.0f}</text>',
                f'<text x="{x + chart_width}" y="{y + chart_height + 22}" '
                f'text-anchor="end" class="axis">{x_max:,.0f}</text>',
            ]
        )
        for index, item in enumerate(series):
            color = palette[index % len(palette)]
            svg.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.2" '
                f'points="{_points(item["values"], (x, y, chart_width, chart_height), (x_min, x_max, y_min, y_max))}"/>'
            )
            legend_x = x + index * 190
            svg.append(
                f'<line x1="{legend_x}" y1="{y + chart_height + 44}" '
                f'x2="{legend_x + 22}" y2="{y + chart_height + 44}" '
                f'stroke="{color}" stroke-width="3"/>'
            )
            svg.append(
                f'<text x="{legend_x + 29}" y="{y + chart_height + 48}" '
                f'class="legend">{_escape(item["name"])}</text>'
            )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _histogram(values: Sequence[float], bins: int = 30) -> list[tuple[float, float]]:
    low, high = min(values), max(values)
    width = max(1e-12, (high - low) / bins)
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / width))
        counts[index] += 1
    maximum = max(counts)
    return [
        (low + (index + 0.5) * width, count / maximum)
        for index, count in enumerate(counts)
    ]


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    width: int,
    font: ImageFont.ImageFont,
    fill: str = "#202938",
    spacing: int = 3,
) -> None:
    average = max(1, int(width / max(6, font.getlength("abcdefghij") / 10)))
    wrapped = textwrap.fill(text, width=average)
    draw.multiline_text(xy, wrapped, font=font, fill=fill, spacing=spacing)


def _human_grid(
    archive: tarfile.TarFile,
    rows: Sequence[dict[str, Any]],
    output: Path,
) -> None:
    selected: list[dict[str, Any]] = []
    per_category: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        category = str(row["category"])
        if per_category[category] < 2:
            selected.append(row)
            per_category[category] += 1
    columns, image_size, cell_width, cell_height = 4, 256, 300, 355
    rows_count = math.ceil(len(selected) / columns)
    canvas = Image.new("RGB", (columns * cell_width, 70 + rows_count * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, prompt_font = _font(24, bold=True), _font(15, bold=True), _font(13)
    draw.text(
        (24, 18),
        "Hemera-Nano: first two human-study prompts per category (not quality-curated)",
        font=title_font,
        fill="#172033",
    )
    for index, row in enumerate(selected):
        column, grid_row = index % columns, index // columns
        x, y = column * cell_width + 22, 70 + grid_row * cell_height
        member = str(row["image"]).removeprefix("/workspace/hemera/")
        image = Image.open(io.BytesIO(_read_bytes(archive, member))).convert("RGB")
        canvas.paste(image.resize((image_size, image_size)), (x, y))
        draw.text(
            (x, y + image_size + 8),
            str(row["category"]).replace("_", " "),
            font=label_font,
            fill="#31559b",
        )
        _draw_wrapped(
            draw,
            (x, y + image_size + 29),
            str(row["prompt"]),
            width=image_size,
            font=prompt_font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def _milestone_grid(milestone_root: Path, prompt_file: Path, output: Path) -> bool:
    milestones = ("25pct", "50pct", "75pct", "95pct")
    directories = {
        label: milestone_root / f"milestone-{label}" / f"canary-{label}"
        for label in milestones
    }
    if not prompt_file.exists() or not all(path.exists() for path in directories.values()):
        return False
    prompts = [line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    image_size, label_width, header_height = 192, 300, 60
    canvas = Image.new(
        "RGB",
        (label_width + image_size * len(milestones), header_height + image_size * len(prompts)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    header_font, prompt_font = _font(18, bold=True), _font(13)
    draw.text((18, 18), "Prompt", font=header_font, fill="#172033")
    for column, label in enumerate(milestones):
        draw.text(
            (label_width + column * image_size + 58, 18),
            label.replace("pct", "%"),
            font=header_font,
            fill="#31559b",
        )
    for row_index, prompt in enumerate(prompts):
        y = header_height + row_index * image_size
        _draw_wrapped(draw, (14, y + 48), prompt, width=label_width - 28, font=prompt_font)
        for column, label in enumerate(milestones):
            source = directories[label] / f"sample-{row_index:03d}.png"
            image = Image.open(source).convert("RGB").resize((image_size, image_size))
            canvas.paste(image, (label_width + column * image_size, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return True


def analyze(
    archive_path: Path,
    output_dir: Path,
    *,
    milestone_root: Path | None = None,
    canary_prompts: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        audit = _read_json(archive, "reports/photonyx-final-audit-verification.json")
        manifest = _read_json(archive, "runs/final/hemera-nano/manifest.json")
        locked = _read_json(archive, "runs/final/locked-inference.json")
        test_metrics_full = _read_json(archive, "runs/final/test/metrics.json")
        test_metrics = _compact_test_metrics(test_metrics_full)
        confirmation = _read_json(archive, "reports/confirmation-comparison.json")
        costs = {
            name: _read_json(archive, member)
            for name, member in {
                "exploratory": "reports/cost-exploratory.json",
                "decision_sprint": "reports/cost-decision-sprint.json",
                "final_instance": "reports/cost-final.json",
            }.items()
        }
        rankings = {
            name: _read_json(archive, member)
            for name, member in RANKING_MEMBERS.items()
        }
        memory = {
            encoder: _read_json(
                archive, f"runs/final/memorization/neighbors-{encoder}.json"
            )
            for encoder in ("clip", "dinov2")
        }
        geneval = _read_json(archive, "runs/final/geneval/evaluator-status.json")
        protocols = {
            name: _read_json(archive, member)
            for name, member in {
                "test": "runs/final/test/generation-summary.json",
                "human_study": "runs/final/human-study/generation-summary.json",
                "safety_bias": "runs/final/safety-bias/generation-summary.json",
                "geneval": "runs/final/geneval/generations/generation-summary.json",
            }.items()
        }
        training_rows = _read_jsonl(archive, "runs/final/hemera-nano/metrics.jsonl")
        human_rows = _read_jsonl(archive, "runs/final/human-study/generations.jsonl")

        train = [row for row in training_rows if "loss" in row]
        validation = [row for row in training_rows if "validation_loss" in row]
        binned = _bin_rows(
            train,
            value_keys=("loss", "lr", "images_per_second", "spent_eur"),
        )
        _write_csv(output_dir / "training-curve-binned.csv", binned)
        _write_csv(output_dir / "validation-curve.csv", validation)
        _write_csv(output_dir / "inference-sweep.csv", locked["candidates"])

        _svg_chart(
            output_dir / "training-curves.svg",
            title="Hemera-Nano final training",
            panels=(
                {
                    "title": "Binned training loss",
                    "series": [{"name": "training loss", "values": [(r["step"], r["loss"]) for r in binned]}],
                },
                {
                    "title": "Validation loss",
                    "series": [
                        {
                            "name": "validation loss",
                            "values": [(float(r["step"]), float(r["validation_loss"])) for r in validation],
                        }
                    ],
                },
                {
                    "title": "Throughput (images/s)",
                    "series": [
                        {
                            "name": "images/s",
                            "values": [(r["step"], r["images_per_second"]) for r in binned],
                        }
                    ],
                },
            ),
        )

        sweep_panels = []
        for metric, title in (
            ("cmmd", "Validation CMMD (lower is better)"),
            ("hpsv2", "Validation HPSv2 (higher is better)"),
            ("clip_score", "Validation CLIP alignment (higher is better)"),
        ):
            sweep_panels.append(
                {
                    "title": title,
                    "series": [
                        {
                            "name": f"CFG {guidance}",
                            "values": [
                                (float(row["nfe"]), float(row[metric]))
                                for row in locked["candidates"]
                                if float(row["guidance"]) == guidance
                            ],
                        }
                        for guidance in (1.0, 2.0, 3.0, 4.0, 5.0)
                    ],
                }
            )
        _svg_chart(
            output_dir / "inference-sweep.svg",
            title="Validation-only inference sweep (15 candidates)",
            panels=sweep_panels,
        )

        memory_panels = []
        memory_summary = {}
        for encoder, payload in memory.items():
            nearest = [
                float(item["neighbors"][0]["cosine_similarity"])
                for item in payload["neighbors"]
            ]
            memory_panels.append(
                {
                    "title": f"{encoder.upper()} nearest-neighbor cosine distribution",
                    "series": [{"name": encoder, "values": _histogram(nearest)}],
                }
            )
            memory_summary[encoder] = {
                key: payload[key]
                for key in (
                    "model_id",
                    "generated_samples",
                    "training_samples",
                    "maximum_similarity",
                    "mean_nearest_similarity",
                    "quantiles",
                )
            }
        _svg_chart(
            output_dir / "memorization-distributions.svg",
            title="Nearest-neighbor diagnostics (diagnostic, not proof of copying)",
            panels=memory_panels,
        )
        _human_grid(archive, human_rows, output_dir / "human-study-sample-grid.png")

    if milestone_root is not None and canary_prompts is not None:
        _milestone_grid(
            milestone_root,
            canary_prompts,
            output_dir / "milestone-progression.png",
        )

    final_training_cost = float(manifest["estimated_eur"])
    complete_instance_cost = float(costs["final_instance"]["estimated_eur"])
    total_logged_cost = sum(float(item["estimated_eur"]) for item in costs.values())
    success_reasons = []
    if not confirmation["candidate_beats_baseline_in_every_seed"]:
        success_reasons.append("confirmation candidate did not win in every seed")
    if not confirmation["aggregate_positive_at_95_confidence"]:
        success_reasons.append("confirmation composite lower confidence bound was not positive")
    success_reasons.append("blinded human preference ratings are pending")
    if not geneval["official_evaluator_run"]:
        success_reasons.append("official GenEval detector scoring is pending")
    summary = {
        "schema_version": 1,
        "dataset": {
            "id": audit["dataset"],
            "revision": audit["dataset_revision"],
            "manifest_sha256": audit["manifest_sha256"],
            "records": audit["records"],
            "by_split": audit["by_split"],
            "by_source": audit["by_source"],
            "by_license": audit["by_license"],
        },
        "model": {
            "parameters": manifest["parameter_count"],
            "active_parameters": manifest["active_parameter_count"],
            "samples_seen": manifest["samples_seen"],
            "mean_training_images_per_second": manifest["mean_throughput"],
            "peak_vram_gib": manifest["peak_vram_bytes"] / 2**30,
            "forward_gflops_per_sample": manifest["estimated_flops"] / 1e9,
            "checkpoint_hashes": manifest["checkpoint_hashes"],
        },
        "selection": locked["winner"],
        "test": test_metrics,
        "memorization": memory_summary,
        "protocols": protocols,
        "confirmation": confirmation,
        "rankings": {
            phase: {
                "winner": rows[0]["name"],
                "winner_score": rows[0]["selection_score"],
                "candidates": len(rows),
            }
            for phase, rows in rankings.items()
        },
        "cost": {
            "training_ledger_eur": final_training_cost,
            "complete_final_instance_wall_estimate_eur": complete_instance_cost,
            "total_logged_project_wall_estimate_eur": total_logged_cost,
            "components": costs,
            "note": (
                "Wall-clock estimates are derived from recorded hourly rates and may "
                "differ from the provider invoice, especially across stopped periods."
            ),
        },
        "pre_registered_success": {
            "passed": False,
            "reasons": success_reasons,
            "checkpoint_reload_and_automated_suite_passed": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("artifacts/final-delivery/hemera-final-results.tar"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/final"))
    parser.add_argument(
        "--milestone-root",
        type=Path,
        default=Path("artifacts/final-run-live"),
    )
    parser.add_argument(
        "--canary-prompts",
        type=Path,
        default=Path("evaluation/prompts/final-canary-v1.txt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(
        args.archive,
        args.output,
        milestone_root=args.milestone_root,
        canary_prompts=args.canary_prompts,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "test_samples": summary["test"]["samples"],
                "locked_inference": summary["selection"],
                "pre_registered_success": summary["pre_registered_success"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
