from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluation import percentile_selection_scores


def collect_runs(root: str | Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for manifest_path in sorted(Path(root).glob("**/manifest.json")):
        if "_failed_attempts" in manifest_path.parts:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        evaluation_path = manifest_path.parent / "evaluation.json"
        evaluation = (
            json.loads(evaluation_path.read_text(encoding="utf-8"))
            if evaluation_path.exists()
            else {}
        )
        runs.append({"path": str(manifest_path.parent), **manifest, "evaluation": evaluation})
    return runs


def write_report(root: str | Path, output: str | Path) -> Path:
    runs = collect_runs(root)
    lines = [
        "# Hemera Experiment Report",
        "",
        "This file is generated from immutable run manifests. Missing metrics are shown as `—`.",
        "",
        "| Run | Backbone | Objective | Total params | Active params | Samples | Cost (€) | img/s | GFLOPs/sample | Peak VRAM GiB | Latent CMMD | Val loss |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        config = run["config"]
        evaluation = run["evaluation"]
        lines.append(
            "| {name} | {backbone} | {objective} | {params:,} | {active_params:,} | {samples:,} | {cost:.3f} | {throughput:.2f} | {flops} | {vram:.2f} | {cmmd} | {loss} |".format(
                name=run["name"],
                backbone=config["model"]["backbone"],
                objective=config["objective"]["name"],
                params=run.get("parameter_count", 0),
                active_params=run.get(
                    "active_parameter_count", run.get("parameter_count", 0)
                ),
                samples=run.get("samples_seen", 0),
                cost=run.get("estimated_eur", 0.0),
                throughput=run.get("mean_throughput", 0.0),
                flops=(
                    f"{run['estimated_flops'] / 1e9:.3f}"
                    if run.get("estimated_flops") is not None
                    else "—"
                ),
                vram=run.get("peak_vram_bytes", 0) / 1024**3,
                cmmd=f"{evaluation['latent_cmmd']:.6f}" if "latent_cmmd" in evaluation else "—",
                loss=f"{evaluation['validation_loss']:.6f}" if "validation_loss" in evaluation else "—",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Screening runs are exploratory. Only three-seed finalist reruns support confirmatory claims.",
            "Image-space CMMD, GenEval, preference metrics, and blinded human results supersede latent diagnostics.",
            "",
        ]
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def write_screening_ranking(root: str | Path, output: str | Path) -> Path:
    rows: list[dict[str, Any]] = []
    for metrics_path in Path(root).glob("**/image-eval/metrics.json"):
        if "_failed_attempts" in metrics_path.parts:
            continue
        manifest_path = metrics_path.parent.parent / "manifest.json"
        if not manifest_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metrics.get("hpsv2_mean") is None:
            raise RuntimeError(f"HPSv2 is missing from {metrics_path}")
        rows.append(
            {
                "name": manifest["name"],
                "path": str(manifest_path.parent),
                "cmmd": metrics["cmmd"],
                "hpsv2": metrics["hpsv2_mean"],
                "clip_score": metrics["clip_score_mean"],
                "images_per_second": manifest.get("mean_throughput", 0.0),
                "peak_vram_bytes": manifest.get("peak_vram_bytes", 0),
            }
        )
    ranked = percentile_selection_scores(rows)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    return target
