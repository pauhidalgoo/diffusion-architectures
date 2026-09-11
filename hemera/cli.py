from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .config import load_config
from .benchmarks import (
    audit_nearest_neighbors,
    build_image_feature_index,
    evaluate_image_pairs,
)
from .data import (
    audit_photonyx,
    precompute_photonyx,
    verify_audit_manifest,
)
from .evaluation import (
    analyze_human_study,
    compare_confirmation_benchmarks,
    compare_paired_benchmarks,
    create_blinded_human_study,
    create_blinded_human_study_from_manifests,
    evaluate_checkpoint,
)
from .eval_generation import (
    generate_evaluation_pairs,
    generate_prompt_suite,
    materialize_audit_images,
)
from .pipeline import HemeraPipeline
from .profiling import (
    profile_attention,
    profile_batch_sizes,
    profile_sweep_batch_sizes,
)
from .protocols import write_protocol_prompts
from .reporting import write_report, write_screening_ranking
from .runtime import resolve_device
from .representation_probe import probe_representation
from .sweep import run_sweep
from .trainer import train_from_config


def _prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        return [
            line.strip()
            for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return args.prompt or ["a small golden sun rising over a calm sea"]


def _save_latent_previews(latents: torch.Tensor, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for index, latent in enumerate(latents):
        channels = latent[:3].float()
        channels = (channels - channels.min()) / (channels.max() - channels.min()).clamp_min(1e-8)
        array = channels.mul(255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(array).resize((256, 256), Image.Resampling.NEAREST).save(
            output / f"latent-{index:03d}.png"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hemera",
        description="Hemera micro-budget text-to-image research framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a configured denoiser")
    train.add_argument("--config", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)

    sample = subparsers.add_parser("sample", help="Sample a checkpoint")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--prompt", action="append")
    sample.add_argument("--prompt-file")
    sample.add_argument("--negative-prompt")
    sample.add_argument("--steps", type=int, default=30)
    sample.add_argument("--guidance", type=float, default=3.0)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--device", default="auto")
    sample.add_argument("--dtype", default="bfloat16")
    sample.add_argument("--output", default="samples")
    sample.add_argument("--latents-only", action="store_true")

    export = subparsers.add_parser(
        "export", help="Export an EMA checkpoint as a portable Safetensors pipeline"
    )
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--device", default="cpu")

    audit = subparsers.add_parser("audit-data", help="Audit Photonyx into a split manifest")
    audit.add_argument("--config", required=True)
    audit.add_argument("--output", default="data/manifests/photonyx.jsonl")

    verify_audit = subparsers.add_parser(
        "verify-audit", help="Validate and summarize an immutable audit manifest"
    )
    verify_audit.add_argument("--config", required=True)
    verify_audit.add_argument("--manifest", required=True)
    verify_audit.add_argument(
        "--output", default="reports/photonyx-audit-verification.json"
    )

    precompute = subparsers.add_parser("precompute", help="Precompute Photonyx representation shards")
    precompute.add_argument("--config", required=True)
    precompute.add_argument("--manifest", required=True)
    precompute.add_argument("--output", default="data/cache")
    precompute.add_argument("--batch-size", type=int, default=32)
    precompute.add_argument(
        "--materialize-workers",
        type=int,
        default=1,
        help="Concurrent pinned Parquet shards used to materialize selected images",
    )
    precompute.add_argument("--device", default="auto")

    preflight = subparsers.add_parser("cloud-preflight", help="Validate real cloud data and encoders")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--output", default="preflight")
    preflight.add_argument("--batch-size", type=int, default=4)
    preflight.add_argument("--items", type=int, default=8)

    representation_probe = subparsers.add_parser(
        "probe-representation",
        help="Measure VAE reconstruction and held-out text-image retrieval",
    )
    representation_probe.add_argument("--config", required=True)
    representation_probe.add_argument("--output", required=True)
    representation_probe.add_argument("--samples", type=int, default=256)
    representation_probe.add_argument("--batch-size", type=int, default=16)
    representation_probe.add_argument("--device", default="auto")
    representation_probe.add_argument(
        "--evaluator", default="openai/clip-vit-large-patch14-336"
    )

    sweep = subparsers.add_parser("sweep", help="Run a budgeted experiment sweep")
    sweep.add_argument("--config", required=True)

    report = subparsers.add_parser("report", help="Generate a report from run manifests")
    report.add_argument("--runs", default="runs")
    report.add_argument("--output", default="reports/EXPERIMENTS.md")

    rank = subparsers.add_parser(
        "rank", help="Rank image-evaluated exploratory runs"
    )
    rank.add_argument("--runs", default="runs/exploratory")
    rank.add_argument("--output", default="reports/screening-ranking.json")

    human_create = subparsers.add_parser("human-study-create", help="Create a blinded study CSV")
    human_create.add_argument("--prompts")
    human_create.add_argument("--model-a-images")
    human_create.add_argument("--model-b-images")
    human_create.add_argument("--model-a-manifest")
    human_create.add_argument("--model-b-manifest")
    human_create.add_argument(
        "--model-a-image-root",
        help="Rebase model-A manifest image basenames under this local directory",
    )
    human_create.add_argument(
        "--model-b-image-root",
        help="Rebase model-B manifest image basenames under this local directory",
    )
    human_create.add_argument("--output", required=True)
    human_create.add_argument("--seed", type=int, default=0)
    human_create.add_argument("--ratings-per-prompt", type=int, default=3)

    human_analyze = subparsers.add_parser("human-study-analyze", help="Analyze completed ratings")
    human_analyze.add_argument("--input", required=True)
    human_analyze.add_argument("--output")

    image_benchmark = subparsers.add_parser(
        "benchmark-images", help="Compute CMMD, alignment, preference, precision and recall"
    )
    image_benchmark.add_argument("--manifest", required=True)
    image_benchmark.add_argument("--output", required=True)
    image_benchmark.add_argument(
        "--model-id", default="openai/clip-vit-large-patch14-336"
    )
    image_benchmark.add_argument(
        "--siglip-model-id",
        help="Optional independent SigLIP alignment diagnostic",
    )
    image_benchmark.add_argument("--batch-size", type=int, default=32)
    image_benchmark.add_argument("--device", default="auto")

    compare_benchmarks = subparsers.add_parser(
        "compare-benchmarks",
        help="Paired-bootstrap two image benchmark feature artifacts",
    )
    compare_benchmarks.add_argument("--candidate", required=True)
    compare_benchmarks.add_argument("--baseline", required=True)
    compare_benchmarks.add_argument("--output", required=True)
    compare_benchmarks.add_argument("--bootstrap-samples", type=int, default=2000)
    compare_benchmarks.add_argument("--seed", type=int, default=0)

    compare_confirmation = subparsers.add_parser(
        "compare-confirmation",
        help="Prompt-cluster bootstrap matched recipes across multiple seeds",
    )
    compare_confirmation.add_argument(
        "--candidate", action="append", required=True
    )
    compare_confirmation.add_argument(
        "--baseline", action="append", required=True
    )
    compare_confirmation.add_argument("--output", required=True)
    compare_confirmation.add_argument(
        "--bootstrap-samples", type=int, default=2000
    )
    compare_confirmation.add_argument("--seed", type=int, default=0)

    generate_eval = subparsers.add_parser(
        "generate-eval", help="Generate deterministic real/model evaluation pairs"
    )
    generate_eval.add_argument("--config", required=True)
    generate_eval.add_argument("--checkpoint", required=True)
    generate_eval.add_argument("--manifest", required=True)
    generate_eval.add_argument("--output", required=True)
    generate_eval.add_argument("--split", choices=["validation", "test"], required=True)
    generate_eval.add_argument("--samples", type=int, default=5000)
    generate_eval.add_argument("--steps", type=int, default=30)
    generate_eval.add_argument("--guidance", type=float, default=3.0)
    generate_eval.add_argument("--seed", type=int, default=1234)
    generate_eval.add_argument("--batch-size", type=int, default=16)
    generate_eval.add_argument("--device", default="auto")
    generate_eval.add_argument("--dtype", default="bfloat16")
    generate_eval.add_argument("--materialize-workers", type=int, default=16)

    materialize = subparsers.add_parser(
        "materialize-images",
        help="Materialize a deterministic audit-manifest image subset",
    )
    materialize.add_argument("--config", required=True)
    materialize.add_argument("--manifest", required=True)
    materialize.add_argument("--output", required=True)
    materialize.add_argument(
        "--split", choices=["train", "validation", "test"], required=True
    )
    materialize.add_argument("--samples", type=int, required=True)
    materialize.add_argument("--seed", type=int, default=1234)

    index_images = subparsers.add_parser(
        "index-images", help="Build a CLIP or DINOv2 image feature index"
    )
    index_images.add_argument("--manifest", required=True)
    index_images.add_argument("--output", required=True)
    index_images.add_argument("--image-field", default="image")
    index_images.add_argument("--encoder", choices=["clip", "dinov2"], required=True)
    index_images.add_argument("--model-id")
    index_images.add_argument("--batch-size", type=int, default=32)
    index_images.add_argument("--device", default="auto")

    neighbors = subparsers.add_parser(
        "nearest-neighbors",
        help="Run exact chunked generated-to-training memorization diagnostics",
    )
    neighbors.add_argument("--generated-index", required=True)
    neighbors.add_argument("--training-index", required=True)
    neighbors.add_argument("--output", required=True)
    neighbors.add_argument("--top-k", type=int, default=5)
    neighbors.add_argument("--chunk-size", type=int, default=16_384)

    prompt_suite = subparsers.add_parser(
        "generate-prompt-suite",
        help="Generate deterministic images for a JSONL prompt suite",
    )
    prompt_suite.add_argument("--checkpoint", required=True)
    prompt_suite.add_argument("--prompts", required=True)
    prompt_suite.add_argument("--output", required=True)
    prompt_suite.add_argument("--generations", type=int, default=1)
    prompt_suite.add_argument("--steps", type=int, default=30)
    prompt_suite.add_argument("--guidance", type=float, default=3.0)
    prompt_suite.add_argument("--seed", type=int, default=1234)
    prompt_suite.add_argument("--device", default="auto")
    prompt_suite.add_argument("--dtype", default="bfloat16")
    prompt_suite.add_argument("--batch-size", type=int, default=16)
    prompt_suite.add_argument("--geneval-layout", action="store_true")

    protocols = subparsers.add_parser(
        "write-protocol-prompts",
        help="Write the frozen human and safety/bias prompt protocols",
    )
    protocols.add_argument("--output", default="evaluation/prompts")

    profile = subparsers.add_parser(
        "profile-attention", help="Gate window attention using end-to-end speed"
    )
    profile.add_argument("--config", required=True)
    profile.add_argument("--output", default="reports/attention-profile.json")
    profile.add_argument("--warmup", type=int, default=5)
    profile.add_argument("--iterations", type=int, default=20)

    profile_batch = subparsers.add_parser(
        "profile-batch", help="Select the fastest safe physical training batch"
    )
    profile_batch.add_argument("--config", required=True)
    profile_batch.add_argument("--output", default="reports/batch-profile.json")
    profile_batch.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 96, 128]
    )
    profile_batch.add_argument("--warmup", type=int, default=5)
    profile_batch.add_argument("--iterations", type=int, default=20)

    profile_sweep = subparsers.add_parser(
        "profile-sweep-batches",
        help="Select safe physical batches for every recipe in a sweep",
    )
    profile_sweep.add_argument("--sweep", required=True)
    profile_sweep.add_argument("--cache-dir", required=True)
    profile_sweep.add_argument("--output", default="reports/sweep-batch-profiles")
    profile_sweep.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[32, 64, 96, 128]
    )
    profile_sweep.add_argument("--warmup", type=int, default=2)
    profile_sweep.add_argument("--iterations", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        checkpoint = train_from_config(load_config(args.config))
        print(checkpoint)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(load_config(args.config), args.checkpoint)
        print(json.dumps(result, indent=2))
    elif args.command == "sample":
        pipeline = HemeraPipeline.from_pretrained(
            args.checkpoint,
            device=args.device,
            dtype=args.dtype,
            load_vae=not args.latents_only,
        )
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
        result = pipeline(
            _prompts(args),
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            output_type="latent" if args.latents_only else "pil",
        )
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        if args.latents_only:
            torch.save(result.cpu(), output / "latents.pt")
            _save_latent_previews(result, output)
        else:
            for index, image in enumerate(result):
                image.save(output / f"sample-{index:03d}.png")
        print(output)
    elif args.command == "export":
        pipeline = HemeraPipeline.from_pretrained(
            args.checkpoint,
            device=args.device,
            dtype="float32",
            load_vae=False,
        )
        print(pipeline.save_pretrained(args.output))
    elif args.command == "audit-data":
        result = audit_photonyx(load_config(args.config), args.output)
        print(json.dumps(result, indent=2))
    elif args.command == "verify-audit":
        result = verify_audit_manifest(
            load_config(args.config), args.manifest, args.output
        )
        print(json.dumps(result, indent=2))
    elif args.command == "precompute":
        config = load_config(args.config)
        result = precompute_photonyx(
            config,
            args.manifest,
            args.output,
            resolve_device(args.device),
            args.batch_size,
            args.materialize_workers,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "cloud-preflight":
        config = load_config(args.config)
        config.data.max_items = args.items
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "manifest.jsonl"
        audit_photonyx(config, manifest)
        result = precompute_photonyx(
            config,
            manifest,
            root / "cache",
            resolve_device(config.train.device),
            args.batch_size,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "probe-representation":
        result = probe_representation(
            load_config(args.config),
            args.output,
            args.samples,
            args.batch_size,
            args.device,
            args.evaluator,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "sweep":
        print(json.dumps(run_sweep(args.config), indent=2))
    elif args.command == "report":
        print(write_report(args.runs, args.output))
    elif args.command == "rank":
        print(write_screening_ranking(args.runs, args.output))
    elif args.command == "human-study-create":
        if args.model_a_manifest and args.model_b_manifest:
            create_blinded_human_study_from_manifests(
                args.model_a_manifest,
                args.model_b_manifest,
                args.output,
                args.seed,
                args.ratings_per_prompt,
                args.model_a_image_root,
                args.model_b_image_root,
            )
        elif args.prompts and args.model_a_images and args.model_b_images:
            prompts = Path(args.prompts).read_text(encoding="utf-8").splitlines()
            model_a = Path(args.model_a_images).read_text(encoding="utf-8").splitlines()
            model_b = Path(args.model_b_images).read_text(encoding="utf-8").splitlines()
            create_blinded_human_study(
                prompts,
                model_a,
                model_b,
                args.output,
                args.seed,
                args.ratings_per_prompt,
            )
        else:
            raise ValueError(
                "Provide both model manifests, or prompts plus both image lists"
            )
        print(args.output)
    elif args.command == "human-study-analyze":
        result = analyze_human_study(args.input)
        rendered = json.dumps(result, indent=2)
        if args.output:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(rendered + "\n", encoding="utf-8")
            temporary.replace(target)
            print(target)
        else:
            print(rendered)
    elif args.command == "benchmark-images":
        result = evaluate_image_pairs(
            args.manifest,
            args.output,
            args.model_id,
            args.batch_size,
            args.device,
            args.siglip_model_id,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "compare-benchmarks":
        result = compare_paired_benchmarks(
            args.candidate,
            args.baseline,
            args.output,
            args.bootstrap_samples,
            args.seed,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "compare-confirmation":
        result = compare_confirmation_benchmarks(
            args.candidate,
            args.baseline,
            args.output,
            args.bootstrap_samples,
            args.seed,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "generate-eval":
        result = generate_evaluation_pairs(
            load_config(args.config),
            args.checkpoint,
            args.manifest,
            args.output,
            args.split,
            args.samples,
            args.steps,
            args.guidance,
            args.seed,
            args.batch_size,
            args.device,
            args.dtype,
            args.materialize_workers,
        )
        print(result)
    elif args.command == "materialize-images":
        print(
            materialize_audit_images(
                load_config(args.config),
                args.manifest,
                args.output,
                args.split,
                args.samples,
                args.seed,
            )
        )
    elif args.command == "index-images":
        print(
            json.dumps(
                build_image_feature_index(
                    args.manifest,
                    args.output,
                    args.image_field,
                    args.encoder,
                    args.model_id,
                    args.batch_size,
                    args.device,
                ),
                indent=2,
            )
        )
    elif args.command == "nearest-neighbors":
        print(
            json.dumps(
                audit_nearest_neighbors(
                    args.generated_index,
                    args.training_index,
                    args.output,
                    args.top_k,
                    args.chunk_size,
                ),
                indent=2,
            )
        )
    elif args.command == "generate-prompt-suite":
        print(
            generate_prompt_suite(
                args.checkpoint,
                args.prompts,
                args.output,
                args.generations,
                args.steps,
                args.guidance,
                args.seed,
                args.device,
                args.dtype,
                args.geneval_layout,
                args.batch_size,
            )
        )
    elif args.command == "write-protocol-prompts":
        print(json.dumps(write_protocol_prompts(args.output), indent=2))
    elif args.command == "profile-attention":
        result = profile_attention(
            load_config(args.config), args.output, args.warmup, args.iterations
        )
        print(json.dumps(result, indent=2))
    elif args.command == "profile-batch":
        result = profile_batch_sizes(
            load_config(args.config),
            args.output,
            args.batch_sizes,
            args.warmup,
            args.iterations,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "profile-sweep-batches":
        result = profile_sweep_batch_sizes(
            args.sweep,
            args.output,
            args.cache_dir,
            args.batch_sizes,
            args.warmup,
            args.iterations,
        )
        print(json.dumps(result, indent=2))
    else:
        raise AssertionError(args.command)
