from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hemera.config import load_config
from hemera.data import _raw_cache_root, image_to_tensor
from hemera.encoders import VAEAdapter, build_text_encoder, torch_dtype


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile frozen VAE+text cache encoding on real normalized images."
    )
    parser.add_argument(
        "--config", action="append", required=True, help="Config to profile"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 96, 128]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Precompute profiling requires CUDA")
    device = torch.device("cuda")
    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "profiles": {},
    }
    for config_path in args.config:
        config = load_config(config_path)
        dtype = torch_dtype(config.representation.precompute_dtype)
        raw_paths = sorted(_raw_cache_root(config).glob("*.png"))
        if not raw_paths:
            # A preprocessing-contract version change intentionally creates a
            # new cache namespace. Throughput profiling may still use an older
            # normalized 256px fixture because it never writes cache contents.
            raw_paths = sorted(
                Path(config.data.raw_cache_dir).glob("**/*.png")
            )
        if not raw_paths:
            raise FileNotFoundError("No real normalized image fixture is cached")
        with Image.open(raw_paths[0]) as handle:
            image = handle.convert("RGB").copy()
        base_tensor = image_to_tensor(image, config.data.image_size)
        vae = VAEAdapter(config.representation, device, dtype)
        text_encoder = build_text_encoder(
            config.representation, config.data, device, dtype
        )
        results: dict[str, Any] = {}
        for batch_size in args.batch_sizes:
            try:
                images = base_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                prompts = [
                    f"a real preprocessing throughput fixture number {index}"
                    for index in range(batch_size)
                ]
                timings: list[float] = []
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                for iteration in range(args.warmup + args.iterations):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        mean, logvar = vae.encode(images)
                        condition = text_encoder.encode(
                            prompts, device, dtype
                        )
                    torch.cuda.synchronize(device)
                    if iteration >= args.warmup:
                        timings.append(time.perf_counter() - started)
                    del mean, logvar, condition
                mean_seconds = statistics.mean(timings)
                results[str(batch_size)] = {
                    "status": "ok",
                    "mean_seconds": mean_seconds,
                    "images_per_second": batch_size / mean_seconds,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
                }
            except (torch.OutOfMemoryError, RuntimeError) as error:
                if (
                    not isinstance(error, torch.OutOfMemoryError)
                    and "out of memory" not in str(error).lower()
                ):
                    raise
                results[str(batch_size)] = {
                    "status": "out_of_memory",
                    "error": str(error).splitlines()[0],
                }
                torch.cuda.empty_cache()
        successful = [
            (int(size), values)
            for size, values in results.items()
            if values["status"] == "ok"
        ]
        report["profiles"][config.name] = {
            "config": config_path,
            "vae": config.representation.vae_id,
            "text_encoder": config.representation.text_encoder_id,
            "fixture_image": str(raw_paths[0]),
            "results": results,
            "recommended_batch_size": max(
                successful, key=lambda item: item[1]["images_per_second"]
            )[0],
        }
        del vae, text_encoder
        torch.cuda.empty_cache()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
