"""Persistent interactive Hemera-Nano prompt playground.

The model and frozen encoders are loaded once, then prompts can be generated
repeatedly without paying model-loading overhead for every command.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hemera.pipeline import HemeraPipeline  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "model" / "hemera-nano"
DEFAULT_OUTPUT = ROOT / "artifacts" / "local-playground"


@dataclass
class Settings:
    steps: int = 15
    guidance: float = 5.0
    seed: int = 1234
    count: int = 1
    negative: str = "blurry, malformed, text, watermark"
    show: bool = False


HELP = """Commands:
  /status                 show current settings
  /steps N                set inference steps (recommended: 15, 30, or 50)
  /cfg X                  set classifier-free guidance (try 2 to 6)
  /seed N                 set the next deterministic seed
  /count N                generate 1 to 8 variants
  /negative TEXT          set the negative prompt; empty text clears it
  /show on|off            open generated images with the system viewer
  /help                   show this help
  /quit                   exit

Any other non-empty line is treated as a prompt.
"""


def apply_command(settings: Settings, line: str) -> str:
    parts = shlex.split(line)
    command = parts[0].lower()
    values = parts[1:]
    if command == "/steps":
        if len(values) != 1 or not 1 <= int(values[0]) <= 200:
            raise ValueError("steps must be an integer from 1 to 200")
        settings.steps = int(values[0])
    elif command == "/cfg":
        if len(values) != 1 or not 0 <= float(values[0]) <= 30:
            raise ValueError("CFG must be a number from 0 to 30")
        settings.guidance = float(values[0])
    elif command == "/seed":
        if len(values) != 1 or int(values[0]) < 0:
            raise ValueError("seed must be a non-negative integer")
        settings.seed = int(values[0])
    elif command == "/count":
        if len(values) != 1 or not 1 <= int(values[0]) <= 8:
            raise ValueError("count must be an integer from 1 to 8")
        settings.count = int(values[0])
    elif command == "/negative":
        settings.negative = " ".join(values)
    elif command == "/show":
        if len(values) != 1 or values[0].lower() not in {"on", "off"}:
            raise ValueError("show must be 'on' or 'off'")
        settings.show = values[0].lower() == "on"
    elif command in {"/status", "/help", "/quit"}:
        pass
    else:
        raise ValueError(f"unknown command: {command}")
    return command


def status(settings: Settings) -> str:
    return (
        f"steps={settings.steps}, cfg={settings.guidance:g}, "
        f"next_seed={settings.seed}, count={settings.count}, "
        f"negative={settings.negative!r}, show={settings.show}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Local export directory or Hugging Face model ID",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument(
        "--negative", default="blurry, malformed, text, watermark"
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        count=args.count,
        negative=args.negative,
        show=args.show,
    )
    # Reuse the same validation as interactive commands.
    for command in (
        f"/steps {settings.steps}",
        f"/cfg {settings.guidance}",
        f"/seed {settings.seed}",
        f"/count {settings.count}",
    ):
        apply_command(settings, command)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "generations.jsonl"
    checkpoint = args.checkpoint
    print(f"Loading Hemera from {checkpoint} ...")
    pipeline = HemeraPipeline.from_pretrained(
        checkpoint,
        device=args.device,
        dtype=args.dtype,
        load_vae=True,
    )
    print(f"Ready on {pipeline.device} ({pipeline.dtype}).")
    print(HELP)
    print(status(settings))

    while True:
        try:
            line = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if not line:
            continue
        if line.startswith("/"):
            try:
                command = apply_command(settings, line)
            except (ValueError, IndexError) as error:
                print(f"Invalid command: {error}")
                continue
            if command == "/quit":
                return
            if command == "/help":
                print(HELP)
            else:
                print(status(settings))
            continue

        seeds = list(range(settings.seed, settings.seed + settings.count))
        generators = [
            torch.Generator(device=pipeline.device).manual_seed(seed)
            for seed in seeds
        ]
        print(
            f"Generating {settings.count} image(s): "
            f"steps={settings.steps}, cfg={settings.guidance:g}, seeds={seeds}"
        )
        images = pipeline(
            [line] * settings.count,
            negative_prompt=settings.negative or None,
            num_inference_steps=settings.steps,
            guidance_scale=settings.guidance,
            generator=generators,
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for index, (seed, image) in enumerate(zip(seeds, images, strict=True)):
            path = output / f"{timestamp}-seed-{seed}-{index:02d}.png"
            image.save(path)
            record = {
                "timestamp_utc": timestamp,
                "prompt": line,
                "negative_prompt": settings.negative,
                "steps": settings.steps,
                "guidance": settings.guidance,
                "seed": seed,
                "image": str(path),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(path)
            if settings.show:
                image.show()
        settings.seed += settings.count


if __name__ == "__main__":
    main()
