# Using Hemera-Nano locally

Hemera-Nano can generate from the bundled export at `model/hemera-nano` or
download the denoiser directly from `pauhidalgoo/hemera-nano` on Hugging Face.
It does not need Photonyx or any training cache. On first use, Hugging Face
also downloads and caches the frozen CLIP text encoder and DC-AE decoder.

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

For an NVIDIA GPU, install a CUDA-enabled PyTorch build appropriate for the
machine before installing the remaining requirements. Confirm the result with:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The pipeline uses CUDA automatically when available and otherwise uses CPU.
CPU execution forces float32 even when `bfloat16` is requested.

## Easiest option: persistent playground

```powershell
python scripts/local_playground.py --show
```

The model loads once. Enter any prompt at `Prompt>`, or change settings:

```text
/steps 30
/cfg 4
/seed 9000
/count 4
/negative blurry, malformed, text, watermark
/show on
/status
/quit
```

Images and a JSONL record of their exact parameters are saved under
`artifacts/local-playground/`. Seeds advance after each batch; use `/seed N`
to reproduce an earlier image.

Useful launch variants:

```powershell
# Explicit CPU mode
python scripts/local_playground.py --device cpu --dtype float32 --show

# CUDA, two variants for every prompt
python scripts/local_playground.py --device cuda --dtype bfloat16 --count 2 --show
```

## One command per prompt

```powershell
python run.py sample `
  --checkpoint model/hemera-nano `
  --prompt "a red fox in fresh snow, detailed wildlife photograph" `
  --negative-prompt "blurry, malformed, text, watermark" `
  --steps 15 `
  --guidance 5 `
  --seed 1234 `
  --device auto `
  --dtype bfloat16 `
  --output artifacts/my-samples/fox
```

Repeat `--prompt` to generate a batch:

```powershell
python run.py sample `
  --checkpoint pauhidalgoo/hemera-nano `
  --prompt "a watercolor harbor at dawn" `
  --prompt "a glass observatory on a snowy mountain" `
  --steps 15 --guidance 5 --seed 2026 `
  --output artifacts/my-samples/batch
```

## Python API

```python
from hemera.pipeline import HemeraPipeline

pipeline = HemeraPipeline.from_pretrained(
    "pauhidalgoo/hemera-nano",
    device="auto",
    dtype="bfloat16",
)

images = pipeline(
    [
        "a small robot tending a rooftop garden at sunrise",
        "an ink illustration of a lighthouse in a storm",
    ],
    negative_prompt="blurry, malformed, text, watermark",
    num_inference_steps=15,
    guidance_scale=5.0,
    seed=1234,
)
for index, image in enumerate(images):
    image.save(f"hemera-{index}.png")
```

For a batch where every image needs a different deterministic seed, pass one
`torch.Generator` per prompt instead of the single `seed` argument.

## Parameter guidance

| Parameter | Practical range | Guidance |
|---|---:|---|
| Inference steps / NFE | 10–50 | **15** is the locked validation winner. Try 30 for comparison; more steps are slower and are not guaranteed to improve rectified-flow samples. |
| CFG | 1–7 | **5** is the locked winner. Lower values are usually more varied/softer; high values can become saturated or brittle. |
| Seed | 0 and above | Same prompt, parameters, software, and device should reproduce the same result. Change it to explore composition. |
| Negative prompt | short phrase | Optional. Long negative prompts are truncated by CLIP just like positive prompts. |
| Count/batch size | 1–8 locally | Larger batches improve GPU throughput but consume more VRAM/RAM. |

Prompts should put the most important subject and relationship early. CLIP
accepts only 77 tokens, and this model uses a pooled text representation, so
long prose and exact multi-object spatial relationships are weak points.

## Verified local behavior

The exact publication export was tested end to end on the current CPU-only
Windows environment:

- first call, including CLIP/DC-AE downloads: approximately 3 minutes;
- subsequent cached 15-step, CFG-5 call: approximately 32 seconds;
- output size: 256×256 PNG.

An NVIDIA GPU should be substantially faster. The first-run downloads still
occur once per Hugging Face cache.
