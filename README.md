# Hemera-Nano

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20model-Hemera--Nano-yellow)](https://huggingface.co/pauhidalgoo/hemera-nano)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-Photonyx-blue)](https://huggingface.co/datasets/pauhidalgoo/photonyx)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Hemera is a from-scratch text-to-image research project built to answer a
deliberately difficult question: **how good can a 256×256 model become on a
micro-budget?** The released Hemera-Nano denoiser has 29.76M parameters and was
trained on [Photonyx](https://huggingface.co/datasets/pauhidalgoo/photonyx)
using frozen CLIP and DC-AE representations.

![A cherry-picked grid of 64 Hemera-Nano generations](assets/showcase/contact-sheet.png)

These are 64 visually reviewed, cherry-picked model outputs—not an uncurated
quality claim. Every image, prompt, negative prompt, seed, CFG value, NFE, hash,
and selection note is preserved in the [full showcase](assets/showcase/README.md)
and [machine-readable manifest](assets/showcase/showcase.jsonl).

## Generate an image in a few lines

Install Hemera directly from GitHub. The first run downloads the 114 MiB
Hemera-Nano denoiser from Hugging Face plus the frozen CLIP text encoder and
DC-AE decoder; subsequent runs use the local Hugging Face cache.

```bash
pip install "git+https://github.com/pauhidalgoo/diffusion-architectures.git@v1.0.0"
```

```python
from hemera import HemeraPipeline

pipe = HemeraPipeline.from_pretrained(
    "pauhidalgoo/hemera-nano",
    device="auto",       # CUDA when available, otherwise CPU
    dtype="bfloat16",    # CPU automatically uses float32
)

image = pipe(
    "a crystal alpine lake reflecting snow capped mountains at sunrise, golden mist",
    negative_prompt="blurry, malformed, distorted, text, watermark, logo",
    num_inference_steps=15,
    guidance_scale=5.0,
    seed=760004,
)[0]
image.save("hemera.png")
```

Or generate from the command line:

```bash
hemera sample \
  --checkpoint pauhidalgoo/hemera-nano \
  --prompt "blue hydrangeas in a white vase, soft window light" \
  --negative-prompt "blurry, malformed, text, watermark" \
  --steps 15 --guidance 5 --seed 1234 \
  --output samples/hydrangeas
```

The validation-locked defaults are **15 inference steps** and **CFG 5**. Change
the seed to explore compositions. The model is strongest on landscapes,
architecture, botanical subjects, still lifes, interiors, and painterly
illustrations; people, hands, legible text, counting, and precise multi-object
layouts are weak points.

## Clone and use the bundled weights

GitHub stores `model/hemera-nano/model.safetensors` with Git LFS because the
file exceeds GitHub's normal 100 MiB limit.

```bash
git lfs install
git clone https://github.com/pauhidalgoo/diffusion-architectures.git
cd diffusion-architectures
git lfs pull
pip install -e .

python run.py sample \
  --checkpoint model/hemera-nano \
  --prompt "sunlit terracotta arches casting long geometric shadows" \
  --negative-prompt "blurry, malformed, text, watermark" \
  --steps 15 --guidance 5 --seed 760600 \
  --output samples/arches
```

For an interactive session that keeps the model loaded:

```bash
python scripts/local_playground.py --show
```

Inside the playground, enter prompts directly or use `/steps 30`, `/cfg 4`,
`/seed 9000`, `/count 4`, `/negative ...`, `/status`, and `/quit`. Generated
images and their exact settings are logged locally. See [LOCAL_USE.md](LOCAL_USE.md)
for GPU setup, batching, reproducibility, and parameter guidance.

## The released model

- Architecture: flat adaLN-Zero DiT, global SDPA, dense SwiGLU
- Conditioning: pooled OpenAI CLIP ViT-B/32 text representation
- Objective: rectified-flow velocity prediction with logit-normal timesteps
- Latent representation: `dc-ai/dc-ae-lite-f32c32-diffusers`
- Denoiser parameters: 29,756,564
- Native output: 256×256 RGB
- Release checkpoint: best-validation EMA at step 320,000
- Training examples processed: 736,104,192
- Mean training throughput: 5,234 images/s on one RTX 5090
- Peak training VRAM: 11.47 GiB
- `model.safetensors` SHA-256:
  `3a033dd0ade8178003adaa43270711b382200ec6dffa6f4e565a6c9d4763c6ee`

| Locked Photonyx test metric | Value |
|---|---:|
| CMMD | 0.55408 |
| FID | 21.83977 |
| Precision / recall | 0.72840 / 0.20621 |
| HPSv2 | 0.19745 |
| CLIP / SigLIP alignment | 0.25158 / 0.12078 |

These are in-domain measurements under the frozen Hemera protocol. The
automated confirmation composite improved at 95% confidence, but the candidate
did not beat the matched baseline in all three seeds. Human preference ratings
and official GenEval detector scoring remain unfinished, so this is presented
as a strong micro-budget research artifact—not a state-of-the-art claim. Read
the [final report](FINAL_REPORT.md) and [model card](MODEL_CARD.md) for the full
methodology and limitations.

## Research framework

Hemera contains reproducible implementations and parameter-matched studies of:

- standard DiT, PixArt-style DiT, U-ViT/XUT, MMDiT, Sana-style linear DiT,
  and a convolutional latent U-Net;
- epsilon prediction with Min-SNR, v-prediction, and rectified flow;
- pooled adaLN, token cross-attention, and joint image/text attention;
- full and sliding-window SDPA, MicroDiT masking, TREAD routing, REPA,
  expert-choice MoE, dense SwiGLU, and source-temperature sampling.

The exploratory study selected DC-AE + CLIP, rectified flow, pooled adaLN,
standard DiT, full SDPA, dense SwiGLU, and source-temperature-0.5 sampling. See
the [research plan](RESEARCH_PLAN.md), [exploratory summary](reports/EXPLORATORY_SUMMARY.md),
and [decision-sprint summary](reports/DECISION_SPRINT_SUMMARY.md).

## Reproduce and test

Local tests use generated fixtures and do not download Photonyx:

```bash
pip install -r requirements.txt
python -m pytest -q
python run.py train --config configs/smoke.yaml
```

All **117 tests** pass, including tiny-model forward/backward, CFG, sampling,
masking, routing, atomic checkpointing, deterministic resume, export/reload,
showcase packaging, and synthetic overfitting. Exact cloud commands and budget
guards live under `scripts/cloud/`.

## Licensing and attribution

Hemera's original code and released denoiser weights are Apache-2.0. Photonyx
is CC-BY as declared by its owner; this release attributes and links the
dataset above. Hemera does not redistribute CLIP or DC-AE weights: the
Transformers and Diffusers libraries download those components from their
upstream repositories on first use, under their respective terms. See
[LICENSE_AUDIT.md](LICENSE_AUDIT.md) for the component-level release record.

The dataset and model can contain aesthetic, demographic, cultural, caption,
and source biases. Do not use Hemera for safety-critical, identity-sensitive,
medical, legal, or factual visual tasks.
