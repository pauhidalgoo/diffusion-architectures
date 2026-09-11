---
library_name: hemera
pipeline_tag: text-to-image
license: apache-2.0
datasets:
- pauhidalgoo/photonyx
tags:
- diffusion
- diffusers
- rectified-flow
- text-to-image
- research
- small-model
---

# Hemera-Nano

![64 cherry-picked Hemera-Nano generations](assets/showcase/contact-sheet.png)

The grid above is deliberately cherry-picked after full-resolution visual
review. Exact prompts, negative prompts, seeds, CFG, NFE, hashes, and selection
notes are available in `assets/showcase/showcase.jsonl`.

Hemera-Nano is a 29,756,564-parameter, 256×256 text-to-image denoiser
trained from scratch on Photonyx under a micro-budget. Frozen pretrained
representation models provide image latents and text embeddings; only the
denoiser was trained.

This release is a completed research artifact, but it does **not** pass every
pre-registered success criterion. The automated confirmation composite was
positive at 95% confidence, but the candidate did not win in every seed.
Blinded human preference ratings and official GenEval detector scoring also
remain pending.

## Model details

- Backbone: standard flat DiT with global self-attention and adaLN-Zero
- Conditioning: pooled frozen CLIP text representation
- Feed-forward network: dense SwiGLU
- Objective: rectified-flow velocity prediction with logit-normal timesteps
- Trainable parameters: 29,756,564
- Forward compute: 2.489 GFLOPs per sample
- Frozen VAE: `dc-ai/dc-ae-lite-f32c32-diffusers`
- Frozen text encoder: `openai/clip-vit-base-patch32`
- Native resolution: 256×256
- Training dataset: Photonyx revision
  `35978bff3d6f38f4e432c25506168aaf8bd7e26f`
- Audited examples: 427,260
- Training examples seen: 736,104,192
- Mean training throughput: 5,234 images/s on an RTX 5090
- Peak training VRAM: 11.47 GiB
- Released checkpoint: best-validation EMA at step 320,000
- Released checkpoint SHA-256:
  `7746f789384ba6349b2798d8959583e9e66aa06cb2e5231c9602d401ada8111d`
- Terminal checkpoint: step 958,469
- Terminal checkpoint SHA-256:
  `58d1996b6183844230a23102a0808e761cdb017c05490ded7de00b594ffdb4ea`
- Exported `model.safetensors` SHA-256:
  `3a033dd0ade8178003adaa43270711b382200ec6dffa6f4e565a6c9d4763c6ee`
- Locked inference: CFG 5.0, NFE 15

The best-validation checkpoint is the release default. The later terminal
checkpoint is retained for analysis and reproducibility, not presented as the
best model.

## Usage

Install the small Hemera inference package directly from its source repository.
The denoiser, VAE, and text encoder are downloaded and cached automatically on
first use.

```bash
pip install "git+https://github.com/pauhidalgoo/diffusion-architectures.git"
```

```python
from hemera import HemeraPipeline

pipeline = HemeraPipeline.from_pretrained(
    "pauhidalgoo/hemera-nano",
    device="auto",
    dtype="bfloat16",
)

images = pipeline(
    [
        "a crystal alpine lake reflecting snow capped mountains at sunrise",
        "blue hydrangea blossoms in a white vase, soft window light",
    ],
    negative_prompt="blurry, malformed, distorted, text, watermark, logo",
    num_inference_steps=15,
    guidance_scale=5.0,
    seed=1234,
)
images[0].save("alpine-lake.png")
images[1].save("hydrangeas.png")
```

The pipeline also supports prompt batches, per-prompt negative prompts,
`torch.Generator` objects, deterministic seeds, latent output, device
selection, and dtype selection.

## Training data

The pinned Photonyx audit accepted 427,260 rows:

| Split | Rows |
|---|---:|
| Train | 418,792 |
| Validation | 4,278 |
| Test | 4,190 |

| Source | Rows |
|---|---:|
| DiffusionDB | 231,217 |
| Safe Commons | 196,043 |

Accepted license labels were CC0, CC0-1.0, and public domain. The immutable
manifest hash is
`7d4fa4f570640747d957876a52b307bfd3aab460477bd74099c65a55d32692cf`.

## Locked test evaluation

Inference settings were selected from 15 CFG/NFE candidates on validation
before the test split was generated once. The realized test set contains every
available test row (4,190), despite the 5,000 requested-sample ceiling.

| Metric | Value |
|---|---:|
| Photonyx test CMMD | 0.55408 |
| Canonical FID | 21.83977 |
| Precision | 0.72840 |
| Recall | 0.20621 |
| HPSv2 | 0.19745 |
| CLIP alignment | 0.25158 |
| SigLIP alignment | 0.12078 |

These values characterize this release under the frozen Hemera protocol. They
should not be compared directly with numbers computed using different
datasets, feature encoders, image sizes, sample counts, or estimators.

### Per-source slices

| Source | Samples | CMMD | Precision | Recall | HPSv2 | CLIP |
|---|---:|---:|---:|---:|---:|---:|
| DiffusionDB | 2,335 | 0.50640 | 0.78630 | 0.29807 | 0.20252 | 0.25005 |
| Safe Commons | 1,855 | 1.00148 | 0.66361 | 0.09704 | 0.19108 | 0.25351 |

The large source gap is a material limitation and likely reflects differences
in captions, content, and image distributions.

## Confirmation and success status

The automated confirmation compared two matched dense recipes across seeds
101, 202, and 303 with 256 paired prompts and 2,000 cluster-bootstrap
resamples.

- Relative composite improvement: 0.00745
- 95% confidence interval: [0.00210, 0.01222]
- Candidate wins by seed: positive, positive, negative
- Pre-registered confirmation gate: **failed**

The checkpoint reload and automated final suite passed. The overall
pre-registered success claim remains false because:

1. the confirmation candidate did not beat its baseline in every seed;
2. blinded human preference ratings are not complete;
3. official GenEval detector scoring is not complete.

This release therefore reports a strong micro-budget artifact and useful
negative/positive results, not a fully confirmed state-of-the-art claim.

## Memorization diagnostics

The locked test generations were compared with a deterministic,
source-stratified subset of 50,000 training images. Similarity is diagnostic
only; a high embedding similarity is not proof of copying.

| Encoder | Mean nearest cosine | 99th percentile | Maximum |
|---|---:|---:|---:|
| CLIP ViT-L/14@336 | 0.82467 | 0.93448 | 0.95997 |
| DINOv2-small | 0.63442 | 0.86550 | 0.92881 |

The highest matches require blinded visual inspection before drawing any
memorization conclusion.

## Additional protocols

- Human-study candidate images: 200 across 10 balanced categories. Matched
  baseline generation, three-rater judgments, confidence intervals, and
  inter-rater agreement remain pending.
- Safety/bias audit images: 400 from 100 prompts. This suite documents behavior
  and does not establish safety.
- Official GenEval-format images: 2,212 (553 prompts × 4). Official detector
  scoring was not run because its pinned PyTorch 1.12/CUDA 11.3 environment is
  incompatible with RTX 5090 execution. The generations and evaluator revision
  are preserved for later isolated scoring.

## Cost

- Exploratory wall-clock estimate: €9.94
- Decision-sprint wall-clock estimate: €0.74
- Final training ledger: €20.25
- Complete final-instance wall-clock estimate, including evaluation and
  packaging: €23.05
- Total recorded project wall-clock estimate: €33.73

The original €10 exploratory target was met. The complete final instance
exceeded the original €20 target by about €3.05 but remained within the later
explicitly authorized contingency. These are estimates derived from recorded
hourly prices; stopped periods and provider billing details can make them
differ from the actual invoice.

## Intended use

Hemera-Nano is intended for research and education around micro-budget
text-to-image training. It is not intended for safety-critical,
identity-sensitive, medical, legal, or factual visual tasks.

## Limitations

Photonyx mixes real, artwork, and synthetic sources and may contain caption,
selection, aesthetic, demographic, and cultural biases. The dataset audit
retained only the two large sources present in the pinned release; smaller
sources described in the dataset README were not present in the audited
revision.

A small 256×256 model struggles with people, faces, hands, text, counting,
precise spatial relations, uncommon concepts, and multi-entity composition.
The deterministic uncurated sample grid shows strong performance on simple
animals, food, landscapes, architecture, and artistic media, alongside clear
failures on people and relational prompts.

The frozen CLIP encoder has a 77-token limit, and pooled conditioning discards
token-level spatial detail. Photonyx captions have a median length of 314
characters and a 95th percentile of 776 characters, so substantial prompt
truncation is expected. This likely limits compositional fidelity.

Outputs may reproduce stereotypes or unsafe associations present in the data
and frozen representation models. Automated safety prompts do not prove that
the model is safe.

## Licensing

Hemera's original code and released denoiser weights are Apache-2.0. Photonyx
is CC-BY as declared by its owner and is attributed by name and repository link
throughout this card. The released Safetensors file contains only Hemera's
29.76M-parameter denoiser; it does not redistribute CLIP or DC-AE weights.
Those frozen dependencies are downloaded separately from their upstream Hub
repositories and remain subject to their respective terms. The component-level
record and release rationale are in [`LICENSE_AUDIT.md`](LICENSE_AUDIT.md).

## Reproducibility

The verified release bundle contains the exact YAML configuration, immutable
dataset and prompt manifests, dependency lock, run manifest, source-tree hash,
checkpoint hashes, telemetry, validation sweep, locked inference record,
generated test pairs, metric features, milestone canaries, failure records,
cost ledgers, source archive, all checkpoints, and exported Safetensors.
