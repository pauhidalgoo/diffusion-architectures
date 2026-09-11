# Hemera: micro-budget text-to-image research plan

## Scope and naming

- **Photonyx** is the training dataset.
- **Hemera** is the research project and experimental framework.
- **Hemera-Nano** is the selected denoiser, with at most 50 million trainable
  parameters.

The research question is: which representation, objective, conditioning
mechanism, backbone, sampling policy, and training-efficiency method gives the
best 256×256 text-to-image quality per euro when the denoiser is initialized
from scratch?

## Outcome addendum

The study and final run are complete. The selected recipe is a 29.76M-parameter
standard DiT with DC-AE latents, pooled CLIP conditioning, rectified flow, full
SDPA, dense SwiGLU blocks, and source-temperature-0.5 sampling. Final training
processed 736,104,192 examples and the validation protocol selected CFG 5 with
15 inference steps.

The completed artifact does not pass the full pre-registered success rule. The
automated confirmation composite had a positive 95% confidence interval, but
the candidate failed to beat its matched baseline in one of three seeds.
Blinded human preference ratings and official GenEval detector scoring also
remain incomplete. The final result must therefore be described as a completed
micro-budget model and mixed research outcome, not a confirmed success under
the rule below.

Hard constraints:

- Photonyx-only generative training data.
- Frozen pretrained VAE and text encoder.
- €10 total exploratory rented compute.
- €20 total final-phase rented compute.
- RTX 5090 when competitively priced; otherwise the fastest CUDA GPU with at
  least 24GB VRAM.
- At least 250GB attached NVMe for the full workflow.
- No local Photonyx or pretrained-weight download.
- Automated comparisons are relative; important claims require blinded human
  evaluation and uncertainty intervals.

## Scientific principles

Every run must preserve its config, source revision, dependency versions, git
state, random seed, hardware, parameters, forward FLOPs, throughput, peak VRAM,
elapsed GPU time, estimated cost, sample count, and checkpoint hashes. Negative,
failed, and stopped runs remain in the ledger.

Screening is exploratory and uses one seed. Dense comparisons must remain within
±5% trainable parameters. Shared data splits, prompt sets, initialization
policy, augmentation, CFG dropout, effective optimizer batch, and evaluation
NFE are fixed unless the variable is the stated intervention. Physical
microbatches are profiled per architecture and gradient accumulation preserves
the shared effective batch. Report both equal-wall-time and equal-sample/FLOP
curves; neither alone is sufficient.

The main selection score is the mean percentile rank of:

1. validation CMMD, lower is better;
2. HPSv2, higher is better;
3. evaluator CLIP alignment, higher is better.

Throughput and then peak VRAM break near-ties. Representation reconstruction and
retrieval scores are diagnostics, not substitutes for downstream generation.
Confirmatory significance uses a paired prompt bootstrap. For each resample,
recompute CMMD and the mean HPSv2 and CLIP scores, express each candidate
improvement relative to the matched baseline, and average the three relative
improvements. The pre-registered automated gate passes only when the lower bound
of its two-sided 95% interval is above zero.

## Data protocol

The cloud audit pins Photonyx revision
`35978bff3d6f38f4e432c25506168aaf8bd7e26f` and streams the dataset. It rejects
corrupt images, empty prompts, unsupported licenses, repeated IDs, and
perceptual duplicates. A pHash cluster cannot cross a split. The immutable
split policy is approximately source-stratified 98/1/1 train/validation/test.
Published reports must include the realized counts by source and split.

Two train sampling policies are compared:

- raw row proportions;
- source-temperature sampling with source weight proportional to
  `source_count^0.5`.

Precomputation is resumable and atomic. Each cache index names its exact
representation contract. Latent posterior moments, token states, masks, pooled
states, and optional DINO targets are stored in checksum-verified Safetensors
with compressed Parquet metadata. On cloud machines, audited global row indices
are mapped back to the pinned Parquet shard layout and selected row groups are
materialized concurrently; every streamed ID and prompt is revalidated before
an image is atomically cached. Training selects one weighted shard per
microbatch to avoid random-read shard thrashing while preserving the desired
row distribution.

The exploratory cache is a deterministic 50,000-example source/split-stratified
subset of the full audited manifest. The four representation probes use the
same nested 20,000-example subset. Selected normalized RGB images are cached
once and reused across representations; the complete final phase overrides the
limit and uses every accepted training row.

## Local readiness gate

Local development uses generated RGB images, correlated synthetic latents, mock
text states, and tiny CPU models. It must not fetch Photonyx or pretrained
weights.

Paid execution is allowed only after:

- every backbone performs forward and backward passes;
- CFG, negative prompts, Euler/Heun sampling, masks, and routing are tested;
- the smoke model reduces a fixed-batch loss;
- interrupted training resumes bit-for-bit;
- checkpoint sampling and Safetensors export/reload are reproducible;
- sharded metadata, hashes, and configuration validation pass;
- a cloud preflight encodes several real Photonyx rows with the selected
  pretrained components.

## Representation study

The four combinations are:

| Image representation | Text representation |
|---|---|
| `stabilityai/sd-vae-ft-mse` | OpenAI CLIP ViT-B/32 token states |
| `stabilityai/sd-vae-ft-mse` | FLAN-T5-small token states |
| `dc-ai/dc-ae-lite-f32c32-diffusers` | OpenAI CLIP ViT-B/32 token states |
| `dc-ai/dc-ae-lite-f32c32-diffusers` | FLAN-T5-small token states |

Measure reconstruction LPIPS, PSNR, reconstruction FID, retrieval diagnostics,
storage per example, encode/decode throughput, and an equal-time generative
probe. The hypothesis is that deeper compression improves transformer
quality-per-euro only if its reconstruction loss does not dominate the small
denoiser’s capacity; token-level T5 may improve compositional alignment but its
longer sequence may lose on this budget. Log latent mean, global and
per-channel scale, and posterior standard deviation as well, so a
representation is not rejected because of an unnoticed normalization mismatch.

## €10 exploratory study

Approximate allocation:

| Phase | Share | Internal run caps |
|---|---:|---:|
| Setup, audit, and representation selection | measured | €3.29 at representation freeze |
| Objective and conditioning | revised | €1.45 |
| Six-backbone comparison | revised | €1.40 |
| Efficiency and data sampling | revised | €1.30 |
| Confirmation | protected | €1.35 |
| Cache building, evaluation, reporting, and transfer | reserve | remaining phase ledger |

The live allocation was revised after the full audit and paired representation
screening: setup through representation freeze consumed an estimated €3.29.
The remaining run caps sum to €5.50, leaving about €1.21 for full-cache
precomputation, profiling, image-level evaluation, reporting, and uploads. The
persistent phase watchdog, not nominal epochs, is authoritative.

Exploratory candidates keep a one-minute atomic-checkpoint margin; the
independent phase watchdog retains a further 60-second stop margin. The
20-minute reserve in the final config remains separate and is not repeatedly
withheld from every small screening arm.

For cost-limited runs, learning-rate warmup/cosine decay and MicroDiT's
unmasked finishing phase are parameterized by normalized usable-euro progress
(the run cap minus its shutdown reserve), not by the deliberately unreachable
step ceiling. Step-limited smoke and debugging runs retain ordinary
step-indexed schedules. This keeps equal-time comparisons trained across the
same complete schedule and guarantees that the final 10% of a MicroDiT run is
actually unmasked.

### Objective hypotheses

Using the same parameter-matched PixArt-style model:

- **ε prediction + Min-SNR:** a robust conventional control, but conversion to
  an inference velocity may be less stable for a tiny model.
- **v prediction + cosine path:** should balance noise levels and make endpoint
  behavior easier than ε prediction.
- **rectified-flow velocity + logit-normal timesteps:** may spend capacity more
  efficiently and sample well at low NFE.

### Conditioning hypotheses

The pooled, cross-attention, and joint-attention candidates use the same
PixArt-style flat backbone and are within 1% parameters:

- pooled text through adaLN is cheapest but may lose attributes and relations;
- token cross-attention should improve local prompt alignment;
- joint text/image self-attention may improve composition but spends quadratic
  compute on text tokens.

### Backbone hypotheses

All six controls are within 5% parameters:

1. latent convolutional U-Net with token cross-attention and no additional
   pooled-text path;
2. **standard DiT** with the original flat global-self-attention/adaLN-Zero
   structure, replacing only the class vector with a pooled frozen text vector;
3. PixArt-style DiT with token cross-attention;
4. U-ViT/XUT-style joint transformer with symmetric long skips;
5. MMDiT with modality-specific projections/FFNs and joint attention;
6. Sana-style linear DiT with Mix-FFN.

Standard DiT is the architectural control. It must not silently receive token
cross-attention or U-shaped skips.

### Efficiency hypotheses

Apply these to the strongest transformer, parameter-matched within 5%:

- dense SwiGLU with full PyTorch SDPA;
- MicroDiT-style 50% image-patch masking with local patch mixing and a short
  unmasked finishing phase;
- TREAD-style token removal and deeper reintroduction;
- REPA alignment to precomputed DINOv2-small patch targets;
- expert-choice MoE in alternate FFNs, with fixed per-expert capacity;
- sliding-window attention only if an end-to-end optimizer-step profile at the
  selected latent length measures at least 15% speedup.

For token reduction, text tokens are never accidentally removed. Report total
parameters, routed/active compute, actual step time, and quality—not theoretical
attention complexity alone.

### Confirmation

Freeze all screening results, then substitute the top two complete recipes into
`configs/sweeps/confirmation.yaml`. Reinitialize each recipe with seeds 101,
202, and 303. Select a final recipe only if it wins in every seed and its paired
bootstrap automated improvement is positive at 95% confidence.

## €20 final phase

Before starting, freeze `configs/final-hemera-nano.yaml`. Reinitialize the
chosen ≤50M denoiser; exploratory weights are forbidden. Train on the complete
cleaned training split and selected source policy.

The phase guard charges bootstrap, preprocessing, training, evaluation, and
upload time. Training receives at most €16 so evaluation and transfer retain a
reserve. Save budget milestones near 25%, 50%, 75%, and 95%, plus the best
validation EMA checkpoint. Stop for cost, never to finish an epoch.

On validation only, evaluate the full grid:

- CFG `{1, 2, 3, 4, 5}`;
- NFE `{15, 30, 50}`.

Write the selected setting and checkpoint hash to `locked-inference.json`.
After that lock, generate the test set exactly once.

## Evaluation

Automated outputs:

- Photonyx test CMMD as primary distribution metric, matching the official
  ViT-L/14@336, L2-normalized-embedding, biased-estimator, σ=10, ×1000
  implementation;
- canonical FID, manifold precision/recall, HPSv2, evaluator CLIP/SigLIP
  alignment, and per-source slices;
- GenEval with four generations per prompt;
- parameter count, forward FLOPs/NFE, throughput, latency, peak VRAM, and
  quality-versus-euro curves;
- DINO and CLIP nearest-neighbor searches against training examples;
- a documented safety/demographic-bias prompt suite, explicitly not presented
  as proof of safety.
- a paired-caption diagnostic that holds latent noise and timestep fixed and
  reports the loss increase and prediction change under shuffled prompts.

Human evaluation uses 200 category-balanced prompts comparing Hemera-Nano with
the strongest matched baseline. Each item receives three blinded, independently
side-randomized ratings for prompt alignment, visual quality, and overall
preference. Report win/tie/loss, prompt-cluster bootstrap 95% intervals, and
Fleiss’ kappa.

Hemera-Nano is called successful only if:

- it beats the matched baseline in every confirmation seed;
- the paired automated-score improvement is positive at 95% confidence;
- the lower human-preference confidence bound exceeds 50%;
- the published checkpoint reloads reproducibly and completes the full suite.

Otherwise publish the negative result and identify the limiting factor.

## Publication artifacts

- source code and passing tests;
- exact configs and complete manifests, including failures;
- final and milestone weights;
- metrics, plots, tables, prompt sets, and sample grids;
- a paper-style report with hypotheses, methods, findings, limitations,
  licensing, bias, safety, and cost accounting;
- Hugging Face model card and reproducible Vast.ai instructions;
- Apache-2.0 original code;
- weights only after verifying compatibility across Photonyx, the chosen
  representation components, and evaluators.

## Primary references

- Peebles & Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748).
- Chen et al., [PixArt-α](https://arxiv.org/abs/2310.00426).
- Bao et al., [All are Worth Words: A ViT Backbone for Diffusion Models](https://arxiv.org/abs/2209.12152).
- Esser et al., [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).
- Xie et al., [SANA](https://arxiv.org/abs/2410.10629).
- Sehwag et al., [Stretching Each Dollar](https://openaccess.thecvf.com/content/CVPR2025/html/Sehwag_Stretching_Each_Dollar_Diffusion_Training_from_Scratch_on_a_Micro-Budget_CVPR_2025_paper.html).
- Krause et al., [TREAD](https://arxiv.org/abs/2501.04765).
- Yu et al., [REPA](https://openreview.net/forum?id=DJSZGGZYVi).
- Zhou et al., [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368).
- Hang et al., [Efficient Diffusion Training via Min-SNR Weighting](https://arxiv.org/abs/2303.09556).
- Jayasumana et al., [Rethinking FID: Towards a Better Evaluation Metric for Image Generation](https://arxiv.org/abs/2401.09603).
- Ghosh et al., [GenEval](https://arxiv.org/abs/2310.11513).
- Xie et al., [SANA 1.5](https://arxiv.org/abs/2501.18427).
- Wang et al., [REPA Works Until It Doesn’t / HASTE](https://openreview.net/forum?id=HK96GI5s7G).
- Becker et al., [EDiT](https://arxiv.org/abs/2503.16726).
- Yeh, [Home-made Diffusion Model from Scratch to Hatch](https://arxiv.org/abs/2509.06068).

## Completion checklist

- [x] Local synthetic, model, cache, sampling, export, and exact-resume tests pass.
- [x] Cloud preflight passes on real Photonyx rows.
- [x] Representation probes and diagnostics complete; DC-AE + CLIP frozen.
- [x] Objective, conditioning, backbone, efficiency, and data sweeps complete.
- [x] Top two recipes confirmed with three fresh seeds; the strict gate failed.
- [x] Final config frozen before final training.
- [x] Final training and validation-only inference tuning complete.
- [x] Locked test, memorization diagnostics, safety/bias generation, and
  GenEval-format generation complete.
- [ ] Blinded human ratings and official GenEval detector scoring complete.
- [x] Weights, logs, manifests, report, model card, and verified delivery bundle
  assembled locally.
- [x] Preliminary component and provenance license audit documented.
- [ ] Audit findings resolved, weight-license decision approved, and release
  published.
