# Hemera decision sprint

Status: complete; final recipe frozen provisionally.

The sprint used one exploratory seed per arm, the same 20,000-example
source-temperature-0.5 cache, rectified flow, DC-AE + CLIP, 256 paired
validation prompts, 30 sampling steps, CFG 3, and an equal €0.12 training
budget per arm. End-to-end phase cost, including batch profiling and image
evaluation, was €0.7439 on one RTX 5090 at €0.348576158/hour.

## Image-space screening

| Arm | Parameters | Samples | Images/s | CMMD ↓ | HPSv2 ↑ | CLIP ↑ | Rank score ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 48M dense DiT | 47,585,012 | 4,460,544 | 3,780.83 | 1.40965 | 0.16400 | 0.21042 | 0.933 |
| 30M dense DiT | 29,756,564 | 5,706,240 | 4,836.86 | **1.37925** | 0.16279 | 0.20947 | 0.867 |
| 15M dense DiT | 15,137,052 | 10,314,240 | 8,744.39 | 1.51241 | 0.15908 | 0.20492 | 0.400 |
| 30M matched MoE | 29,393,004 total / 18,081,588 active | 5,924,352 | 5,021.95 | 1.43623 | 0.15709 | 0.20385 | 0.333 |
| 30M token cross-attention | 29,959,716 | 5,160,192 | 4,374.29 | 1.49131 | 0.15548 | 0.20415 | 0.267 |
| 30M matched MicroDiT | 29,324,861 | 8,911,872 | 7,554.53 | 1.54340 | 0.16048 | 0.20303 | 0.200 |

The ordinal rank score places 48M first because it leads HPSv2 and CLIP.
The paired analysis below shows that its small advantage is not distinguishable
from the 30M control, while 30M trains 27.9% faster.

## Paired-prompt uncertainty

Each comparison uses 2,000 bootstrap resamples of the same 256 prompts. Values
are relative composite improvement of the candidate over the named baseline.

| Candidate vs baseline | Observed | 95% CI | Interpretation |
|---|---:|---:|---|
| 30M dense vs 15M dense | +4.45% | +2.84% to +5.88% | 30M wins |
| 48M dense vs 30M dense | -0.34% | -1.95% to +1.26% | no detectable difference |
| MicroDiT vs 30M dense | -5.47% | -7.20% to -3.59% | dense wins |
| MoE vs 30M dense | -3.44% | -5.08% to -1.85% | dense wins |
| Token cross-attention vs pooled 30M dense | -5.05% | -6.81% to -3.29% | pooled dense wins |

## Decision

Freeze Hemera-Nano as a 29,756,564-parameter standard adaLN-Zero DiT with
pooled CLIP conditioning, dense SwiGLU, full SDPA, rectified flow, DC-AE
latents, and source-temperature-0.5 sampling.

The 15M arm gives up statistically clear quality. The 48M arm does not show a
paired quality improvement and processes 21.8% fewer images per second than
30M. Correctly parameter-matched MicroDiT, MoE, and token cross-attention all
lose to dense pooled conditioning with confidence intervals entirely below
zero. Dense also beat REPA in the earlier aggregate three-seed confirmation,
although that preregistered gate failed because it did not win every seed.

This is a cost-aware engineering decision, not a claim that 30M is universally
optimal. The new size/efficiency/conditioning arms have one training seed.
A publication-strength scaling claim would rerun at least 30M and 48M with
three seeds, but that is not required to begin the explicitly exploratory
€20 final run from scratch.

## Preserved artifacts

- `decision-sprint-ranking.json`: preregistered six-arm rank.
- `decision-sprint-comparisons/`: five paired bootstrap comparisons.
- `DECISION_SPRINT.md`: generated parameter, throughput, FLOP, VRAM, and latent
  diagnostic table.
- `cost-decision-sprint.json`: end-to-end cost ledger.
- `runs/decision-sprint/*`: exact configs, manifests, metrics, logs, hashes,
  paired features, and image manifests.
