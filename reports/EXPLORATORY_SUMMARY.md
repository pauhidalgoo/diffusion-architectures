# Hemera exploratory findings

Status: exploratory phase complete; confirmation gate failed.

Cost: €9.9416 estimated on one RTX 5090 at €0.348576158/hour.

## Frozen stage decisions

| Stage | Selected result | Key evidence |
|---|---|---|
| Representation | DC-AE + CLIP | Best predeclared three-metric representation score |
| Objective | Rectified flow | CMMD 1.5439, HPSv2 0.1552, CLIP 0.2027 |
| Conditioning | Pooled adaLN | CMMD 1.4536, HPSv2 0.1598, CLIP 0.2055 |
| Backbone | Standard DiT | CMMD 1.4143, HPSv2 0.1660, CLIP 0.2086 |
| Attention | Full SDPA | Sliding window was 13.6% slower; training gate failed |
| Efficiency | Dense control | Screening score 0.9167; REPA was second at 0.8333 |
| Data sampling | Source temperature 0.5 | Won CMMD, HPSv2, and CLIP screening metrics |

The backbone decision uses decoded-image metrics. PixArt and U-ViT had stronger
latent diagnostics in some runs, but those diagnostics were not the
preregistered architecture-selection endpoint.

## Corrected confirmation

The valid comparison is dense versus REPA, trained from scratch with seeds 101,
202, and 303 under equal per-run euro budgets. Each checkpoint was evaluated on
the same 256 validation prompts. The earlier six dense-versus-dense runs were
caused by a stale confirmation YAML and are retained under
`runs/confirmatory-invalid-dense-vs-dense-20260727`.

The completed MicroDiT and MoE screening arms are also confounded: their actual
parameter counts (22.94M and 24.60M) were outside the planned ±5% match to the
29.76M dense control. Their measurements are preserved, but they must not be
used for a parameter-controlled claim. The sweep config is corrected to 29.32M
and 29.39M respectively for any future rerun.

Dense minus REPA:

| Metric | Observed improvement | 95% prompt-cluster bootstrap CI |
|---|---:|---:|
| CMMD | 0.01812 | -0.00215 to 0.03731 |
| HPSv2 | 0.00100 | 0.00038 to 0.00160 |
| CLIP | 0.00075 | -0.00050 to 0.00197 |
| Relative composite | 0.745% | 0.210% to 1.222% |

The relative composite improved significantly in aggregate, but dense did not
beat REPA in every seed: seed 303 had a -0.030% composite difference. The
preregistered confirmation gate therefore failed. Dense remains the practical
leader for a subsequent decision because it was faster, used less VRAM, and
had the stronger aggregate result, but it is not a confirmed universal winner.

## Reproducibility artifacts

- `confirmation-checkpoints.json`: six validated checkpoint hashes and configs.
- `confirmation-ranking.json`: per-seed image-space rankings.
- `confirmation-comparison.json`: clustered-bootstrap result and gate decision.
- `CONFIRMATION.md`: generated run-accounting table.
- `cost-exploratory.json`: final exploratory cost ledger.
