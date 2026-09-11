# Hemera-Nano human preference evaluation

This guide covers the frozen 200-prompt, three-rater, blinded pairwise
protocol. It is designed to produce the pre-registered alignment, visual
quality, and overall-preference statistics—not an informal sample survey.

## Important current limitation

The final delivery contains all 200 Hemera-Nano images, but it does **not**
contain images or a restorable checkpoint for a full-budget matched baseline.
Therefore the pre-registered comparative human gate cannot be completed from
the local artifacts alone.

The confirmatory baseline was the REPA recipe (`finalist-b`), but its
confirmatory checkpoints saw only about 2.5 million examples. Hemera-Nano saw
736.1 million examples. Comparing those checkpoints would mostly measure
training duration, so it must not be reported as the primary matched human
comparison.

For the defensible primary comparison, first train the REPA baseline with the
same frozen data, representation, parameter ceiling, budget or examples seen,
and validation-only inference tuning as Hemera-Nano. If a cheaper or external
comparator is used instead, label the study secondary and name the mismatch.

## 1. Extract the existing Hemera images

From the repository root in PowerShell:

```powershell
New-Item -ItemType Directory -Force artifacts/human-study/candidate
tar -xf artifacts/final-delivery/hemera-final-results.tar `
  -C artifacts/human-study/candidate `
  runs/final/human-study
```

The important files will be:

- `artifacts/human-study/candidate/runs/final/human-study/generations.jsonl`
- `artifacts/human-study/candidate/runs/final/human-study/images/`

These are the locked Hemera generations: CFG 5, 15 inference steps, base seed
240200, one generation for every frozen prompt.

## 2. Generate the baseline images

After producing a portable export of the appropriately trained baseline, tune
its CFG and inference steps on validation prompts only. Then freeze those
settings and generate the human suite. Use the same base seed so each pair
starts from corresponding noise:

```powershell
python run.py generate-prompt-suite `
  --checkpoint <PATH-TO-BASELINE-EXPORT> `
  --prompts evaluation/prompts/human-study-v1.jsonl `
  --output artifacts/human-study/baseline `
  --generations 1 `
  --steps <LOCKED-BASELINE-NFE> `
  --guidance <LOCKED-BASELINE-CFG> `
  --seed 240200 `
  --device auto `
  --dtype bfloat16 `
  --batch-size 16
```

Use `--batch-size 1` on CPU or if memory is tight. Do not tune parameters after
looking at the human-study images.

## 3. Create the blinded assignments

Model A is Hemera-Nano, so the analyzer's `model_a_*` fields are the Hemera
results. The image-root argument safely rebases the remote paths stored in the
downloaded candidate manifest:

```powershell
python run.py human-study-create `
  --model-a-manifest artifacts/human-study/candidate/runs/final/human-study/generations.jsonl `
  --model-a-image-root artifacts/human-study/candidate/runs/final/human-study/images `
  --model-b-manifest artifacts/human-study/baseline/generations.jsonl `
  --model-b-image-root artifacts/human-study/baseline/images `
  --output artifacts/human-study/study.csv `
  --seed 20260730 `
  --ratings-per-prompt 3
```

This creates:

- `study.csv`: randomized image sides and empty rating columns;
- `study.key.csv`: the private A/B unblinding key.

Do not show `study.key.csv` to raters, rename model folders to reveal their
identity, or inspect the key before ratings are frozen.

## 4. Collect three independent ratings

Each rater evaluates all 200 prompts in exactly one assignment slot. Use
pseudonymous IDs rather than names or email addresses.

Rater one:

```powershell
python scripts/human_rating_server.py `
  --study artifacts/human-study/study.csv `
  --rater-id rater-01 `
  --slot 0
```

Rater two:

```powershell
python scripts/human_rating_server.py `
  --study artifacts/human-study/study.csv `
  --rater-id rater-02 `
  --slot 1
```

Rater three:

```powershell
python scripts/human_rating_server.py `
  --study artifacts/human-study/study.csv `
  --rater-id rater-03 `
  --slot 2
```

The command opens a local browser at `http://127.0.0.1:8080`. It saves each
response atomically and resumes at the first incomplete item. A rater chooses
left, tie, or right separately for:

1. prompt alignment;
2. visual quality independent of the prompt;
3. overall preference.

Do not let one person fill all three slots. Raters should work independently
and should not discuss examples until all ratings are frozen. Record
recruitment, compensation, instructions, exclusions, missing responses, and
whether raters knew the project.

## 5. Analyze the completed study

```powershell
python run.py human-study-analyze `
  --input artifacts/human-study/study.csv `
  --output artifacts/human-study/results.json
```

The output reports win/tie/loss counts, Hemera win rate excluding ties,
prompt-cluster bootstrap 95% confidence intervals, and Fleiss' kappa for all
three dimensions.

Before using the headline result, verify:

- 200 rated prompts and 600 ratings per dimension;
- exactly three independent rater IDs;
- no missing or invalid votes;
- `model_a_wins + ties + model_a_losses = 600`;
- the comparison model and inference settings are disclosed;
- the unblinding key and final CSV are checksum-preserved.

The pre-registered human gate passes only if the lower endpoint of the
**overall-preference** 95% confidence interval exceeds 0.50. Alignment and
quality remain separate reported outcomes.
