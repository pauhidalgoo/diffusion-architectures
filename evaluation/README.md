# Hemera evaluation protocols

The files in `prompts/` are frozen before architecture selection. Their SHA-256
digests are recorded in `protocol-manifest.json`; changing a prompt creates a
new protocol version.

## Human comparison

`human-study-v1.jsonl` contains 200 prompts, balanced across ten categories.
Generate one locked-seed image from Hemera-Nano and the strongest
parameter-matched baseline. Create three independently assigned ratings per
prompt with `human-study-create`. Raters see randomized left/right sides and
judge prompt alignment, visual quality, and overall preference separately.
They may choose left, right, or tie. The analysis reports win/tie/loss,
prompt-cluster bootstrap intervals, and Fleiss' kappa. Disclose rater
recruitment, compensation, exclusions, and missing responses in the report.
See [`../HUMAN_EVALUATION_GUIDE.md`](../HUMAN_EVALUATION_GUIDE.md) for the
baseline prerequisite, exact generation and assignment commands, and the
dependency-free local rating interface.

## Safety and demographic-bias audit

`safety-bias-v1.jsonl` contains 100 neutral or non-graphic prompts across ten
categories. Generate at least four independent seeds per prompt. Two reviewers
should code visible demographic attributes, prompt fidelity, image defects,
stereotypical associations, and inappropriate refusal or unsafe escalation.
Report category-wise counts and representative failure modes. Do not collapse
this suite into a model-ranking score and do not describe passing it as proof
that the model is safe or unbiased.

The `unmarked_professions` prompts intentionally leave demographic attributes
unspecified to expose defaults. Identity-marked prompts test whether explicit,
benign descriptions are followed. The `ambiguous_safety` group tests benign
contexts that contain objects or activities sometimes over-blocked by safety
systems; it is not an unsafe-content benchmark.

## Memorization diagnostics

Materialize a deterministic, source-stratified training subset (50,000 examples
unless storage permits the full split), and index it independently with CLIP
ViT-L/14@336 and DINOv2-small. Index all locked test generations with the same
encoders, then run exact chunked top-5 search. Visually inspect the highest 100
matches for each encoder while blinded to similarity. Embedding similarity is
a triage signal, not proof of copying or its absence.

## GenEval

Use the unmodified official `evaluation_metadata.jsonl`, four images per prompt,
and the locked Hemera inference settings. `scripts/cloud/run_geneval.sh`
materializes the official directory layout and invokes the official evaluator.
Record the GenEval repository commit and detector checkpoint hash.
