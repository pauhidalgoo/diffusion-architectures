#!/usr/bin/env bash
set -euo pipefail

source scripts/cloud/budget_guard.sh
hemera_budget_init \
  "${HEMERA_PHASE_CAP_EUR:-23.54}" \
  "${HEMERA_HOURLY_EUR:-0.4905}" \
  final
trap hemera_budget_report EXIT

CONFIG="${1:-configs/final-hemera-nano.yaml}"
CHECKPOINT="${2:-runs/final/hemera-nano/checkpoint-best.pt}"
MANIFEST="${3:-data/manifests/photonyx.jsonl}"
EVAL_BATCH="${HEMERA_EVAL_BATCH:-64}"

if [[ ! -f "$CHECKPOINT" ]]; then
  CHECKPOINT="runs/final/hemera-nano/checkpoint-last.pt"
fi

for guidance in 1 2 3 4 5; do
  for nfe in 15 30 50; do
    tag="cfg-${guidance}-nfe-${nfe}"
    hemera_run python run.py generate-eval \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" \
      --output "runs/final/validation/$tag" --split validation --samples 128 \
      --steps "$nfe" --guidance "$guidance" --batch-size "$EVAL_BATCH"
    hemera_run python run.py benchmark-images \
      --manifest "runs/final/validation/$tag/pairs.jsonl" \
      --output "runs/final/validation/$tag/metrics.json"
  done
done

hemera_run python - <<'PY'
import hashlib
import json
from pathlib import Path

rows = []
for path in sorted(Path("runs/final/validation").glob("*/metrics.json")):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("hpsv2_mean") is None:
        raise SystemExit("HPSv2 is required before inference settings can be locked")
    parts = path.parent.name.split("-")
    rows.append({
        "guidance": float(parts[1]),
        "nfe": int(parts[3]),
        "cmmd": metrics["cmmd"],
        "hpsv2": metrics["hpsv2_mean"],
        "clip_score": metrics["clip_score_mean"],
    })
for metric, higher in (("cmmd", False), ("hpsv2", True), ("clip_score", True)):
    order = sorted(range(len(rows)), key=lambda i: rows[i][metric], reverse=higher)
    denominator = max(1, len(rows) - 1)
    for rank, index in enumerate(order):
        rows[index].setdefault("percentiles", {})[metric] = 1 - rank / denominator
for row in rows:
    row["selection_score"] = sum(row["percentiles"].values()) / 3
winner = max(rows, key=lambda row: row["selection_score"])
checkpoint = Path("runs/final/hemera-nano/checkpoint-best.pt")
if not checkpoint.exists():
    checkpoint = Path("runs/final/hemera-nano/checkpoint-last.pt")
winner.update({
    "selection_split": "validation",
    "candidate_count": len(rows),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
})
Path("runs/final/locked-inference.json").write_text(
    json.dumps({"winner": winner, "candidates": rows}, indent=2), encoding="utf-8"
)
PY

GUIDANCE="$(python -c "import json; print(json.load(open('runs/final/locked-inference.json'))['winner']['guidance'])")"
NFE="$(python -c "import json; print(json.load(open('runs/final/locked-inference.json'))['winner']['nfe'])")"

hemera_run python run.py generate-eval \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" \
  --output runs/final/test --split test --samples 5000 \
  --steps "$NFE" --guidance "$GUIDANCE" --batch-size "$EVAL_BATCH"
hemera_run python run.py benchmark-images \
  --manifest runs/final/test/pairs.jsonl \
  --output runs/final/test/metrics.json \
  --siglip-model-id google/siglip-base-patch16-256

# Memorization is a diagnostic, not a proof of copying. Compare every locked
# test generation against a deterministic, source-stratified 50k subset of the
# training split with both semantic (CLIP) and appearance (DINOv2) encoders.
# Cached source images are hard-linked, so this does not duplicate the 45 GB
# raw-image cache or re-download Photonyx.
hemera_run python run.py materialize-images \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --output data/memorization-train \
  --split train \
  --samples 50000 \
  --seed 1234
for encoder in clip dinov2; do
  hemera_run python run.py index-images \
    --manifest data/memorization-train/images.jsonl \
    --output "runs/final/memorization/training-${encoder}.json" \
    --encoder "$encoder" \
    --batch-size "$EVAL_BATCH"
  hemera_run python run.py index-images \
    --manifest runs/final/test/pairs.jsonl \
    --image-field generated_image \
    --output "runs/final/memorization/generated-${encoder}.json" \
    --encoder "$encoder" \
    --batch-size "$EVAL_BATCH"
  hemera_run python run.py nearest-neighbors \
    --generated-index "runs/final/memorization/generated-${encoder}.json" \
    --training-index "runs/final/memorization/training-${encoder}.json" \
    --output "runs/final/memorization/neighbors-${encoder}.json" \
    --top-k 5 \
    --chunk-size 16384
done

hemera_run python run.py generate-prompt-suite \
  --checkpoint "$CHECKPOINT" \
  --prompts evaluation/prompts/human-study-v1.jsonl \
  --output runs/final/human-study \
  --generations 1 \
  --steps "$NFE" \
  --guidance "$GUIDANCE" \
  --seed 240200 \
  --batch-size "$EVAL_BATCH"

hemera_run python run.py generate-prompt-suite \
  --checkpoint "$CHECKPOINT" \
  --prompts evaluation/prompts/safety-bias-v1.jsonl \
  --output runs/final/safety-bias \
  --generations 4 \
  --steps "$NFE" \
  --guidance "$GUIDANCE" \
  --seed 240300 \
  --batch-size "$EVAL_BATCH"

GENEVAL_DIR="${GENEVAL_DIR:-/workspace/geneval-official}"
GENEVAL_MODEL_PATH="${GENEVAL_MODEL_PATH:-/workspace/geneval-models}"
GENEVAL_METADATA="$GENEVAL_DIR/prompts/evaluation_metadata.jsonl"
if [[ -f "$GENEVAL_METADATA" ]]; then
  hemera_run python run.py generate-prompt-suite \
    --checkpoint "$CHECKPOINT" \
    --prompts "$GENEVAL_METADATA" \
    --output runs/final/geneval/generations \
    --generations 4 \
    --steps "$NFE" \
    --guidance "$GUIDANCE" \
    --seed 240113 \
    --batch-size "$EVAL_BATCH" \
    --geneval-layout
  git -C "$GENEVAL_DIR" rev-parse HEAD \
    > runs/final/geneval/official-commit.txt
  sha256sum "$GENEVAL_MODEL_PATH"/* \
    > runs/final/geneval/detector-SHA256SUMS
  python - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "official_evaluator_run": False,
    "generation_complete": True,
    "reason": (
        "The pinned official GenEval environment uses PyTorch 1.12/CUDA 11.3, "
        "while RTX 5090 execution requires a CUDA >=12.8 build. The official "
        "four-image-per-prompt suite is preserved for scoring in a compatible "
        "isolated evaluator; the training environment was not downgraded or "
        "mutated."
    ),
    "official_commit": Path(
        "runs/final/geneval/official-commit.txt"
    ).read_text(encoding="utf-8").strip(),
    "detector_sha256s": Path(
        "runs/final/geneval/detector-SHA256SUMS"
    ).read_text(encoding="utf-8").splitlines(),
}
target = Path("runs/final/geneval/evaluator-status.json")
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
os.replace(temporary, target)
PY
else
  echo "Official GenEval metadata is unavailable; generation was not run." \
    > reports/geneval-generation-not-run.txt
fi

hemera_run python run.py export \
  --checkpoint "$CHECKPOINT" --output runs/final/hemera-nano/export
