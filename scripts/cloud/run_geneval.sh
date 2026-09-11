#!/usr/bin/env bash
set -euo pipefail
export PATH="/venv/main/bin:$PATH"

GENEVAL_DIR="${GENEVAL_DIR:?Set GENEVAL_DIR to a checkout of the official GenEval repository}"
GENEVAL_MODEL_PATH="${GENEVAL_MODEL_PATH:?Set GENEVAL_MODEL_PATH to the downloaded GenEval detector directory}"
CHECKPOINT="${1:?Usage: run_geneval.sh <checkpoint> [output-directory] [steps] [guidance]}"
OUTPUT="${2:-reports/geneval}"
STEPS="${3:-30}"
GUIDANCE="${4:-3}"
METADATA="$GENEVAL_DIR/prompts/evaluation_metadata.jsonl"

mkdir -p "$OUTPUT"
python run.py generate-prompt-suite \
  --checkpoint "$CHECKPOINT" \
  --prompts "$METADATA" \
  --output "$OUTPUT/generations" \
  --generations 4 \
  --steps "$STEPS" \
  --guidance "$GUIDANCE" \
  --seed 240113 \
  --geneval-layout
python "$GENEVAL_DIR/evaluation/evaluate_images.py" \
  "$OUTPUT/generations" \
  --outfile "$OUTPUT/results.jsonl" \
  --model-path "$GENEVAL_MODEL_PATH"
python "$GENEVAL_DIR/evaluation/summary_scores.py" "$OUTPUT/results.jsonl" \
  > "$OUTPUT/summary.txt"
