#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
source scripts/cloud/budget_guard.sh

CONFIG="${1:-configs/final-hemera-nano.yaml}"
OUTPUT_DIR="$(
  python -c "from hemera.config import load_config; print(load_config('$CONFIG').train.output_dir)"
)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HEMERA_HOURLY_EUR="${HEMERA_HOURLY_EUR:-0.4905}"
export HEMERA_PHASE_CAP_EUR="${HEMERA_PHASE_CAP_EUR:-23.54}"

hemera_budget_init "$HEMERA_PHASE_CAP_EUR" "$HEMERA_HOURLY_EUR" final

mkdir -p reports
on_exit() {
  status=$?
  printf '%s\n' "$status" > reports/final-postprocess-exit-code
  hemera_budget_report
  exit "$status"
}
trap on_exit EXIT

train_status="$(cat reports/final-train-exit-code)"
if [[ "$train_status" != "0" ]]; then
  echo "Final training did not complete successfully: exit $train_status" >&2
  exit 65
fi
if [[ ! -f "$OUTPUT_DIR/manifest.json" ]]; then
  echo "Final training manifest is missing" >&2
  exit 66
fi

CHECKPOINT="$OUTPUT_DIR/checkpoint-best.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  CHECKPOINT="$OUTPUT_DIR/checkpoint-last.pt"
fi

hemera_run python run.py evaluate \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT"
bash scripts/cloud/tune_and_evaluate.sh "$CONFIG" "$CHECKPOINT"
hemera_run python run.py report --runs runs --output reports/FINAL.md
hemera_budget_report
hemera_run python scripts/cloud/finalize_model_card.py \
  --config "$CONFIG" \
  --output MODEL_CARD.md
bash scripts/cloud/package_final_delivery.sh

echo "Final evaluation, locked inference sweep, export, and packaging completed."
