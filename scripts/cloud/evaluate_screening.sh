#!/usr/bin/env bash
set -euo pipefail

source scripts/cloud/budget_guard.sh
hemera_budget_init \
  "${HEMERA_BUDGET_CAP_EUR:-10}" \
  "${HEMERA_HOURLY_EUR:-0.348576158}" \
  "${HEMERA_BUDGET_PHASE:-exploratory}"
trap hemera_budget_report EXIT

MANIFEST="${1:-data/manifests/photonyx.jsonl}"
RUN_ROOT="${2:-runs/exploratory}"
RANKING_OUTPUT="${3:-reports/screening-ranking.json}"
SAMPLES="${4:-128}"

while IFS= read -r checkpoint; do
  run_dir="$(dirname "$checkpoint")"
  config="$run_dir/config.yaml"
  output="$run_dir/image-eval"
  test -f "$config" || continue
  if [[ -f "$output/metrics.json" ]]; then
    echo "Skipping completed image evaluation: $run_dir"
    continue
  fi
  hemera_run python run.py generate-eval \
    --config "$config" --checkpoint "$checkpoint" --manifest "$MANIFEST" \
    --output "$output" --split validation --samples "$SAMPLES" \
    --batch-size "${HEMERA_EVAL_BATCH:-64}" --materialize-workers 16 \
    --steps 30 --guidance 3
  hemera_run python run.py benchmark-images \
    --manifest "$output/pairs.jsonl" --output "$output/metrics.json"
done < <(
  find "$RUN_ROOT" -type d -name _failed_attempts -prune -o \
    -name checkpoint-last.pt -print | sort
)

hemera_run python run.py rank --runs "$RUN_ROOT" --output "$RANKING_OUTPUT"
