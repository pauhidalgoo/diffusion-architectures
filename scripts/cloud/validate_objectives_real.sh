#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
source scripts/cloud/budget_guard.sh
hemera_budget_init 10 "${HEMERA_HOURLY_EUR:-0.5}" exploratory
trap hemera_budget_report EXIT

for objective in epsilon v; do
  config="configs/gpu-preflight-${objective}.yaml"
  run_dir="preflight/gpu-${objective}-overfit"
  hemera_run python run.py train --config "$config"
  hemera_run python scripts/cloud/validate_real_run.py \
    --config "$config" \
    --checkpoint "$run_dir/checkpoint-last.pt" \
    > "$run_dir/objective-validation.json"
  hemera_run python run.py sample \
    --checkpoint "$run_dir/checkpoint-last.pt" \
    --prompt "a red fox resting in green grass" \
    --prompt "a blue ceramic vase on a wooden table" \
    --steps 15 --guidance 3 --seed 20260726 \
    --output "$run_dir/samples"
done
