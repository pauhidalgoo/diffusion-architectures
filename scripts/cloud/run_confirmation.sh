#!/usr/bin/env bash
set -euo pipefail

source scripts/cloud/budget_guard.sh
hemera_budget_init 10 "${HEMERA_HOURLY_EUR:-0.348576158}" exploratory
trap hemera_budget_report EXIT

CONFIG="${1:-configs/sweeps/confirmation.yaml}"
if python scripts/cloud/validate_completed_confirmation.py \
  --output reports/confirmation-checkpoints.json; then
  echo "All confirmation checkpoints are complete and checksum-verified; skipping training."
else
  hemera_run python run.py sweep --config "$CONFIG"
  python scripts/cloud/validate_completed_confirmation.py \
    --output reports/confirmation-checkpoints.json
fi
bash scripts/cloud/evaluate_screening.sh \
  data/manifests/photonyx.jsonl runs/confirmatory \
  reports/confirmation-ranking.json "${HEMERA_CONFIRMATION_SAMPLES:-512}"

candidate_args=()
baseline_args=()
for seed in 101 202 303; do
  candidate_args+=(--candidate "runs/confirmatory/finalist-a-seed-${seed}/image-eval/metrics.json")
  baseline_args+=(--baseline "runs/confirmatory/finalist-b-seed-${seed}/image-eval/metrics.json")
done
hemera_run python run.py compare-confirmation \
  "${candidate_args[@]}" "${baseline_args[@]}" \
  --output reports/confirmation-comparison.json \
  --bootstrap-samples 2000 --seed 20260726
hemera_run python run.py report --runs runs/confirmatory \
  --output reports/CONFIRMATION.md
