#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
source scripts/cloud/budget_guard.sh

export HEMERA_HOURLY_EUR="${HEMERA_HOURLY_EUR:-0.348576158}"
export HEMERA_GPU_DATA_CACHE="${HEMERA_GPU_DATA_CACHE:-1}"
export HEMERA_EVAL_BATCH="${HEMERA_EVAL_BATCH:-64}"
export HF_HUB_OFFLINE="${HEMERA_HF_DATA_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE=1
export HEMERA_BUDGET_CAP_EUR=1.5
export HEMERA_BUDGET_PHASE=decision-sprint

hemera_budget_init "${HEMERA_BUDGET_CAP_EUR}" "${HEMERA_HOURLY_EUR}" "${HEMERA_BUDGET_PHASE}"
mkdir -p reports/batch-profiles
rm -f reports/decision-sprint-exit-code

record_exit() {
  status=$?
  printf '%s\n' "${status}" > reports/decision-sprint-exit-code
  hemera_budget_report
}
trap record_exit EXIT

python run.py profile-sweep-batches \
  --sweep configs/sweeps/decision-sprint.yaml \
  --cache-dir data/cache \
  --output reports/batch-profiles/decision-sprint \
  --batch-sizes 64 96 128 192 256 384 768 \
  --warmup 2 --iterations 5

python scripts/cloud/freeze_stage.py apply-batches \
  --sweep configs/sweeps/decision-sprint.yaml \
  --profiles reports/batch-profiles/decision-sprint/summary.json

hemera_run python run.py sweep --config configs/sweeps/decision-sprint.yaml

bash scripts/cloud/evaluate_screening.sh \
  data/manifests/photonyx.jsonl \
  runs/decision-sprint \
  reports/decision-sprint-ranking.json \
  "${HEMERA_DECISION_SAMPLES:-256}"

hemera_run python run.py report \
  --runs runs/decision-sprint \
  --output reports/DECISION_SPRINT.md
