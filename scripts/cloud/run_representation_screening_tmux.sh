#!/usr/bin/env bash
set -uo pipefail

cd /workspace/hemera
export PATH="/venv/main/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HEMERA_HOURLY_EUR="${HEMERA_HOURLY_EUR:-0.348576158}"

rm -f reports/representation-screening-exit-code
bash scripts/cloud/evaluate_screening.sh \
  data/manifests/photonyx.jsonl \
  runs/exploratory/representations \
  reports/representation-ranking.json \
  128 \
  2>&1 | tee reports/representation-screening-live.log
status="${PIPESTATUS[0]}"
printf '%s\n' "$status" > reports/representation-screening-exit-code
exit "$status"
