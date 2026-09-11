#!/usr/bin/env bash
set -uo pipefail

cd /workspace/hemera
export PATH="/venv/main/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HEMERA_HOURLY_EUR="${HEMERA_HOURLY_EUR:-0.348576158}"

rm -f reports/representations-exit-code
bash scripts/cloud/run_representation_probes.sh \
  data/manifests/photonyx.jsonl \
  2>&1 | tee reports/representations-live.log
status="${PIPESTATUS[0]}"
printf '%s\n' "$status" > reports/representations-exit-code
exit "$status"
