#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera

export HEMERA_HOURLY_EUR="${HEMERA_HOURLY_EUR:-0.348576158}"
export HEMERA_CONFIRMATION_SAMPLES="${HEMERA_CONFIRMATION_SAMPLES:-256}"
export HEMERA_GPU_DATA_CACHE="${HEMERA_GPU_DATA_CACHE:-1}"
# Validation-image materialization may need uncached Photonyx shards. Model
# weights remain cached locally, but the dataset transport must be allowed
# online unless the caller explicitly proves the full evaluation set is cached.
export HF_HUB_OFFLINE="${HEMERA_HF_DATA_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE=1

mkdir -p reports
rm -f reports/corrected-confirmation-exit-code

record_exit() {
  status=$?
  printf '%s\n' "${status}" > reports/corrected-confirmation-exit-code
}
trap record_exit EXIT

bash scripts/cloud/run_confirmation.sh
