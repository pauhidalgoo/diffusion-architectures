#!/usr/bin/env bash
set -euo pipefail

AUDIT_PID="${1:?Usage: wait_then_representations.sh <audit-pid>}"
while kill -0 "$AUDIT_PID" 2>/dev/null; do
  sleep 30
done

test -f data/manifests/photonyx.summary.json || {
  echo "Audit process ended without photonyx.summary.json" >&2
  exit 1
}

export PATH="/venv/main/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HEMERA_HOURLY_EUR="${HEMERA_HOURLY_EUR:-0.348576158}"

set +e
bash scripts/cloud/run_representation_probes.sh \
  data/manifests/photonyx.jsonl \
  2>&1 | tee reports/representations-live.log
status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$status" > reports/representations-exit-code
exit "$status"
