#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
source scripts/cloud/budget_guard.sh
hemera_budget_init 10 "${HEMERA_HOURLY_EUR:-0.348576158}" exploratory

mkdir -p logs reports
exec >> logs/exploratory-supervisor.log 2>&1

max_restarts="${HEMERA_SUPERVISOR_RESTARTS:-3}"
restarts=0

while true; do
  pipeline_pid="$(
    pgrep -f '^bash scripts/cloud/run_full_exploratory_pipeline.sh$' \
      | head -n 1 || true
  )"
  if [[ -n "$pipeline_pid" ]]; then
    echo "$(date --iso-8601=seconds) monitoring pipeline pid=$pipeline_pid"
    while kill -0 "$pipeline_pid" 2>/dev/null; do
      sleep 15
    done
  fi

  status="$(cat reports/full-pipeline-exit-code 2>/dev/null || echo 1)"
  if [[ "$status" == "0" ]]; then
    echo "$(date --iso-8601=seconds) exploratory pipeline completed"
    exit 0
  fi
  if (( $(hemera_remaining_seconds) <= 300 )); then
    echo "$(date --iso-8601=seconds) budget nearly exhausted; not restarting"
    exit 75
  fi
  if (( restarts >= max_restarts )); then
    echo "$(date --iso-8601=seconds) restart limit reached status=$status"
    exit "$status"
  fi

  restarts="$((restarts + 1))"
  echo "$(date --iso-8601=seconds) restarting resumable pipeline attempt=$restarts"
  bash scripts/cloud/run_full_exploratory_pipeline.sh || true
done
