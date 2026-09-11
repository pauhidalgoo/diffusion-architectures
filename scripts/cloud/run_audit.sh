#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
source scripts/cloud/budget_guard.sh
hemera_budget_init 10 "${HEMERA_HOURLY_EUR:-0.5}" exploratory
trap hemera_budget_report EXIT

hemera_run python run.py audit-data \
  --config configs/audit-full.yaml \
  --output data/manifests/photonyx.jsonl
