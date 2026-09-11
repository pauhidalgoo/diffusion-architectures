#!/usr/bin/env bash
set -euo pipefail

source scripts/cloud/budget_guard.sh
hemera_budget_init 10 "${HEMERA_HOURLY_EUR:-0.5}" exploratory
trap hemera_budget_report EXIT

MANIFEST="${1:-data/manifests/photonyx.jsonl}"
MATERIALIZE_WORKERS="${HEMERA_MATERIALIZE_WORKERS:-16}"
test -f "$MANIFEST" || {
  echo "Missing $MANIFEST; run audit-data first." >&2
  exit 1
}

hemera_run python run.py verify-audit \
  --config configs/audit-full.yaml \
  --manifest "$MANIFEST" \
  --output reports/photonyx-audit-verification.json

for name in sd-clip dc-clip sd-t5 dc-t5; do
  config="configs/representations/${name}.yaml"
  cache="data/representation-cache/${name}"
  case "$name" in
    sd-clip) precompute_batch=64 ;;
    dc-clip) precompute_batch=128 ;;
    sd-t5) precompute_batch=96 ;;
    dc-t5) precompute_batch=128 ;;
  esac
  hemera_run python run.py precompute --config "$config" \
    --manifest "$MANIFEST" --output "$cache" \
    --batch-size "$precompute_batch" \
    --materialize-workers "$MATERIALIZE_WORKERS"
  hemera_run python run.py probe-representation --config "$config" \
    --output "runs/exploratory/representations/${name}/representation-probe.json" \
    --samples 256 --batch-size 16
  hemera_run python run.py train --config "$config"
  hemera_run python run.py evaluate --config "$config" \
    --checkpoint "runs/exploratory/representations/${name}/checkpoint-last.pt"
done

bash scripts/cloud/evaluate_screening.sh \
  "$MANIFEST" runs/exploratory/representations reports/representation-ranking.json
hemera_run python run.py report --runs runs/exploratory/representations \
  --output reports/REPRESENTATIONS.md
