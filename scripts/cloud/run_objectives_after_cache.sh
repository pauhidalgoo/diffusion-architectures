#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
mkdir -p logs reports
exec > >(tee -a logs/objective-pipeline.log) 2>&1
trap 'status=$?; printf "%s\n" "$status" > reports/objective-pipeline-exit-code' EXIT
rm -f reports/objective-pipeline-exit-code

while [[ ! -f reports/exploratory-cache-exit-code ]]; do
  sleep 15
done
if [[ "$(cat reports/exploratory-cache-exit-code)" != "0" ]]; then
  echo "Exploratory cache failed; refusing to spend objective budget." >&2
  exit 1
fi
python - <<'PY'
import json

index = json.load(open("data/cache/index.json", encoding="utf-8"))
progress = json.load(
    open("data/cache/precompute-progress.json", encoding="utf-8")
)
representation = index.get("representation", {})
if progress.get("stage") != "complete":
    raise SystemExit("Cache success marker exists but progress is not complete")
if index.get("statistics", {}).get("total_examples") != 30_000:
    raise SystemExit("Objective sweep requires exactly 30,000 cached examples")
if representation.get("selection_count") != 30_000:
    raise SystemExit("Cache selection contract is not the frozen 30k subset")
if not representation.get("repa_encoder"):
    raise SystemExit("Cache is missing frozen REPA targets")
PY

HEMERA_HOURLY_EUR="${HEMERA_HOURLY_EUR:-0.348576158}" \
  bash scripts/cloud/run_exploratory.sh objectives
