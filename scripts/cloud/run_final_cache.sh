#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
source scripts/cloud/budget_guard.sh

export HEMERA_HOURLY_EUR="${HEMERA_HOURLY_EUR:-0.4905}"
export HEMERA_PHASE_CAP_EUR="${HEMERA_PHASE_CAP_EUR:-23.54}"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

hemera_budget_init \
  "${HEMERA_PHASE_CAP_EUR}" \
  "${HEMERA_HOURLY_EUR}" \
  final

record_exit() {
  status=$?
  printf '%s\n' "${status}" > reports/final-cache-exit-code
  hemera_budget_report
}
trap record_exit EXIT

python - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = Path("data/manifests/photonyx.jsonl")
summary = json.loads(
    Path("data/manifests/photonyx.summary.json").read_text(encoding="utf-8")
)
observed = hashlib.sha256(manifest.read_bytes()).hexdigest()
if observed != summary["manifest_sha256"]:
    raise SystemExit("Final audit manifest checksum mismatch")
if int(summary["accepted"]) != 427_260:
    raise SystemExit("Unexpected final audit row count")
print(f"Verified final audit manifest: {observed}")
PY

hemera_run python run.py precompute \
  --config configs/final-hemera-nano.yaml \
  --manifest data/manifests/photonyx.jsonl \
  --output data/final-cache \
  --batch-size "${HEMERA_PRECOMPUTE_BATCH:-128}" \
  --materialize-workers "${HEMERA_MATERIALIZE_WORKERS:-16}"
