#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
source scripts/cloud/budget_guard.sh

BASE_CONFIG="${1:-configs/final-hemera-nano.yaml}"
RESUME_CONFIG="${2:-configs/final-hemera-nano-resume.yaml}"
CHECKPOINT="${3:-runs/final/hemera-nano/checkpoint-last.pt}"
METRICS="runs/final/hemera-nano/metrics.jsonl"

export HEMERA_GPU_DATA_CACHE="${HEMERA_GPU_DATA_CACHE:-1}"
export HEMERA_SHARD_CACHE_GIB="${HEMERA_SHARD_CACHE_GIB:-8}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HEMERA_HOURLY_EUR="${HEMERA_HOURLY_EUR:-0.4905}"

hemera_budget_init \
  "${HEMERA_PHASE_CAP_EUR:-23.54}" \
  "$HEMERA_HOURLY_EUR" \
  final

mkdir -p reports
rm -f reports/final-train-exit-code \
  reports/final-postprocess-exit-code

on_exit() {
  status=$?
  printf '%s\n' "$status" > reports/final-train-exit-code
  hemera_budget_report
  exit "$status"
}
trap on_exit EXIT

if [[ ! -f "$CHECKPOINT" ]] || [[ ! -f "$METRICS" ]]; then
  echo "Final resume requires an existing checkpoint and metrics ledger" >&2
  exit 68
fi

/venv/main/bin/python scripts/cloud/inspect_checkpoint.py "$CHECKPOINT" \
  --output reports/final-resume-checkpoint.json

/venv/main/bin/python - \
  "$BASE_CONFIG" "$RESUME_CONFIG" "$CHECKPOINT" "$METRICS" <<'PY'
import json
import sys
from pathlib import Path

from hemera.config import load_config, save_config

base, destination, checkpoint, metrics_path = sys.argv[1:]
config = load_config(base)
rows = [
    json.loads(line)
    for line in Path(metrics_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
spent = max(
    (
        float(row["spent_eur"])
        for row in rows
        if row.get("spent_eur") is not None
    ),
    default=0.0,
)
config.train.resume = checkpoint
config.budget.already_spent_eur = max(
    float(config.budget.already_spent_eur), spent
)
save_config(config, destination)
integrity_path = Path("reports/final-resume-checkpoint.json")
integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
event = {
    "schema_version": 1,
    "resume_checkpoint": checkpoint,
    "resume_checkpoint_sha256": integrity["sha256"],
    "resume_from_step": int(integrity["step"]),
    "last_logged_step_before_resume": max(
        (int(row["step"]) for row in rows if row.get("step") is not None),
        default=0,
    ),
    "already_spent_eur": config.budget.already_spent_eur,
    "resume_config": destination,
}
target = Path("reports/final-resume-event.json")
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(event, indent=2), encoding="utf-8")
temporary.replace(target)
print(json.dumps(event, indent=2))
PY

hemera_run /venv/main/bin/python run.py train --config "$RESUME_CONFIG"
