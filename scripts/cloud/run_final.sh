#!/usr/bin/env bash
set -euo pipefail

source scripts/cloud/budget_guard.sh
hemera_budget_init \
  "${HEMERA_PHASE_CAP_EUR:-23.54}" \
  "${HEMERA_HOURLY_EUR:-0.4905}" \
  final
trap hemera_budget_report EXIT

CONFIG="${1:-configs/final-hemera-nano.yaml}"
PRECOMPUTE_BATCH="${HEMERA_PRECOMPUTE_BATCH:-128}"
MATERIALIZE_WORKERS="${HEMERA_MATERIALIZE_WORKERS:-16}"
CACHE_DIR="$(python -c "from hemera.config import load_config; print(load_config('$CONFIG').data.cache_dir)")"
if [[ ! -f "$CACHE_DIR/index.json" ]]; then
  if [[ -f data/manifests/photonyx.jsonl ]] &&
     [[ -f data/manifests/photonyx.summary.json ]]; then
    hemera_run python - "$CONFIG" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from hemera.config import load_config

config = load_config(sys.argv[1])
manifest = Path("data/manifests/photonyx.jsonl")
summary = json.loads(
    Path("data/manifests/photonyx.summary.json").read_text(encoding="utf-8")
)
observed = hashlib.sha256(manifest.read_bytes()).hexdigest()
if summary.get("dataset") != config.data.dataset_id:
    raise SystemExit("Preserved audit manifest dataset mismatch")
if summary.get("dataset_revision") != config.data.dataset_revision:
    raise SystemExit("Preserved audit manifest revision mismatch")
if summary.get("manifest_sha256") != observed:
    raise SystemExit("Preserved audit manifest checksum mismatch")
if int(summary.get("accepted", 0)) <= 50_000:
    raise SystemExit("Preserved audit manifest is not full-dataset")
print(
    f"Reusing verified audit manifest: {summary['accepted']} rows, "
    f"sha256={observed}"
)
PY
  else
    hemera_run python run.py audit-data --config "$CONFIG" \
      --output data/manifests/photonyx.jsonl
  fi
  hemera_run python run.py precompute --config "$CONFIG" \
    --manifest data/manifests/photonyx.jsonl --output "$CACHE_DIR" \
    --batch-size "$PRECOMPUTE_BATCH" \
    --materialize-workers "$MATERIALIZE_WORKERS"
fi

hemera_run python - "$CONFIG" "$CACHE_DIR/index.json" <<'PY'
import json
import sys
from hemera.config import load_config

config = load_config(sys.argv[1])
index = json.load(open(sys.argv[2], encoding="utf-8"))
representation = index.get("representation", {})
if config.data.max_items is not None:
    raise SystemExit("Final config must leave data.max_items unset")
if representation.get("selection_count", 0) <= 50_000:
    raise SystemExit(
        "Final cache is not full-dataset: selection_count must exceed 50,000"
    )
if index.get("revision") != config.data.dataset_revision:
    raise SystemExit("Final cache dataset revision mismatch")
PY

hemera_run python run.py train --config "$CONFIG"
hemera_run python run.py evaluate --config "$CONFIG" \
  --checkpoint runs/final/hemera-nano/checkpoint-last.pt
bash scripts/cloud/tune_and_evaluate.sh "$CONFIG"
hemera_run python run.py report --runs runs --output reports/FINAL.md

echo "Final local artifacts are ready. Sync them before terminating the instance."
