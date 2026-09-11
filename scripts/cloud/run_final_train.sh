#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
source scripts/cloud/budget_guard.sh

CONFIG="${1:-configs/final-hemera-nano.yaml}"
CACHE_DIR="$(
  python -c "from hemera.config import load_config; print(load_config('$CONFIG').data.cache_dir)"
)"
OUTPUT_DIR="$(
  python -c "from hemera.config import load_config; print(load_config('$CONFIG').train.output_dir)"
)"

export HEMERA_GPU_DATA_CACHE="${HEMERA_GPU_DATA_CACHE:-1}"
export HEMERA_SHARD_CACHE_GIB="${HEMERA_SHARD_CACHE_GIB:-8}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

hemera_budget_init \
  "${HEMERA_PHASE_CAP_EUR:-23.54}" \
  "${HEMERA_HOURLY_EUR:-0.4905}" \
  final

mkdir -p reports
on_exit() {
  status=$?
  printf '%s\n' "$status" > reports/final-train-exit-code
  hemera_budget_report
  exit "$status"
}
trap on_exit EXIT

if [[ -e "$OUTPUT_DIR/checkpoint-last.pt" ]] ||
   [[ -e "$OUTPUT_DIR/checkpoint-best.pt" ]]; then
  echo "Refusing to overwrite or resume an existing final run: $OUTPUT_DIR" >&2
  exit 64
fi

python - "$CONFIG" "$CACHE_DIR/index.json" <<'PY'
import json
import sys
from pathlib import Path

from hemera.config import load_config
from hemera.runtime import sha256_file

config = load_config(sys.argv[1])
index_path = Path(sys.argv[2])
index = json.loads(index_path.read_text(encoding="utf-8"))
representation = index.get("representation", {})

if config.data.max_items is not None:
    raise SystemExit("Final config must leave data.max_items unset")
if int(representation.get("selection_count", 0)) != 427_260:
    raise SystemExit("Final cache must contain exactly 427,260 audited records")
if index.get("revision") != config.data.dataset_revision:
    raise SystemExit("Final cache dataset revision mismatch")
if representation.get("text_storage") != "pooled-token-v1":
    raise SystemExit("Final cache is not the frozen pooled-text representation")
if representation.get("vae") != config.representation.vae_id:
    raise SystemExit("Final cache VAE mismatch")
if representation.get("text_encoder") != config.representation.text_encoder_id:
    raise SystemExit("Final cache text encoder mismatch")

for shard in index["shards"]:
    tensor_path = index_path.parent / shard["file"]
    metadata_path = index_path.parent / shard["metadata"]
    if sha256_file(tensor_path) != shard["sha256"]:
        raise SystemExit(f"Tensor shard checksum mismatch: {tensor_path}")
    if sha256_file(metadata_path) != shard["metadata_sha256"]:
        raise SystemExit(f"Metadata shard checksum mismatch: {metadata_path}")
print("Verified every final-cache shard and the frozen representation contract.")
PY

hemera_run python run.py train --config "$CONFIG"
