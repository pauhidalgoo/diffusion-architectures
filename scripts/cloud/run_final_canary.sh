#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
source scripts/cloud/budget_guard.sh

TARGET_STEP="${HEMERA_CANARY_STEP:-10000}"
METRICS="runs/final/hemera-nano/metrics.jsonl"
CHECKPOINT="runs/final/hemera-nano/checkpoint-last.pt"
OUTPUT="runs/final/canary-step-${TARGET_STEP}"
PROMPTS="evaluation/prompts/final-canary-v1.txt"

mkdir -p reports
rm -f reports/final-canary-exit-code

while true; do
  if grep -q "\"step\": ${TARGET_STEP}, .*\"validation_loss\"" \
      "$METRICS" 2>/dev/null; then
    break
  fi
  if [[ -f reports/final-train-exit-code ]]; then
    echo "Training ended before canary step ${TARGET_STEP}" >&2
    printf '%s\n' 68 > reports/final-canary-exit-code
    exit 68
  fi
  sleep 15
done

train_pid="$(
  pgrep -f '^python run.py train --config configs/final-hemera-nano.yaml$' \
    | head -n 1
)"
if [[ -z "$train_pid" ]]; then
  echo "Could not identify the final training process" >&2
  printf '%s\n' 69 > reports/final-canary-exit-code
  exit 69
fi

resume_training() {
  kill -CONT "$train_pid" 2>/dev/null || true
}
on_exit() {
  status=$?
  resume_training
  printf '%s\n' "$status" > reports/final-canary-exit-code
  exit "$status"
}
trap on_exit EXIT
kill -STOP "$train_pid"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/venv/main/bin/python scripts/cloud/inspect_checkpoint.py "$CHECKPOINT" \
  --output reports/final-canary-checkpoint.json

checkpoint_step="$(
  /venv/main/bin/python -c \
    "import json; print(json.load(open('reports/final-canary-checkpoint.json'))['step'])"
)"
if (( checkpoint_step < TARGET_STEP )); then
  echo "Canary checkpoint is stale: ${checkpoint_step} < ${TARGET_STEP}" >&2
  exit 70
fi

/venv/main/bin/python run.py sample \
  --checkpoint "$CHECKPOINT" \
  --prompt-file "$PROMPTS" \
  --steps 30 \
  --guidance 3 \
  --seed 2026 \
  --dtype bfloat16 \
  --output "$OUTPUT"

/venv/main/bin/python - "$OUTPUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

root = Path(sys.argv[1])
paths = sorted(root.glob("sample-*.png"))
if len(paths) != 8:
    raise SystemExit(f"Expected eight canary images, found {len(paths)}")
rows = []
for path in paths:
    with Image.open(path) as image:
        image.load()
        if image.size != (256, 256) or image.mode != "RGB":
            raise SystemExit(
                f"Unexpected canary image contract: {path} "
                f"{image.size} {image.mode}"
            )
    rows.append(
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
Path("reports/final-canary-images.json").write_text(
    json.dumps({"images": rows}, indent=2), encoding="utf-8"
)
PY

resume_training
trap - EXIT
printf '%s\n' 0 > reports/final-canary-exit-code
echo "Final canary completed at checkpoint step ${checkpoint_step}."
